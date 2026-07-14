#!/usr/bin/env python3

from __future__ import annotations

"""Evaluate trained G1 grass policies and write one CSV row per episode."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

REQUESTED_HELP = any(arg in {"-h", "--help"} for arg in sys.argv[1:])

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[3]
RSL_RL_SCRIPT_DIR = REPO_ROOT / "scripts" / "rsl_rl"
if str(RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

import cli_args  # isort: skip


SCENARIOS = (
    "eval_flat_to_grass",
    "eval_mild_grass",
    "eval_wet_grass",
    "eval_soft_grass",
    "eval_hard_hidden_bumps",
    "eval_extreme_coupled",
)

CSV_COLUMNS = (
    "method",
    "train_seed",
    "eval_seed",
    "scenario",
    "episode",
    "success",
    "fall_count",
    "distance_m",
    "mean_tracking_error",
    "touchdown_timing_error_mean",
    "foot_slip_ratio",
    "missed_delayed_support_ratio",
    "stance_duration_deviation_mean",
    "unexpected_contact_count",
    "unexpected_contact_ratio",
    "contact_window_iou",
    "roll_rms",
    "pitch_rms",
    "base_ang_vel_rms",
    "com_height_fluctuation",
    "recovery_time_s",
    "ankle_action_mean",
    "ankle_action_max",
    "torque_peak",
    "torque_rms",
    "torque_saturation_ratio",
    "joint_limit_margin_min",
    "action_jerk",
    "compensation_phase_alignment",
    "compensation_efficiency",
)

GAIT_PERIOD_S = 0.8
GAIT_OFFSETS = (0.0, 0.5)
STANCE_THRESHOLD = 0.55
FOOT_SLIP_VELOCITY_THRESHOLD = 0.20
CONTACT_FORCE_THRESHOLD_N = 1.0
COMPENSATION_EFFICIENCY_EVENT_TYPES = ("foot_slip", "unexpected_contact", "missed_support")
COMPENSATION_EFFICIENCY_WINDOW_S = (-0.25, 0.50)
STABLE_TRACKING_ERROR = 0.20
UNSTABLE_TRACKING_ERROR = 0.35
STABLE_TILT_RAD = 0.20
UNSTABLE_TILT_RAD = 0.35


parser = argparse.ArgumentParser(description="Evaluate a trained G1 grass RSL-RL policy.")
parser.add_argument("--task", type=str, required=True, help="Name of the Gym task to evaluate.")
parser.add_argument("--method", type=str, required=True, help="Method label written to the CSV.")
parser.add_argument("--seed", type=int, required=True, help="Training seed for the checkpoint being evaluated.")
parser.add_argument(
    "--eval-seed",
    type=int,
    default=None,
    help="Evaluation random seed. Defaults to --seed when omitted.",
)
parser.add_argument("--episodes", type=int, default=200, help="Number of completed episodes to write.")
parser.add_argument("--scenario", type=str, required=True, choices=SCENARIOS, help="Evaluation scenario.")
parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel evaluation environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
if REQUESTED_HELP:
    parser.print_help()
    raise SystemExit(0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive.")
if args_cli.eval_seed is None:
    args_cli.eval_seed = args_cli.seed

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab_platform_compat import patch_conda_forge_sys_version_for_isaaclab

patch_conda_forge_sys_version_for_isaaclab()

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.research.g1_grass.g1_grass_env_cfg import GRASS_TERRAIN_SCHEDULE
from unitree_rl_lab.tasks.locomotion.research.g1_grass.risk_metrics import (
    action_jerk as action_jerk_metric,
)
from unitree_rl_lab.tasks.locomotion.research.g1_grass.risk_metrics import (
    joint_limit_margin as joint_limit_margin_metric,
)
from unitree_rl_lab.tasks.locomotion.research.g1_grass.risk_metrics import (
    torque_saturation_ratio as torque_saturation_ratio_metric,
)
from unitree_rl_lab.tasks.locomotion.research.g1_grass.terrain_cfg import GrassTerrainStage
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scenario_stage(scenario: str):
    schedule = GRASS_TERRAIN_SCHEDULE
    if scenario == "eval_flat_to_grass":
        return schedule.mild_grass
    if scenario == "eval_mild_grass":
        return schedule.mild_grass
    if scenario == "eval_wet_grass":
        return GrassTerrainStage(
            name="wet_grass_eval",
            lambda_level=0.75,
            height_range=schedule.wet_soft_grass.height_range,
            friction_range=(0.30, 0.60),
            stiffness_range=(0.60, 1.00),
            damping_range=(0.50, 1.30),
            transition_probability=schedule.wet_soft_grass.transition_probability,
        )
    if scenario == "eval_soft_grass":
        return GrassTerrainStage(
            name="soft_grass_eval",
            lambda_level=0.75,
            height_range=schedule.wet_soft_grass.height_range,
            friction_range=(0.55, 0.90),
            stiffness_range=(0.25, 0.55),
            damping_range=(1.20, 1.80),
            transition_probability=schedule.wet_soft_grass.transition_probability,
        )
    if scenario == "eval_hard_hidden_bumps":
        return GrassTerrainStage(
            name="hard_hidden_bumps_eval",
            lambda_level=0.80,
            height_range=(0.04, 0.10),
            friction_range=(0.55, 0.95),
            stiffness_range=(0.80, 1.20),
            damping_range=(0.50, 1.20),
            transition_probability=0.30,
        )
    if scenario == "eval_extreme_coupled":
        return schedule.extreme_coupled
    raise ValueError(f"Unknown scenario: {scenario}")


def apply_eval_scenario(env_cfg: Any, scenario: str) -> None:
    stage = scenario_stage(scenario)
    env_cfg.seed = args_cli.eval_seed
    env_cfg.grass_eval_scenario = scenario
    env_cfg.grass_eval_distribution = stage
    env_cfg.grass_eval_distribution_log = stage.to_dict()

    if hasattr(env_cfg, "grass_active_distribution"):
        env_cfg.grass_active_distribution = stage
    if hasattr(env_cfg, "grass_final_distribution"):
        env_cfg.grass_final_distribution = stage

    material = getattr(env_cfg.scene.terrain, "physics_material", None)
    if material is not None:
        friction_midpoint = 0.5 * (stage.friction_range[0] + stage.friction_range[1])
        material.static_friction = friction_midpoint
        material.dynamic_friction = friction_midpoint

    physics_material_event = getattr(env_cfg.events, "physics_material", None)
    if physics_material_event is not None:
        physics_material_event.params["static_friction_range"] = stage.friction_range
        physics_material_event.params["dynamic_friction_range"] = stage.friction_range

    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is not None and hasattr(terrain_generator, "difficulty_range"):
        if scenario == "eval_flat_to_grass":
            terrain_generator.difficulty_range = (0.0, stage.lambda_level)
        elif scenario == "eval_hard_hidden_bumps":
            terrain_generator.difficulty_range = (0.65, 0.90)
        else:
            terrain_generator.difficulty_range = (stage.lambda_level, stage.lambda_level)


def load_policy(env: RslRlVecEnvWrapper, agent_cfg: RslRlOnPolicyRunnerCfg, checkpoint: str):
    resume_path = retrieve_file_path(checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    return runner.get_inference_policy(device=env.unwrapped.device)


def normalize_observations(observations: Any) -> torch.Tensor:
    return observations[0] if isinstance(observations, tuple) else observations


def unpack_step(step_result: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if len(step_result) == 4:
        obs, reward, dones, extras = step_result
    elif len(step_result) == 5:
        obs, reward, terminated, truncated, extras = step_result
        dones = terminated | truncated
        extras = dict(extras)
        extras.setdefault("time_outs", truncated)
    else:
        raise RuntimeError(f"Unsupported env.step return length: {len(step_result)}")
    return normalize_observations(obs), reward, dones.reshape(-1).bool(), extras


def tensor_like(value: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def zeros(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(num_envs, device=device, dtype=torch.float32)


def event_window_step_counts(dt: float) -> tuple[int, int]:
    pre_window_s, post_window_s = COMPENSATION_EFFICIENCY_WINDOW_S
    pre_steps = max(1, int(math.ceil(abs(min(pre_window_s, 0.0)) / dt)))
    post_steps = max(1, int(math.ceil(max(post_window_s, dt) / dt)))
    return pre_steps, post_steps


def resolve_ids(entity: Any, finder_name: str, pattern: str) -> list[int]:
    finder = getattr(entity, finder_name, None)
    if finder is not None:
        try:
            ids, _ = finder(pattern)
        except TypeError:
            ids, _ = finder([pattern])
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        return list(ids)

    names = getattr(entity, "body_names", None) or getattr(entity, "joint_names", None)
    if names is None and hasattr(entity, "data"):
        names = getattr(entity.data, "body_names", None) or getattr(entity.data, "joint_names", None)
    if names is None:
        return []

    import re

    regex = re.compile(pattern)
    return [index for index, name in enumerate(names) if regex.fullmatch(name) or regex.search(name)]


def get_scene_sensor(unwrapped_env: Any, name: str) -> Any | None:
    sensors = getattr(unwrapped_env.scene, "sensors", None)
    if sensors is not None:
        if hasattr(sensors, "get"):
            sensor = sensors.get(name)
            if sensor is not None:
                return sensor
        try:
            return sensors[name]
        except Exception:
            pass
    try:
        return unwrapped_env.scene[name]
    except Exception:
        return None


def resolve_eval_ids(unwrapped_env: Any) -> dict[str, list[int]]:
    robot = unwrapped_env.scene["robot"]
    contact_sensor = get_scene_sensor(unwrapped_env, "contact_forces")

    foot_body_ids = resolve_ids(robot, "find_bodies", ".*ankle_roll.*")
    ankle_joint_ids = resolve_ids(robot, "find_joints", ".*ankle.*")
    if contact_sensor is not None:
        foot_contact_body_ids = resolve_ids(contact_sensor, "find_bodies", ".*ankle_roll.*")
    else:
        foot_contact_body_ids = []
    if not foot_contact_body_ids:
        foot_contact_body_ids = foot_body_ids

    return {
        "foot_body_ids": foot_body_ids,
        "foot_contact_body_ids": foot_contact_body_ids,
        "ankle_joint_ids": ankle_joint_ids,
    }


def quat_to_roll_pitch(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quat = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(quat_wxyz.dtype).eps)
    w, x, y, z = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def get_command(unwrapped_env: Any, name: str = "base_velocity") -> torch.Tensor:
    try:
        return unwrapped_env.command_manager.get_command(name)
    except Exception:
        return torch.zeros((unwrapped_env.num_envs, 3), device=unwrapped_env.device)


def get_torque(robot: Any) -> torch.Tensor:
    for attr_name in ("applied_torque", "applied_effort", "computed_torque", "computed_effort"):
        value = getattr(robot.data, attr_name, None)
        if value is not None:
            return value
    return torch.zeros_like(robot.data.joint_vel)


def get_torque_limit(robot: Any) -> torch.Tensor | None:
    for attr_name in (
        "torque_limits",
        "effort_limits",
        "joint_effort_limits",
        "soft_joint_effort_limits",
        "applied_torque_limits",
        "computed_torque_limits",
    ):
        value = getattr(robot.data, attr_name, None)
        if value is not None:
            return torch.abs(value)

    actuator_limits = []
    for actuator in getattr(robot, "actuators", {}).values():
        for attr_name in ("effort_limit", "effort_limit_sim", "_effort_y2", "_effort_y1"):
            value = getattr(actuator, attr_name, None)
            if value is not None:
                actuator_limits.append(value)
                break
    if actuator_limits:
        try:
            device = getattr(robot, "device", robot.data.joint_pos.device)
            return torch.cat([tensor_like(value, device=device).reshape(-1) for value in actuator_limits])
        except Exception:
            return None
    return None


def get_joint_pos_limits(robot: Any) -> torch.Tensor | None:
    value = getattr(robot.data, "soft_joint_pos_limits", None)
    if value is not None:
        return value
    value = getattr(robot.data, "joint_pos_limits", None)
    if value is not None:
        return value
    return None


def collect_state(unwrapped_env: Any, actions: torch.Tensor, eval_ids: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    robot = unwrapped_env.scene["robot"]
    device = unwrapped_env.device
    num_envs = unwrapped_env.num_envs

    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    root_lin_vel = getattr(robot.data, "root_lin_vel_b", getattr(robot.data, "root_lin_vel_w", None))
    root_ang_vel = getattr(robot.data, "root_ang_vel_b", getattr(robot.data, "root_ang_vel_w", None))
    if root_lin_vel is None:
        root_lin_vel = torch.zeros((num_envs, 3), device=device)
    if root_ang_vel is None:
        root_ang_vel = torch.zeros((num_envs, 3), device=device)

    roll, pitch = quat_to_roll_pitch(root_quat)
    command = get_command(unwrapped_env)
    torque = get_torque(robot)
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    foot_body_ids = eval_ids["foot_body_ids"]
    if foot_body_ids:
        foot_xy_velocity = robot.data.body_lin_vel_w[:, foot_body_ids, :2]
    else:
        foot_xy_velocity = torch.zeros((num_envs, 0, 2), device=device)

    contact_sensor = get_scene_sensor(unwrapped_env, "contact_forces")
    contact_body_ids = eval_ids["foot_contact_body_ids"]
    if contact_sensor is not None and contact_body_ids:
        contact_time = contact_sensor.data.current_contact_time[:, contact_body_ids]
        contact_mask = contact_time > 0.0
        net_forces_w = getattr(contact_sensor.data, "net_forces_w", None)
        if net_forces_w is not None:
            contact_force_z = torch.abs(net_forces_w[:, contact_body_ids, 2])
        else:
            contact_force_z = torch.full_like(contact_time, float("nan"))
        has_contact_observation = torch.ones(num_envs, device=device, dtype=torch.bool)
    elif foot_body_ids:
        contact_time = torch.zeros((num_envs, len(foot_body_ids)), device=device)
        contact_mask = torch.zeros_like(contact_time, dtype=torch.bool)
        contact_force_z = torch.full_like(contact_time, float("nan"))
        has_contact_observation = torch.zeros(num_envs, device=device, dtype=torch.bool)
    else:
        contact_time = torch.zeros((num_envs, 0), device=device)
        contact_mask = torch.zeros((num_envs, 0), device=device, dtype=torch.bool)
        contact_force_z = torch.zeros((num_envs, 0), device=device)
        has_contact_observation = torch.zeros(num_envs, device=device, dtype=torch.bool)

    return {
        "root_pos": root_pos,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "roll": roll,
        "pitch": pitch,
        "command": command,
        "actions": actions,
        "torque": torque,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "foot_xy_velocity": foot_xy_velocity,
        "foot_contact_time": contact_time,
        "foot_contact_mask": contact_mask,
        "foot_contact_force_z": contact_force_z,
        "has_contact_observation": has_contact_observation,
    }


def select_state(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    selected = {}
    for key, after_value in after.items():
        before_value = before[key]
        view_shape = (mask.shape[0],) + (1,) * (after_value.ndim - 1)
        selected[key] = torch.where(mask.reshape(view_shape), before_value, after_value)
    return selected


def gait_expectation(elapsed_s: torch.Tensor, num_feet: int) -> tuple[torch.Tensor, torch.Tensor]:
    if num_feet == 0:
        empty = torch.zeros((elapsed_s.shape[0], 0), device=elapsed_s.device)
        return empty.bool(), empty

    offsets = torch.tensor(GAIT_OFFSETS, device=elapsed_s.device, dtype=torch.float32)
    if num_feet != offsets.numel():
        repeats = int(math.ceil(num_feet / offsets.numel()))
        offsets = offsets.repeat(repeats)[:num_feet]

    phase = ((elapsed_s.unsqueeze(1) / GAIT_PERIOD_S) + offsets.unsqueeze(0)) % 1.0
    expected_contact = phase < STANCE_THRESHOLD
    expected_stance_elapsed = phase * GAIT_PERIOD_S
    return expected_contact, expected_stance_elapsed


def init_accumulators(
    num_envs: int,
    device: torch.device,
    root_pos: torch.Tensor,
    actions: torch.Tensor,
    dt: float,
) -> dict[str, torch.Tensor]:
    action_dim = actions.shape[1]
    pre_window_steps, post_window_steps = event_window_step_counts(dt)
    accum = {
        "elapsed_s": zeros(num_envs, device),
        "distance_m": zeros(num_envs, device),
        "tracking_error_sum": zeros(num_envs, device),
        "tracking_count": zeros(num_envs, device),
        "touchdown_timing_error_sum": zeros(num_envs, device),
        "touchdown_timing_error_count": zeros(num_envs, device),
        "stance_duration_deviation_sum": zeros(num_envs, device),
        "stance_duration_deviation_count": zeros(num_envs, device),
        "slip_count": zeros(num_envs, device),
        "contact_count": zeros(num_envs, device),
        "missed_support_count": zeros(num_envs, device),
        "expected_support_count": zeros(num_envs, device),
        "support_force_observed_count": zeros(num_envs, device),
        "unexpected_contact_count": zeros(num_envs, device),
        "unexpected_contact_denominator": zeros(num_envs, device),
        "contact_iou_intersection": zeros(num_envs, device),
        "contact_iou_union": zeros(num_envs, device),
        "contact_iou_observed_count": zeros(num_envs, device),
        "roll_sq_sum": zeros(num_envs, device),
        "pitch_sq_sum": zeros(num_envs, device),
        "base_ang_vel_sq_sum": zeros(num_envs, device),
        "posture_count": zeros(num_envs, device),
        "height_sum": zeros(num_envs, device),
        "height_sq_sum": zeros(num_envs, device),
        "height_count": zeros(num_envs, device),
        "recovery_time_s": zeros(num_envs, device),
        "ankle_action_abs_sum": zeros(num_envs, device),
        "ankle_action_count": zeros(num_envs, device),
        "ankle_action_max": zeros(num_envs, device),
        "torque_peak": zeros(num_envs, device),
        "torque_sq_sum": zeros(num_envs, device),
        "torque_count": zeros(num_envs, device),
        "torque_saturation_sum": zeros(num_envs, device),
        "torque_saturation_count": zeros(num_envs, device),
        "joint_limit_margin_min": torch.full((num_envs,), float("inf"), device=device),
        "action_jerk_sum": zeros(num_envs, device),
        "action_jerk_count": zeros(num_envs, device),
        "comp_energy_total": zeros(num_envs, device),
        "comp_energy_valid_window": zeros(num_envs, device),
        "event_pre_contact_history": torch.zeros((num_envs, pre_window_steps), device=device),
        "event_pre_posture_history": torch.zeros((num_envs, pre_window_steps), device=device),
        "event_pre_valid_history": torch.zeros((num_envs, pre_window_steps), device=device),
        "event_window_count": torch.zeros((num_envs, post_window_steps), device=device),
        "event_window_pre_contact_sum": torch.zeros((num_envs, post_window_steps), device=device),
        "event_window_pre_posture_sum": torch.zeros((num_envs, post_window_steps), device=device),
        "event_window_pre_count": torch.zeros((num_envs, post_window_steps), device=device),
        "event_window_post_contact_sum": torch.zeros((num_envs, post_window_steps), device=device),
        "event_window_post_posture_sum": torch.zeros((num_envs, post_window_steps), device=device),
        "event_window_joint_energy": torch.zeros((num_envs, post_window_steps), device=device),
        "event_window_post_count": torch.zeros((num_envs, post_window_steps), device=device),
        "event_completed_pre_contact_sum": zeros(num_envs, device),
        "event_completed_pre_posture_sum": zeros(num_envs, device),
        "event_completed_pre_count": zeros(num_envs, device),
        "event_completed_post_contact_sum": zeros(num_envs, device),
        "event_completed_post_posture_sum": zeros(num_envs, device),
        "event_completed_joint_energy": zeros(num_envs, device),
        "event_completed_post_count": zeros(num_envs, device),
        "recovery_active": torch.zeros(num_envs, device=device, dtype=torch.bool),
        "prev_root_xy": root_pos[:, :2].clone(),
        "prev_actions": actions.clone(),
        "prev_prev_actions": torch.zeros((num_envs, action_dim), device=device),
    }
    return accum


def reset_accumulator(accum: dict[str, torch.Tensor], env_id: int, root_pos: torch.Tensor, actions: torch.Tensor) -> None:
    for key, value in accum.items():
        if key in {"prev_root_xy", "prev_actions", "prev_prev_actions"}:
            continue
        if value.dtype == torch.bool:
            value[env_id] = False
        elif key == "joint_limit_margin_min":
            value[env_id] = float("inf")
        else:
            value[env_id] = 0.0
    accum["prev_root_xy"][env_id] = root_pos[env_id, :2]
    accum["prev_actions"][env_id] = actions[env_id]
    accum["prev_prev_actions"][env_id] = 0.0


def update_event_window_accumulators(
    accum: dict[str, torch.Tensor],
    contact_risk_step: torch.Tensor,
    posture_risk_step: torch.Tensor,
    joint_energy_step: torch.Tensor,
    event_count_step: torch.Tensor,
) -> None:
    event_count_step = event_count_step.to(dtype=contact_risk_step.dtype)
    pre_contact_history = accum["event_pre_contact_history"]
    pre_posture_history = accum["event_pre_posture_history"]
    pre_valid_history = accum["event_pre_valid_history"]

    pre_contact_sum = pre_contact_history.sum(dim=-1)
    pre_posture_sum = pre_posture_history.sum(dim=-1)
    pre_count = pre_valid_history.sum(dim=-1)

    if torch.any(event_count_step > 0.0):
        accum["event_window_count"][:, 0] += event_count_step
        accum["event_window_pre_contact_sum"][:, 0] += pre_contact_sum * event_count_step
        accum["event_window_pre_posture_sum"][:, 0] += pre_posture_sum * event_count_step
        accum["event_window_pre_count"][:, 0] += pre_count * event_count_step

    window_count = accum["event_window_count"]
    contact_view = contact_risk_step.reshape(-1, 1)
    posture_view = posture_risk_step.reshape(-1, 1)
    energy_view = joint_energy_step.reshape(-1, 1)
    accum["event_window_post_contact_sum"] += contact_view * window_count
    accum["event_window_post_posture_sum"] += posture_view * window_count
    accum["event_window_joint_energy"] += energy_view * window_count
    accum["event_window_post_count"] += window_count

    last_index = window_count.shape[1] - 1
    completed_pre_count = accum["event_window_pre_count"][:, last_index]
    completed_post_count = accum["event_window_post_count"][:, last_index]
    completed_valid = ((completed_pre_count > 0.0) & (completed_post_count > 0.0)).to(dtype=contact_risk_step.dtype)
    accum["event_completed_pre_contact_sum"] += accum["event_window_pre_contact_sum"][:, last_index] * completed_valid
    accum["event_completed_pre_posture_sum"] += accum["event_window_pre_posture_sum"][:, last_index] * completed_valid
    accum["event_completed_pre_count"] += completed_pre_count * completed_valid
    accum["event_completed_post_contact_sum"] += accum["event_window_post_contact_sum"][:, last_index] * completed_valid
    accum["event_completed_post_posture_sum"] += accum["event_window_post_posture_sum"][:, last_index] * completed_valid
    accum["event_completed_joint_energy"] += accum["event_window_joint_energy"][:, last_index] * completed_valid
    accum["event_completed_post_count"] += completed_post_count * completed_valid

    for key in (
        "event_window_count",
        "event_window_pre_contact_sum",
        "event_window_pre_posture_sum",
        "event_window_pre_count",
        "event_window_post_contact_sum",
        "event_window_post_posture_sum",
        "event_window_joint_energy",
        "event_window_post_count",
    ):
        value = accum[key]
        shifted = torch.zeros_like(value)
        if value.shape[1] > 1:
            shifted[:, 1:] = value[:, :-1]
        value[:] = shifted

    pre_contact_history[:, :-1] = pre_contact_history[:, 1:].clone()
    pre_contact_history[:, -1] = contact_risk_step
    pre_posture_history[:, :-1] = pre_posture_history[:, 1:].clone()
    pre_posture_history[:, -1] = posture_risk_step
    pre_valid_history[:, :-1] = pre_valid_history[:, 1:].clone()
    pre_valid_history[:, -1] = 1.0


def update_accumulators(
    accum: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    eval_ids: dict[str, list[int]],
    dt: float,
    torque_limit: torch.Tensor | None,
    joint_pos_limits: torch.Tensor | None,
) -> None:
    num_envs = state["root_pos"].shape[0]
    device = state["root_pos"].device
    actions = state["actions"]
    command = state["command"]
    actual_velocity = torch.cat((state["root_lin_vel"][:, :2], state["root_ang_vel"][:, 2:3]), dim=-1)
    tracking_error = torch.linalg.vector_norm(actual_velocity - command[:, :3], dim=-1)

    root_xy = state["root_pos"][:, :2]
    accum["distance_m"] += torch.linalg.vector_norm(root_xy - accum["prev_root_xy"], dim=-1)
    accum["prev_root_xy"][:] = root_xy
    accum["tracking_error_sum"] += tracking_error
    accum["tracking_count"] += 1.0

    num_feet = state["foot_contact_mask"].shape[1]
    expected_contact, expected_stance_elapsed = gait_expectation(accum["elapsed_s"], num_feet)
    contact_mask = state["foot_contact_mask"]
    contact_time = state["foot_contact_time"]
    contact_force_z = state["foot_contact_force_z"]
    has_contact_observation = state["has_contact_observation"].bool()
    contact_risk_step = zeros(num_envs, device)
    event_count_step = zeros(num_envs, device)

    if num_feet:
        foot_speed = torch.linalg.vector_norm(state["foot_xy_velocity"], dim=-1)
        slip_mask = contact_mask & (foot_speed > FOOT_SLIP_VELOCITY_THRESHOLD)
        slip_count_step = slip_mask.float().sum(dim=-1)
        contact_count_step = contact_mask.float().sum(dim=-1)
        accum["slip_count"] += slip_count_step
        accum["contact_count"] += contact_count_step

        timing_mask = expected_contact | contact_mask
        timing_error = torch.abs(contact_time - expected_stance_elapsed)
        timing_error_sum_step = (timing_error * timing_mask.float()).sum(dim=-1)
        timing_error_count_step = timing_mask.float().sum(dim=-1)
        accum["touchdown_timing_error_sum"] += timing_error_sum_step
        accum["touchdown_timing_error_count"] += timing_error_count_step

        stance_mask = expected_contact & contact_mask
        stance_error = torch.abs(contact_time - expected_stance_elapsed)
        accum["stance_duration_deviation_sum"] += (stance_error * stance_mask.float()).sum(dim=-1)
        accum["stance_duration_deviation_count"] += stance_mask.float().sum(dim=-1)

        unexpected_contact = contact_mask & ~expected_contact
        unexpected_contact_count_step = unexpected_contact.float().sum(dim=-1)
        accum["unexpected_contact_count"] += unexpected_contact_count_step
        # Denominator for E_unexpected: number of steps outside expected contact window
        swing_step_count = (~expected_contact).float().sum(dim=-1)
        accum["unexpected_contact_denominator"] += swing_step_count

        force_observed = torch.isfinite(contact_force_z)
        support_force_observed = force_observed.any(dim=-1)
        expected_with_force = expected_contact & force_observed
        missed_support = expected_with_force & (contact_force_z < CONTACT_FORCE_THRESHOLD_N)
        missed_support_count_step = missed_support.float().sum(dim=-1)
        expected_support_count_step = expected_with_force.float().sum(dim=-1)
        accum["missed_support_count"] += missed_support_count_step
        accum["expected_support_count"] += expected_support_count_step
        accum["support_force_observed_count"] += support_force_observed.float()

        real_contact_mask = torch.where(force_observed, contact_force_z > CONTACT_FORCE_THRESHOLD_N, contact_mask)
        observed_contact = has_contact_observation.reshape(num_envs, 1)
        contact_union = (expected_contact | real_contact_mask) & observed_contact
        contact_intersection = expected_contact & real_contact_mask & observed_contact
        accum["contact_iou_intersection"] += contact_intersection.float().sum(dim=-1)
        accum["contact_iou_union"] += contact_union.float().sum(dim=-1)
        accum["contact_iou_observed_count"] += has_contact_observation.float()

        touchdown_risk_step = torch.where(
            timing_error_count_step > 0.0,
            timing_error_sum_step / timing_error_count_step.clamp_min(1.0),
            zeros(num_envs, device),
        )
        slip_ratio_step = torch.where(
            contact_count_step > 0.0,
            slip_count_step / contact_count_step.clamp_min(1.0),
            zeros(num_envs, device),
        )
        missed_support_ratio_step = torch.where(
            expected_support_count_step > 0.0,
            missed_support_count_step / expected_support_count_step.clamp_min(1.0),
            zeros(num_envs, device),
        )
        unexpected_contact_ratio_step = torch.where(
            swing_step_count > 0.0,
            unexpected_contact_count_step / swing_step_count.clamp_min(1.0),
            zeros(num_envs, device),
        )
        contact_risk_step = (
            touchdown_risk_step + slip_ratio_step + unexpected_contact_ratio_step + missed_support_ratio_step
        ) / 4.0
        event_count_step = (
            slip_mask.any(dim=-1).float()
            + unexpected_contact.any(dim=-1).float()
            + missed_support.any(dim=-1).float()
        )
    else:
        foot_speed = torch.zeros((num_envs, 0), device=device)

    roll = state["roll"]
    pitch = state["pitch"]
    base_ang_vel_mag = torch.linalg.vector_norm(state["root_ang_vel"], dim=-1)
    posture_risk_step = torch.abs(roll) + torch.abs(pitch) + 0.1 * base_ang_vel_mag
    accum["roll_sq_sum"] += roll.square()
    accum["pitch_sq_sum"] += pitch.square()
    accum["base_ang_vel_sq_sum"] += base_ang_vel_mag.square()
    accum["posture_count"] += 1.0

    base_height = state["root_pos"][:, 2]
    accum["height_sum"] += base_height
    accum["height_sq_sum"] += base_height.square()
    accum["height_count"] += 1.0

    ankle_joint_ids = eval_ids["ankle_joint_ids"]
    if ankle_joint_ids and actions.shape[1] >= max(ankle_joint_ids) + 1:
        ankle_actions = actions[:, ankle_joint_ids]
    else:
        ankle_actions = actions
    accum["ankle_action_abs_sum"] += torch.abs(ankle_actions).sum(dim=-1)
    accum["ankle_action_count"] += ankle_actions.shape[1]
    accum["ankle_action_max"] = torch.maximum(accum["ankle_action_max"], torch.abs(ankle_actions).amax(dim=-1))

    torque = state["torque"]
    accum["torque_peak"] = torch.maximum(accum["torque_peak"], torch.abs(torque).amax(dim=-1))
    accum["torque_sq_sum"] += torque.square().sum(dim=-1)
    accum["torque_count"] += torque.shape[1]
    if torque_limit is not None:
        try:
            saturation = torque_saturation_ratio_metric(torque, torque_limit).reshape(-1)
            accum["torque_saturation_sum"] += saturation
            accum["torque_saturation_count"] += 1.0
        except Exception:
            pass

    if joint_pos_limits is not None:
        try:
            lower_limits = joint_pos_limits[:, :, 0] if joint_pos_limits.ndim == 3 else joint_pos_limits[:, 0]
            upper_limits = joint_pos_limits[:, :, 1] if joint_pos_limits.ndim == 3 else joint_pos_limits[:, 1]
            margin = joint_limit_margin_metric(state["joint_pos"], lower_limits, upper_limits).reshape(-1)
            accum["joint_limit_margin_min"] = torch.minimum(accum["joint_limit_margin_min"], margin)
        except Exception:
            pass

    jerk = action_jerk_metric(actions, accum["prev_actions"], accum["prev_prev_actions"]).reshape(-1)
    accum["action_jerk_sum"] += jerk
    accum["action_jerk_count"] += 1.0
    accum["prev_prev_actions"][:] = accum["prev_actions"]
    accum["prev_actions"][:] = actions

    comp_energy = (state["joint_vel"].abs() * torque.abs()).sum(dim=-1) * dt
    accum["comp_energy_total"] += comp_energy
    valid_window = (
        (expected_contact & contact_mask).any(dim=-1)
        if num_feet
        else torch.zeros(num_envs, device=device).bool()
    )
    accum["comp_energy_valid_window"] += comp_energy * valid_window.float()
    update_event_window_accumulators(accum, contact_risk_step, posture_risk_step, comp_energy, event_count_step)

    unstable = (
        (tracking_error > UNSTABLE_TRACKING_ERROR)
        | (torch.abs(roll) > UNSTABLE_TILT_RAD)
        | (torch.abs(pitch) > UNSTABLE_TILT_RAD)
    )
    stable = (
        (tracking_error < STABLE_TRACKING_ERROR)
        & (torch.abs(roll) < STABLE_TILT_RAD)
        & (torch.abs(pitch) < STABLE_TILT_RAD)
    )
    active = accum["recovery_active"] | unstable
    accum["recovery_time_s"] += active.float() * dt
    accum["recovery_active"][:] = active & ~stable
    accum["elapsed_s"] += dt


def safe_mean(sum_value: torch.Tensor, count_value: torch.Tensor, env_id: int) -> float:
    count = float(count_value[env_id].detach().cpu())
    if count <= 0.0:
        return float("nan")
    return float((sum_value[env_id] / count_value[env_id]).detach().cpu())


def safe_rms(sum_sq_value: torch.Tensor, count_value: torch.Tensor, env_id: int) -> float:
    count = float(count_value[env_id].detach().cpu())
    if count <= 0.0:
        return float("nan")
    return float(torch.sqrt(sum_sq_value[env_id] / count_value[env_id]).detach().cpu())


def safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, env_id: int) -> float:
    denom = float(denominator[env_id].detach().cpu())
    if denom <= 0.0:
        return float("nan")
    return float((numerator[env_id] / denominator[env_id]).detach().cpu())


def safe_ratio_with_observation(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    observed_count: torch.Tensor,
    env_id: int,
    *,
    empty_value: float,
) -> float:
    observed = float(observed_count[env_id].detach().cpu())
    if observed <= 0.0:
        return float("nan")
    denom = float(denominator[env_id].detach().cpu())
    if denom <= 0.0:
        return empty_value
    return float((numerator[env_id] / denominator[env_id]).detach().cpu())


def event_compensation_efficiency(accum: dict[str, torch.Tensor], env_id: int) -> float:
    pre_count = float(accum["event_completed_pre_count"][env_id].detach().cpu())
    post_count = float(accum["event_completed_post_count"][env_id].detach().cpu())
    joint_energy = float(accum["event_completed_joint_energy"][env_id].detach().cpu())
    if pre_count <= 0.0 or post_count <= 0.0 or joint_energy <= 0.0:
        return float("nan")

    pre_contact = float(accum["event_completed_pre_contact_sum"][env_id].detach().cpu()) / pre_count
    pre_posture = float(accum["event_completed_pre_posture_sum"][env_id].detach().cpu()) / pre_count
    post_contact = float(accum["event_completed_post_contact_sum"][env_id].detach().cpu()) / post_count
    post_posture = float(accum["event_completed_post_posture_sum"][env_id].detach().cpu()) / post_count
    return ((pre_contact - post_contact) + (pre_posture - post_posture)) / joint_energy


def scalar(value: torch.Tensor, env_id: int) -> float:
    output = float(value[env_id].detach().cpu())
    return output if math.isfinite(output) else float("nan")


def finalize_row(
    accum: dict[str, torch.Tensor],
    env_id: int,
    episode_number: int,
    success: bool,
) -> dict[str, Any]:
    height_mean = safe_mean(accum["height_sum"], accum["height_count"], env_id)
    height_sq_mean = safe_mean(accum["height_sq_sum"], accum["height_count"], env_id)
    if math.isnan(height_mean) or math.isnan(height_sq_mean):
        com_height_fluctuation = float("nan")
    else:
        com_height_fluctuation = math.sqrt(max(height_sq_mean - height_mean * height_mean, 0.0))

    joint_margin = scalar(accum["joint_limit_margin_min"], env_id)
    if math.isinf(joint_margin):
        joint_margin = float("nan")

    comp_phase_alignment = safe_ratio(accum["comp_energy_valid_window"], accum["comp_energy_total"], env_id)
    missed_delayed_support_ratio = safe_ratio_with_observation(
        accum["missed_support_count"],
        accum["expected_support_count"],
        accum["support_force_observed_count"],
        env_id,
        empty_value=0.0,
    )
    contact_window_iou = safe_ratio_with_observation(
        accum["contact_iou_intersection"],
        accum["contact_iou_union"],
        accum["contact_iou_observed_count"],
        env_id,
        empty_value=1.0,
    )

    return {
        "method": args_cli.method,
        "train_seed": args_cli.seed,
        "eval_seed": args_cli.eval_seed,
        "scenario": args_cli.scenario,
        "episode": episode_number,
        "success": int(success),
        "fall_count": int(not success),
        "distance_m": scalar(accum["distance_m"], env_id),
        "mean_tracking_error": safe_mean(accum["tracking_error_sum"], accum["tracking_count"], env_id),
        "touchdown_timing_error_mean": safe_mean(
            accum["touchdown_timing_error_sum"], accum["touchdown_timing_error_count"], env_id
        ),
        "foot_slip_ratio": safe_ratio(accum["slip_count"], accum["contact_count"], env_id),
        "missed_delayed_support_ratio": missed_delayed_support_ratio,
        "stance_duration_deviation_mean": safe_mean(
            accum["stance_duration_deviation_sum"], accum["stance_duration_deviation_count"], env_id
        ),
        "unexpected_contact_count": scalar(accum["unexpected_contact_count"], env_id),
        "unexpected_contact_ratio": safe_ratio(
            accum["unexpected_contact_count"], accum["unexpected_contact_denominator"], env_id
        ),
        "contact_window_iou": contact_window_iou,
        "roll_rms": safe_rms(accum["roll_sq_sum"], accum["posture_count"], env_id),
        "pitch_rms": safe_rms(accum["pitch_sq_sum"], accum["posture_count"], env_id),
        "base_ang_vel_rms": safe_rms(accum["base_ang_vel_sq_sum"], accum["posture_count"], env_id),
        "com_height_fluctuation": com_height_fluctuation,
        "recovery_time_s": scalar(accum["recovery_time_s"], env_id),
        "ankle_action_mean": safe_ratio(accum["ankle_action_abs_sum"], accum["ankle_action_count"], env_id),
        "ankle_action_max": scalar(accum["ankle_action_max"], env_id),
        "torque_peak": scalar(accum["torque_peak"], env_id),
        "torque_rms": safe_rms(accum["torque_sq_sum"], accum["torque_count"], env_id),
        "torque_saturation_ratio": safe_mean(
            accum["torque_saturation_sum"], accum["torque_saturation_count"], env_id
        ),
        "joint_limit_margin_min": joint_margin,
        "action_jerk": safe_mean(accum["action_jerk_sum"], accum["action_jerk_count"], env_id),
        "compensation_phase_alignment": comp_phase_alignment,
        "compensation_efficiency": event_compensation_efficiency(accum, env_id),
    }


def extract_time_outs(
    dones: torch.Tensor,
    extras: dict[str, Any],
    unwrapped_env: Any,
    pre_episode_lengths: torch.Tensor,
) -> torch.Tensor:
    for key in ("time_outs", "timeouts", "time_out"):
        value = extras.get(key)
        if value is not None:
            return value.reshape(-1).to(device=dones.device).bool()

    termination_manager = getattr(unwrapped_env, "termination_manager", None)
    if termination_manager is not None:
        for attr_name in ("time_outs", "_time_out_buf", "time_out_buf"):
            value = getattr(termination_manager, attr_name, None)
            if value is not None:
                return value.reshape(-1).to(device=dones.device).bool()

    max_episode_length = getattr(unwrapped_env, "max_episode_length", None)
    if max_episode_length is not None:
        return dones & (pre_episode_lengths >= int(max_episode_length) - 1)
    return torch.zeros_like(dones)


def write_header_if_needed(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def main() -> None:
    set_reproducible_seed(args_cli.eval_seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    apply_eval_scenario(env_cfg, args_cli.scenario)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.seed = args_cli.eval_seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        policy = load_policy(env, agent_cfg, args_cli.checkpoint)
        unwrapped_env = env.unwrapped
        eval_ids = resolve_eval_ids(unwrapped_env)
        dt = float(unwrapped_env.step_dt)
        num_envs = int(unwrapped_env.num_envs)
        device = unwrapped_env.device

        obs = normalize_observations(env.get_observations())
        with torch.inference_mode():
            action_template = policy(obs)
        actions = torch.zeros_like(action_template)
        initial_state = collect_state(unwrapped_env, actions, eval_ids)
        torque_limit = get_torque_limit(unwrapped_env.scene["robot"])
        joint_pos_limits = get_joint_pos_limits(unwrapped_env.scene["robot"])
        accum = init_accumulators(num_envs, device, initial_state["root_pos"], actions, dt)

        write_header_if_needed(args_cli.output)
        episodes_written = 0
        with args_cli.output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)

            while episodes_written < args_cli.episodes and simulation_app.is_running():
                pre_episode_lengths = unwrapped_env.episode_length_buf.clone()
                with torch.inference_mode():
                    actions = policy(obs)
                    before_state = collect_state(unwrapped_env, actions, eval_ids)
                    step_result = env.step(actions)
                    obs, _, dones, extras = unpack_step(step_result)
                    after_state = collect_state(unwrapped_env, actions, eval_ids)

                post_episode_lengths = unwrapped_env.episode_length_buf.clone()
                reset_before_metrics = dones & (post_episode_lengths <= 1)
                state = select_state(before_state, after_state, reset_before_metrics)
                update_accumulators(accum, state, eval_ids, dt, torque_limit, joint_pos_limits)

                if dones.any():
                    time_outs = extract_time_outs(dones, extras, unwrapped_env, pre_episode_lengths)
                    done_env_ids = torch.nonzero(dones, as_tuple=False).flatten().detach().cpu().tolist()
                    for env_id in done_env_ids:
                        if episodes_written >= args_cli.episodes:
                            break
                        episodes_written += 1
                        row = finalize_row(accum, env_id, episodes_written, bool(time_outs[env_id].detach().cpu()))
                        writer.writerow(row)
                        stream.flush()
                        reset_accumulator(accum, env_id, after_state["root_pos"], actions)

        if episodes_written < args_cli.episodes:
            print(
                f"[WARN] Wrote {episodes_written}/{args_cli.episodes} episodes before the simulation stopped.",
                flush=True,
            )
        else:
            print(f"[INFO] Wrote {episodes_written} evaluation episodes to {args_cli.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
