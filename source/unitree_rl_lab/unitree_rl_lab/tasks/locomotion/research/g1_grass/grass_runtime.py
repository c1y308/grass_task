"""Isaac Lab runtime integration for grass-risk diagnostics and curriculum."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from isaaclab.managers import (
    CurriculumTermCfg,
    DatasetExportMode,
    ManagerTermBase,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

from .curriculum_runtime import CurriculumRuntimeCfg, EpisodeSummary, GrassCurriculumController
from .risk_curriculum import CurriculumState
from .risk_metrics import action_jerk, compensation_risk, contact_risk, posture_risk


STAGE_NAMES = ("flat_rigid", "mild_grass", "uneven_grass", "wet_soft_grass", "extreme_coupled")
DECISION_COLUMNS = (
    "control_step",
    "aggregate_transitions",
    "local_ppo_iteration",
    "stage_before",
    "stage_after",
    "dwell_transitions",
    "window_episodes",
    "success_rate",
    "contact_risk",
    "posture_risk",
    "compensation_risk",
    "dwell_pass",
    "success_pass",
    "contact_pass",
    "posture_pass",
    "compensation_pass",
    "gate_pass",
    "consecutive_passes",
    "promoted",
    "hold_reason",
)


def _quat_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(quat.dtype).eps)
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator > 0.0, numerator / denominator.clamp_min(1.0), torch.zeros_like(numerator))


def _mean_or_blank(values: list[float | int]) -> float | str:
    return sum(values) / len(values) if values else ""


class GrassRuntimeContext:
    """Shared state between the recorder and curriculum manager terms."""

    def __init__(self, env):
        cfg = env.cfg.grass_runtime_cfg
        runtime_cfg = CurriculumRuntimeCfg(
            num_stages=len(STAGE_NAMES),
            window_episodes=cfg["window_episodes"],
            evaluation_interval_transitions=cfg["evaluation_interval_transitions"],
            required_consecutive_passes=cfg["required_consecutive_passes"],
            rollout_steps_per_env=cfg["rollout_steps_per_env"],
        )
        initial_stage = int(getattr(env.cfg, "grass_initial_stage_index", 0))
        self.controller = GrassCurriculumController(
            env.cfg.grass_curriculum_progression,
            runtime_cfg,
            CurriculumState(current_stage=initial_stage),
        )
        self.pending: list[EpisodeSummary] = []
        self.latest_decision: dict[str, Any] = {}
        self.num_envs = env.num_envs
        self.device = env.device
        self.env_stage = torch.full((env.num_envs,), initial_stage, device=env.device, dtype=torch.long)
        self.transition_tile = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.height_amplitude = torch.zeros(env.num_envs, device=env.device)
        self.friction = torch.ones(env.num_envs, device=env.device)
        self.stiffness = torch.zeros(env.num_envs, device=env.device)
        self.damping = torch.zeros(env.num_envs, device=env.device)
        self.sample_counts: Counter[tuple[int, bool]] = Counter()
        self.sampling_since_evaluation: list[EpisodeSummary] = []
        self.latest_sampling_fractions = {stage_id: 0.0 for stage_id in range(len(STAGE_NAMES))}
        self.collect_episode_summaries = bool(getattr(env.cfg, "grass_collect_episode_summaries", False))
        self.reference_summaries: list[EpisodeSummary] = []
        self.log_dir = Path(getattr(env.cfg, "grass_runtime_log_dir", "")) if getattr(
            env.cfg, "grass_runtime_log_dir", ""
        ) else None
        self.terrain_bank = getattr(env.scene.terrain, "grass_tile_bank", None)
        self.schedule = env.cfg.grass_terrain_schedule
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, summaries: list[EpisodeSummary]) -> None:
        self.pending.extend(summaries)
        self.sampling_since_evaluation.extend(summaries)
        if self.collect_episode_summaries:
            self.reference_summaries.extend(summaries)
        for summary in summaries:
            self.sample_counts[(summary.stage_id, bool(summary.transition_tile))] += 1

    def assign_terrain(self, env_ids: torch.Tensor) -> None:
        stage_id = self.controller.state.current_stage
        stage = env_ids.new_full((len(env_ids),), stage_id)
        if self.terrain_bank is not None:
            sample = self.terrain_bank.sample(stage, env_ids)
            self.transition_tile[env_ids] = sample["transition"]
            self.height_amplitude[env_ids] = sample["height_amplitude"]
            self.friction[env_ids] = sample["friction"]
            self.stiffness[env_ids] = sample["stiffness"]
            self.damping[env_ids] = sample["damping"]
        else:
            stage_cfg = self.schedule.get_stage(stage_id)
            self.height_amplitude[env_ids] = stage_cfg.height_range[1]
            self.friction[env_ids] = 0.5 * sum(stage_cfg.friction_range)
        self.env_stage[env_ids] = stage_id

    def log_decision(self, record: dict[str, Any]) -> None:
        self.latest_decision = dict(record)
        sampling_rows = self._consume_sampling_rows(record)
        if self.log_dir is None:
            return
        decision_path = self.log_dir / "curriculum_decisions.csv"
        write_header = not decision_path.exists()
        with decision_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=DECISION_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        if record.get("promoted"):
            with (self.log_dir / "promotion_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._write_sampling_rows(sampling_rows)

    def _consume_sampling_rows(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        summaries = self.sampling_since_evaluation
        self.sampling_since_evaluation = []
        total = len(summaries)
        self.latest_sampling_fractions = {
            stage_id: sum(item.stage_id == stage_id for item in summaries) / total if total else 0.0
            for stage_id in range(len(STAGE_NAMES))
        }
        rows = []
        for stage_id, stage_name in enumerate(STAGE_NAMES):
            for transition in (False, True):
                items = [
                    item
                    for item in summaries
                    if item.stage_id == stage_id and bool(item.transition_tile) == transition
                ]
                terminations = Counter(item.termination_reason for item in items)
                row: dict[str, Any] = {
                    "aggregate_transitions": record["aggregate_transitions"],
                    "local_ppo_iteration": record["local_ppo_iteration"],
                    "decision_stage": record["stage_before"],
                    "sampled_stage": stage_id,
                    "terrain_name": stage_name,
                    "tile_kind": "transition" if transition else "steady",
                    "episodes": len(items),
                    "cumulative_episodes": self.sample_counts[(stage_id, transition)],
                    "success_rate": _mean_or_blank([item.success for item in items]),
                    "episode_reward_mean": _mean_or_blank([item.episode_reward for item in items]),
                    "episode_length_mean": _mean_or_blank([item.episode_length for item in items]),
                    "timeout_count": terminations["time_out"],
                    "base_height_count": terminations["base_height"],
                    "bad_orientation_count": terminations["bad_orientation"],
                    "other_termination_count": sum(
                        count
                        for name, count in terminations.items()
                        if name not in ("time_out", "base_height", "bad_orientation")
                    ),
                }
                for field in ("height_amplitude", "friction", "stiffness", "damping"):
                    values = [getattr(item, field) for item in items]
                    row[f"{field}_min"] = min(values) if values else ""
                    row[f"{field}_mean"] = _mean_or_blank(values)
                    row[f"{field}_max"] = max(values) if values else ""
                rows.append(row)
        return rows

    def _write_sampling_rows(self, rows: list[dict[str, Any]]) -> None:
        path = self.log_dir / "terrain_sampling.csv"
        fields = list(rows[0])
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


def _runtime_log_state(context: GrassRuntimeContext) -> dict[str, float]:
    controller = context.controller
    decision = context.latest_decision
    stage = controller.state.current_stage
    output = {
        "stage_id": float(stage),
        "lambda_level": stage / max(len(STAGE_NAMES) - 1, 1),
        "control_step": float(controller.state.control_step),
        "aggregate_transitions": float(controller.state.aggregate_transitions),
        "local_ppo_iteration": float(
            max(controller.state.control_step - 1, 0) // controller.cfg.rollout_steps_per_env
        ),
        "window_episodes": float(controller.window_size),
        "transition_tile_fraction": float(context.transition_tile.float().mean().item()),
        "promotion_event": float(
            bool(decision.get("promoted", False))
            and decision.get("aggregate_transitions") == controller.state.aggregate_transitions
        ),
    }
    for stage_id, name in enumerate(STAGE_NAMES):
        output[f"target_probability_{name}"] = float(stage_id == stage)
        output[f"actual_fraction_{name}"] = context.latest_sampling_fractions[stage_id]
    for name in (
        "success_rate",
        "contact_risk",
        "posture_risk",
        "compensation_risk",
        "dwell_pass",
        "success_pass",
        "contact_pass",
        "posture_pass",
        "compensation_pass",
        "gate_pass",
    ):
        output[name] = float(decision.get(name, 0.0))
    return output


class GrassRiskRecorderTerm(RecorderTerm):
    """Accumulate per-step diagnostics without adding them to observations or rewards."""

    def __init__(self, cfg: RecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        if not hasattr(env, "grass_runtime_context"):
            env.grass_runtime_context = GrassRuntimeContext(env)
        self.context: GrassRuntimeContext = env.grass_runtime_context
        self.robot = env.scene["robot"]
        self.sensor = env.scene.sensors["contact_forces"]
        self.foot_body_ids, self.foot_names = self.robot.find_bodies(".*ankle_roll.*")
        self.contact_body_ids, contact_names = self.sensor.find_bodies(".*ankle_roll.*")
        if len(self.foot_body_ids) != 2 or len(self.contact_body_ids) != 2:
            raise RuntimeError("Grass diagnostics require exactly two ankle-roll foot bodies.")
        if sorted(self.foot_names) != sorted(contact_names):
            raise RuntimeError("Robot and contact-sensor foot ordering do not describe the same bodies.")
        self.dt = float(env.step_dt)
        self.num_joints = self.robot.data.joint_pos.shape[1]
        self.contact_force_threshold: torch.Tensor | None = None
        self.robot_material_verified = False
        self.event_window_steps = max(1, round(0.50 / self.dt))
        self.valid_window_steps = max(1, round(0.30 / self.dt))
        self.refractory_steps = max(1, round(0.25 / self.dt))
        self._allocate()

    def _allocate(self) -> None:
        n, d = self.num_envs, self.device
        zero = lambda: torch.zeros(n, device=d)
        self.values = {
            name: zero()
            for name in (
                "touchdown_sum",
                "touchdown_count",
                "slip_num",
                "slip_den",
                "unexpected_num",
                "unexpected_den",
                "missed_num",
                "missed_den",
                "roll_sq",
                "pitch_sq",
                "posture_count",
                "ang_vel_peak",
                "torque_sat_sum",
                "joint_proximity_sum",
                "jerk_sum",
                "comp_count",
                "event_energy",
                "valid_event_energy",
                "xy_error_sum",
                "xy_error_count",
                "progress_num",
                "progress_den",
                "yaw_error_sum",
                "yaw_error_count",
            )
        }
        self.prev_expected = torch.zeros((n, 2), device=d, dtype=torch.bool)
        self.prev_contact = torch.zeros((n, 2), device=d, dtype=torch.bool)
        self.pending_touchdown = torch.zeros((n, 2), device=d, dtype=torch.bool)
        self.expected_touchdown_time = torch.zeros((n, 2), device=d)
        self.prev_event = torch.zeros(n, device=d, dtype=torch.bool)
        self.refractory = torch.zeros(n, device=d, dtype=torch.long)
        self.event_remaining = torch.zeros(n, device=d, dtype=torch.long)
        self.valid_remaining = torch.zeros(n, device=d, dtype=torch.long)
        self.history_steps = torch.zeros(n, device=d, dtype=torch.long)
        target = self.robot.data.joint_pos_target.clone()
        self.prev_target = target
        self.prev_prev_target = target.clone()

    def record_post_step(self):
        if not self.robot_material_verified:
            properties = self.robot.root_physx_view.get_material_properties().to(self.device)
            expected = torch.full_like(properties[..., :2], 1.5)
            if not torch.allclose(properties[..., :2], expected, atol=1e-5, rtol=0.0):
                actual_min = float(properties[..., :2].min().item())
                actual_max = float(properties[..., :2].max().item())
                raise RuntimeError(
                    "Grass tasks require fixed robot friction 1.5; PhysX reported "
                    f"range [{actual_min:.6f}, {actual_max:.6f}]."
                )
            self.robot_material_verified = True
        controller = self.context.controller
        controller.update_counters(self._env.common_step_counter, self.num_envs)
        if controller.progression is not None:
            decision = controller.evaluate_if_due()
            if decision is not None:
                self.context.log_decision(decision)
        elapsed = self._env.episode_length_buf.float() * self.dt
        offsets = torch.tensor((0.0, 0.5), device=self.device)
        phase = ((elapsed[:, None] / 0.8) + offsets[None, :]) % 1.0
        expected = phase < 0.55

        if self.contact_force_threshold is None:
            masses = self.robot.root_physx_view.get_masses().to(self.device).sum(-1)
            self.contact_force_threshold = 0.05 * masses * 9.81
        forces = self.sensor.data.net_forces_w[:, self.contact_body_ids, 2].abs()
        contact = forces > self.contact_force_threshold[:, None]
        expected_rise = expected & ~self.prev_expected
        contact_rise = contact & ~self.prev_contact
        self.pending_touchdown |= expected_rise
        self.expected_touchdown_time = torch.where(expected_rise, elapsed[:, None], self.expected_touchdown_time)
        matched = contact_rise & self.pending_touchdown
        error = ((elapsed[:, None] - self.expected_touchdown_time).abs() / 0.10).clamp(0.0, 1.0)
        self.values["touchdown_sum"] += (error * matched.float()).sum(-1)
        self.values["touchdown_count"] += matched.float().sum(-1)
        self.pending_touchdown &= ~matched
        expired = self.pending_touchdown & ((elapsed[:, None] - self.expected_touchdown_time) > 0.10)
        self.values["touchdown_sum"] += expired.float().sum(-1)
        self.values["touchdown_count"] += expired.float().sum(-1)
        self.pending_touchdown &= ~expired

        foot_speed = torch.linalg.vector_norm(self.robot.data.body_lin_vel_w[:, self.foot_body_ids, :2], dim=-1)
        slip = contact & (foot_speed > 0.20)
        unexpected = contact & ~expected
        missed = expected & ~contact
        self.values["slip_num"] += slip.float().sum(-1)
        self.values["slip_den"] += contact.float().sum(-1)
        self.values["unexpected_num"] += unexpected.float().sum(-1)
        self.values["unexpected_den"] += (~expected).float().sum(-1)
        self.values["missed_num"] += missed.float().sum(-1)
        self.values["missed_den"] += expected.float().sum(-1)

        roll, pitch = _quat_roll_pitch(self.robot.data.root_quat_w)
        ang_vel_xy = torch.linalg.vector_norm(self.robot.data.root_ang_vel_b[:, :2], dim=-1)
        self.values["roll_sq"] += roll.square()
        self.values["pitch_sq"] += pitch.square()
        self.values["posture_count"] += 1.0
        self.values["ang_vel_peak"] = torch.maximum(self.values["ang_vel_peak"], ang_vel_xy)

        torque = self.robot.data.applied_torque
        torque_limit = self._torque_limit(torque)
        self.values["torque_sat_sum"] += (torque.abs() > 0.85 * torque_limit).float().mean(-1)
        limits = self.robot.data.soft_joint_pos_limits
        joint_range = (limits[..., 1] - limits[..., 0]).clamp_min(1e-6)
        margin = torch.minimum(
            (self.robot.data.joint_pos - limits[..., 0]) / joint_range,
            (limits[..., 1] - self.robot.data.joint_pos) / joint_range,
        )
        self.values["joint_proximity_sum"] += (margin < 0.05).float().mean(-1)
        target = self.robot.data.joint_pos_target
        jerk = action_jerk(target, self.prev_target, self.prev_prev_target, self.dt, 100.0)
        valid_history = self.history_steps >= 2
        self.values["jerk_sum"] += jerk * valid_history.float()
        self.values["comp_count"] += valid_history.float()
        self.prev_prev_target[:] = self.prev_target
        self.prev_target[:] = target
        self.history_steps += 1

        raw_event = slip.any(-1) | unexpected.any(-1) | missed.any(-1)
        new_event = raw_event & ~self.prev_event & (self.refractory <= 0)
        self.event_remaining = torch.where(new_event, self.event_window_steps, self.event_remaining)
        self.valid_remaining = torch.where(new_event, self.valid_window_steps, self.valid_remaining)
        self.refractory = torch.where(new_event, self.refractory_steps, torch.clamp(self.refractory - 1, min=0))
        energy = (self.robot.data.joint_vel.abs() * torque.abs()).sum(-1) * self.dt
        self.values["event_energy"] += energy * (self.event_remaining > 0).float()
        stance_response = (expected & contact).any(-1)
        self.values["valid_event_energy"] += energy * ((self.valid_remaining > 0) & stance_response).float()
        self.event_remaining = torch.clamp(self.event_remaining - 1, min=0)
        self.valid_remaining = torch.clamp(self.valid_remaining - 1, min=0)
        self.prev_event[:] = raw_event

        command = self._env.command_manager.get_command("base_velocity")
        actual_xy = self.robot.data.root_lin_vel_b[:, :2]
        command_xy = command[:, :2]
        command_norm = torch.linalg.vector_norm(command_xy, dim=-1)
        moving = command_norm > 0.10
        xy_error = torch.linalg.vector_norm(actual_xy - command_xy, dim=-1)
        self.values["xy_error_sum"] += xy_error * moving.float()
        self.values["xy_error_count"] += moving.float()
        direction = command_xy / command_norm[:, None].clamp_min(1e-6)
        self.values["progress_num"] += (actual_xy * direction).sum(-1) * self.dt * moving.float()
        self.values["progress_den"] += command_norm * self.dt * moving.float()
        turning = command[:, 2].abs() > 0.05
        yaw_error = (self.robot.data.root_ang_vel_b[:, 2] - command[:, 2]).abs()
        self.values["yaw_error_sum"] += yaw_error * turning.float()
        self.values["yaw_error_count"] += turning.float()

        self.prev_expected[:] = expected
        self.prev_contact[:] = contact
        log = self._env.extras.setdefault("log", {})
        for name in tuple(log):
            if name.startswith("Curriculum/grass_runtime/"):
                del log[name]
        if self._env.common_step_counter % controller.cfg.rollout_steps_per_env == 0:
            log.update(
                {
                    f"Curriculum/grass_runtime/{name}": value
                    for name, value in _runtime_log_state(self.context).items()
                }
            )
        return None, None

    def record_pre_reset(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # ManagerBasedEnv calls record_pre_reset during the initial reset as well.
        # There is no completed episode to submit until at least one control step ran.
        ids = ids[self._env.episode_length_buf[ids] > 0]
        if ids.numel() == 0:
            return None, None
        pending = self.pending_touchdown[ids].float().sum(-1)
        self.values["touchdown_sum"][ids] += pending
        self.values["touchdown_count"][ids] += pending
        touchdown = _safe_ratio(self.values["touchdown_sum"][ids], self.values["touchdown_count"][ids])
        slip = _safe_ratio(self.values["slip_num"][ids], self.values["slip_den"][ids])
        unexpected = _safe_ratio(self.values["unexpected_num"][ids], self.values["unexpected_den"][ids])
        missed = _safe_ratio(self.values["missed_num"][ids], self.values["missed_den"][ids])
        contact_value = contact_risk(touchdown, slip, unexpected, missed)

        count = self.values["posture_count"][ids].clamp_min(1.0)
        roll_sq_mean = self.values["roll_sq"][ids] / count
        pitch_sq_mean = self.values["pitch_sq"][ids] / count
        roll_per_env = torch.sqrt(roll_sq_mean).unsqueeze(-1)
        pitch_per_env = torch.sqrt(pitch_sq_mean).unsqueeze(-1)
        ang_vel_per_env = self.values["ang_vel_peak"][ids].unsqueeze(-1).unsqueeze(-1)
        posture_value = posture_risk(
            roll_per_env,
            pitch_per_env,
            ang_vel_per_env,
            roll_reference_rad=0.20,
            pitch_reference_rad=0.20,
            ang_vel_reference_rad_s=2.0,
        )
        comp_count = self.values["comp_count"][ids].clamp_min(1.0)
        phase = torch.where(
            self.values["event_energy"][ids] > 0,
            self.values["valid_event_energy"][ids] / self.values["event_energy"][ids].clamp_min(1e-6),
            torch.ones_like(comp_count),
        )
        compensation_value = compensation_risk(
            self.values["torque_sat_sum"][ids] / comp_count,
            self.values["joint_proximity_sum"][ids] / comp_count,
            self.values["jerk_sum"][ids] / comp_count,
            phase,
        )

        xy_error = _safe_ratio(self.values["xy_error_sum"][ids], self.values["xy_error_count"][ids])
        progress_den = self.values["progress_den"][ids]
        progress = torch.where(
            progress_den > 0.0,
            self.values["progress_num"][ids] / progress_den.clamp_min(1e-6),
            torch.ones_like(progress_den),
        )
        yaw_error = _safe_ratio(self.values["yaw_error_sum"][ids], self.values["yaw_error_count"][ids])
        timeout = self._env.termination_manager.time_outs[ids]
        success = timeout & (xy_error <= 0.35) & (progress >= 0.50) & (yaw_error <= 0.25)

        episode_reward = sum(self._env.reward_manager._episode_sums.values())[ids]
        episode_length = self._env.episode_length_buf[ids]
        termination_masks = {
            name: self._env.termination_manager.get_term(name)[ids]
            for name in self._env.termination_manager.active_terms
        }
        termination_reasons = []
        for index in range(len(ids)):
            if bool(timeout[index].item()):
                termination_reasons.append("time_out")
                continue
            termination_reasons.append(
                next(
                    (
                        name
                        for name, mask in termination_masks.items()
                        if name != "time_out" and bool(mask[index].item())
                    ),
                    "unknown",
                )
            )

        cpu = lambda value: value.detach().cpu().tolist()
        summaries = [
            EpisodeSummary(*values)
            for values in zip(
                cpu(self.context.env_stage[ids]),
                cpu(success.float()),
                cpu(contact_value),
                cpu(posture_value),
                cpu(compensation_value),
                cpu(self.context.transition_tile[ids].float()),
                cpu(self.context.height_amplitude[ids]),
                cpu(self.context.friction[ids]),
                cpu(self.context.stiffness[ids]),
                cpu(self.context.damping[ids]),
                cpu(episode_reward),
                cpu(episode_length),
                termination_reasons,
            )
        ]
        self.context.submit(summaries)
        return None, None

    def reset(self, env_ids=None):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for value in self.values.values():
            value[ids] = 0.0
        for value in (
            self.prev_expected,
            self.prev_contact,
            self.pending_touchdown,
            self.prev_event,
            self.refractory,
            self.event_remaining,
            self.valid_remaining,
            self.history_steps,
        ):
            value[ids] = 0
        target = self.robot.data.joint_pos_target[ids]
        self.prev_target[ids] = target
        self.prev_prev_target[ids] = target

    def _torque_limit(self, torque: torch.Tensor) -> torch.Tensor:
        for name in ("joint_effort_limits", "soft_joint_effort_limits", "default_joint_effort_limits"):
            value = getattr(self.robot.data, name, None)
            if value is not None:
                return value.abs().to(torque.device).clamp_min(1e-6)
        return torch.full_like(torque, float("inf"))


class GrassCurriculumTerm(ManagerTermBase):
    """Consume completed episodes, evaluate gates, and assign reset terrain tiles."""

    def __init__(self, cfg: CurriculumTermCfg, env) -> None:
        super().__init__(cfg, env)
        if not hasattr(env, "grass_runtime_context"):
            env.grass_runtime_context = GrassRuntimeContext(env)
        self.context: GrassRuntimeContext = env.grass_runtime_context

    def __call__(self, env, env_ids):
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        controller = self.context.controller
        controller.update_counters(env.common_step_counter, env.num_envs)
        if self.context.pending:
            controller.submit(self.context.pending)
            self.context.pending.clear()
        self.context.assign_terrain(ids)
        if env.common_step_counter % controller.cfg.rollout_steps_per_env == 0:
            return self._log_state()
        return {}

    def _log_state(self) -> dict[str, float]:
        return _runtime_log_state(self.context)


@configclass
class GrassRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    grass_risk = RecorderTermCfg(class_type=GrassRiskRecorderTerm)


def grass_curriculum_term_cfg() -> CurriculumTermCfg:
    return CurriculumTermCfg(func=GrassCurriculumTerm)
