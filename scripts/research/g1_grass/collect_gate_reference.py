#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect flat-reference episodes for grass gate calibration.")
parser.add_argument("--task", default="Unitree-G1-29dof-Grass-RiskGate")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=4096)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab_platform_compat import patch_conda_forge_sys_version_for_isaaclab

patch_conda_forge_sys_version_for_isaaclab()

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def observations(value):
    return value[0] if isinstance(value, tuple) else value


def main() -> None:
    torch.manual_seed(args.seed)
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = args.seed
    env_cfg.grass_collect_episode_summaries = True
    env_cfg.grass_runtime_log_dir = ""
    env_cfg.grass_curriculum_progression = None
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = args.device
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(
        str(args.checkpoint.resolve()),
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": True},
    )
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    obs = observations(wrapped.get_observations())
    context = wrapped.unwrapped.grass_runtime_context
    try:
        while len(context.reference_summaries) < args.episodes and simulation_app.is_running():
            with torch.inference_mode():
                action = policy(obs)
                step_result = wrapped.step(action)
                obs = observations(step_result[0])
        rows = [asdict(item) for item in context.reference_summaries[: args.episodes]]
        if len(rows) < args.episodes:
            raise RuntimeError(f"Simulation stopped after {len(rows)}/{args.episodes} reference episodes.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Wrote {len(rows)} reference episodes to {args.output}")
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
