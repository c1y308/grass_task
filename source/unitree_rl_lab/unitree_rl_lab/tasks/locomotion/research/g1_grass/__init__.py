"""Research configuration helpers and task registration for G1 grass experiments."""

import gymnasium as gym

from .terrain_cfg import GrassTerrainSchedule, GrassTerrainStage


def _register_grass_task(task_id: str, env_cfg: str, play_env_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_grass_env_cfg:{env_cfg}",
            "play_env_cfg_entry_point": f"{__name__}.g1_grass_env_cfg:{play_env_cfg}",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )


_register_grass_task(
    "Unitree-G1-29dof-Grass-FlatRigid",
    "G1GrassFlatRigidEnvCfg",
    "G1GrassFlatRigidPlayEnvCfg",
)
_register_grass_task(
    "Unitree-G1-29dof-Grass-CoupledRandom",
    "G1GrassCoupledRandomEnvCfg",
    "G1GrassCoupledRandomPlayEnvCfg",
)
_register_grass_task(
    "Unitree-G1-29dof-Grass-FixedSchedule",
    "G1GrassFixedScheduleEnvCfg",
    "G1GrassFixedSchedulePlayEnvCfg",
)
_register_grass_task(
    "Unitree-G1-29dof-Grass-SuccessGate",
    "G1GrassSuccessGateEnvCfg",
    "G1GrassSuccessGatePlayEnvCfg",
)
_register_grass_task(
    "Unitree-G1-29dof-Grass-RiskGate",
    "G1GrassRiskGateEnvCfg",
    "G1GrassRiskGatePlayEnvCfg",
)

__all__ = ["GrassTerrainSchedule", "GrassTerrainStage"]
