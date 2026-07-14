from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0) -> torch.Tensor:
    """Binary foot contact flags for each foot body in the contact sensor.

    Returns a tensor of shape (num_envs, num_bodies) with 1.0 for active
    contact and 0.0 otherwise.  On the real G1 platform, binary contact
    can be obtained from joint-torque or foot-force sensors, making this
    a proprioceptive observation consistent with the paper's requirement
    that the policy include "可由平台获得的接触相关状态".
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs()
    threshold = force_threshold
    return (net_forces_z > threshold).float()
