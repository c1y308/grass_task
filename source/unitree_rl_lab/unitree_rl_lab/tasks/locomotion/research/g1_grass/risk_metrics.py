"""Vectorized risk metrics for G1 grass-like locomotion research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


__all__ = [
    "touchdown_timing_error",
    "foot_slip_ratio",
    "stance_duration_deviation",
    "unexpected_contact_count",
    "missed_support_ratio",
    "contact_window_iou",
    "contact_risk",
    "event_window_delta_risk",
    "posture_risk",
    "action_jerk",
    "torque_saturation_ratio",
    "joint_limit_margin",
    "compensation_phase_alignment",
    "compensation_efficiency",
]


def _as_tensor(value: Any, like: torch.Tensor | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value if like is None else value.to(device=like.device, dtype=like.dtype)
    if like is None:
        return torch.as_tensor(value)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _as_float_tensor(value: Any, like: torch.Tensor | None = None) -> torch.Tensor:
    value = _as_tensor(value, like=like)
    return value if value.is_floating_point() else value.to(dtype=torch.float32)


def _non_batch_dims(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, value.ndim))


def _sum_non_batch(value: torch.Tensor) -> torch.Tensor:
    dims = _non_batch_dims(value)
    return value.sum(dim=dims) if dims else value


def _mean_non_batch(value: torch.Tensor) -> torch.Tensor:
    dims = _non_batch_dims(value)
    return value.mean(dim=dims) if dims else value


def _max_non_batch(value: torch.Tensor) -> torch.Tensor:
    dims = _non_batch_dims(value)
    return value.amax(dim=dims) if dims else value


def _min_non_batch(value: torch.Tensor) -> torch.Tensor:
    dims = _non_batch_dims(value)
    return value.amin(dim=dims) if dims else value


def _rms_non_batch(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_mean_non_batch(torch.square(value)))


def _l2_non_batch(value: torch.Tensor) -> torch.Tensor:
    dims = _non_batch_dims(value)
    return torch.linalg.vector_norm(value, dim=dims) if dims else torch.abs(value)


def _weight(
    weights: Mapping[str, float | torch.Tensor] | Sequence[float | torch.Tensor] | None,
    name: str,
    index: int,
    like: torch.Tensor,
) -> torch.Tensor:
    if weights is None:
        return torch.ones((), device=like.device, dtype=like.dtype)
    if isinstance(weights, Mapping):
        return _as_tensor(weights.get(name, 1.0), like=like)
    return _as_tensor(weights[index], like=like)


def touchdown_timing_error(
    real_touchdown_time: torch.Tensor,
    expected_touchdown_time: torch.Tensor,
) -> torch.Tensor:
    """Contact-risk metric: absolute touchdown timing mismatch in the paper."""
    real_touchdown_time = _as_float_tensor(real_touchdown_time)
    expected_touchdown_time = _as_float_tensor(expected_touchdown_time, like=real_touchdown_time)
    return torch.abs(real_touchdown_time - expected_touchdown_time)


def foot_slip_ratio(
    contact_mask: torch.Tensor,
    foot_xy_velocity: torch.Tensor,
    slip_velocity_threshold: float = 0.20,
) -> torch.Tensor:
    """Contact-risk metric: ratio of stance samples whose foot XY speed exceeds the slip threshold."""
    foot_xy_velocity = _as_float_tensor(foot_xy_velocity)
    contact_mask = _as_tensor(contact_mask, like=foot_xy_velocity).bool()

    if foot_xy_velocity.ndim > 0 and foot_xy_velocity.shape[-1] == 2:
        foot_speed = torch.linalg.vector_norm(foot_xy_velocity, dim=-1)
    else:
        foot_speed = torch.abs(foot_xy_velocity)

    contact_mask, foot_speed = torch.broadcast_tensors(contact_mask, foot_speed)
    slip_mask = contact_mask & (foot_speed > slip_velocity_threshold)
    slip_count = _sum_non_batch(slip_mask.to(dtype=foot_speed.dtype))
    contact_count = _sum_non_batch(contact_mask.to(dtype=foot_speed.dtype))
    return torch.where(contact_count > 0.0, slip_count / contact_count.clamp_min(1.0), torch.zeros_like(slip_count))


def stance_duration_deviation(
    real_stance_time: torch.Tensor,
    expected_stance_time: torch.Tensor,
) -> torch.Tensor:
    """Contact-risk metric: absolute deviation between measured and expected stance duration."""
    real_stance_time = _as_float_tensor(real_stance_time)
    expected_stance_time = _as_float_tensor(expected_stance_time, like=real_stance_time)
    return torch.abs(real_stance_time - expected_stance_time)


def unexpected_contact_count(swing_contact_mask: torch.Tensor) -> torch.Tensor:
    """Contact-risk metric: count of contacts observed during expected swing phase."""
    swing_contact_mask = _as_tensor(swing_contact_mask).bool()
    return _sum_non_batch(swing_contact_mask.to(dtype=torch.float32))


def missed_support_ratio(
    expected_contact_mask: torch.Tensor,
    contact_force_z: torch.Tensor,
    force_threshold: float | torch.Tensor,
) -> torch.Tensor:
    """Contact-risk metric: expected support samples whose vertical contact force is below threshold."""
    contact_force_z = _as_float_tensor(contact_force_z)
    expected_contact_mask = _as_tensor(expected_contact_mask, like=contact_force_z).bool()
    expected_contact_mask, contact_force_z = torch.broadcast_tensors(expected_contact_mask, contact_force_z)
    force_threshold = _as_float_tensor(force_threshold, like=contact_force_z)

    missed_support = expected_contact_mask & (contact_force_z < force_threshold)
    missed_count = _sum_non_batch(missed_support.to(dtype=contact_force_z.dtype))
    expected_count = _sum_non_batch(expected_contact_mask.to(dtype=contact_force_z.dtype))
    return torch.where(
        expected_count > 0.0,
        missed_count / expected_count.clamp_min(1.0),
        torch.zeros_like(missed_count),
    )


def contact_window_iou(
    expected_contact_mask: torch.Tensor,
    real_contact_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Contact-risk metric: intersection over union between expected and real contact windows."""
    real_contact_mask = _as_tensor(real_contact_mask)
    expected_contact_mask = _as_tensor(expected_contact_mask).to(device=real_contact_mask.device).bool()
    if real_contact_mask.dtype == torch.bool:
        real_contact_mask = real_contact_mask.bool()
        result_dtype = torch.float32
    else:
        result_dtype = real_contact_mask.dtype if real_contact_mask.is_floating_point() else torch.float32
        real_contact_mask = real_contact_mask > 0.0
    expected_contact_mask, real_contact_mask = torch.broadcast_tensors(expected_contact_mask, real_contact_mask)

    intersection = expected_contact_mask & real_contact_mask
    union = expected_contact_mask | real_contact_mask
    intersection_count = _sum_non_batch(intersection.to(dtype=result_dtype))
    union_count = _sum_non_batch(union.to(dtype=result_dtype))
    return torch.where(
        union_count > 0.0,
        intersection_count / union_count.clamp_min(eps),
        torch.ones_like(intersection_count),
    )


def contact_risk(
    touchdown_error: torch.Tensor,
    slip_ratio: torch.Tensor,
    unexpected_contact: torch.Tensor,
    missed_support: torch.Tensor,
    weights: Mapping[str, float | torch.Tensor] | Sequence[float | torch.Tensor] | None = None,
) -> torch.Tensor:
    """Aggregate contact risk from touchdown, slip, unexpected-contact, and missed-support terms."""
    touchdown_error = _as_float_tensor(touchdown_error)
    slip_ratio = _as_float_tensor(slip_ratio, like=touchdown_error)
    unexpected_contact = _as_float_tensor(unexpected_contact, like=touchdown_error)
    missed_support = _as_float_tensor(missed_support, like=touchdown_error)
    touchdown_error, slip_ratio, unexpected_contact, missed_support = torch.broadcast_tensors(
        touchdown_error,
        slip_ratio,
        unexpected_contact,
        missed_support,
    )
    return (
        _weight(weights, "touchdown_error", 0, touchdown_error) * touchdown_error
        + _weight(weights, "slip_ratio", 1, slip_ratio) * slip_ratio
        + _weight(weights, "unexpected_contact", 2, unexpected_contact) * unexpected_contact
        + _weight(weights, "missed_support", 3, missed_support) * missed_support
    )


def event_window_delta_risk(pre_window_risk: torch.Tensor, post_window_risk: torch.Tensor) -> torch.Tensor:
    """Event-window risk reduction: pre-event window risk minus post-event window risk."""
    pre_window_risk = _as_float_tensor(pre_window_risk)
    post_window_risk = _as_float_tensor(post_window_risk, like=pre_window_risk)
    return _mean_non_batch(pre_window_risk) - _mean_non_batch(post_window_risk)


def posture_risk(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    base_ang_vel: torch.Tensor,
    weights: Mapping[str, float | torch.Tensor] | Sequence[float | torch.Tensor] | None = None,
) -> torch.Tensor:
    """Posture-risk metric: weighted RMS roll, RMS pitch, and peak base angular velocity."""
    roll = _as_float_tensor(roll)
    pitch = _as_float_tensor(pitch, like=roll)
    base_ang_vel = _as_float_tensor(base_ang_vel, like=roll)

    if base_ang_vel.ndim >= 2 and base_ang_vel.shape[-1] in (2, 3):
        base_ang_vel_mag = torch.linalg.vector_norm(base_ang_vel, dim=-1)
    else:
        base_ang_vel_mag = torch.abs(base_ang_vel)

    roll_risk = _rms_non_batch(roll)
    pitch_risk = _rms_non_batch(pitch)
    ang_vel_risk = _max_non_batch(torch.abs(base_ang_vel_mag))

    return (
        _weight(weights, "roll", 0, roll_risk) * roll_risk
        + _weight(weights, "pitch", 1, pitch_risk) * pitch_risk
        + _weight(weights, "ang_vel", 2, ang_vel_risk) * ang_vel_risk
    )


def action_jerk(
    actions_t: torch.Tensor,
    actions_t_minus_1: torch.Tensor,
    actions_t_minus_2: torch.Tensor,
) -> torch.Tensor:
    """Compensation-risk metric: L2 magnitude of the second finite difference of policy actions."""
    actions_t = _as_float_tensor(actions_t)
    actions_t_minus_1 = _as_float_tensor(actions_t_minus_1, like=actions_t)
    actions_t_minus_2 = _as_float_tensor(actions_t_minus_2, like=actions_t)
    jerk = actions_t - 2.0 * actions_t_minus_1 + actions_t_minus_2
    return _l2_non_batch(jerk)


def torque_saturation_ratio(
    torque: torch.Tensor,
    torque_limit: torch.Tensor,
    rho: float = 0.85,
) -> torch.Tensor:
    """Compensation safety-boundary metric: P_tau, the ratio of torques above rho times the limit."""
    torque = _as_float_tensor(torque)
    torque_limit = torch.abs(_as_float_tensor(torque_limit, like=torque))
    saturation = torch.abs(torque) > rho * torque_limit
    return _mean_non_batch(saturation.to(dtype=torque.dtype))


def joint_limit_margin(
    joint_pos: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
) -> torch.Tensor:
    """Compensation safety-boundary metric: worst normalized distance to the nearest joint limit."""
    joint_pos = _as_float_tensor(joint_pos)
    lower_limits = _as_float_tensor(lower_limits, like=joint_pos)
    upper_limits = _as_float_tensor(upper_limits, like=joint_pos)

    joint_range = (upper_limits - lower_limits).clamp_min(torch.finfo(joint_pos.dtype).eps)
    lower_margin = (joint_pos - lower_limits) / joint_range
    upper_margin = (upper_limits - joint_pos) / joint_range
    margin = torch.minimum(lower_margin, upper_margin).clamp_min(0.0)
    return _min_non_batch(margin)


def compensation_phase_alignment(
    comp_energy_valid_window: torch.Tensor,
    comp_energy_total: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compensation-quality metric: energy in the valid stance-response window over total compensation energy."""
    comp_energy_valid_window = _as_float_tensor(comp_energy_valid_window)
    comp_energy_total = _as_float_tensor(comp_energy_total, like=comp_energy_valid_window)
    valid_energy = _sum_non_batch(comp_energy_valid_window)
    total_energy = _sum_non_batch(comp_energy_total)
    return valid_energy / total_energy.clamp_min(eps)


def compensation_efficiency(
    delta_contact_risk: torch.Tensor,
    delta_posture_risk: torch.Tensor,
    joint_energy: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Event-window diagnostic: contact plus posture risk reduction per unit joint energy."""
    delta_contact_risk = _as_float_tensor(delta_contact_risk)
    delta_posture_risk = _as_float_tensor(delta_posture_risk, like=delta_contact_risk)
    joint_energy = _as_float_tensor(joint_energy, like=delta_contact_risk)
    risk_reduction = _sum_non_batch(delta_contact_risk) + _sum_non_batch(delta_posture_risk)
    joint_energy = _sum_non_batch(joint_energy)
    return risk_reduction / joint_energy.clamp_min(eps)


if __name__ == "__main__":
    batch = 2
    feet = 2
    joints = 4
    actions = 6

    real_touchdown = torch.tensor([[0.11, 0.52], [0.09, 0.48]])
    expected_touchdown = torch.tensor([[0.10, 0.50], [0.10, 0.50]])
    contact = torch.tensor([[True, False], [True, True]])
    foot_vel = torch.tensor([[[0.25, 0.00], [0.05, 0.00]], [[0.10, 0.00], [0.30, 0.00]]])
    roll = torch.tensor([[0.02, -0.03, 0.01], [0.04, 0.01, -0.02]])
    pitch = torch.tensor([[0.01, 0.02, -0.02], [0.03, -0.01, 0.02]])
    base_ang_vel = torch.zeros(batch, 3, 3)
    action_t = torch.ones(batch, actions)
    action_t_minus_1 = torch.full((batch, actions), 0.8)
    action_t_minus_2 = torch.full((batch, actions), 0.7)
    torque = torch.tensor([[0.1, 0.9, 0.4, 0.2], [0.2, 0.3, 0.95, 0.1]])
    torque_limit = torch.ones(joints)
    joint_pos = torch.zeros(batch, joints)
    lower = -torch.ones(joints)
    upper = torch.ones(joints)

    metrics = {
        "touchdown_timing_error": touchdown_timing_error(real_touchdown, expected_touchdown),
        "foot_slip_ratio": foot_slip_ratio(contact, foot_vel),
        "stance_duration_deviation": stance_duration_deviation(real_touchdown, expected_touchdown),
        "unexpected_contact_count": unexpected_contact_count(contact),
        "missed_support_ratio": missed_support_ratio(contact, torch.tensor([[1.2, 0.0], [0.4, 0.8]]), 0.5),
        "contact_window_iou": contact_window_iou(contact, torch.tensor([[True, False], [False, True]])),
        "contact_risk": contact_risk(
            touchdown_timing_error(real_touchdown, expected_touchdown),
            foot_slip_ratio(contact, foot_vel),
            unexpected_contact_count(contact),
            missed_support_ratio(contact, torch.tensor([[1.2, 0.0], [0.4, 0.8]]), 0.5),
        ),
        "event_window_delta_risk": event_window_delta_risk(torch.ones(batch, feet), torch.zeros(batch, feet)),
        "posture_risk": posture_risk(roll, pitch, base_ang_vel),
        "action_jerk": action_jerk(action_t, action_t_minus_1, action_t_minus_2),
        "torque_saturation_ratio": torque_saturation_ratio(torque, torque_limit),
        "joint_limit_margin": joint_limit_margin(joint_pos, lower, upper),
        "compensation_phase_alignment": compensation_phase_alignment(torch.ones(batch, feet), torch.ones(batch) * 4.0),
        "compensation_efficiency": compensation_efficiency(
            torch.ones(batch), torch.ones(batch) * 0.5, torch.ones(batch) * 3.0
        ),
    }
    print({name: tuple(metric.shape) for name, metric in metrics.items()})
