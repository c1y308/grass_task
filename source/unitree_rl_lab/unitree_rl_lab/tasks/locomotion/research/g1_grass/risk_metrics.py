"""Dimensionless, vectorized risk metrics for G1 grass locomotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


Weights = Mapping[str, float | torch.Tensor] | Sequence[float | torch.Tensor] | None

__all__ = [
    "action_jerk",
    "compensation_efficiency",
    "compensation_phase_alignment",
    "compensation_risk",
    "contact_risk",
    "contact_window_iou",
    "event_window_delta_risk",
    "foot_slip_ratio",
    "joint_limit_margin",
    "joint_limit_proximity_ratio",
    "missed_support_ratio",
    "normalized_touchdown_error",
    "posture_risk",
    "rising_edge",
    "stance_duration_deviation",
    "torque_saturation_ratio",
    "touchdown_timing_error",
    "unexpected_contact_count",
    "unexpected_contact_ratio",
]


def _as_tensor(value: Any, like: torch.Tensor | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value if like is None else value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value) if like is None else torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _float(value: Any, like: torch.Tensor | None = None) -> torch.Tensor:
    value = _as_tensor(value, like)
    return value if value.is_floating_point() else value.float()


def _dims(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, value.ndim))


def _sum(value: torch.Tensor) -> torch.Tensor:
    return value.sum(dim=_dims(value)) if value.ndim > 1 else value


def _mean(value: torch.Tensor) -> torch.Tensor:
    return value.mean(dim=_dims(value)) if value.ndim > 1 else value


def _max(value: torch.Tensor) -> torch.Tensor:
    return value.amax(dim=_dims(value)) if value.ndim > 1 else value


def _min(value: torch.Tensor) -> torch.Tensor:
    return value.amin(dim=_dims(value)) if value.ndim > 1 else value


def _weighted_mean(values: list[torch.Tensor], names: list[str], weights: Weights) -> torch.Tensor:
    reference = values[0]
    if weights is None:
        weight_values = [torch.ones((), device=reference.device, dtype=reference.dtype) for _ in values]
    elif isinstance(weights, Mapping):
        weight_values = [_as_tensor(weights.get(name, 1.0), reference) for name in names]
    else:
        weight_values = [_as_tensor(value, reference) for value in weights]
    total = sum(weight_values)
    if float(total.detach().cpu()) <= 0.0:
        raise ValueError("risk weights must sum to a positive value")
    return sum(weight * value for weight, value in zip(weight_values, values)) / total


def rising_edge(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    current = _as_tensor(current).bool()
    previous = _as_tensor(previous).to(current.device).bool()
    return current & ~previous


def touchdown_timing_error(real_touchdown_time: torch.Tensor, expected_touchdown_time: torch.Tensor) -> torch.Tensor:
    real = _float(real_touchdown_time)
    expected = _float(expected_touchdown_time, real)
    return torch.abs(real - expected)


def normalized_touchdown_error(
    real_touchdown_time: torch.Tensor,
    expected_touchdown_time: torch.Tensor,
    tolerance_s: float = 0.10,
) -> torch.Tensor:
    if tolerance_s <= 0.0:
        raise ValueError("tolerance_s must be positive")
    return torch.clamp(touchdown_timing_error(real_touchdown_time, expected_touchdown_time) / tolerance_s, 0.0, 1.0)


def foot_slip_ratio(
    contact_mask: torch.Tensor,
    foot_xy_velocity: torch.Tensor,
    slip_velocity_threshold: float = 0.20,
) -> torch.Tensor:
    velocity = _float(foot_xy_velocity)
    speed = torch.linalg.vector_norm(velocity, dim=-1) if velocity.shape[-1:] == (2,) else torch.abs(velocity)
    contact = _as_tensor(contact_mask).to(speed.device).bool()
    contact, speed = torch.broadcast_tensors(contact, speed)
    slip = contact & (speed > slip_velocity_threshold)
    denominator = _sum(contact.to(speed.dtype))
    return torch.where(denominator > 0, _sum(slip.to(speed.dtype)) / denominator.clamp_min(1.0), 0.0)


def stance_duration_deviation(real_stance_time: torch.Tensor, expected_stance_time: torch.Tensor) -> torch.Tensor:
    return touchdown_timing_error(real_stance_time, expected_stance_time)


def unexpected_contact_count(swing_contact_mask: torch.Tensor) -> torch.Tensor:
    return _sum(_as_tensor(swing_contact_mask).bool().float())


def unexpected_contact_ratio(expected_contact_mask: torch.Tensor, real_contact_mask: torch.Tensor) -> torch.Tensor:
    expected = _as_tensor(expected_contact_mask).bool()
    real = _as_tensor(real_contact_mask).to(expected.device).bool()
    expected, real = torch.broadcast_tensors(expected, real)
    swing = ~expected
    denominator = _sum(swing.float())
    numerator = _sum((real & swing).float())
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1.0), 0.0)


def missed_support_ratio(
    expected_contact_mask: torch.Tensor,
    contact_force_z: torch.Tensor,
    force_threshold: float | torch.Tensor,
) -> torch.Tensor:
    force = _float(contact_force_z)
    expected = _as_tensor(expected_contact_mask).to(force.device).bool()
    expected, force = torch.broadcast_tensors(expected, force)
    threshold = _float(force_threshold, force)
    denominator = _sum(expected.to(force.dtype))
    numerator = _sum((expected & (force < threshold)).to(force.dtype))
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1.0), 0.0)


def contact_window_iou(expected_contact_mask: torch.Tensor, real_contact_mask: torch.Tensor, eps: float = 1e-6):
    expected = _as_tensor(expected_contact_mask).bool()
    real = _as_tensor(real_contact_mask).to(expected.device).bool()
    expected, real = torch.broadcast_tensors(expected, real)
    intersection = _sum((expected & real).float())
    union = _sum((expected | real).float())
    return torch.where(union > 0, intersection / union.clamp_min(eps), torch.ones_like(union))


def contact_risk(
    touchdown_error: torch.Tensor,
    slip_ratio: torch.Tensor,
    unexpected_contact: torch.Tensor,
    missed_support: torch.Tensor,
    weights: Weights = None,
) -> torch.Tensor:
    values = torch.broadcast_tensors(
        _float(touchdown_error),
        _float(slip_ratio, _float(touchdown_error)),
        _float(unexpected_contact, _float(touchdown_error)),
        _float(missed_support, _float(touchdown_error)),
    )
    values = [torch.clamp(value, 0.0, 1.0) for value in values]
    return _weighted_mean(values, ["touchdown", "slip", "unexpected", "missed"], weights)


def event_window_delta_risk(pre_window_risk: torch.Tensor, post_window_risk: torch.Tensor) -> torch.Tensor:
    pre = _float(pre_window_risk)
    post = _float(post_window_risk, pre)
    return _mean(pre) - _mean(post)


def posture_risk(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    base_ang_vel_xy: torch.Tensor,
    weights: Weights = None,
    roll_reference_rad: float = 0.20,
    pitch_reference_rad: float = 0.20,
    ang_vel_reference_rad_s: float = 2.0,
) -> torch.Tensor:
    roll = _float(roll)
    pitch = _float(pitch, roll)
    ang_vel = _float(base_ang_vel_xy, roll)
    if ang_vel.ndim >= 2 and ang_vel.shape[-1] in (2, 3):
        ang_vel = torch.linalg.vector_norm(ang_vel[..., :2], dim=-1)
    values = [
        torch.clamp(torch.sqrt(_mean(roll.square())) / roll_reference_rad, 0.0, 1.0),
        torch.clamp(torch.sqrt(_mean(pitch.square())) / pitch_reference_rad, 0.0, 1.0),
        torch.clamp(_max(torch.abs(ang_vel)) / ang_vel_reference_rad_s, 0.0, 1.0),
    ]
    return _weighted_mean(values, ["roll", "pitch", "ang_vel"], weights)


def action_jerk(
    targets_t: torch.Tensor,
    targets_t_minus_1: torch.Tensor,
    targets_t_minus_2: torch.Tensor,
    dt: float = 1.0,
    reference_acceleration: float | None = None,
) -> torch.Tensor:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    current = _float(targets_t)
    previous = _float(targets_t_minus_1, current)
    previous_previous = _float(targets_t_minus_2, current)
    acceleration = (current - 2.0 * previous + previous_previous) / (dt * dt)
    mean_square = _mean(acceleration.square())
    if reference_acceleration is None:
        return mean_square
    if reference_acceleration <= 0.0:
        raise ValueError("reference_acceleration must be positive")
    return torch.clamp(mean_square / (reference_acceleration**2), 0.0, 1.0)


def torque_saturation_ratio(torque: torch.Tensor, torque_limit: torch.Tensor, rho: float = 0.85) -> torch.Tensor:
    torque = _float(torque)
    limit = torch.abs(_float(torque_limit, torque))
    return _mean((torch.abs(torque) > rho * limit).to(torque.dtype))


def joint_limit_margin(joint_pos: torch.Tensor, lower_limits: torch.Tensor, upper_limits: torch.Tensor):
    joint_pos = _float(joint_pos)
    lower = _float(lower_limits, joint_pos)
    upper = _float(upper_limits, joint_pos)
    joint_range = (upper - lower).clamp_min(torch.finfo(joint_pos.dtype).eps)
    margin = torch.minimum((joint_pos - lower) / joint_range, (upper - joint_pos) / joint_range).clamp_min(0.0)
    return _min(margin)


def joint_limit_proximity_ratio(
    joint_pos: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    margin_threshold: float = 0.05,
) -> torch.Tensor:
    joint_pos = _float(joint_pos)
    lower = _float(lower_limits, joint_pos)
    upper = _float(upper_limits, joint_pos)
    joint_range = (upper - lower).clamp_min(torch.finfo(joint_pos.dtype).eps)
    margin = torch.minimum((joint_pos - lower) / joint_range, (upper - joint_pos) / joint_range)
    return _mean((margin < margin_threshold).to(joint_pos.dtype))


def compensation_phase_alignment(valid_window_energy: torch.Tensor, total_window_energy: torch.Tensor, eps=1e-6):
    valid = _float(valid_window_energy)
    total = _float(total_window_energy, valid)
    return torch.clamp(_sum(valid) / _sum(total).clamp_min(eps), 0.0, 1.0)


def compensation_risk(
    torque_saturation: torch.Tensor,
    joint_limit_proximity: torch.Tensor,
    normalized_action_jerk: torch.Tensor,
    phase_alignment: torch.Tensor,
    weights: Weights = None,
) -> torch.Tensor:
    reference = _float(torque_saturation)
    values = [
        torch.clamp(reference, 0.0, 1.0),
        torch.clamp(_float(joint_limit_proximity, reference), 0.0, 1.0),
        torch.clamp(_float(normalized_action_jerk, reference), 0.0, 1.0),
        1.0 - torch.clamp(_float(phase_alignment, reference), 0.0, 1.0),
    ]
    return _weighted_mean(values, ["torque", "joint_limit", "jerk", "phase"], weights)


def compensation_efficiency(
    delta_contact_risk: torch.Tensor,
    delta_posture_risk: torch.Tensor,
    joint_energy: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    contact = _float(delta_contact_risk)
    posture = _float(delta_posture_risk, contact)
    energy = _float(joint_energy, contact)
    return (_sum(contact) + _sum(posture)) / _sum(energy).clamp_min(eps)
