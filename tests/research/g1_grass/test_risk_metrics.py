from __future__ import annotations

import pytest
import torch

import risk_metrics as metrics


def test_contact_components_are_dimensionless():
    expected = torch.tensor([[True, True, False, False]])
    real = torch.tensor([[True, False, True, False]])
    force = torch.tensor([[20.0, 0.0, 20.0, 0.0]])
    assert metrics.unexpected_contact_ratio(expected, real).item() == pytest.approx(0.5)
    assert metrics.missed_support_ratio(expected, force, 10.0).item() == pytest.approx(0.5)
    assert metrics.contact_window_iou(expected, real).item() == pytest.approx(1.0 / 3.0)
    risk = metrics.contact_risk(
        torch.tensor([0.2]), torch.tensor([0.4]), torch.tensor([0.6]), torch.tensor([0.8])
    )
    assert risk.item() == pytest.approx(0.5)


def test_touchdown_normalization_and_unmatched_cap():
    value = metrics.normalized_touchdown_error(torch.tensor([0.18]), torch.tensor([0.0]), 0.10)
    assert value.item() == 1.0


def test_rising_edge_counts_persistent_contact_once():
    sequence = [False, True, True, True, False, True]
    previous = torch.tensor([False])
    count = 0
    for item in sequence:
        current = torch.tensor([item])
        count += int(metrics.rising_edge(current, previous).item())
        previous = current
    assert count == 2


def test_action_jerk_is_mean_squared_physical_acceleration():
    current = torch.tensor([[0.04, 0.04]])
    previous = torch.tensor([[0.01, 0.01]])
    previous_previous = torch.zeros_like(current)
    value = metrics.action_jerk(current, previous, previous_previous, dt=0.02)
    assert value.item() == pytest.approx(2500.0)
    normalized = metrics.action_jerk(
        current, previous, previous_previous, dt=0.02, reference_acceleration=100.0
    )
    assert normalized.item() == pytest.approx(0.25)


def test_compensation_risk_uses_phase_misalignment():
    risk = metrics.compensation_risk(
        torch.tensor([0.1]), torch.tensor([0.2]), torch.tensor([0.3]), torch.tensor([0.2])
    )
    assert risk.item() == pytest.approx(0.35)


def test_posture_risk_normalization():
    roll = torch.tensor([[0.2, 0.2]])
    pitch = torch.zeros_like(roll)
    ang_vel = torch.zeros((1, 2, 2))
    assert metrics.posture_risk(roll, pitch, ang_vel).item() == pytest.approx(1.0 / 3.0)
