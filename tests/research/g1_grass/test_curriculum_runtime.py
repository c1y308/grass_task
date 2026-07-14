from __future__ import annotations

from curriculum_runtime import CurriculumRuntimeCfg, EpisodeSummary, GrassCurriculumController
from risk_curriculum import (
    CurriculumState,
    FixedScheduleProgression,
    RiskGateProgression,
    SuccessRateGateProgression,
)


def episode(stage=0, success=1.0, contact=0.1, posture=0.1, compensation=0.1):
    return EpisodeSummary(stage, success, contact, posture, compensation)


def test_aggregate_transition_counter_and_fixed_schedule():
    state = CurriculumState()
    controller = GrassCurriculumController(
        FixedScheduleProgression([98_304] * 4),
        CurriculumRuntimeCfg(window_episodes=1, evaluation_interval_transitions=1, required_consecutive_passes=1),
        state,
    )
    controller.update_counters(control_step=24, num_envs=4096)
    controller.submit([episode()])
    record = controller.evaluate()
    assert record["aggregate_transitions"] == 98_304
    assert record["local_ppo_iteration"] == 0
    assert record["promoted"] is True
    assert state.current_stage == 1


def test_success_gate_requires_three_consecutive_windows():
    progression = SuccessRateGateProgression(0.85, min_transitions_per_stage=0)
    controller = GrassCurriculumController(
        progression,
        CurriculumRuntimeCfg(window_episodes=2, evaluation_interval_transitions=1, required_consecutive_passes=3),
    )
    controller.submit([episode(), episode()])
    assert controller.evaluate()["promoted"] is False
    assert controller.evaluate()["promoted"] is False
    assert controller.evaluate()["promoted"] is True


def test_risk_gate_holds_when_success_gate_would_pass():
    state = CurriculumState(aggregate_transitions=10, last_promotion_transition=0)
    metrics = {"success_rate": 0.90, "contact_risk": 0.45, "posture_risk": 0.1, "compensation_risk": 0.1}
    success = SuccessRateGateProgression(0.85, 10).should_promote(state, metrics)
    risk = RiskGateProgression(0.85, 0.40, 0.40, 0.40, 10).should_promote(state, metrics)
    assert success.promote is True
    assert risk.promote is False
    assert risk.reason == "contact_failed"
    assert risk.gate_flags["success_pass"] is True


def test_stale_episodes_are_excluded_after_promotion():
    controller = GrassCurriculumController(
        FixedScheduleProgression([0] * 4),
        CurriculumRuntimeCfg(window_episodes=1, evaluation_interval_transitions=1, required_consecutive_passes=1),
    )
    controller.submit([episode(stage=0)])
    assert controller.evaluate()["promoted"] is True
    controller.submit([episode(stage=0), episode(stage=1, contact=0.2)])
    assert controller.window_size == 1
    assert controller.metrics()["contact_risk"] == 0.2


def test_final_stage_never_promotes():
    state = CurriculumState(current_stage=4)
    controller = GrassCurriculumController(
        FixedScheduleProgression([0] * 4),
        CurriculumRuntimeCfg(window_episodes=1, evaluation_interval_transitions=1, required_consecutive_passes=1),
        state,
    )
    controller.submit([episode(stage=4)])
    record = controller.evaluate()
    assert record["hold_reason"] == "final_stage"
    assert record["promoted"] is False
