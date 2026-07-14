"""Pure-Python progression rules for the G1 grass curriculum."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


Metrics = Mapping[str, Any]


__all__ = [
    "CurriculumState",
    "FixedScheduleProgression",
    "PromotionDecision",
    "RiskGateProgression",
    "SuccessRateGateProgression",
]


@dataclass
class PromotionDecision:
    """One auditable curriculum decision."""

    promote: bool
    reason: str
    failed_gates: list[str] = field(default_factory=list)
    gate_flags: dict[str, bool] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "reason": self.reason,
            "failed_gates": list(self.failed_gates),
            "gate_flags": dict(self.gate_flags),
        }


@dataclass
class CurriculumState:
    """Global monotonic curriculum state measured in aggregate transitions."""

    current_stage: int = 0
    control_step: int = 0
    aggregate_transitions: int = 0
    last_promotion_transition: int = 0
    consecutive_passes: int = 0
    promotion_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        for name in (
            "current_stage",
            "control_step",
            "aggregate_transitions",
            "last_promotion_transition",
            "consecutive_passes",
        ):
            setattr(self, name, int(getattr(self, name)))
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}.")
        if self.last_promotion_transition > self.aggregate_transitions:
            raise ValueError("last_promotion_transition must not exceed aggregate_transitions.")
        self.promotion_history = [dict(item) for item in self.promotion_history]

    @property
    def dwell_transitions(self) -> int:
        return self.aggregate_transitions - self.last_promotion_transition

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "control_step": self.control_step,
            "aggregate_transitions": self.aggregate_transitions,
            "last_promotion_transition": self.last_promotion_transition,
            "dwell_transitions": self.dwell_transitions,
            "consecutive_passes": self.consecutive_passes,
            "promotion_history": [dict(item) for item in self.promotion_history],
        }


@dataclass
class FixedScheduleProgression:
    """Promote after fixed aggregate-transition dwell counts."""

    stage_transition_counts: list[int]

    def __post_init__(self):
        self.stage_transition_counts = [int(value) for value in self.stage_transition_counts]
        if any(value < 0 for value in self.stage_transition_counts):
            raise ValueError("stage_transition_counts must be non-negative.")

    def should_promote(self, state: CurriculumState, metrics: Metrics) -> PromotionDecision:
        del metrics
        if state.current_stage >= len(self.stage_transition_counts):
            return PromotionDecision(False, "final_stage", gate_flags={"dwell_pass": False})
        dwell_pass = state.dwell_transitions >= self.stage_transition_counts[state.current_stage]
        return PromotionDecision(
            dwell_pass,
            "fixed_schedule_passed" if dwell_pass else "insufficient_dwell",
            [] if dwell_pass else ["dwell"],
            {"dwell_pass": dwell_pass},
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "step_unit": "aggregate_transitions",
            "stage_transition_counts": list(self.stage_transition_counts),
        }


@dataclass
class SuccessRateGateProgression:
    success_threshold: float
    min_transitions_per_stage: int

    def __post_init__(self):
        self.success_threshold = float(self.success_threshold)
        self.min_transitions_per_stage = int(self.min_transitions_per_stage)
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in [0, 1].")
        if self.min_transitions_per_stage < 0:
            raise ValueError("min_transitions_per_stage must be non-negative.")

    def should_promote(self, state: CurriculumState, metrics: Metrics) -> PromotionDecision:
        success_rate = _metric_value(metrics, "success_rate")
        flags = {
            "dwell_pass": state.dwell_transitions >= self.min_transitions_per_stage,
            "success_pass": success_rate is not None and success_rate >= self.success_threshold,
            "contact_pass": True,
            "posture_pass": True,
            "compensation_pass": True,
        }
        failed = [name.removesuffix("_pass") for name, passed in flags.items() if not passed]
        reason = _first_hold_reason(flags)
        return PromotionDecision(not failed, "success_gate_passed" if not failed else reason, failed, flags)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "step_unit": "aggregate_transitions",
            "success_threshold": self.success_threshold,
            "min_transitions_per_stage": self.min_transitions_per_stage,
        }


@dataclass
class RiskGateProgression:
    success_threshold: float
    contact_threshold: float
    posture_threshold: float
    compensation_threshold: float
    min_transitions_per_stage: int

    def __post_init__(self):
        self.success_threshold = float(self.success_threshold)
        self.contact_threshold = float(self.contact_threshold)
        self.posture_threshold = float(self.posture_threshold)
        self.compensation_threshold = float(self.compensation_threshold)
        self.min_transitions_per_stage = int(self.min_transitions_per_stage)
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in [0, 1].")
        for name in ("contact_threshold", "posture_threshold", "compensation_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.min_transitions_per_stage < 0:
            raise ValueError("min_transitions_per_stage must be non-negative.")

    def should_promote(self, state: CurriculumState, metrics: Metrics) -> PromotionDecision:
        success_rate = _metric_value(metrics, "success_rate")
        contact_risk = _metric_value(metrics, "contact_risk")
        posture_risk = _metric_value(metrics, "posture_risk")
        compensation_risk = _metric_value(metrics, "compensation_risk")
        flags = {
            "dwell_pass": state.dwell_transitions >= self.min_transitions_per_stage,
            "success_pass": success_rate is not None and success_rate >= self.success_threshold,
            "contact_pass": contact_risk is not None and contact_risk <= self.contact_threshold,
            "posture_pass": posture_risk is not None and posture_risk <= self.posture_threshold,
            "compensation_pass": compensation_risk is not None
            and compensation_risk <= self.compensation_threshold,
        }
        failed = [name.removesuffix("_pass") for name, passed in flags.items() if not passed]
        reason = _first_hold_reason(flags)
        return PromotionDecision(not failed, "risk_gate_passed" if not failed else reason, failed, flags)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "step_unit": "aggregate_transitions",
            "success_threshold": self.success_threshold,
            "contact_threshold": self.contact_threshold,
            "posture_threshold": self.posture_threshold,
            "compensation_threshold": self.compensation_threshold,
            "min_transitions_per_stage": self.min_transitions_per_stage,
        }


def _first_hold_reason(flags: Mapping[str, bool]) -> str:
    priority = (
        ("dwell_pass", "insufficient_dwell"),
        ("success_pass", "success_failed"),
        ("contact_pass", "contact_failed"),
        ("posture_pass", "posture_failed"),
        ("compensation_pass", "compensation_failed"),
    )
    return next((reason for flag, reason in priority if flag in flags and not flags[flag]), "gate_passed")


def _metric_value(metrics: Metrics, key: str) -> float | None:
    if key not in metrics:
        return None
    value = metrics[key]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)
