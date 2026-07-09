"""Pure-Python curriculum progression helpers for G1 grass-risk experiments."""

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
    """Decision returned by a curriculum progression rule."""

    promote: bool
    reason: str
    failed_gates: list[str] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly log dictionary."""
        return {
            "promote": self.promote,
            "reason": self.reason,
            "failed_gates": list(self.failed_gates),
        }


@dataclass
class CurriculumState:
    """Mutable curriculum state owned by the future environment integration layer."""

    current_stage: int = 0
    global_step: int = 0
    last_promotion_step: int = 0
    promotion_history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.current_stage = int(self.current_stage)
        self.global_step = int(self.global_step)
        self.last_promotion_step = int(self.last_promotion_step)
        self.promotion_history = [dict(item) for item in self.promotion_history]

        if self.current_stage < 0:
            raise ValueError(f"current_stage must be non-negative, got {self.current_stage}.")
        if self.global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {self.global_step}.")
        if self.last_promotion_step < 0:
            raise ValueError(f"last_promotion_step must be non-negative, got {self.last_promotion_step}.")
        if self.last_promotion_step > self.global_step:
            raise ValueError(
                "last_promotion_step must be <= global_step, "
                f"got {self.last_promotion_step} > {self.global_step}."
            )

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly log dictionary."""
        return {
            "current_stage": self.current_stage,
            "global_step": self.global_step,
            "last_promotion_step": self.last_promotion_step,
            "promotion_history": [dict(item) for item in self.promotion_history],
        }


@dataclass
class FixedScheduleProgression:
    """Fixed-step schedule that promotes when the next cumulative stage boundary is reached."""

    stage_steps: list[int]

    def __post_init__(self):
        self.stage_steps = [int(step) for step in self.stage_steps]
        if any(step < 0 for step in self.stage_steps):
            raise ValueError(f"stage_steps must be non-negative, got {self.stage_steps}.")

    @property
    def stage_boundaries(self) -> list[int]:
        """Cumulative global-step boundaries for each promotion."""
        boundaries: list[int] = []
        total = 0
        for step in self.stage_steps:
            total += step
            boundaries.append(total)
        return boundaries

    def should_promote(self, state: CurriculumState, metrics: Metrics) -> PromotionDecision:
        """Promote only when global_step reaches the next fixed schedule boundary."""
        del metrics
        if state.current_stage >= len(self.stage_steps):
            return PromotionDecision(False, "no_next_stage_boundary", ["stage_boundary"])

        next_boundary = self.stage_boundaries[state.current_stage]
        if state.global_step >= next_boundary:
            return PromotionDecision(True, "fixed_schedule_boundary_reached")
        return PromotionDecision(False, "fixed_schedule_boundary_not_reached", ["stage_boundary"])

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly log dictionary."""
        return {
            "type": self.__class__.__name__,
            "stage_steps": list(self.stage_steps),
            "stage_boundaries": self.stage_boundaries,
        }


@dataclass
class SuccessRateGateProgression:
    """Success-rate gated progression with a minimum dwell time per stage."""

    success_threshold: float
    min_steps_per_stage: int

    def __post_init__(self):
        self.success_threshold = float(self.success_threshold)
        self.min_steps_per_stage = int(self.min_steps_per_stage)
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError(f"success_threshold must be in [0, 1], got {self.success_threshold}.")
        if self.min_steps_per_stage < 0:
            raise ValueError(f"min_steps_per_stage must be non-negative, got {self.min_steps_per_stage}.")

    def should_promote(self, state: CurriculumState, metrics: Metrics) -> PromotionDecision:
        """Promote only when success_rate passes threshold and the stage dwell time is satisfied."""
        failed_gates: list[str] = []
        success_rate = _metric_value(metrics, "success_rate")

        if _steps_since_promotion(state) < self.min_steps_per_stage:
            failed_gates.append("min_steps_per_stage")
        if success_rate is None or success_rate < self.success_threshold:
            failed_gates.append("success_rate")

        if failed_gates:
            return PromotionDecision(False, "success_rate_gate_failed", failed_gates)
        return PromotionDecision(True, "success_rate_gate_passed")

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly log dictionary."""
        return {
            "type": self.__class__.__name__,
            "success_threshold": self.success_threshold,
            "min_steps_per_stage": self.min_steps_per_stage,
        }


@dataclass
class RiskGateProgression:
    """Risk-gated progression with success, contact, posture, and compensation gates.

    ``contact_risk`` is expected to be the aggregate of touchdown timing error, slip ratio,
    unexpected contact, and missed support. ``compensation_risk`` must be a safety-boundary
    risk such as torque saturation, joint-limit proximity/margin, action jerk, or
    ``1 - compensation_phase_alignment``; compensation efficiency is diagnostic-only.
    """

    success_threshold: float
    contact_threshold: float
    posture_threshold: float
    compensation_threshold: float
    min_steps_per_stage: int

    def __post_init__(self):
        self.success_threshold = float(self.success_threshold)
        self.contact_threshold = float(self.contact_threshold)
        self.posture_threshold = float(self.posture_threshold)
        self.compensation_threshold = float(self.compensation_threshold)
        self.min_steps_per_stage = int(self.min_steps_per_stage)

        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError(f"success_threshold must be in [0, 1], got {self.success_threshold}.")
        for name in ("contact_threshold", "posture_threshold", "compensation_threshold"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.min_steps_per_stage < 0:
            raise ValueError(f"min_steps_per_stage must be non-negative, got {self.min_steps_per_stage}.")

    def should_promote(self, state: CurriculumState, metrics: Metrics) -> PromotionDecision:
        """Promote only when success, three named risk gates, and minimum dwell time all pass."""
        failed_gates: list[str] = []
        success_rate = _metric_value(metrics, "success_rate")
        contact_risk = _metric_value(metrics, "contact_risk")
        posture_risk = _metric_value(metrics, "posture_risk")
        compensation_risk = _metric_value(metrics, "compensation_risk")

        if _steps_since_promotion(state) < self.min_steps_per_stage:
            failed_gates.append("min_steps_per_stage")
        if success_rate is None or success_rate < self.success_threshold:
            failed_gates.append("success_rate")
        if contact_risk is None or contact_risk > self.contact_threshold:
            failed_gates.append("contact_risk")
        if posture_risk is None or posture_risk > self.posture_threshold:
            failed_gates.append("posture_risk")
        if compensation_risk is None or compensation_risk > self.compensation_threshold:
            failed_gates.append("compensation_risk")

        if failed_gates:
            return PromotionDecision(False, "risk_gate_failed", failed_gates)
        return PromotionDecision(True, "risk_gate_passed")

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly log dictionary."""
        return {
            "type": self.__class__.__name__,
            "success_threshold": self.success_threshold,
            "contact_threshold": self.contact_threshold,
            "posture_threshold": self.posture_threshold,
            "compensation_threshold": self.compensation_threshold,
            "min_steps_per_stage": self.min_steps_per_stage,
        }


def _steps_since_promotion(state: CurriculumState) -> int:
    return state.global_step - state.last_promotion_step


def _metric_value(metrics: Metrics, key: str) -> float | None:
    if key not in metrics:
        return None

    value = metrics[key]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


if __name__ == "__main__":
    state = CurriculumState(current_stage=1, global_step=250, last_promotion_step=100)
    metrics = {
        "success_rate": 0.92,
        "contact_risk": 0.08,
        "posture_risk": 0.10,
        "compensation_risk": 0.12,
    }
    progressions = {
        "fixed": FixedScheduleProgression(stage_steps=[100, 150, 200]),
        "success": SuccessRateGateProgression(success_threshold=0.9, min_steps_per_stage=100),
        "risk": RiskGateProgression(
            success_threshold=0.9,
            contact_threshold=0.1,
            posture_threshold=0.15,
            compensation_threshold=0.2,
            min_steps_per_stage=100,
        ),
    }
    decisions = {
        name: progression.should_promote(state, metrics).to_log_dict()
        for name, progression in progressions.items()
    }
    print(decisions)
