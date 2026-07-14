"""Framework-independent rolling-window controller for grass curricula."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

try:
    from .risk_curriculum import CurriculumState, PromotionDecision
except ImportError:  # Allows direct loading in pure-Python tests.
    from risk_curriculum import CurriculumState, PromotionDecision


@dataclass(frozen=True)
class EpisodeSummary:
    stage_id: int
    success: float
    contact_risk: float
    posture_risk: float
    compensation_risk: float
    transition_tile: float = 0.0
    height_amplitude: float = 0.0
    friction: float = 1.0
    stiffness: float = 0.0
    damping: float = 0.0
    episode_reward: float = 0.0
    episode_length: int = 0
    termination_reason: str = "unknown"


@dataclass(frozen=True)
class CurriculumRuntimeCfg:
    num_stages: int = 5
    window_episodes: int = 2048
    evaluation_interval_transitions: int = 1_000_000
    required_consecutive_passes: int = 3
    rollout_steps_per_env: int = 24

    def __post_init__(self):
        for name in (
            "num_stages",
            "window_episodes",
            "evaluation_interval_transitions",
            "required_consecutive_passes",
            "rollout_steps_per_env",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class GrassCurriculumController:
    """Owns the global stage and rejects stale summaries after promotion."""

    def __init__(self, progression: Any, cfg: CurriculumRuntimeCfg, state: CurriculumState | None = None):
        self.progression = progression
        self.cfg = cfg
        self.state = state or CurriculumState()
        self._window: deque[EpisodeSummary] = deque(maxlen=cfg.window_episodes)
        self._next_evaluation = cfg.evaluation_interval_transitions

    @property
    def window_size(self) -> int:
        return len(self._window)

    def update_counters(self, control_step: int, num_envs: int) -> None:
        self.state.control_step = int(control_step)
        self.state.aggregate_transitions = int(control_step) * int(num_envs)

    def submit(self, summaries: list[EpisodeSummary]) -> None:
        self._window.extend(summary for summary in summaries if summary.stage_id == self.state.current_stage)

    def metrics(self) -> dict[str, float]:
        if not self._window:
            return {name: float("nan") for name in self._metric_fields()}
        values = {name: fmean(getattr(item, name) for item in self._window) for name in self._metric_fields()}
        return {
            "success_rate": values["success"],
            "contact_risk": values["contact_risk"],
            "posture_risk": values["posture_risk"],
            "compensation_risk": values["compensation_risk"],
        }

    def evaluation_due(self) -> bool:
        return self.state.aggregate_transitions >= self._next_evaluation

    def evaluate_if_due(self) -> dict[str, Any] | None:
        if not self.evaluation_due():
            return None
        while self._next_evaluation <= self.state.aggregate_transitions:
            self._next_evaluation += self.cfg.evaluation_interval_transitions
        return self.evaluate()

    def evaluate(self) -> dict[str, Any]:
        metrics = self.metrics()
        stage_before = self.state.current_stage
        dwell_transitions = self.state.dwell_transitions
        window_episodes = self.window_size
        if stage_before >= self.cfg.num_stages - 1:
            decision = PromotionDecision(False, "final_stage", gate_flags=_empty_gate_flags())
        elif self.window_size < self.cfg.window_episodes:
            decision = PromotionDecision(
                False,
                "insufficient_window",
                ["window"],
                {**_empty_gate_flags(), "window_pass": False},
            )
        else:
            decision = self.progression.should_promote(self.state, metrics)

        gate_pass = decision.promote
        self.state.consecutive_passes = self.state.consecutive_passes + 1 if gate_pass else 0
        decision_consecutive_passes = self.state.consecutive_passes
        promote = gate_pass and decision_consecutive_passes >= self.cfg.required_consecutive_passes
        if promote:
            self.state.current_stage += 1
            self.state.last_promotion_transition = self.state.aggregate_transitions
            self.state.consecutive_passes = 0
            self._window.clear()

        record = {
            "control_step": self.state.control_step,
            "aggregate_transitions": self.state.aggregate_transitions,
            "local_ppo_iteration": max(self.state.control_step - 1, 0) // self.cfg.rollout_steps_per_env,
            "stage_before": stage_before,
            "stage_after": self.state.current_stage,
            "dwell_transitions": dwell_transitions,
            "window_episodes": window_episodes,
            **metrics,
            **decision.gate_flags,
            "gate_pass": gate_pass,
            "consecutive_passes": decision_consecutive_passes,
            "promoted": promote,
            "hold_reason": "" if gate_pass else decision.reason,
        }
        if promote:
            self.state.promotion_history.append(dict(record))
        return record

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_log_dict(),
            "window": [asdict(item) for item in self._window],
            "next_evaluation": self._next_evaluation,
        }

    @staticmethod
    def _metric_fields() -> tuple[str, ...]:
        return ("success", "contact_risk", "posture_risk", "compensation_risk")


def _empty_gate_flags() -> dict[str, bool]:
    return {
        "dwell_pass": False,
        "success_pass": False,
        "contact_pass": False,
        "posture_pass": False,
        "compensation_pass": False,
    }
