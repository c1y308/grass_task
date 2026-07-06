"""Paper-level grass terrain stage schedule for G1 locomotion experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Range = tuple[float, float]


@dataclass(frozen=True)
class GrassTerrainStage:
    """Single grass terrain curriculum stage."""

    name: str
    lambda_level: float
    height_range: Range
    friction_range: Range
    stiffness_range: Range
    damping_range: Range
    transition_probability: float

    def __post_init__(self):
        object.__setattr__(self, "lambda_level", float(self.lambda_level))
        object.__setattr__(self, "transition_probability", float(self.transition_probability))

        if not 0.0 <= self.lambda_level <= 1.0:
            raise ValueError(f"lambda_level must be in [0, 1], got {self.lambda_level}.")
        if not 0.0 <= self.transition_probability <= 1.0:
            raise ValueError(
                f"transition_probability must be in [0, 1], got {self.transition_probability}."
            )

        for attr_name in ("height_range", "friction_range", "stiffness_range", "damping_range"):
            range_value = getattr(self, attr_name)
            if len(range_value) != 2:
                raise ValueError(f"{attr_name} must contain exactly two values, got {range_value}.")

            lower, upper = float(range_value[0]), float(range_value[1])
            if lower > upper:
                raise ValueError(f"{attr_name} lower bound must be <= upper bound, got {range_value}.")
            object.__setattr__(self, attr_name, (lower, upper))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation."""
        return {
            "name": self.name,
            "lambda_level": self.lambda_level,
            "height_range": list(self.height_range),
            "friction_range": list(self.friction_range),
            "stiffness_range": list(self.stiffness_range),
            "damping_range": list(self.damping_range),
            "transition_probability": self.transition_probability,
        }


@dataclass(frozen=True)
class GrassTerrainSchedule:
    """Fixed five-stage grass terrain schedule for research experiments."""

    flat_rigid: GrassTerrainStage = field(
        default_factory=lambda: GrassTerrainStage(
            name="flat_rigid",
            lambda_level=0.00,
            height_range=(0.0, 0.0),
            friction_range=(0.9, 1.1),
            stiffness_range=(1.0, 1.0),
            damping_range=(1.0, 1.0),
            transition_probability=0.0,
        )
    )
    mild_grass: GrassTerrainStage = field(
        default_factory=lambda: GrassTerrainStage(
            name="mild_grass",
            lambda_level=0.25,
            height_range=(0.0, 0.03),
            friction_range=(0.7, 1.0),
            stiffness_range=(0.7, 1.0),
            damping_range=(0.7, 1.2),
            transition_probability=0.10,
        )
    )
    uneven_grass: GrassTerrainStage = field(
        default_factory=lambda: GrassTerrainStage(
            name="uneven_grass",
            lambda_level=0.50,
            height_range=(0.0, 0.06),
            friction_range=(0.55, 1.0),
            stiffness_range=(0.5, 1.0),
            damping_range=(0.5, 1.4),
            transition_probability=0.20,
        )
    )
    wet_soft_grass: GrassTerrainStage = field(
        default_factory=lambda: GrassTerrainStage(
            name="wet_soft_grass",
            lambda_level=0.75,
            height_range=(0.0, 0.09),
            friction_range=(0.40, 0.9),
            stiffness_range=(0.35, 0.9),
            damping_range=(0.4, 1.6),
            transition_probability=0.35,
        )
    )
    extreme_coupled: GrassTerrainStage = field(
        default_factory=lambda: GrassTerrainStage(
            name="extreme_coupled",
            lambda_level=1.00,
            height_range=(0.0, 0.12),
            friction_range=(0.30, 0.85),
            stiffness_range=(0.25, 0.85),
            damping_range=(0.3, 1.8),
            transition_probability=0.50,
        )
    )

    @property
    def stages(self) -> tuple[GrassTerrainStage, ...]:
        return (
            self.flat_rigid,
            self.mild_grass,
            self.uneven_grass,
            self.wet_soft_grass,
            self.extreme_coupled,
        )

    def get_stage(self, index: int) -> GrassTerrainStage:
        """Return a stage by zero-based index."""
        if index < 0 or index >= self.num_stages():
            raise IndexError(f"stage index {index} out of range [0, {self.num_stages() - 1}].")
        return self.stages[index]

    def num_stages(self) -> int:
        """Return the number of schedule stages."""
        return len(self.stages)

    def final_distribution(self) -> GrassTerrainStage:
        """Return the final terrain distribution stage."""
        return self.stages[-1]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation."""
        return {"stages": [stage.to_dict() for stage in self.stages]}
