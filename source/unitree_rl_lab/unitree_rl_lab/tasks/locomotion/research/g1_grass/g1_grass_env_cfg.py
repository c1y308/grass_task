"""G1 29-DoF grass-like locomotion task variants.

These task configs intentionally inherit the upstream
``Unitree-G1-29dof-Velocity`` environment and only add research metadata for
grass-like terrain distributions and curriculum progression. The actor policy
observation group remains proprioceptive-only.
"""

from __future__ import annotations

import importlib
from typing import Any

from isaaclab.utils import configclass
from isaaclab.managers import ObsTermCfg
from isaaclab.scene import SceneEntityCfg

from .grass_runtime import GrassRecorderCfg, grass_curriculum_term_cfg
from .risk_curriculum import FixedScheduleProgression, RiskGateProgression, SuccessRateGateProgression
from .terrain_bank import GrassTerrainImporter, spawn_grass_robot
from .terrain_cfg import GrassTerrainSchedule, GrassTerrainStage

from unitree_rl_lab.tasks.locomotion.mdp.observations import foot_contact


_BASE_G1_MODULE = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)
_BaseG1EnvCfg = _BASE_G1_MODULE.RobotEnvCfg

GRASS_TERRAIN_SCHEDULE = GrassTerrainSchedule()
GRASS_FINAL_DISTRIBUTION = GRASS_TERRAIN_SCHEDULE.final_distribution()
GRASS_POLICY_OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "foot_contact",
)

DEFAULT_FIXED_STAGE_TRANSITIONS = [5_000_000] * (GRASS_TERRAIN_SCHEDULE.num_stages() - 1)
DEFAULT_SUCCESS_THRESHOLD = 0.85
DEFAULT_MIN_TRANSITIONS_PER_STAGE = 5_000_000
DEFAULT_CONTACT_THRESHOLD = 0.50
DEFAULT_POSTURE_THRESHOLD = 0.50
DEFAULT_COMPENSATION_THRESHOLD = 0.50
PILOT_RUNTIME_CFG = {
    "step_unit": "aggregate_transitions",
    "window_episodes": 2048,
    "evaluation_interval_transitions": 1_000_000,
    "required_consecutive_passes": 3,
    "rollout_steps_per_env": 24,
}
DEFAULT_TERRAIN_SEED = 20260710
DEFAULT_TILES_PER_KIND = 16
DEFAULT_TILE_SIZE = 8.0
DEFAULT_COMPLIANT_K_REF = 165_000.0
DEFAULT_EFFECTIVE_FOOT_MASS = 0.5 * 33.34114202
DEFAULT_COMPLIANT_C_REF = 2.0 * (DEFAULT_COMPLIANT_K_REF * DEFAULT_EFFECTIVE_FOOT_MASS) ** 0.5


def _stage_by_name(name: str) -> GrassTerrainStage:
    for stage in GRASS_TERRAIN_SCHEDULE.stages:
        if stage.name == name:
            return stage
    raise ValueError(f"Unknown grass terrain stage: {name}.")


def _midpoint(value_range: tuple[float, float]) -> float:
    return 0.5 * (value_range[0] + value_range[1])


def _configure_grass_variant(
    cfg: Any,
    *,
    variant_name: str,
    distribution_mode: str,
    active_stage: GrassTerrainStage,
    initial_stage: GrassTerrainStage,
    final_stage: GrassTerrainStage,
    progression: Any | None = None,
) -> None:
    cfg.grass_variant_name = variant_name
    cfg.grass_distribution_mode = distribution_mode
    cfg.grass_terrain_schedule = GRASS_TERRAIN_SCHEDULE
    cfg.grass_terrain_schedule_log = GRASS_TERRAIN_SCHEDULE.to_dict()
    cfg.grass_initial_stage = initial_stage
    cfg.grass_initial_stage_index = next(
        index for index, stage in enumerate(GRASS_TERRAIN_SCHEDULE.stages) if stage.name == initial_stage.name
    )
    cfg.grass_active_distribution = active_stage
    cfg.grass_final_distribution = final_stage
    cfg.grass_curriculum_progression = progression
    cfg.grass_curriculum_progression_log = None if progression is None else progression.to_log_dict()
    cfg.grass_actor_policy_observation_terms = GRASS_POLICY_OBSERVATION_TERMS
    cfg.grass_runtime_cfg = dict(PILOT_RUNTIME_CFG)
    cfg.grass_runtime_log_dir = ""
    cfg.grass_log_dict = {
        "variant_name": variant_name,
        "distribution_mode": distribution_mode,
        "initial_stage": initial_stage.to_dict(),
        "active_distribution": active_stage.to_dict(),
        "final_distribution": final_stage.to_dict(),
        "terrain_schedule": GRASS_TERRAIN_SCHEDULE.to_dict(),
        "curriculum_progression": cfg.grass_curriculum_progression_log,
        "actor_policy_observation_terms": list(GRASS_POLICY_OBSERVATION_TERMS),
    }
    cfg.sim.physx.gpu_collision_stack_size = 2**28

    _keep_actor_policy_proprioceptive(cfg)
    _configure_terrain_bank(cfg)
    _disable_upstream_terrain_curriculum(cfg)
    _configure_runtime_managers(cfg)
    _fix_robot_contact_material(cfg)
    _apply_fixed_contact_material(cfg, active_stage)


def _keep_actor_policy_proprioceptive(cfg: Any) -> None:
    cfg.scene.height_scanner = None
    policy_cfg = cfg.observations.policy
    for terrain_obs_name in ("height_scanner", "height_scan", "terrain_scan", "visual_observations"):
        if hasattr(policy_cfg, terrain_obs_name):
            setattr(policy_cfg, terrain_obs_name, None)
    # Add foot contact observation as required by the paper:
    # "可由平台获得的接触相关状态" (contact-related states obtainable from the platform).
    # Binary foot contact is proprioceptive — on the real G1, it is obtained
    # from joint-torque or foot-force sensors.
    if not hasattr(policy_cfg, "foot_contact") or getattr(policy_cfg, "foot_contact", None) is None:
        setattr(
            policy_cfg,
            "foot_contact",
            ObsTermCfg(
                func=foot_contact,
                params={
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                    "force_threshold": 1.0,
                },
            ),
        )


def _configure_terrain_bank(cfg: Any) -> None:
    cfg.scene.robot.spawn.func = spawn_grass_robot
    terrain = cfg.scene.terrain
    terrain.class_type = GrassTerrainImporter
    terrain.grass_schedule = GRASS_TERRAIN_SCHEDULE
    terrain.grass_seed = DEFAULT_TERRAIN_SEED
    terrain.grass_variants_per_kind = DEFAULT_TILES_PER_KIND
    terrain.grass_tile_size = DEFAULT_TILE_SIZE
    terrain.grass_k_ref = DEFAULT_COMPLIANT_K_REF
    terrain.grass_c_ref = DEFAULT_COMPLIANT_C_REF
    terrain.grass_initial_stage_index = cfg.grass_initial_stage_index
    terrain.max_init_terrain_level = cfg.grass_initial_stage_index


def _disable_upstream_terrain_curriculum(cfg: Any) -> None:
    if getattr(cfg.curriculum, "terrain_levels", None) is not None:
        cfg.curriculum.terrain_levels = None
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False


def _configure_runtime_managers(cfg: Any) -> None:
    cfg.recorders = GrassRecorderCfg()
    cfg.curriculum.grass_runtime = grass_curriculum_term_cfg()


def _fix_robot_contact_material(cfg: Any) -> None:
    cfg.events.physics_material = None


def _apply_fixed_contact_material(cfg: Any, stage: GrassTerrainStage) -> None:
    friction_range = stage.friction_range
    material = cfg.scene.terrain.physics_material
    material.static_friction = _midpoint(friction_range)
    material.dynamic_friction = _midpoint(friction_range)
    material.friction_combine_mode = "min"


def _apply_play_overrides(cfg: Any) -> None:
    cfg.scene.num_envs = 32
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges
    cfg.recorders = {}
    cfg.curriculum.grass_runtime = None


def _make_fixed_schedule_progression() -> FixedScheduleProgression:
    return FixedScheduleProgression(stage_transition_counts=list(DEFAULT_FIXED_STAGE_TRANSITIONS))


def _make_success_gate_progression() -> SuccessRateGateProgression:
    return SuccessRateGateProgression(
        success_threshold=DEFAULT_SUCCESS_THRESHOLD,
        min_transitions_per_stage=DEFAULT_MIN_TRANSITIONS_PER_STAGE,
    )


def _make_risk_gate_progression() -> RiskGateProgression:
    return RiskGateProgression(
        success_threshold=DEFAULT_SUCCESS_THRESHOLD,
        contact_threshold=DEFAULT_CONTACT_THRESHOLD,
        posture_threshold=DEFAULT_POSTURE_THRESHOLD,
        compensation_threshold=DEFAULT_COMPENSATION_THRESHOLD,
        min_transitions_per_stage=DEFAULT_MIN_TRANSITIONS_PER_STAGE,
    )


def _apply_flat_rigid_variant(cfg: Any) -> None:
    flat_stage = _stage_by_name("flat_rigid")
    _configure_grass_variant(
        cfg,
        variant_name="B0_flat_rigid",
        distribution_mode="single_stage_flat_rigid",
        active_stage=flat_stage,
        initial_stage=flat_stage,
        final_stage=flat_stage,
    )


def _apply_coupled_random_variant(cfg: Any) -> None:
    _configure_grass_variant(
        cfg,
        variant_name="B1_coupled_random",
        distribution_mode="full_range_coupled_random",
        active_stage=GRASS_FINAL_DISTRIBUTION,
        initial_stage=GRASS_FINAL_DISTRIBUTION,
        final_stage=GRASS_FINAL_DISTRIBUTION,
    )


def _apply_fixed_schedule_variant(cfg: Any) -> None:
    flat_stage = _stage_by_name("flat_rigid")
    _configure_grass_variant(
        cfg,
        variant_name="B2_fixed_schedule",
        distribution_mode="fixed_schedule_curriculum",
        active_stage=flat_stage,
        initial_stage=flat_stage,
        final_stage=GRASS_FINAL_DISTRIBUTION,
        progression=_make_fixed_schedule_progression(),
    )


def _apply_success_gate_variant(cfg: Any) -> None:
    flat_stage = _stage_by_name("flat_rigid")
    _configure_grass_variant(
        cfg,
        variant_name="B3_success_gate",
        distribution_mode="success_rate_gated_curriculum",
        active_stage=flat_stage,
        initial_stage=flat_stage,
        final_stage=GRASS_FINAL_DISTRIBUTION,
        progression=_make_success_gate_progression(),
    )


def _apply_risk_gate_variant(cfg: Any) -> None:
    flat_stage = _stage_by_name("flat_rigid")
    _configure_grass_variant(
        cfg,
        variant_name="Ours_risk_gate",
        distribution_mode="risk_gated_curriculum",
        active_stage=flat_stage,
        initial_stage=flat_stage,
        final_stage=GRASS_FINAL_DISTRIBUTION,
        progression=_make_risk_gate_progression(),
    )


@configclass
class G1GrassFlatRigidEnvCfg(_BaseG1EnvCfg):
    """Flat-rigid grass experiment baseline."""

    def __post_init__(self):
        super().__post_init__()
        _apply_flat_rigid_variant(self)


@configclass
class G1GrassCoupledRandomEnvCfg(_BaseG1EnvCfg):
    """Full-range coupled grass randomization baseline."""

    def __post_init__(self):
        super().__post_init__()
        _apply_coupled_random_variant(self)


@configclass
class G1GrassFixedScheduleEnvCfg(_BaseG1EnvCfg):
    """Fixed-step grass terrain curriculum baseline."""

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_schedule_variant(self)


@configclass
class G1GrassSuccessGateEnvCfg(_BaseG1EnvCfg):
    """Success-rate gated grass terrain curriculum baseline."""

    def __post_init__(self):
        super().__post_init__()
        _apply_success_gate_variant(self)


@configclass
class G1GrassRiskGateEnvCfg(_BaseG1EnvCfg):
    """Risk-gated grass terrain curriculum task variant."""

    def __post_init__(self):
        super().__post_init__()
        _apply_risk_gate_variant(self)


@configclass
class G1GrassFlatRigidPlayEnvCfg(G1GrassFlatRigidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class G1GrassCoupledRandomPlayEnvCfg(G1GrassCoupledRandomEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class G1GrassFixedSchedulePlayEnvCfg(G1GrassFixedScheduleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class G1GrassSuccessGatePlayEnvCfg(G1GrassSuccessGateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class G1GrassRiskGatePlayEnvCfg(G1GrassRiskGateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)
