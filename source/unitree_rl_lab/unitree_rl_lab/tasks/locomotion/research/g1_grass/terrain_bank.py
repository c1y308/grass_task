"""Deterministic multi-material terrain tile bank for grass curricula."""

from __future__ import annotations

import math

import numpy as np
import torch
import trimesh
from pxr import UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.utils import create_prim_from_mesh


@sim_utils.clone
def spawn_grass_robot(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the G1 with one fixed high-friction, min-combine contact material."""
    prim = sim_utils.spawn_from_urdf.__wrapped__(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    material_path = f"{prim_path}/grassRobotContactMaterial"
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.5,
        dynamic_friction=1.5,
        restitution=0.0,
        friction_combine_mode="min",
        restitution_combine_mode="min",
    )
    material.func(material_path, material)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(
        UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path)),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    return prim


class GrassTerrainImporter(TerrainImporter):
    """Import individually materialized terrain tiles and sample them at reset."""

    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device
        self.terrain_prim_paths: list[str] = []
        self._terrain_flat_patches = {}
        self._rng = np.random.default_rng(int(cfg.grass_seed))
        self._torch_generator = torch.Generator(device=self.device)
        self._torch_generator.manual_seed(int(cfg.grass_seed))
        self.schedule = cfg.grass_schedule
        self.num_variants = int(cfg.grass_variants_per_kind)
        self.tile_size = float(cfg.grass_tile_size)
        self.k_ref = float(cfg.grass_k_ref)
        self.c_ref = float(cfg.grass_c_ref)
        self._build_bank()
        self.grass_tile_bank = self
        initial_stage = int(cfg.grass_initial_stage_index)
        initial_columns = torch.arange(cfg.num_envs, device=self.device) % self.num_variants
        self.terrain_levels = torch.full((cfg.num_envs,), initial_stage, device=self.device, dtype=torch.long)
        self.terrain_types = initial_columns.clone()
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
        self.set_debug_vis(cfg.debug_vis)

    def _build_bank(self) -> None:
        num_stages = self.schedule.num_stages()
        num_columns = 2 * self.num_variants
        meshes: list[trimesh.Trimesh] = []
        material_groups: list[tuple[int, int, sim_utils.RigidBodyMaterialCfg]] = []
        face_offset = 0
        origins = np.zeros((num_stages, num_columns, 3), dtype=np.float32)
        parameters = {
            name: np.zeros((num_stages, num_columns), dtype=np.float32)
            for name in ("height_amplitude", "friction", "stiffness", "damping", "transition")
        }
        for stage_id, stage in enumerate(self.schedule.stages):
            for column in range(num_columns):
                transition = column >= self.num_variants
                height = self._rng.uniform(*stage.height_range)
                friction = self._rng.uniform(*stage.friction_range)
                if stage.stiffness_range == (0.0, 0.0) and stage.damping_range == (0.0, 0.0):
                    stiffness = damping = 0.0
                else:
                    stiffness = self.k_ref * self._rng.uniform(*stage.stiffness_range)
                    damping = self.c_ref * self._rng.uniform(*stage.damping_range)
                x_offset = stage_id * self.tile_size
                y_offset = column * self.tile_size
                mesh = self._height_mesh(height, transition)
                mesh.apply_translation((x_offset, y_offset, 0.0))
                grass_material = self._material(friction, stiffness, damping)
                if transition and stage_id > 0:
                    flat_faces, grass_faces = self._split_transition_faces(mesh)
                    for faces, material in (
                        (flat_faces, self._material(1.0, 0.0, 0.0)),
                        (grass_faces, grass_material),
                    ):
                        submesh = trimesh.Trimesh(mesh.vertices, faces, process=False)
                        meshes.append(submesh)
                        material_groups.append((face_offset, len(submesh.faces), material))
                        face_offset += len(submesh.faces)
                else:
                    meshes.append(mesh)
                    material_groups.append((face_offset, len(mesh.faces), grass_material))
                    face_offset += len(mesh.faces)
                origins[stage_id, column] = (x_offset + self.tile_size / 2, y_offset + self.tile_size / 2, 0.0)
                parameters["height_amplitude"][stage_id, column] = height
                parameters["friction"][stage_id, column] = friction
                parameters["stiffness"][stage_id, column] = stiffness
                parameters["damping"][stage_id, column] = damping
                parameters["transition"][stage_id, column] = float(transition)
        bank_path = f"{self.cfg.prim_path}/grass_tile_bank"
        create_prim_from_mesh(
            bank_path,
            trimesh.util.concatenate(meshes),
            visual_material=self.cfg.visual_material,
            physics_material=self._material(1.0, 0.0, 0.0),
        )
        mesh_prim = sim_utils.get_current_stage().GetPrimAtPath(f"{bank_path}/mesh")
        UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr().Set("none")
        for index, (start, count, material_cfg) in enumerate(material_groups):
            material_path = f"{bank_path}/physics_material_{index}"
            material_cfg.func(material_path, material_cfg)
            subset = UsdGeom.Subset.Define(mesh_prim.GetStage(), f"{bank_path}/mesh/material_subset_{index}")
            subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
            subset.CreateFamilyNameAttr().Set(UsdShade.Tokens.materialBind)
            subset.CreateIndicesAttr().Set(list(range(start, start + count)))
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
                UsdShade.Material(mesh_prim.GetStage().GetPrimAtPath(material_path)),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        self.terrain_prim_paths = [bank_path]
        self.terrain_origins = torch.tensor(origins, device=self.device)
        self.parameters = {name: torch.tensor(value, device=self.device) for name, value in parameters.items()}
        self.max_terrain_level = num_stages

    def sample(self, stage_ids: torch.Tensor, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        probabilities = torch.tensor(
            [stage.transition_probability for stage in self.schedule.stages],
            device=self.device,
            dtype=torch.float,
        )
        random_values = torch.rand(len(env_ids), device=self.device, generator=self._torch_generator)
        transition = random_values < probabilities[stage_ids]
        variants = torch.randint(
            self.num_variants,
            (len(env_ids),),
            device=self.device,
            generator=self._torch_generator,
        )
        columns = variants + transition.long() * self.num_variants
        self.terrain_levels[env_ids] = stage_ids
        self.terrain_types[env_ids] = columns
        self.env_origins[env_ids] = self.terrain_origins[stage_ids, columns]
        return {
            "transition": transition,
            "tile_column": columns,
            **{name: values[stage_ids, columns] for name, values in self.parameters.items() if name != "transition"},
        }

    def _height_mesh(self, amplitude: float, transition: bool) -> trimesh.Trimesh:
        resolution = 0.20
        points = int(round(self.tile_size / resolution)) + 1
        coarse_points = 9
        coarse = self._rng.uniform(-amplitude, amplitude, (coarse_points, coarse_points))
        coarse_axis = np.linspace(0.0, self.tile_size, coarse_points)
        axis = np.linspace(0.0, self.tile_size, points)
        intermediate = np.vstack([np.interp(axis, coarse_axis, row) for row in coarse])
        heights = np.vstack([np.interp(axis, coarse_axis, intermediate[:, index]) for index in range(points)]).T
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        radius = np.sqrt((xx - self.tile_size / 2) ** 2 + (yy - self.tile_size / 2) ** 2)
        flat_radius = 1.25 if transition else 0.55
        blend_width = 0.50
        blend = np.clip((radius - flat_radius) / blend_width, 0.0, 1.0)
        heights *= blend
        vertices = np.column_stack((xx.ravel(), yy.ravel(), heights.ravel()))
        faces = []
        for i in range(points - 1):
            for j in range(points - 1):
                a = i * points + j
                b = (i + 1) * points + j
                faces.append((a, b, a + 1))
                faces.append((a + 1, b, b + 1))
        return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)

    def _split_transition_faces(self, mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
        center = np.array(
            [
                math.floor(mesh.vertices[:, 0].min() / self.tile_size) * self.tile_size + self.tile_size / 2,
                math.floor(mesh.vertices[:, 1].min() / self.tile_size) * self.tile_size + self.tile_size / 2,
            ]
        )
        centroids = mesh.vertices[mesh.faces].mean(axis=1)[:, :2]
        flat_mask = np.linalg.norm(centroids - center, axis=1) <= 1.50
        return mesh.faces[flat_mask], mesh.faces[~flat_mask]

    @staticmethod
    def _material(friction: float, stiffness: float, damping: float):
        return sim_utils.RigidBodyMaterialCfg(
            static_friction=float(friction),
            dynamic_friction=float(friction),
            restitution=0.0,
            friction_combine_mode="min",
            restitution_combine_mode="min",
            compliant_contact_stiffness=float(stiffness),
            compliant_contact_damping=float(damping),
        )
