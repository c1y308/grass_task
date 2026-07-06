# G1 29-DoF Locomotion Repo Inspection

本文检查当前 `Unitree-G1-29dof-Velocity` 任务，并总结如何在不破坏上游任务的前提下新增一个 grass-risk research variant。本次检查只新增文档，不修改训练代码。

## 1. Task registration 机制

全仓库的任务注册依赖 Python import side effect：

- `source/unitree_rl_lab/unitree_rl_lab/tasks/__init__.py:5-10` 从 `isaaclab_tasks.utils` 导入 `import_packages`，并执行 `import_packages(__name__, _BLACKLIST_PKGS)`。
- 当 package scan/import 触发 `unitree_rl_lab.tasks.locomotion.robots.g1.29dof` 的 `__init__.py` 时，该文件执行 `gym.register(...)`。

G1 29-DoF locomotion 的实际 Gym registration 位于：

`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py`

注册内容：

- `id="Unitree-G1-29dof-Velocity"`
- `entry_point="isaaclab.envs:ManagerBasedRLEnv"`
- `disable_env_checker=True`
- `kwargs["env_cfg_entry_point"] = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg:RobotEnvCfg"`
- `kwargs["play_env_cfg_entry_point"] = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg:RobotPlayEnvCfg"`
- `kwargs["rsl_rl_cfg_entry_point"] = "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"`

Implication for a research variant:

- 不要复用或改写 `Unitree-G1-29dof-Velocity` 这个 Gym ID。
- 不要直接改 `29dof/velocity_env_cfg.py` 的上游 class，避免改变基线任务行为。
- 新增 variant 应该有独立 Gym ID，例如 `Unitree-G1-29dof-GrassRisk-Velocity`。
- 最稳妥做法是新增独立 package 并在新 package 的 `__init__.py` 注册新 ID，让 `tasks/__init__.py` 的 package scan 发现它。

## 2. `Unitree-G1-29dof-Velocity` 使用的 config classes

主配置文件：

`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`

核心配置关系：

- `RobotEnvCfg(ManagerBasedRLEnvCfg)`
  - `scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)`
  - `observations: ObservationsCfg = ObservationsCfg()`
  - `actions: ActionsCfg = ActionsCfg()`
  - `commands: CommandsCfg = CommandsCfg()`
  - `rewards: RewardsCfg = RewardsCfg()`
  - `terminations: TerminationsCfg = TerminationsCfg()`
  - `events: EventCfg = EventCfg()`
  - `curriculum: CurriculumCfg = CurriculumCfg()`
- `RobotPlayEnvCfg(RobotEnvCfg)`
  - `scene.num_envs = 32`
  - `scene.terrain.terrain_generator.num_rows = 2`
  - `scene.terrain.terrain_generator.num_cols = 10`
  - `commands.base_velocity.ranges = commands.base_velocity.limit_ranges`

Nested classes and important config objects:

- `COBBLESTONE_ROAD_CFG`: `terrain_gen.TerrainGeneratorCfg`，当前 `sub_terrains` 只有 `"flat": MeshPlaneTerrainCfg(proportion=0.5)`。
- `RobotSceneCfg`: terrain, robot, height scanner, contact sensor, sky light。
- `EventCfg`: startup material/mass randomization, reset, interval push。
- `CommandsCfg`: `mdp.UniformLevelVelocityCommandCfg`，初始 command range 较小，`limit_ranges` 是 curriculum 上限。
- `ActionsCfg`: `mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)`。
- `ObservationsCfg.PolicyCfg`: policy obs，history length 5，启用 corruption 和 concatenate。
- `ObservationsCfg.CriticCfg`: privileged critic obs，history length 5。
- `RewardsCfg`: 速度跟踪、存活、base/joint penalty、姿态、足端 gait/slide/clearance、undesired contact。
- `TerminationsCfg`: timeout, root height minimum, bad orientation。
- `CurriculumCfg`: terrain level curriculum 和 linear velocity command curriculum。

`RobotEnvCfg.__post_init__` 还设置：

- `decimation = 4`
- `episode_length_s = 20.0`
- `sim.dt = 0.005`
- `sim.render_interval = decimation`
- `sim.physics_material = scene.terrain.physics_material`
- `scene.contact_forces.update_period = sim.dt`
- `scene.height_scanner.update_period = decimation * sim.dt`
- 如果 `curriculum.terrain_levels` 存在，则打开 `scene.terrain.terrain_generator.curriculum`。

## 3. 已有 curriculum hooks

G1 29-DoF 任务当前 hook：

- `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`
- `lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)`

`mdp.lin_vel_cmd_levels` 位于：

`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`

行为：

- 默认读取 reward term `track_lin_vel_xy`。
- 在 episode 边界检查平均 episode reward。
- 当 `reward > reward_term.weight * 0.8` 时，将 `base_velocity.ranges.lin_vel_x` 和 `lin_vel_y` 同时扩张 `[-0.1, +0.1]`。
- 扩张结果 clamp 到 `base_velocity.limit_ranges`。
- 返回当前 `lin_vel_x` 上界。

同文件还存在但 G1 29-DoF 当前未接入的 hook：

- `ang_vel_cmd_levels(env, env_ids, reward_term_name="track_ang_vel_z")`
- 行为与 `lin_vel_cmd_levels` 类似，但只扩张 `ranges.ang_vel_z`。

Research variant 注意点：

- 如果沿用 `lin_vel_cmd_levels`，应保留 `track_lin_vel_xy` 这个 reward term 名称，或显式传入新的 `reward_term_name`。
- 如果新增 grass-risk curriculum，建议新增独立 term，例如 `grass_risk_levels = CurrTerm(func=grass_mdp.grass_risk_levels)`，不要改变上游 `lin_vel_cmd_levels` 默认逻辑。

## 4. 已有 contact sensor 和 foot body 命名

Scene 中已有 contact sensor：

- 名称：`contact_forces`
- 类型：`ContactSensorCfg`
- `prim_path="{ENV_REGEX_NS}/Robot/.*"`
- `history_length=3`
- `track_air_time=True`
- update period 在 `RobotEnvCfg.__post_init__` 中设为 `sim.dt`

当前足端 selector 使用的是 body regex，而不是硬编码 body id：

- gait: `SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")`
- feet slide: `SceneEntityCfg("robot", body_names=".*ankle_roll.*")` 和 `SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")`
- feet clearance: `SceneEntityCfg("robot", body_names=".*ankle_roll.*")`
- undesired contacts: `SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"])`

资产侧对应的 ankle roll link/joint 命名可从 G1 29-DoF URDF/MJCF 看出：

- `left_ankle_roll_link`
- `right_ankle_roll_link`
- `left_ankle_roll_joint`
- `right_ankle_roll_joint`

因此，新增 grass-risk reward 或 curriculum 时，优先复用现有 selector：

- 足端 body: `body_names=".*ankle_roll.*"`
- contact sensor: `SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")`

这能和已有 gait、slide、contact-time 逻辑保持一致。

## 5. 与 slip、gait、energy、action rate、posture 相关的已有 rewards

### Slip / slide

已配置 reward term：

- term name: `feet_slide`
- func: `mdp.feet_slide`
- weight: `-0.2`
- asset selector: `SceneEntityCfg("robot", body_names=".*ankle_roll.*")`
- sensor selector: `SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")`

`mdp.feet_slide` 不是本仓库 `tasks/locomotion/mdp/rewards.py` 中定义的函数；它经由 `tasks/locomotion/mdp/__init__.py` 从 `isaaclab_tasks.manager_based.locomotion.velocity.mdp` wildcard import 暴露。

### Gait

已配置 reward term：

- term name: `gait`
- func: `mdp.feet_gait`
- weight: `0.5`
- `period = 0.8`
- `offset = [0.0, 0.5]`
- `threshold = 0.55`
- `command_name = "base_velocity"`
- sensor selector: `SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")`

本仓库 `mdp.feet_gait` 的实现使用 `contact_sensor.data.current_contact_time` 判断足端是否接触，并用 episode time 生成左右腿 phase。传入 `command_name` 时，只在 command norm 大于 `0.1` 时生效。

### Energy

已配置 reward term：

- term name: `energy`
- func: `mdp.energy`
- weight: `-2e-5`

本仓库 `mdp.energy` 的实现为：

- 读取 `asset.data.joint_vel`
- 读取 `asset.data.applied_torque`
- 返回 `sum(abs(qvel) * abs(qfrc))`

### Action rate

已配置 reward term：

- term name: `action_rate`
- func: `mdp.action_rate_l2`
- weight: `-0.05`

`mdp.action_rate_l2` 由上游 IsaacLab MDP namespace 暴露，不在本仓库 `rewards.py` 中定义。

### Posture / pose regularization

已配置 posture 相关 terms：

- `joint_deviation_arms`
  - func: `mdp.joint_deviation_l1`
  - weight: `-0.1`
  - joint regex: `.*_shoulder_.*_joint`, `.*_elbow_joint`, `.*_wrist_.*`
- `joint_deviation_waists`
  - func: `mdp.joint_deviation_l1`
  - weight: `-1`
  - joint regex: `waist.*`
- `joint_deviation_legs`
  - func: `mdp.joint_deviation_l1`
  - weight: `-1.0`
  - joint regex: `.*_hip_roll_joint`, `.*_hip_yaw_joint`
- `flat_orientation_l2`
  - func: `mdp.flat_orientation_l2`
  - weight: `-5.0`
- `base_height`
  - func: `mdp.base_height_l2`
  - weight: `-10`
  - `target_height = 0.78`
- `dof_pos_limits`
  - func: `mdp.joint_pos_limits`
  - weight: `-5.0`

本仓库 `rewards.py` 中还定义了一些当前 G1 29-DoF env 未接入的 posture/feet helper，可供 research variant 选择性复用：

- `stand_still`
- `orientation_l2`
- `upward`
- `joint_position_penalty`
- `feet_stumble`
- `feet_height_body`
- `feet_too_near`
- `feet_contact_without_cmd`
- `air_time_variance_penalty`
- `joint_mirror`

## 6. 新 grass-risk research task 推荐文件路径

首选隔离方案：新增独立 package，不修改 `g1/29dof/velocity_env_cfg.py`，也不改现有 `Unitree-G1-29dof-Velocity` registration。

推荐路径：

- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/grass_29dof/__init__.py`
  - 注册新 Gym ID：`Unitree-G1-29dof-GrassRisk-Velocity`
  - `entry_point` 继续使用 `isaaclab.envs:ManagerBasedRLEnv`
  - `env_cfg_entry_point` 指向本 package 的 `velocity_env_cfg:GrassRiskRobotEnvCfg`
  - `play_env_cfg_entry_point` 指向本 package 的 `velocity_env_cfg:GrassRiskRobotPlayEnvCfg`
  - `rsl_rl_cfg_entry_point` 可先复用 `unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/grass_29dof/velocity_env_cfg.py`
  - 通过继承上游 `RobotEnvCfg` / `RobotPlayEnvCfg` 派生 research config。
  - 只 override grass terrain、risk curriculum、risk rewards 或 command ranges。
  - 不在 import 时 mutate 上游 class-level config object。
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/grass_29dof/grass_mdp.py`
  - 放 research-only grass risk reward/curriculum helper。
  - cfg 里直接 `from . import grass_mdp`，避免把实验函数塞进全局 `tasks/locomotion/mdp/__init__.py`。
- 可选：`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/grass_29dof/rsl_rl_ppo_cfg.py`
  - 只有当 PPO runner 参数也要实验化时再新增。
- 文档路径继续放在 `docs/research/g1_grass/`。

注意：现有 package 名 `29dof` 以数字开头，不能直接写 `from ...g1.29dof.velocity_env_cfg import RobotEnvCfg` 这种 Python import 语法。若新 package 需要继承上游 29-DoF cfg，建议在 `grass_29dof/velocity_env_cfg.py` 中用 `importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")` 取得 base module，然后派生 class。

备选小 diff 方案：

- 新增 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/grass_velocity_env_cfg.py`
- 在 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py` 追加第二个 `gym.register(...)`

该方案可以用相对 import `from .velocity_env_cfg import RobotEnvCfg`，实现更简单；但它会触碰现有 `29dof/__init__.py`。如果 research variant 会长期演进，独立 `grass_29dof` package 更利于隔离上游任务。

## 最小安全原则

- 保持 `Unitree-G1-29dof-Velocity` ID、entry points、上游 `RobotEnvCfg` 和 `RobotPlayEnvCfg` 不变。
- grass-risk variant 只新增新 ID、新 cfg、新 research helper。
- 复用已有 `contact_forces` sensor 和 `.*ankle_roll.*` 足端 selector。
- 如果沿用 command curriculum，保留 `track_lin_vel_xy` / `track_ang_vel_z` reward term 名称，或在 curriculum params 中显式指向新 reward term。
- 所有 grass-risk 参数放到派生 cfg 中，不在 module import 阶段修改上游 config object。
