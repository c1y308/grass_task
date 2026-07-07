# G1 Grass Task Registration Smoke Report

Date: 2026-07-07

Environment: `env_isaaclab`

This smoke test intentionally did not run a full experiment. The shortest train launch used `--max_iterations 1` and `--num_envs 1`.

## Summary

| Check | Status |
| --- | --- |
| `./unitree_rl_lab.sh -l` lists new G1 grass tasks | PASS |
| `train_matrix.py --dry-run --methods Ours_risk_gate --seeds 1` emits expected command | PASS |
| `train.py --help` exposes short-run CLI options | PASS |
| shortest `Unitree-G1-29dof-Grass-RiskGate` training launch | BLOCKED BY LOCAL ISAAC SIM ENVIRONMENT |

Notes:

- The first short-training attempt exposed a real config-loader issue: `GrassTerrainStage` and `GrassTerrainSchedule` were frozen dataclasses, which caused `FrozenInstanceError` during Isaac Lab/Hydra `env_cfg.from_dict(...)`.
- Fixed in `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/research/g1_grass/terrain_cfg.py` by changing those dataclasses to mutable dataclasses while preserving validation and `to_dict()` behavior.
- After that fix, the launch progressed through task registration, Hydra config parsing, logging setup, base environment setup, and scene creation.
- Final blocker is local Isaac Sim installation/runtime state, not G1 grass task registration: missing/incompatible URDF importer extension (`isaacsim.asset.importer.urdf` requested `2.4.31`, local install has `2.4.30`) plus no NVIDIA driver in this container.

## 1. List Registered Environments

Command requested:

```bash
./unitree_rl_lab.sh -l
```

Executed in `env_isaaclab`:

```bash
conda run -n env_isaaclab ./unitree_rl_lab.sh -l
```

Exit code: `0`

Output:

```text
+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                           Available Environments in Unitree RL Lab                                                          |
+--------+---------------------------------------+---------------------------------+--------------------------------------------------------------------------+
| S. No. | Task Name                             | Entry Point                     | Config                                                                   |
+--------+---------------------------------------+---------------------------------+--------------------------------------------------------------------------+
|   1    | Unitree-G1-29dof-Velocity             | isaaclab.envs:ManagerBasedRLEnv | locomotion.robots.g1.29dof.velocity_env_cfg:RobotEnvCfg                  |
|   2    | Unitree-Go2-Velocity                  | isaaclab.envs:ManagerBasedRLEnv | locomotion.robots.go2.velocity_env_cfg:RobotEnvCfg                       |
|   3    | Unitree-H1-Velocity                   | isaaclab.envs:ManagerBasedRLEnv | locomotion.robots.h1.velocity_env_cfg:RobotEnvCfg                        |
|   4    | Unitree-G1-29dof-Grass-FlatRigid      | isaaclab.envs:ManagerBasedRLEnv | locomotion.research.g1_grass.g1_grass_env_cfg:G1GrassFlatRigidEnvCfg     |
|   5    | Unitree-G1-29dof-Grass-CoupledRandom  | isaaclab.envs:ManagerBasedRLEnv | locomotion.research.g1_grass.g1_grass_env_cfg:G1GrassCoupledRandomEnvCfg |
|   6    | Unitree-G1-29dof-Grass-FixedSchedule  | isaaclab.envs:ManagerBasedRLEnv | locomotion.research.g1_grass.g1_grass_env_cfg:G1GrassFixedScheduleEnvCfg |
|   7    | Unitree-G1-29dof-Grass-SuccessGate    | isaaclab.envs:ManagerBasedRLEnv | locomotion.research.g1_grass.g1_grass_env_cfg:G1GrassSuccessGateEnvCfg   |
|   8    | Unitree-G1-29dof-Grass-RiskGate       | isaaclab.envs:ManagerBasedRLEnv | locomotion.research.g1_grass.g1_grass_env_cfg:G1GrassRiskGateEnvCfg      |
|   9    | Unitree-G1-29dof-Mimic-Dance-102      | isaaclab.envs:ManagerBasedRLEnv | mimic.robots.g1_29dof.dance_102.tracking_env_cfg:RobotEnvCfg             |
|   10   | Unitree-G1-29dof-Mimic-Gangnanm-Style | isaaclab.envs:ManagerBasedRLEnv | mimic.robots.g1_29dof.gangnanm_style.tracking_env_cfg:RobotEnvCfg        |
+--------+---------------------------------------+---------------------------------+--------------------------------------------------------------------------+
```

Result: the five G1 grass task registrations are visible.

## 2. Matrix Runner Dry Run

Command requested:

```bash
python scripts/research/g1_grass/train_matrix.py --dry-run --methods Ours_risk_gate --seeds 1
```

Executed in `env_isaaclab`:

```bash
conda run -n env_isaaclab python scripts/research/g1_grass/train_matrix.py --dry-run --methods Ours_risk_gate --seeds 1
```

Exit code: `0`

Output:

```text
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Grass-RiskGate --seed 1
```

Result: the matrix runner selects `Ours_risk_gate` and emits the expected training command without running training.

## 3. Train CLI Help

Command:

```bash
conda run -n env_isaaclab python scripts/rsl_rl/train.py --help
```

Exit code: `0`

Relevant output:

```text
usage: train.py [-h] [--video] [--video_length VIDEO_LENGTH]
                [--video_interval VIDEO_INTERVAL] [--num_envs NUM_ENVS]
                [--task {Unitree-G1-29dof-Velocity,Unitree-Go2-Velocity,Unitree-H1-Velocity,Unitree-G1-29dof-Grass-FlatRigid,Unitree-G1-29dof-Grass-CoupledRandom,Unitree-G1-29dof-Grass-FixedSchedule,Unitree-G1-29dof-Grass-SuccessGate,Unitree-G1-29dof-Grass-RiskGate,Unitree-G1-29dof-Mimic-Dance-102,Unitree-G1-29dof-Mimic-Gangnanm-Style}]
                [--seed SEED] [--max_iterations MAX_ITERATIONS]
                [--distributed] [--experiment_name EXPERIMENT_NAME]
                [--run_name RUN_NAME] [--resume] [--load_run LOAD_RUN]
                [--checkpoint CHECKPOINT]
                [--logger {tensorboard,wandb,neptune}]
                [--log_project_name LOG_PROJECT_NAME] [--headless]
                [--livestream {0,1,2}] [--enable_cameras] [--xr]
                [--device DEVICE] [--verbose] [--info]
                [--experience EXPERIENCE]
                [--rendering_mode {performance,balanced,quality}]
                [--kit_args KIT_ARGS] [--anim_recording_enabled]
                [--anim_recording_start_time ANIM_RECORDING_START_TIME]
                [--anim_recording_stop_time ANIM_RECORDING_STOP_TIME]

  --num_envs NUM_ENVS   Number of environments to simulate.
  --task {...,Unitree-G1-29dof-Grass-RiskGate,...}
                        Name of the task.
  --seed SEED           Seed used for the environment
  --max_iterations MAX_ITERATIONS
                        RL Policy training iterations.
  --headless            Force display off at all times.
  --device DEVICE       The device to run the simulation on. Can be "cpu",
                        "cuda", "cuda:N", where N is the device ID
```

Result: safe short-run options are available: `--max_iterations`, `--num_envs`, `--headless`, and `--device`.

## 4. Shortest Training Launch

Initial shortest CUDA command:

```bash
conda run -n env_isaaclab python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Grass-RiskGate --seed 1 --num_envs 1 --max_iterations 1 --experiment_name smoke_g1_grass_risk_gate --run_name smoke_seed1_iter1
```

Exit code: `1`

Key output:

```text
[INFO][AppLauncher]: Using device: cuda:0
[INFO]: Parsing configuration from: unitree_rl_lab.tasks.locomotion.research.g1_grass.g1_grass_env_cfg:G1GrassRiskGateEnvCfg
[INFO]: Parsing configuration from: unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg
dataclasses.FrozenInstanceError: cannot assign to field 'name'
```

Action taken:

```text
Changed GrassTerrainStage and GrassTerrainSchedule from frozen dataclasses to regular dataclasses so Isaac Lab/Hydra can update config fields during env_cfg.from_dict(...).
```

Retried shortest CPU command with writable temporary HOME/XDG cache:

```bash
env HOME=/tmp XDG_DATA_HOME=/tmp/xdg-data XDG_CACHE_HOME=/tmp/xdg-cache MPLCONFIGDIR=/tmp/matplotlib \
  conda run -n env_isaaclab python scripts/rsl_rl/train.py \
  --headless --device cpu \
  --task Unitree-G1-29dof-Grass-RiskGate \
  --seed 1 --num_envs 1 --max_iterations 1 \
  --experiment_name smoke_g1_grass_risk_gate \
  --run_name smoke_seed1_iter1_cpu_tmp_home
```

Exit code: `1`

Relevant output:

```text
[INFO][AppLauncher]: Using device: cpu
[INFO]: Parsing configuration from: unitree_rl_lab.tasks.locomotion.research.g1_grass.g1_grass_env_cfg:G1GrassRiskGateEnvCfg
[INFO]: Parsing configuration from: unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg
[INFO] Logging experiment in directory: /root/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_grass_riskgate
[INFO]: Base environment:
        Environment device    : cpu
        Environment seed      : 1
        Physics step-size     : 0.005
        Rendering step-size   : 0.02
        Environment step-size : 0.02
[INFO]: Time taken for scene creation : 2.925170 seconds
```

Final blocker:

```text
Failed to resolve extension dependencies. Failure hints:
        Can't find extension to satisfy dependency: 'isaacsim.asset.importer.urdf' = { version='=2.4.31' }
 Available versions:
        - [isaacsim.asset.importer.urdf-2.4.30+107.3.3.lx64.r.cp311] (/isaac-sim/exts/isaacsim.asset.importer.urdf)
 Synced registries:
        - kit/default         : found 0 packages (couldn't connect or empty)
        - kit/sdk             : found 0 packages (couldn't connect or empty)
        - kit/community       : found 0 packages (couldn't connect or empty)

ModuleNotFoundError: No module named 'isaacsim.asset'
```

Additional local runtime constraints observed:

```text
NVIDIA driver is not loaded.
No CUDA devices found.
Unable to create PxCudaContextManager.
Failed to acquire exclusive lock to data store at /isaac-sim/kit/cache/DerivedDataCache.
```

Result: the short train launch validates G1 grass registration and config parsing, but it cannot complete one training iteration in this local environment because Isaac Sim cannot satisfy the URDF importer extension dependency and no NVIDIA driver is available.

## Local Code Change Made During Smoke

File:

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/research/g1_grass/terrain_cfg.py
```

Change:

```text
@dataclass(frozen=True) -> @dataclass
```

Applied to:

```text
GrassTerrainStage
GrassTerrainSchedule
```

Validation:

```bash
conda run -n env_isaaclab python -m py_compile \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/research/g1_grass/terrain_cfg.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/research/g1_grass/g1_grass_env_cfg.py \
  scripts/rsl_rl/train.py \
  scripts/research/g1_grass/train_matrix.py
```

Exit code: `0`
