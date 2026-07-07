# G1 Grass Full Training Runbook

This runbook describes how to launch the full G1 grass experiment matrix. It is intended for the actual training machine, inside the Isaac Lab environment.

## 1. Environment Setup

Use the environment setup flow from the project `README.md`.

Install Isaac Lab first by following the Isaac Lab installation guide. Then install this repository as an Isaac Lab standalone environment:

```bash
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd unitree_rl_lab
conda activate env_isaaclab
./unitree_rl_lab.sh -i
# restart your shell to activate the environment changes
```

Download robot description files.

Recommended URDF method for Isaac Sim >= 5.0:

```bash
git clone https://github.com/unitreerobotics/unitree_ros.git
```

Then configure `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`:

```python
UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
```

Alternative USD method:

```bash
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
```

Then configure `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`:

```python
UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
```

Before launching training, confirm the active shell is using the Isaac Lab environment:

```bash
conda activate env_isaaclab
python -c "import isaaclab, gymnasium; print('isaaclab ok')"
```

## 2. Verify Task Registration

List available Unitree RL Lab tasks:

```bash
./unitree_rl_lab.sh -l
```

The list must include:

- `Unitree-G1-29dof-Grass-FlatRigid`
- `Unitree-G1-29dof-Grass-CoupledRandom`
- `Unitree-G1-29dof-Grass-FixedSchedule`
- `Unitree-G1-29dof-Grass-SuccessGate`
- `Unitree-G1-29dof-Grass-RiskGate`

Do not start full training until all five tasks are visible.

## 3. Dry Run

Dry-run the full matrix:

```bash
python scripts/research/g1_grass/train_matrix.py --dry-run
```

Dry-run only the main method:

```bash
python scripts/research/g1_grass/train_matrix.py --dry-run --methods Ours_risk_gate --seeds 1
```

The dry run should print exact `python scripts/rsl_rl/train.py --headless --task ... --seed ...` commands and must not launch Isaac Sim training.

## 4. Full Execute Command

Run the full matrix with:

```bash
python scripts/research/g1_grass/train_matrix.py --execute
```

This command launches all configured methods and seeds from `configs/research/g1_grass/experiment_matrix.yaml`.

Use a persistent shell session such as `tmux` or `screen`. Record the hostname, GPU model, start time, git commit, and active conda environment before starting.

## 5. Run Only Part Of The Matrix

Limit to selected methods:

```bash
python scripts/research/g1_grass/train_matrix.py --execute --methods Ours_risk_gate
```

Limit to selected methods and seeds:

```bash
python scripts/research/g1_grass/train_matrix.py --execute --methods B3_success_gate,Ours_risk_gate --seeds 1,2
```

Override training iterations only when intentionally doing a short pilot:

```bash
python scripts/research/g1_grass/train_matrix.py --execute --methods Ours_risk_gate --seeds 1 --train-steps 100
```

Do not use `--train-steps` for paper results unless the same override is applied to every compared method and documented in the run ledger.

## 6. Expected Output Paths

Matrix command ledger:

```text
logs/research/g1_grass/commands_train.jsonl
```

RSL-RL training logs are written by `scripts/rsl_rl/train.py` under:

```text
logs/rsl_rl/<task-derived-experiment-name>/<timestamp>_<run_name>/
```

Expected task-derived experiment names include:

```text
logs/rsl_rl/unitree_g1_29dof_grass_flatrigid/
logs/rsl_rl/unitree_g1_29dof_grass_coupledrandom/
logs/rsl_rl/unitree_g1_29dof_grass_fixedschedule/
logs/rsl_rl/unitree_g1_29dof_grass_successgate/
logs/rsl_rl/unitree_g1_29dof_grass_riskgate/
```

Each run directory should contain RSL-RL logs, checkpoints, copied environment configuration, and exported deployment configuration when training reaches those code paths.

Recommended manual run ledger:

```text
logs/research/g1_grass/run_ledger.csv
```

Suggested columns:

```text
method,task,seed,status,start_time,end_time,host,gpu,git_commit,run_dir,notes
```

Use `status=running`, `completed`, `failed_seed`, `stopped_for_resume`, or `excluded`.

## 7. Safe Stop And Resume

The repository training CLI supports resume through:

```text
--resume
--load_run
--checkpoint
```

The locomotion PPO config uses:

```text
save_interval = 100
max_iterations = 50000
```

Safe stop procedure:

1. Prefer stopping between seeds, after the current `train.py` process exits.
2. If a run must be interrupted, stop it once a checkpoint has recently been written.
3. Record the interrupted `method`, `task`, `seed`, run directory, last checkpoint, wall-clock time, and reason in `run_ledger.csv`.
4. Do not delete partial logs.

Resume a single interrupted run with the repository training CLI:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Grass-RiskGate \
  --seed 1 \
  --resume \
  --load_run <timestamp_run_dir_name> \
  --checkpoint <checkpoint_file_name>
```

Example:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-G1-29dof-Grass-RiskGate \
  --seed 1 \
  --resume \
  --load_run 2026-07-07_20-34-20 \
  --checkpoint model_100.pt
```

`train_matrix.py` currently launches fresh matrix commands and does not expose `--resume`, `--load_run`, or `--checkpoint`. For an interrupted seed, resume with `scripts/rsl_rl/train.py` directly, then mark the manual resume in the run ledger.

## 8. GPU And Time Logging

Before training:

```bash
date -Is
hostname
git rev-parse HEAD
conda info --envs
nvidia-smi
```

During training, record GPU utilization periodically:

```bash
nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 60
```

Save GPU monitoring output to a run-specific file, for example:

```text
logs/research/g1_grass/gpu_logs/<method>_seed<seed>_gpu.csv
```

Record wall-clock start and end time for every seed in `run_ledger.csv`. If training is launched from `tmux`, record the session name in the ledger notes.

## 9. Exception Handling

Crash:

- Do not automatically retry.
- Inspect the relevant run directory under `logs/rsl_rl/...`.
- Inspect Isaac Sim and Isaac Lab logs.
- Record `status=failed_seed` or `status=stopped_for_resume` in `run_ledger.csv`.
- Only relaunch after the failure cause is understood and documented.

Stalled training:

- Record the stall time, method, seed, and last visible training iteration.
- Check GPU activity with `nvidia-smi`.
- Continue only if GPU utilization or memory activity indicates the job is still alive.
- If GPU activity is absent and logs are not advancing, stop the seed and mark the ledger with the observed stall condition.

NaN loss:

- Stop that seed.
- Do not automatically retry with changed hyperparameters.
- Mark `status=failed_seed` in `run_ledger.csv`.
- Preserve the run directory for diagnosis.
- Exclude the failed seed from paper aggregation unless the experiment protocol explicitly defines a replacement policy before results are inspected.

## 10. Pre-Launch Checklist

- [ ] `conda activate env_isaaclab` is active.
- [ ] `./unitree_rl_lab.sh -l` lists all five G1 grass tasks.
- [ ] `python scripts/research/g1_grass/train_matrix.py --dry-run` prints the expected full command list.
- [ ] Git commit hash is recorded.
- [ ] `nvidia-smi` output is recorded.
- [ ] `logs/research/g1_grass/run_ledger.csv` is initialized.
- [ ] Enough disk space is available for `logs/rsl_rl`.
- [ ] Full command is ready:

```bash
python scripts/research/g1_grass/train_matrix.py --execute
```
