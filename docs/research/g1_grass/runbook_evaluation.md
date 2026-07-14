# G1 Grass Evaluation Runbook

This runbook describes how to evaluate trained G1 grass policies, aggregate per-episode metrics, export paper tables, and generate event-aligned compensation plots.

Run all commands from the repository root inside the Isaac Lab environment:

```bash
conda activate env_isaaclab
cd ~/grass_task
```

## 1. Locate Checkpoints

Training checkpoints are written by `scripts/rsl_rl/train.py` under:

```text
logs/rsl_rl/<task-derived-experiment-name>/<timestamp>_<optional_run_name>/
```

For the five G1 grass methods, expected experiment directories are:

```text
logs/rsl_rl/unitree_g1_29dof_grass_flatrigid/
logs/rsl_rl/unitree_g1_29dof_grass_coupledrandom/
logs/rsl_rl/unitree_g1_29dof_grass_fixedschedule/
logs/rsl_rl/unitree_g1_29dof_grass_successgate/
logs/rsl_rl/unitree_g1_29dof_grass_riskgate/
```

Locate recent Ours checkpoints:

```bash
find logs/rsl_rl/unitree_g1_29dof_grass_riskgate -type f -name "model_*.pt" | sort
```

Inspect recent run directories:

```bash
ls -td logs/rsl_rl/unitree_g1_29dof_grass_riskgate/*
```

Record the selected checkpoint in the evaluation ledger:

```text
method,train_seed,task,checkpoint,eval_seed,scenario,status,output_csv,notes
```

For paper results, use the intended final checkpoint for every method and seed. If a run was resumed, record the final resumed checkpoint path, not the interrupted checkpoint.

## 2. Evaluate One Checkpoint

Example for `Unitree-G1-29dof-Grass-RiskGate` / `Ours_risk_gate`, training seed `1`, scenario `eval_mild_grass`:

```bash
python scripts/research/g1_grass/evaluate_policy.py \
  --headless \
  --task Unitree-G1-29dof-Grass-RiskGate \
  --method Ours_risk_gate \
  --seed 1 \
  --eval-seed 1001 \
  --episodes 200 \
  --scenario eval_mild_grass \
  --checkpoint logs/rsl_rl/unitree_g1_29dof_grass_riskgate/<run_dir>/model_<iter>.pt \
  --output results/research/g1_grass/evaluation/Ours_risk_gate_seed1_eval_mild_grass.csv
```

Notes:

- `--seed` is the training seed associated with the checkpoint.
- `--eval-seed` is the held-out terrain/evaluation seed. Use the same held-out evaluation seeds for every method.
- `--episodes 200` is the minimum per method-scenario pair unless compute limits are explicitly reported.
- The output CSV must contain the columns documented in `docs/research/g1_grass/evaluation_schema.md`.

## 3. Evaluate All Scenarios

Scenarios:

```text
eval_flat_to_grass
eval_mild_grass
eval_wet_grass
eval_soft_grass
eval_hard_hidden_bumps
eval_extreme_coupled
```

Example loop for one Ours checkpoint:

```bash
CHECKPOINT="logs/rsl_rl/unitree_g1_29dof_grass_riskgate/<run_dir>/model_<iter>.pt"
METHOD="Ours_risk_gate"
TASK="Unitree-G1-29dof-Grass-RiskGate"
TRAIN_SEED=1
EVAL_SEED=1001
OUTPUT_DIR="results/research/g1_grass/evaluation"

mkdir -p "${OUTPUT_DIR}"

for SCENARIO in \
  eval_flat_to_grass \
  eval_mild_grass \
  eval_wet_grass \
  eval_soft_grass \
  eval_hard_hidden_bumps \
  eval_extreme_coupled
do
  python scripts/research/g1_grass/evaluate_policy.py \
    --headless \
    --task "${TASK}" \
    --method "${METHOD}" \
    --seed "${TRAIN_SEED}" \
    --eval-seed "${EVAL_SEED}" \
    --episodes 200 \
    --scenario "${SCENARIO}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT_DIR}/${METHOD}_seed${TRAIN_SEED}_${SCENARIO}.csv"
done
```

Repeat the same process for every method and training seed:

| Method | Task |
| --- | --- |
| `B0_flat_rigid` | `Unitree-G1-29dof-Grass-FlatRigid` |
| `B1_coupled_random` | `Unitree-G1-29dof-Grass-CoupledRandom` |
| `B2_fixed_schedule` | `Unitree-G1-29dof-Grass-FixedSchedule` |
| `B3_success_gate` | `Unitree-G1-29dof-Grass-SuccessGate` |
| `Ours_risk_gate` | `Unitree-G1-29dof-Grass-RiskGate` |

## 4. Validate Evaluation CSVs

Check that all expected columns are present:

```bash
python -c "import pandas as pd; from pathlib import Path; expected=['method','train_seed','eval_seed','scenario','episode','success','distance_m','mean_tracking_error','touchdown_timing_error_mean','foot_slip_ratio','missed_delayed_support_ratio','stance_duration_deviation_mean','unexpected_contact_count','contact_window_iou','roll_rms','pitch_rms','base_ang_vel_rms','com_height_fluctuation','recovery_time_s','ankle_action_mean','ankle_action_max','torque_peak','torque_rms','torque_saturation_ratio','joint_limit_margin_min','action_jerk','compensation_phase_alignment','compensation_efficiency']; missing={str(p): [c for c in expected if c not in pd.read_csv(p, nrows=1).columns] for p in Path('results/research/g1_grass/evaluation').glob('*.csv')}; print({k:v for k,v in missing.items() if v})"
```

The printed dictionary must be empty.

Check episode counts:

```bash
python -c "import pandas as pd; from pathlib import Path; rows=[]; [rows.append((p.name, len(pd.read_csv(p)))) for p in Path('results/research/g1_grass/evaluation').glob('*.csv')]; print([item for item in rows if item[1] < 200])"
```

The printed list must be empty unless compute limits are explicitly reported.

## 5. Aggregate Results

Aggregate all evaluation CSVs:

```bash
python scripts/research/g1_grass/aggregate_results.py \
  results/research/g1_grass/evaluation \
  --output results/research/g1_grass/aggregated_metrics.csv
```

The output contains one row per `method,scenario`, including:

- `n_episodes`
- `n_success`
- `success_rate`
- `mean`
- `std`
- `median`
- `ci95_low`
- `ci95_high`

## 6. Export Paper Tables

Export paper-ready CSV tables:

```bash
python scripts/research/g1_grass/export_paper_tables.py \
  results/research/g1_grass/aggregated_metrics.csv \
  --output-dir results/research/g1_grass/paper_tables
```

Expected outputs:

```text
results/research/g1_grass/paper_tables/table_success_rate.csv
results/research/g1_grass/paper_tables/table_contact_risk.csv
results/research/g1_grass/paper_tables/table_posture_risk.csv
results/research/g1_grass/paper_tables/table_compensation_quality.csv
results/research/g1_grass/paper_tables/table_b3_vs_ours.csv
```

`table_b3_vs_ours.csv` explicitly compares `B3_success_gate` against `Ours_risk_gate`.

## 7. Generate Event-Aligned Plots

Event-aligned plots require per-step diagnostic CSV or parquet files. These are separate from the per-episode evaluation CSVs.

Supported event types:

```text
touchdown_error
foot_slip
unexpected_contact
missed_support
terrain_transition
```

Example for an Ours per-step diagnostic CSV:

```bash
python scripts/research/g1_grass/plot_event_aligned.py \
  results/research/g1_grass/diagnostics/Ours_risk_gate_seed1_eval_mild_grass_steps.csv \
  --event-types touchdown_error,foot_slip,unexpected_contact,missed_support,terrain_transition \
  --window -0.5 1.0 \
  --output-dir results/research/g1_grass/figures \
  --prefix Ours_risk_gate_seed1_eval_mild_grass
```

Expected outputs include PNG and PDF figures:

```text
results/research/g1_grass/figures/Ours_risk_gate_seed1_eval_mild_grass_touchdown_error.png
results/research/g1_grass/figures/Ours_risk_gate_seed1_eval_mild_grass_touchdown_error.pdf
results/research/g1_grass/figures/Ours_risk_gate_seed1_eval_mild_grass_foot_slip.png
results/research/g1_grass/figures/Ours_risk_gate_seed1_eval_mild_grass_foot_slip.pdf
results/research/g1_grass/figures/Ours_risk_gate_seed1_eval_mild_grass_terrain_transition.png
results/research/g1_grass/figures/Ours_risk_gate_seed1_eval_mild_grass_terrain_transition.pdf
```

The figure panels include:

- Slip velocity aligned to event time.
- Roll/pitch error aligned to event time.
- Ankle action amplitude aligned to event time.
- Torque saturation indicator aligned to event time.
- Action jerk aligned to event time.

## 8. Minimum Acceptance Criteria For Paper Results

Results may enter the paper only if:

- All methods are evaluated with the same held-out terrain seeds.
- Evaluation CSV columns are complete, with no missing schema columns.
- Each method has at least three training seeds.
- Each `method,scenario` pair has at least 200 evaluation episodes, unless compute limits are explicitly reported.
- Failed seeds are marked in the evaluation ledger and not silently replaced after looking at results.
- Aggregated metrics and paper tables are regenerated from committed scripts.
- Any missing event-aligned diagnostic plot is reported as unavailable rather than implied.

## 9. Recommended Directory Layout

```text
results/research/g1_grass/evaluation/
results/research/g1_grass/diagnostics/
results/research/g1_grass/aggregated_metrics.csv
results/research/g1_grass/paper_tables/
results/research/g1_grass/figures/
logs/research/g1_grass/evaluation_ledger.csv
```
