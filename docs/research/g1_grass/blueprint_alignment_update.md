# G1 Grass Blueprint Alignment Update

本文件记录本次按最新版论文蓝图/审计清单完成的增量修订。此次修订只更新实验包代码、评估导出脚本和文档，不重跑完整训练。

## 1. 对齐的论文蓝图版本要点

- 接触风险不再只依赖 touchdown timing error、foot slip ratio 和 unexpected contact count，还需要覆盖 missed/delayed support 与接触窗口重合质量。
- `contact_risk` 的聚合项明确为 touchdown timing error、slip ratio、unexpected contact、missed support 四项。
- `RiskGateProgression` 仍只读取 `metrics["contact_risk"]`、`metrics["posture_risk"]`、`metrics["compensation_risk"]`。
- `compensation_risk` 只能来自安全边界或动作质量项，例如 `torque_saturation_ratio`、`joint_limit_proximity` 或 `joint_limit_margin`、`action_jerk`、`1 - compensation_phase_alignment`。
- `compensation_efficiency` 改为事件触发固定前后窗口诊断指标，作为 paper/table/plot diagnostic-only 量，不进入课程晋级 gate。

## 2. 修改的文件列表

- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/research/g1_grass/risk_metrics.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/research/g1_grass/risk_curriculum.py`
- `scripts/research/g1_grass/evaluate_policy.py`
- `scripts/research/g1_grass/aggregate_results.py`
- `scripts/research/g1_grass/export_paper_tables.py`
- `scripts/research/g1_grass/plot_event_aligned.py`
- `docs/research/g1_grass/evaluation_schema.md`
- `docs/research/g1_grass/experiment_protocol.md`
- `docs/research/g1_grass/manuscript_results_handoff.md`
- `docs/research/g1_grass/runbook_evaluation.md`
- `docs/research/g1_grass/figure_plan.md`

## 3. 新增指标

### `missed_delayed_support_ratio`

定义为 `expected_contact_mask` 内 missed/delayed support 的比例：

- 分母：`expected_contact_mask` 内的期望支撑时间步数量。
- 分子：同一窗口内 `contact_force_z < force_threshold` 的时间步数量。
- 空期望支撑窗口返回 `0`，避免 NaN。
- 当前 per-episode evaluator 只有在 contact sensor 暴露 z 向接触力时可靠计算；缺少接触力诊断时输出 `nan`。

### `contact_window_iou`

定义为期望接触窗口与真实接触窗口的 IoU：

- `real_contact_mask` 优先由 `contact_force_z > force_threshold` 得到。
- 若没有接触力但已有 contact flag，则使用已有 contact flag。
- `IoU = intersection / union`。
- `union == 0` 时返回 `1`，避免 NaN 并表示期望/真实窗口同为空。

## 4. `compensation_efficiency` 新定义

`compensation_efficiency` 现在定义为事件触发固定前后窗口诊断指标：

```text
compensation_efficiency =
  (delta_contact_risk + delta_posture_risk) / joint_energy
```

其中：

- 事件类型至少包括 `foot_slip`、`unexpected_contact`、`missed_support`。
- `delta_contact_risk = pre_window_contact_risk - post_window_contact_risk`。
- `delta_posture_risk = pre_window_posture_risk - post_window_posture_risk`。
- `joint_energy` 使用事件后固定窗口内的关节能量。
- 如果没有 per-step diagnostics 支持完整事件窗口计算，per-episode evaluation CSV 必须输出 `nan`，不得用整 episode 平均值或旧 proxy 冒充。

## 5. Gate 说明

`compensation_efficiency` 不进入 `R_comp` gate。

`R_comp`/`metrics["compensation_risk"]` 应由安全边界或动作质量指标构成，例如：

- `torque_saturation_ratio`
- `joint_limit_proximity` 或 `joint_limit_margin`
- `action_jerk`
- `1 - compensation_phase_alignment`

## 6. 已运行验证命令和结果摘要

- `python -m compileall -q ...`
  - 结果：通过，目标 Python 文件可编译。
- `python -m py_compile scripts/research/g1_grass/evaluate_policy.py`
  - 结果：通过。
- `conda run -n env_isaaclab python scripts/research/g1_grass/evaluate_policy.py --help`
  - 结果：通过，CLI help 正常输出，不启动训练。
- `conda run -n env_isaaclab python scripts/research/g1_grass/aggregate_results.py --help`
  - 结果：通过。
- `conda run -n env_isaaclab python scripts/research/g1_grass/export_paper_tables.py --help`
  - 结果：通过。
- `conda run -n env_isaaclab python scripts/research/g1_grass/plot_event_aligned.py --help`
  - 结果：通过，事件类型包含 `touchdown_error`、`foot_slip`、`unexpected_contact`、`missed_support`、`terrain_transition`。
- `conda run -n env_isaaclab python source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/research/g1_grass/risk_metrics.py`
  - 结果：通过，新增/既有 metric 自检输出 shape 正常。
- `conda run -n env_isaaclab python source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/research/g1_grass/risk_curriculum.py`
  - 结果：通过，fixed/success/risk gate 示例均正常。
- 使用 `importlib` 直接加载 `risk_metrics.py` 运行新增指标数值断言。
  - 结果：通过，`missed_support_ratio`、`contact_window_iou`、`contact_risk`、`event_window_delta_risk`、`compensation_efficiency` 的核心数值断言均通过。
- 使用 `/tmp/g1_grass_smoke/eval.csv` 运行：
  - `conda run -n env_isaaclab python scripts/research/g1_grass/aggregate_results.py /tmp/g1_grass_smoke/eval.csv --output /tmp/g1_grass_smoke/aggregated_metrics.csv --bootstrap-samples 0`
  - 结果：通过，写出 2 个 method/scenario 聚合组。
- 使用 `/tmp/g1_grass_smoke/aggregated_metrics.csv` 运行：
  - `conda run -n env_isaaclab python scripts/research/g1_grass/export_paper_tables.py /tmp/g1_grass_smoke/aggregated_metrics.csv --output-dir /tmp/g1_grass_smoke/paper_tables`
  - 结果：通过，`table_contact_risk.csv` 包含 `missed_delayed_support_ratio` 和 `contact_window_iou`，`table_compensation_quality.csv` 标注 `compensation_efficiency_role=diagnostic-only`。
- 使用 `/tmp/g1_grass_smoke/steps.csv` 运行：
  - `conda run -n env_isaaclab python scripts/research/g1_grass/plot_event_aligned.py /tmp/g1_grass_smoke/steps.csv --event-types foot_slip,unexpected_contact,missed_support --window -0.05 0.05 --bin-size 0.05 --output-dir /tmp/g1_grass_smoke/figures --prefix smoke`
  - 结果：通过，生成事件对齐 PNG/PDF 和 `smoke_compensation_efficiency_summary.csv`。

注：系统默认 Python 缺少 `numpy`/`matplotlib`，因此脚本 help 与 smoke tests 使用 `env_isaaclab` 环境运行。

## 7. 后续需要重跑的实验

- 旧 evaluation CSV 不包含 `missed_delayed_support_ratio` 和 `contact_window_iou` 新列，必须重新评估 checkpoint。
- 如果训练时 risk gate 的 `contact_risk` 或 `compensation_risk` 已按本次蓝图发生改变，则 `Ours_risk_gate` 至少需要重新训练。
- 如果旧训练只缺评估列，且训练 gate 逻辑没有变化，则可以先复用旧 checkpoint 重新评估。
