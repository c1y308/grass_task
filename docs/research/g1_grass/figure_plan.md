# G1 Grass Figure Plan

## Figure 1: Method Overview Diagram

- 展示整体方法链路：grass terrain parameter schedule、risk-gated curriculum promotion、proprioceptive policy training、held-out scenario evaluation。
- 重点标出本文方法和 baseline 的差异：B0 flat rigid、B1 coupled random、B2 fixed schedule、B3 success gate、Ours risk gate。
- 建议输出形式：流程图或系统框图，主线从 terrain parameters 到 policy，再到 risk diagnostics。

## Figure 2: Grass Terrain Parameter Curriculum

- 展示课程阶段中 terrain height、friction、stiffness、damping、transition probability 的变化。
- 横轴为 curriculum stage 或 normalized training progress，纵轴为参数范围或归一化难度。
- 对比 fixed schedule、success gate、risk gate 的 stage promotion 轨迹。

## Figure 3: Success/Contact/Posture Metrics By Method

- 使用 aggregated evaluation metrics，按 method 和 scenario 展示 success rate、contact risk、posture risk。
- 建议包含 bootstrap 95% CI。
- contact risk 可包含 touchdown timing error、foot slip ratio、unexpected contact count、missed/delayed support ratio、contact-window IoU。
- posture risk 可包含 roll RMS、pitch RMS、base angular velocity RMS、COM height fluctuation、recovery time。

## Figure 4: Event-Aligned Compensation Analysis

- 使用 per-step diagnostic CSV/parquet，经 `scripts/research/g1_grass/plot_event_aligned.py` 生成。
- 分别围绕 `touchdown_error`、`foot_slip`、`unexpected_contact`、`missed_support`、`terrain_transition` 对齐时间轴，默认窗口为 event 前 `0.5 s` 到 event 后 `1.0 s`。
- 子图包含 expected contact mask、real contact mask、roll/pitch error、ankle action amplitude、torque saturation indicator、action jerk。
- 同步输出 event-window compensation efficiency summary；该 summary 是 diagnostic-only，不进入 curriculum gate。
- 目标是解释 Ours 是否在接触扰动后产生更及时、更低饱和、更平滑的补偿。

## Figure 5: Sim2Sim Or Real G1 Qualitative Sequence If Available

- 若有 sim2sim 或真实 G1 数据，展示代表性草地扰动序列。
- 推荐使用 4 到 6 帧时间序列，标注 terrain transition、foot slip 或 recovery 事件。
- 若真实机器人实验暂不可用，则使用 Isaac Sim qualitative rollout，并明确标注为 simulation-only qualitative result。
