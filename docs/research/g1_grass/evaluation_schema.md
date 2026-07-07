# G1 Grass Policy Evaluation CSV Schema

`scripts/research/g1_grass/evaluate_policy.py` 用于对训练好的 G1 grass policy 做可复现评估。脚本复用仓库现有 RSL-RL play 路径来创建环境、加载 checkpoint、获得 inference policy，并额外按 episode 累积指标。CSV 中每一行对应一个完成的 episode。

默认评估长度为 `--episodes 200`。支持的 `--scenario` 为：

- `eval_flat_to_grass`
- `eval_mild_grass`
- `eval_wet_grass`
- `eval_soft_grass`
- `eval_hard_hidden_bumps`
- `eval_extreme_coupled`

`--seed` 记录为 `train_seed`，表示 checkpoint 对应的训练 seed。`eval_seed` 默认等于 `train_seed`，也可以用 `--eval-seed` 覆盖，用于评估环境初始化和 rollout 随机性的复现。

## Columns

| Column | 含义 | 单位 |
| --- | --- | --- |
| `method` | 方法名，由 `--method` 传入，例如 `B0_flat_rigid` 或 `Ours_risk_gate`。 | 类别 |
| `train_seed` | checkpoint 对应的训练 seed，即 CLI `--seed`。 | 整数 |
| `eval_seed` | 评估 rollout 使用的随机 seed，默认等于 `train_seed`。 | 整数 |
| `scenario` | 评估场景名称。 | 类别 |
| `episode` | 本次评估中的 episode 序号，从 1 开始。 | 计数 |
| `success` | episode 是否成功结束；timeout 记为 `1`，低高度、姿态失稳等失败终止记为 `0`。 | 二值 |
| `distance_m` | episode 内 base 在水平 XY 平面累计走过的路径长度。 | m |
| `mean_tracking_error` | 指令 base velocity `[vx, vy, yaw_rate]` 与实测 base velocity 的范数误差均值。 | 混合速度范数 |
| `touchdown_timing_error_mean` | 基于 gait clock 的期望 stance timing 与 ankle-roll 足端实测接触时间的平均绝对误差。 | s |
| `foot_slip_ratio` | stance contact 样本中，ankle-roll 足端水平速度超过 `0.20 m/s` 的比例。 | 比例 |
| `stance_duration_deviation_mean` | 期望 stance elapsed time 与实测接触持续时间之间的平均绝对偏差。 | s |
| `unexpected_contact_count` | 期望 swing phase 中检测到 ankle-roll 足端接触的样本数。 | 计数 |
| `roll_rms` | base/root roll 的 RMS。 | rad |
| `pitch_rms` | base/root pitch 的 RMS。 | rad |
| `base_ang_vel_rms` | base/root 角速度模长的 RMS。 | rad/s |
| `com_height_fluctuation` | base/root 高度在 episode 内的标准差，作为质心高度波动 proxy。 | m |
| `recovery_time_s` | tracking error 或机身倾角超过不稳定阈值后，处于 recovery 状态的累计时间。 | s |
| `ankle_action_mean` | ankle joint policy action 的平均绝对值；若 action 维度无法和 joint 对齐，则退化为全部 action 维度均值。 | action unit |
| `ankle_action_max` | ankle joint policy action 的最大绝对值；选择规则同 `ankle_action_mean`。 | action unit |
| `torque_peak` | episode 内观测到的最大绝对 applied joint torque。 | N m |
| `torque_rms` | 所有关节、所有 step 的 applied joint torque RMS。 | N m |
| `torque_saturation_ratio` | 绝对 torque 超过 `0.85 * torque_limit` 的关节比例均值；若当前 articulation 不暴露 torque limit，则写 `nan`。 | 比例 |
| `joint_limit_margin_min` | 到最近 joint position limit 的最小归一化距离；`0` 表示贴近限位，越大越安全。若无 joint limit tensor，则写 `nan`。 | 比例 |
| `action_jerk` | policy action 二阶有限差分的 L2 范数均值。 | action unit/step^2 |
| `compensation_phase_alignment` | 期望 stance 与实测接触重叠窗口内消耗的关节机械能，占总关节机械能的比例。 | 比例 |
| `compensation_efficiency` | 派生 tracking/posture/slip risk proxy 的正向下降量除以关节机械能。 | risk/J |

`nan` 表示该指标无法从当前 Isaac Lab 或 articulation 暴露的字段中可靠计算。常见情况是机器人对象没有 torque limit 或 joint limit tensor。
