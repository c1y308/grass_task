# G1 Grass Manuscript Results Handoff

This document is filled after experiments are complete. Do not add inferred results, informal impressions, or unsupported claims. Use only completed logs, evaluation CSVs, aggregated metrics, exported paper tables, figures, and safety records.

## Material Passport

| field | value |
| --- | --- |
| robot model | |
| simulator | |
| Isaac Lab version | |
| Unitree RL Lab source | |
| terrain assets/source | |
| policy framework | |
| evaluation scripts | |
| aggregation scripts | |
| plotting scripts | |
| hardware used for training | |
| hardware used for evaluation | |
| notes | |

## Experiment Package Version

| field | value |
| --- | --- |
| package name | |
| package version/tag | |
| config path | |
| experiment matrix path | |
| training runbook path | |
| evaluation runbook path | |
| protocol path | |
| safety checklist path | |

## Git Commit Hash

| field | value |
| --- | --- |
| repository | |
| commit hash | |
| branch | |
| dirty worktree at run time | |
| uncommitted files included in run | |
| command used to record hash | |

## Training Matrix Summary

| method | task | train seed | checkpoint path | start time | end time | total environment steps | max iterations | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_flat_rigid | Unitree-G1-29dof-Grass-FlatRigid | | | | | | | | |
| B1_coupled_random | Unitree-G1-29dof-Grass-CoupledRandom | | | | | | | | |
| B2_fixed_schedule | Unitree-G1-29dof-Grass-FixedSchedule | | | | | | | | |
| B3_success_gate | Unitree-G1-29dof-Grass-SuccessGate | | | | | | | | |
| Ours_risk_gate | Unitree-G1-29dof-Grass-RiskGate | | | | | | | | |

## Evaluation Matrix Summary

| method | train seed | eval seed | scenario | episodes | csv path | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0_flat_rigid | | | eval_flat_to_grass | | | | |
| B0_flat_rigid | | | eval_mild_grass | | | | |
| B0_flat_rigid | | | eval_wet_grass | | | | |
| B0_flat_rigid | | | eval_soft_grass | | | | |
| B0_flat_rigid | | | eval_hard_hidden_bumps | | | | |
| B0_flat_rigid | | | eval_extreme_coupled | | | | |
| B1_coupled_random | | | eval_flat_to_grass | | | | |
| B1_coupled_random | | | eval_mild_grass | | | | |
| B1_coupled_random | | | eval_wet_grass | | | | |
| B1_coupled_random | | | eval_soft_grass | | | | |
| B1_coupled_random | | | eval_hard_hidden_bumps | | | | |
| B1_coupled_random | | | eval_extreme_coupled | | | | |
| B2_fixed_schedule | | | eval_flat_to_grass | | | | |
| B2_fixed_schedule | | | eval_mild_grass | | | | |
| B2_fixed_schedule | | | eval_wet_grass | | | | |
| B2_fixed_schedule | | | eval_soft_grass | | | | |
| B2_fixed_schedule | | | eval_hard_hidden_bumps | | | | |
| B2_fixed_schedule | | | eval_extreme_coupled | | | | |
| B3_success_gate | | | eval_flat_to_grass | | | | |
| B3_success_gate | | | eval_mild_grass | | | | |
| B3_success_gate | | | eval_wet_grass | | | | |
| B3_success_gate | | | eval_soft_grass | | | | |
| B3_success_gate | | | eval_hard_hidden_bumps | | | | |
| B3_success_gate | | | eval_extreme_coupled | | | | |
| Ours_risk_gate | | | eval_flat_to_grass | | | | |
| Ours_risk_gate | | | eval_mild_grass | | | | |
| Ours_risk_gate | | | eval_wet_grass | | | | |
| Ours_risk_gate | | | eval_soft_grass | | | | |
| Ours_risk_gate | | | eval_hard_hidden_bumps | | | | |
| Ours_risk_gate | | | eval_extreme_coupled | | | | |

## Excluded Or Failed Seeds With Reasons

| method | train seed | eval seed | scenario | stage | reason | evidence path | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | |

## Table 1 Success Metrics

| method | scenario | n_episodes | n_success | success_rate_mean | success_rate_std | success_rate_median | success_rate_ci95_low | success_rate_ci95_high | distance_m_mean | distance_m_std | distance_m_median | distance_m_ci95_low | distance_m_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | | | |

## Table 2 Contact Risk Metrics

| method | scenario | n_episodes | foot_slip_ratio_mean | foot_slip_ratio_std | foot_slip_ratio_median | foot_slip_ratio_ci95_low | foot_slip_ratio_ci95_high | unexpected_contact_count_mean | unexpected_contact_count_std | unexpected_contact_count_median | unexpected_contact_count_ci95_low | unexpected_contact_count_ci95_high | touchdown_timing_error_mean_mean | stance_duration_deviation_mean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | | | | |

## Table 3 Posture Risk Metrics

| method | scenario | n_episodes | roll_rms_mean | roll_rms_std | roll_rms_median | roll_rms_ci95_low | roll_rms_ci95_high | pitch_rms_mean | pitch_rms_std | pitch_rms_median | pitch_rms_ci95_low | pitch_rms_ci95_high | base_ang_vel_rms_mean | com_height_fluctuation_mean | recovery_time_s_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | | | | | |

## Table 4 Compensation Quality Metrics

| method | scenario | n_episodes | ankle_action_mean_mean | ankle_action_max_mean | torque_peak_mean | torque_rms_mean | torque_saturation_ratio_mean | joint_limit_margin_min_mean | action_jerk_mean | compensation_phase_alignment_mean | compensation_efficiency_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | |

## Figure Inventory

| figure | title | source data | script | output path | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 1 | method overview diagram | | | | | |
| Figure 2 | grass terrain parameter curriculum | | | | | |
| Figure 3 | success/contact/posture metrics by method | | | | | |
| Figure 4 | event-aligned compensation analysis | | | | | |
| Figure 5 | sim2sim or real G1 qualitative sequence if available | | | | | |

## Claims Supported By Data

| claim | supporting table/figure | metric(s) | scenario(s) | evidence path | manuscript wording |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Claims Not Supported By Data

| claim considered | reason not supported | conflicting or missing evidence | manuscript action |
| --- | --- | --- | --- |
| | | | |

## Limitations To Report In The Manuscript

| limitation | affected result | evidence path | manuscript wording |
| --- | --- | --- | --- |
| | | | |
