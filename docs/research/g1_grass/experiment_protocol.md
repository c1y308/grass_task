# G1 Grass Paper Experiment Protocol

This protocol defines the controlled experiment used for the G1 grass risk-curriculum paper. Its purpose is to make the comparison between curriculum variants reproducible, fair, and easy to audit before results are used in the manuscript.

## Methods

The experiment matrix contains five methods:

| Method | Name | Task |
| --- | --- | --- |
| B0 | `B0_flat_rigid` | `Unitree-G1-29dof-Grass-FlatRigid` |
| B1 | `B1_coupled_random` | `Unitree-G1-29dof-Grass-CoupledRandom` |
| B2 | `B2_fixed_schedule` | `Unitree-G1-29dof-Grass-FixedSchedule` |
| B3 | `B3_success_gate` | `Unitree-G1-29dof-Grass-SuccessGate` |
| Ours | `Ours_risk_gate` | `Unitree-G1-29dof-Grass-RiskGate` |

B0 is the flat-rigid baseline. B1 trains with full coupled randomization. B2, B3, and Ours are curriculum-based methods and must be compared under the controlled conditions below.

## Controlled Conditions

All curriculum-based methods must use identical settings for:

- Policy architecture.
- PPO hyperparameters.
- Reward function.
- Terrain parameter upper bound.
- Total environment steps.
- Evaluation seeds.
- Command velocity range.
- Episode length.
- Termination conditions.

**All curriculum-based methods share the same terrain stages and final grass-like parameter distribution. The only difference lies in the progression criterion.**

For B2, the progression criterion is a fixed schedule. For B3, the progression criterion is success-rate gating. For Ours, the progression criterion is risk-gated promotion using contact, posture, and compensation-risk diagnostics.

## Evaluation Protocol

Each trained policy must be evaluated on the same held-out scenarios and seeds:

- `eval_flat_to_grass`
- `eval_mild_grass`
- `eval_wet_grass`
- `eval_soft_grass`
- `eval_hard_hidden_bumps`
- `eval_extreme_coupled`

Each method should use the same number of training seeds, the same number of evaluation episodes per seed, and the same evaluation command distribution. Per-episode CSV outputs should be aggregated by `method` and `scenario`, with bootstrap confidence intervals reported for paper tables.

## Expected Claims

The primary expected claim is that Ours improves contact risk under grass-like perturbations. Contact-risk evidence should come from touchdown timing error, foot slip ratio, unexpected contact count, missed/delayed support ratio, and contact-window IoU.

Posture risk should be interpreted as a downstream improvement, not the central mechanism. Reduced roll RMS, pitch RMS, base angular velocity RMS, height fluctuation, or recovery time can support the claim, but should not replace the contact-risk argument.

Compensation amplitude does not necessarily always decrease. A robust policy may sometimes use larger ankle actions or torque responses when terrain disturbances are severe.

Compensation quality should be judged by phase alignment, recovery effectiveness, safety-boundary respect, and duration. In particular, event-aligned analysis should consider whether compensation occurs in the correct stance-response window, reduces slip or posture deviation, avoids torque saturation and joint-limit proximity, and returns to nominal behavior quickly.

`compensation_efficiency` is diagnostic-only and must not enter curriculum promotion gates. The Ours gate may read only `metrics["contact_risk"]`, `metrics["posture_risk"]`, and `metrics["compensation_risk"]`; `contact_risk` is the aggregate of touchdown timing error, slip ratio, unexpected contact, and missed support, while `compensation_risk` must be built from safety-boundary terms such as torque saturation ratio, joint-limit proximity or margin, action jerk, and `1 - compensation_phase_alignment`.

## Paper Inclusion Checklist

- [ ] All five methods B0, B1, B2, B3, and Ours were trained or explicitly marked unavailable.
- [ ] B2, B3, and Ours use the same policy architecture.
- [ ] B2, B3, and Ours use the same PPO hyperparameters.
- [ ] B2, B3, and Ours use the same reward function.
- [ ] B2, B3, and Ours use the same terrain stages and final grass-like parameter distribution.
- [ ] B2, B3, and Ours differ only in progression criterion.
- [ ] Total environment steps are identical across compared curriculum methods.
- [ ] Evaluation seeds are identical across all methods.
- [ ] Command velocity range is identical during evaluation.
- [ ] Episode length and termination conditions are identical during evaluation.
- [ ] Per-episode evaluation CSVs were produced for all included method-scenario pairs.
- [ ] Aggregated metrics include `n_episodes`, `n_success`, means, standard deviations, medians, and 95% confidence intervals.
- [ ] B3 vs Ours comparison table was generated.
- [ ] Contact-risk metrics support the main claim, including missed/delayed support ratio and contact-window IoU.
- [ ] Posture-risk metrics are reported as downstream evidence.
- [ ] Compensation-quality plots include event-aligned expected/real contact masks, roll/pitch error, ankle action amplitude, torque saturation, action jerk, and compensation-efficiency summaries.
- [ ] Any missing sim2sim or real G1 qualitative result is clearly labeled as unavailable rather than implied.
- [ ] All figure and table outputs can be regenerated from committed scripts and recorded input paths.
