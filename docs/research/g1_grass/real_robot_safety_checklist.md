# G1 Grass Real-Robot Safety Checklist

This checklist is for optional sim2sim and real Unitree G1 validation of the G1 grass policies. It follows the deployment order in the Unitree RL Lab README:

1. Train and evaluate in Isaac Lab.
2. Test the trained policy in Mujoco sim2sim.
3. Deploy to the physical G1 only after sim2sim passes.

Real-robot validation is optional. Do not use it to replace the paper evaluation protocol, and do not advance to the next stage if any hard gate fails.

## Stage Order

### 1. Isaac Lab Training And Evaluation

- [ ] The policy was trained in Isaac Lab using the documented G1 grass experiment matrix.
- [ ] The checkpoint path, method name, training seed, and task name are recorded.
- [ ] Evaluation CSVs were generated with `scripts/research/g1_grass/evaluate_policy.py`.
- [ ] Aggregated metrics were generated with `scripts/research/g1_grass/aggregate_results.py`.
- [ ] Paper tables were generated with `scripts/research/g1_grass/export_paper_tables.py`.
- [ ] Event-aligned figures were generated when per-step diagnostics are available.

### 2. Mujoco Sim2Sim

Use the README deployment order. After training is complete, start Mujoco first, then start the G1 controller.

```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
```

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
```

README operation sequence:

- Press `[L2 + Up]` to stand up.
- Click the Mujoco window, then press `8` to make the robot feet touch the ground.
- Press `[R1 + X]` to run the policy.
- Click the Mujoco window, then press `9` to disable the elastic band.

Before moving to real hardware:

- [ ] Sim2sim starts cleanly.
- [ ] Sim2sim stops cleanly.
- [ ] The policy can stand, walk at low command velocity, and return to stop without controller instability.
- [ ] The operator has rehearsed the stop sequence.
- [ ] Any abnormal sim2sim behavior is documented before proceeding.

### 3. Physical G1

The README notes that the real controller can be started with:

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network eth0
```

Only run this after the hard gates below pass. Confirm the correct network interface and make sure the on-board control program has been closed, as described in the README.

## Hard Gates

All hard gates must pass before real-robot grass validation.

- [ ] **Simulation evaluation complete:** all required evaluation scenarios are complete for the target checkpoint.
- [ ] **Extreme coupled safety boundary:** in `eval_extreme_coupled`, `Ours_risk_gate` is not worse than `B3_success_gate` on safety-boundary metrics.
- [ ] **Safety-boundary metric check:** compare at least `foot_slip_ratio`, `unexpected_contact_count`, `roll_rms`, `pitch_rms`, `base_ang_vel_rms`, `torque_peak`, `torque_rms`, `torque_saturation_ratio`, and `joint_limit_margin_min`.
- [ ] **Sim2sim launch and shutdown:** Mujoco sim2sim can start, run the policy, stop the policy, and shut down normally.
- [ ] **Emergency stop tested:** emergency stop has been tested before the policy is allowed to command motion.
- [ ] **Safe surface:** the robot is on a flat, dry, non-slip, obstacle-free safety surface for initial trials.
- [ ] **Low command velocity first:** the first real-robot trials use low command velocity and short duration.
- [ ] **Harness or human support:** the robot has a harness, overhead support, or trained human support during first motion trials.
- [ ] **Artificial turf before outdoor grass:** outdoor grass testing is not allowed until artificial turf testing passes.
- [ ] **Operator readiness:** at least one operator is dedicated to safety monitoring and can stop the run immediately.

Recommended initial command limits for first real-robot checks:

- Linear velocity: start near zero and increase only after stable repeated trials.
- Yaw velocity: start near zero.
- Trial duration: start with short trials, then increase only if no stop condition appears.

## Stop Conditions

Stop the trial immediately if any condition below occurs. Do not restart until the event is reviewed and recorded.

- [ ] Repeated visible foot slip.
- [ ] Base roll or pitch exceeds the pre-declared conservative threshold.
- [ ] Suggested initial threshold: stop if absolute roll or pitch exceeds `8 deg`, or if roll/pitch oscillation grows across the trial.
- [ ] Torque saturation warning.
- [ ] Joint limit warning.
- [ ] Unexpected contact with the ground, harness, support frame, operator, or nearby object.
- [ ] Abnormal sound, vibration, odor, or heat.
- [ ] Controller network instability, packet loss, delayed command response, or intermittent disconnect.
- [ ] Delayed stop response.
- [ ] Operator discomfort or loss of confidence in the trial.

If a stop condition occurs repeatedly for the same checkpoint, mark the checkpoint as failed for real-robot validation and return to simulation diagnostics.

## Real-Robot Trial Procedure

### Pre-Trial

- [ ] Confirm checkpoint, method, training seed, and task name.
- [ ] Confirm robot firmware version and controller build.
- [ ] Confirm emergency stop works.
- [ ] Confirm the robot starts from a stable nominal pose.
- [ ] Confirm the test surface is safe and documented.
- [ ] Confirm video recording is active.
- [ ] Confirm the operator has a clear stop command and physical access path.

### Trial

- [ ] Start with zero or near-zero command.
- [ ] Increase command velocity only after stable stance and low-speed motion.
- [ ] Keep the first trial short.
- [ ] Watch feet, ankle behavior, torso roll/pitch, cable/harness interaction, and controller warnings.
- [ ] Stop immediately if any stop condition appears.

### Post-Trial

- [ ] Save logs, video, and operator notes.
- [ ] Record all safety events, including near misses.
- [ ] Inspect motors, joints, feet, harness, and surface.
- [ ] Decide whether to repeat, reduce command range, return to sim2sim, or stop this checkpoint.

## Surface Progression

Use the following order. Do not skip stages.

1. Flat dry safety surface with harness or support.
2. Flat dry safety surface without increasing command range.
3. Indoor artificial turf with harness or support.
4. Indoor artificial turf at low command velocity.
5. Outdoor grass only after artificial turf passes.

Artificial turf is considered passed only when repeated low-speed trials complete without visible repeated slip, safety warnings, abnormal sound or heat, operator discomfort, or delayed stop response.

## Real-Robot Log Sheet Template

Use one row per trial.

| date | robot firmware | policy checkpoint | method | train seed | task | command range | surface description | trial duration | safety events | video filename | operator notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD | firmware/version string | path/to/model.pt | Ours_risk_gate | 1 | Unitree-G1-29dof-Grass-RiskGate | vx/vy/yaw min-max | flat dry mat / artificial turf / grass details | seconds | none / stopped reason | filename.mp4 | notes |

Required fields from the paper checklist are `date`, `robot firmware`, `policy checkpoint`, `command range`, `surface description`, `trial duration`, `safety events`, and `video filename`. The additional fields help connect the trial to the experiment matrix and evaluation results.

## Minimum Evidence Before Reporting Real-Robot Results

- [ ] The exact checkpoint is traceable to training logs and evaluation CSVs.
- [ ] The sim2sim result for the same checkpoint is recorded.
- [ ] The real-robot log sheet is complete.
- [ ] Safety events are reported, including failed or stopped trials.
- [ ] The result is described as qualitative unless enough repeated trials exist for a quantitative claim.
