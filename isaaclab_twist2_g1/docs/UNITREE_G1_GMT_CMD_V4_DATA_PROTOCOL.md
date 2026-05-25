# Unitree G1 GMT Command V4 Data Protocol

## 1. Goal

V4 defines a backend-agnostic command protocol for a VLA policy that can drive multiple G1 GMT backends, starting with SONIC and TWIST2.

The central rule is:

```text
VLA action must represent a short-horizon command in the current robot frame.
It must not represent a backend tracking residual.
```

This is the main difference from v3. V3 stores a robot-current-local target rotation:

```text
action.root_current_local_target_rot6d = rot6d(inverse(Q_robot_current_t) * Q_target_t)
```

That is a target-pose residual. It works when `Q_target_t` closely tracks the robot, but it is not a stable common interface when different GMT backends have different tracking behavior.

The v4 schema name should be:

```text
unitree_g1_gmt_cmd_v4
```

## 2. Why V3 Is Not Sufficient

The SONIC and TWIST2 v3 converters both write a 40D action, but their root targets come from different physical semantics.

### SONIC V3

SONIC conversion builds a target pose from the source body pose:

```text
Q_target_t = aligned SONIC/body root orientation at t
action_rot = inverse(Q_robot_current_t) * Q_target_t
```

In the current data, this residual is usually small because the SONIC target pose and robot root pose stay close. The field therefore looks numerically benign.

### TWIST2 V3

TWIST2 raw action is a 35D mimic command:

```text
[xy_vel(2), z_pos, roll, pitch, yaw_vel, joint_targets(29)]
```

The yaw field is a velocity command, not an absolute heading target. V3 converts it by integrating a hidden target yaw:

```text
yaw_target_t = yaw_target_{t-1} + yaw_vel_t * dt
action_rot = inverse(Q_robot_current_t) * Q_target_t
```

If a recording has:

```text
Pico / command heading: 90 deg
G1 actual heading:      45 deg
tracking residual:      45 deg
```

then the original TWIST2 backend only sees a smooth `yaw_vel` command. It does not directly consume the 45 deg residual. V3, however, converts that residual into the action rotation label. Over an episode, this creates large yaw residuals that are absent from SONIC statistics.

Observed v3 symptom on current datasets:

```text
HOI_pp_box action abs yaw:
  SONIC   p90 ~= 12.6 deg, p99 ~= 23.8 deg
  TWIST2  p90 ~= 44.2 deg, p99 ~= 91.9 deg

TWIST2 yaw residual grows with episode time:
  first 20%: median ~= 1.9 deg,  p90 ~= 10.9 deg
  last  20%: median ~= 29.5 deg, p90 ~= 49.2 deg
```

This is not mainly a video/trajectory quality problem. It is a protocol problem: v3 exposes backend tracking residual to the VLA action label.

## 3. V4 Design Principle

V4 should encode what the upper-level VLA wants the robot to do during the next short horizon, not how far the backend failed to track a previous hidden target.

The root command should mean:

```text
From the current robot state, command the next small motion.
```

It should not mean:

```text
Drive toward a separately integrated or externally accumulated target pose.
```

Consequences:

1. No cumulative `yaw_target` should be stored in the canonical action.
2. No `target_yaw - robot_current_yaw` residual should be stored in the canonical action.
3. Runtime adapters may convert the short-horizon command into backend-specific payloads, but they must not reintroduce a hidden long-horizon target integrator.
4. Per-task SONIC/TWIST2 action statistics should be comparable before normalization.

## 4. Coordinate Frames

Use explicit frame names:

```text
W      Isaac world frame
B_t    current robot root frame at control step t
H_t    current robot heading frame at control step t, yaw-only version of B_t
S_t    source/SONIC body target frame at t
T_t    one-step command target frame produced by the v4 adapter at t
```

V4 root planar velocity is expressed in `H_t`, not in an accumulated target frame:

```text
[vx_heading, vy_heading] = inverse(H_t) * world_xy_velocity
```

V4 yaw is a command rate:

```text
yaw_rate = desired short-horizon yaw velocity, rad/s
```

Equivalently, runtime can use:

```text
yaw_delta = yaw_rate * dt
```

but the stored action field is `yaw_rate` so the semantic is not tied to a particular control frequency.

## 5. Data Format

Keep the first implementation storage-compatible with the existing 40D LeRobot/VLA plumbing:

```text
observation.state: float32[64]
action:            float32[40]
fps:               Isaac control fps, normally 50
schema:            unitree_g1_gmt_cmd_v4
```

The logical command has 37 meaningful dimensions. The remaining 3 dimensions are reserved zeros to preserve the existing 40D action width and keep `joint_pos` and `hand_binary` at the familiar indices.

### 5.1 Observation State

Reuse the v3 observation state unless a later migration proves otherwise:

```text
state[0:6]    state.root_heading_canonical_rot6d
state[6:35]   state.dof_pos.<29 canonical joints>
state[35:64]  state.dof_vel.<29 canonical joints>
```

Root state formula:

```text
Q_state_root_t = inverse(heading(Q_robot_0)) * Q_robot_t
```

The observation describes the current robot. It should not include hidden target yaw, accumulated command yaw, or backend tracking residual.

### 5.2 Action Layout

Use a new set of feature names. Do not call `action[3:9]` rot6d in v4.

```text
action[0]     action.root_heading_local_vx        # m/s
action[1]     action.root_heading_local_vy        # m/s
action[2]     action.root_z                       # m
action[3]     action.root_roll                    # rad, target roll for this step
action[4]     action.root_pitch                   # rad, target pitch for this step
action[5]     action.root_yaw_rate                # rad/s, short-horizon command
action[6]     action.reserved_0                   # must be 0.0
action[7]     action.reserved_1                   # must be 0.0
action[8]     action.reserved_2                   # must be 0.0
action[9:38]  action.joint_pos.<29 canonical joints>
action[38]    action.hand_binary.left
action[39]    action.hand_binary.right
```

Rationale for keeping 40D:

- Existing policy heads, LeRobot metadata, servers, and action chunk code already assume 40D in many places.
- Keeping joints at `[9:38]` and hands at `[38:40]` reduces migration risk.
- Reserved dims give space for future root commands, but must be zero in v4 data and checked by the verifier.

Rationale for using velocity instead of residual:

- TWIST2 already consumes `xy_vel` and `yaw_vel` natively.
- SONIC target-pose data can be converted into velocities through finite differences.
- Velocity/rate commands do not accumulate backend tracking error into the supervised label.

## 6. Offline Conversion Rules

### 6.1 Shared Rules

For every backend:

1. Use the actual Isaac control `dt` from the recording metadata.
2. Express planar velocity in the current robot heading frame `H_t`.
3. Store `root_yaw_rate` as a short-horizon rate, not as cumulative heading residual.
4. Store joints in canonical 29-DOF order.
5. Set `action[6:9] = 0.0` exactly.
6. Reset all finite differences at episode boundaries.
7. Use angle wrapping for per-step yaw deltas:

```text
yaw_rate_t = wrap_to_pi(yaw_t - yaw_{t-1}) / dt
```

Do not compute:

```text
wrap_to_pi(yaw_target_integrated_t - yaw_robot_current_t)
```

### 6.2 TWIST2 To V4

TWIST2 should convert from its raw mimic command without integrating yaw:

```python
mimic = robot_action_mimic[t]

action[0] = mimic[0]                         # vx, current/root heading local, m/s
action[1] = mimic[1]                         # vy, current/root heading local, m/s
action[2] = mimic[2]                         # root z
action[3] = mimic[3]                         # roll
action[4] = mimic[4]                         # pitch
action[5] = mimic[5]                         # yaw_rate, rad/s
action[6:9] = 0.0
action[9:38] = reorder_twist2_to_canonical_29(mimic[6:35])
action[38:40] = hand_binary
```

Do not do this in v4:

```python
yaw_target += mimic[5] * dt
action_yaw = yaw_target - yaw(robot_root_orientation[t])
```

The fact that a TWIST2 recording may contain `command heading = 90 deg` while `robot heading = 45 deg` is allowed. That residual is backend tracking behavior and must not become the VLA action.

### 6.3 SONIC To V4

SONIC should also be converted to the same command semantics. It must not keep the v3 target-pose residual as the canonical action.

First construct the aligned source target pose as in v3:

```python
Q_target_t = source_to_robot_heading * Q_body_t
P_target_t = aligned body position in Isaac/world coordinates
```

Then convert the target pose trajectory into short-horizon commands:

```python
world_vel_xy_t = (P_target_t.xy - P_target_{t-1}.xy) / dt
[vx, vy] = inverse(heading(Q_robot_current_t)) * [world_vel_xy_t.x, world_vel_xy_t.y, 0]

roll, pitch = roll_pitch(Q_target_t)
yaw_rate = wrap_to_pi(yaw(Q_target_t) - yaw(Q_target_{t-1})) / dt
```

For the first frame of an episode:

```text
vx = 0
vy = 0
yaw_rate = 0
```

Use the current robot heading frame for `vx/vy`, not the target heading frame. This keeps the command tied to the current state observed by the VLA and avoids target-frame drift.

### 6.4 Optional Smoothing

Finite differences from SONIC pose targets may be noisier than TWIST2 native commands. If smoothing is needed, it must be a documented converter option and applied consistently before statistics are computed.

Rules:

1. Do not smooth across episode boundaries.
2. Do not introduce future leakage if the model is meant to be causal.
3. Report stats before and after smoothing.
4. Use the same smoothing policy for all backends only if it preserves native semantics. Otherwise prefer no smoothing plus robust training normalization.

## 7. Runtime Adapter Rules

The shared v4 runtime should parse the action into a command frame:

```text
GMTCommandFrameV4(
  vx_heading,
  vy_heading,
  root_z,
  root_roll,
  root_pitch,
  root_yaw_rate,
  joint_pos_canonical_29,
  hand_binary,
)
```

It should not maintain an accumulated root target pose. Hidden accumulation is the main failure mode v4 is designed to remove.

### 7.1 TWIST2 Adapter

TWIST2 receives the native mimic observation:

```python
mimic_obs = [
    vx_heading,
    vy_heading,
    root_z,
    root_roll,
    root_pitch,
    root_yaw_rate,
    *reorder_canonical_to_twist2_29(joint_pos_canonical_29),
]
```

This should round-trip TWIST2 v4 data almost exactly:

```text
v4 action -> TWIST2 adapter -> mimic_obs ~= original robot_action_mimic
```

### 7.2 SONIC Adapter

SONIC needs a short-horizon body target payload. Build it from the current robot pose each control step:

```python
xy_delta_world = heading(Q_robot_current_t) * [vx_heading * dt, vy_heading * dt, 0]
body_pos = [robot_xy_current + xy_delta_world, root_z]

target_yaw = yaw(Q_robot_current_t) + root_yaw_rate * dt
body_quat_w = quat_from_roll_pitch_yaw(root_roll, root_pitch, target_yaw)

payload = {
    "body_pos": body_pos,
    "body_quat_w": body_quat_w,
    "joint_pos": joint_pos_canonical_29,
    "joint_vel": finite_difference(joint_pos_canonical_29),
}
```

Do not set SONIC `body_quat_w` from an accumulated target yaw. Do not accumulate `body_pos` in a target-only frame that can drift away from the robot.

This means the SONIC adapter receives one-step targets close to the current robot, while the TWIST2 adapter receives native velocity commands. Both are derived from the same v4 command.

## 8. Statistical Compatibility Requirements

A v4 verifier must compare SONIC and TWIST2 per task before training. The verifier should fail loudly when one backend has a much larger distribution than the other.

Recommended verifier:

```text
tools/data_tools/verify_lerobot_gmt_cmd_v4.py
```

Required per-task, per-backend report:

```text
rows, episodes, fps
abs(root_heading_local_vx), abs(root_heading_local_vy), xy_speed
root_z
abs(root_roll), abs(root_pitch)
abs(root_yaw_rate)
joint_pos mean/std/p99 per joint
hand_binary rates
reserved max_abs
```

Required episode-bin report:

```text
For each backend and task, split each episode into 5 equal time bins.
Report p50/p90/p99 of abs(root_yaw_rate), xy_speed, abs(root_roll), abs(root_pitch).
```

The key anti-regression rule is:

```text
No root command statistic should grow monotonically through the episode because of hidden target accumulation.
```

Suggested hard gates for same-task SONIC vs TWIST2 comparison:

```text
reserved max_abs == 0.0
abs(root_yaw_rate) p99 backend ratio <= 3.0, unless both p99 < 0.25 rad/s
xy_speed p99 backend ratio <= 3.0, unless both p99 < 0.05 m/s
abs(root_roll) p99 backend ratio <= 3.0, unless both p99 < 0.05 rad
abs(root_pitch) p99 backend ratio <= 3.0, unless both p99 < 0.05 rad
last_bin abs(root_yaw_rate) p90 / first_bin p90 <= 2.5
last_bin xy_speed p90 / first_bin p90 <= 2.5
```

Suggested warning gates:

```text
abs(root_yaw_rate) p99 > 3.0 rad/s
abs(root_yaw_rate) max > 6.0 rad/s
xy_speed p99 > 2.0 m/s
abs(root_roll) p99 > 0.6 rad
abs(root_pitch) p99 > 0.6 rad
any joint p99 outside known G1 joint limits
```

These thresholds should be treated as initial guardrails, not physics laws. The important rule is that any large SONIC/TWIST2 discrepancy must be explained and signed off before training.

## 9. Round-Trip And Adapter Tests

### 9.1 TWIST2 Native Round-Trip

For each converted TWIST2 episode, reconstruct mimic obs from v4 action and compare to the original raw mimic command:

```text
max_abs(vx/vy/z/roll/pitch/yaw_rate) < 1e-5
max_abs(joint_pos after order conversion) < 1e-5
hand binary exact
```

This verifies that v4 did not accidentally integrate yaw or change units.

### 9.2 SONIC Command Reconstruction

For SONIC, reconstruct one-step pose targets from v4 using the current robot pose and compare the implied short-horizon command to the stored action:

```text
reconstructed vx/vy matches action[0:2]
reconstructed yaw_rate matches action[5]
roll/pitch match action[3:5]
joint_pos match action[9:38]
```

Do not require reconstructed one-step target pose to equal the original accumulated SONIC target pose exactly. V4 intentionally converts pose-target trajectories into command semantics.

### 9.3 No Hidden Residual Test

For every backend, compute:

```text
command_yaw_delta_t = action.root_yaw_rate_t * dt
robot_yaw_delta_t = yaw(robot_t) - yaw(robot_{t-1})
```

The verifier may report their correlation, but it must not construct or supervise:

```text
integrated_command_yaw_t - robot_yaw_t
```

That residual is a diagnostic only, never an action label.

## 10. Training And Normalization Rules

1. Use the same feature names and indices for all backends.
2. Compute and inspect raw physical-unit statistics before normalization.
3. Then compute merged normalization stats only after the backend compatibility report passes.
4. Do not let normalization hide a backend semantic mismatch. A field with TWIST2 p99 ten times larger than SONIC p99 is a protocol bug unless explicitly justified.
5. Store `schema=unitree_g1_gmt_cmd_v4` in metadata so v3 models cannot be accidentally evaluated with v4 adapters.

If training one merged model across backends, do not include backend-specific residuals or hidden target states. If the same visual state maps to genuinely different commands for different backends after v4 conversion, either fix the converters/adapters or explicitly condition the model on backend. Do not rely on the model to infer backend from action statistics.

## 11. Migration Plan

Recommended new files:

```text
action_provider/vla_gmt_cmd_runtime_v4.py
tools/data_tools/smpl_lerobot_gmt_cmd_v4_common.py
tools/data_tools/sonic2lerobot_gmt_cmd_v4.py
tools/data_tools/twist2lerobot_gmt_cmd_v4.py
tools/data_tools/verify_lerobot_gmt_cmd_v4.py
tools/data_tools/batch_convert_gmt_cmd_v4.sh
```

Recommended runtime changes:

```text
action_provider/action_provider_sonic.py
  - Add robot_type=unitree_g1_gmt_cmd_v4 path.
  - Convert v4 command into one-step SONIC body target payload.

action_provider/action_provider_wh_twist2.py
  - Add robot_type=unitree_g1_gmt_cmd_v4 path.
  - Convert v4 command directly into TWIST2 mimic_obs.
```

Recommended eval defaults:

```text
ROBOT_TYPE=unitree_g1_gmt_cmd_v4
RESULTS_TAG_PREFIX should include gmt-cmd-v4
```

Do not modify v2/v3 converters in place. Keep them as historical protocols.

## 12. Main Risk Items

### Risk 1: Accidentally reintroducing target accumulation

The most important failure mode is a runtime or converter that secretly maintains:

```text
yaw_target += yaw_rate * dt
```

and then supervises or executes:

```text
yaw_target - robot_current_yaw
```

This recreates the v3 TWIST2 problem. Accumulated yaw may be used for debug plots only, not for canonical action labels.

### Risk 2: SONIC and TWIST2 still using different root semantics

If SONIC keeps target-pose residual while TWIST2 uses velocity command, the merged dataset will still be semantically inconsistent. V4 requires both to use short-horizon command fields.

### Risk 3: Velocity frame mismatch

TWIST2 native `xy_vel` is root/base local. V4 standardizes on current robot heading-local planar velocity. The converter and runtime must not rotate this field twice, and must not use target-heading-local velocity.

### Risk 4: Finite-difference noise in SONIC

SONIC pose targets converted to velocities may introduce spikes. Use angle wrapping, episode-boundary resets, and optional documented smoothing. Verify p99/max values before training.

### Risk 5: FPS and dt mismatch

Because v4 stores velocities/rates, every converter and runtime must use the correct `dt` when deriving or applying short-horizon targets. Metadata must preserve fps/control_dt.

### Risk 6: Reserved dims leaking values

`action[6:9]` are reserved to preserve 40D width. They must be zero in data and ignored by runtime. Nonzero values should fail verification.

### Risk 7: Normalization hiding bad semantics

Merged normalization can make two incompatible distributions look numerically trainable. Always inspect raw physical-unit stats first.

### Risk 8: Backend-specific duplicate demonstrations

If SONIC and TWIST2 recordings of the same task encode different user behavior, a merged policy may see multi-modal labels. The first fix should be converter semantics and stats. If true backend-specific behavior remains, add explicit backend conditioning rather than relying on hidden distribution differences.

## 13. Acceptance Checklist

Before training v4 models, require:

```text
[ ] All datasets declare schema=unitree_g1_gmt_cmd_v4.
[ ] Feature names match the v4 action layout exactly.
[ ] Reserved dims max_abs == 0.0.
[ ] TWIST2 v4->adapter round-trip to robot_action_mimic passes.
[ ] SONIC finite-difference command reconstruction passes.
[ ] Per-task SONIC/TWIST2 raw stats report is generated.
[ ] No same-task root field has unexplained p99 backend ratio > 3x.
[ ] No episode-bin yaw_rate or xy_speed drift indicates hidden accumulation.
[ ] Eval scripts use ROBOT_TYPE=unitree_g1_gmt_cmd_v4.
[ ] V3 models are not evaluated with v4 runtime adapters.
```

## 14. Summary

V4 should make the common VLA-GMT interface a short-horizon egocentric command protocol:

```text
[vx, vy, z, roll, pitch, yaw_rate, reserved(3), joint_pos_29, hand_binary_2]
```

TWIST2 maps to this almost directly from native mimic commands. SONIC maps to it by finite-differencing aligned pose targets into current-robot-frame velocities and yaw rates. Runtime adapters then turn the same command into backend-specific payloads.

This removes the v3 failure mode where backend tracking residual, especially TWIST2 integrated yaw residual, becomes a supervised VLA action. The protocol is only accepted if raw SONIC/TWIST2 action statistics are comparable per task before normalization.
