# Unitree G1 GMT Command V4 Data Protocol

## 1. Goal

V4 defines a backend-agnostic command protocol for a VLA policy that can drive multiple G1 GMT backends, starting with SONIC and TWIST2.

The central rule is:

```text
VLA action must represent a short-horizon command in the current robot frame.
It must not represent a backend tracking residual.
```

The v4 schema name should be:

```text
unitree_g1_gmt_cmd_v4
```

V4 is not only a data-format migration. It is a semantic migration. In particular, it intentionally moves the root command away from the v3 target-pose residual interface:

```text
v3: rot6d(inverse(Q_robot_current_t) * Q_target_t)
v4: short-horizon velocity/rate command expressed in the current robot base frame
```

This is meant to support a common VLA output across GMT backends whose tracking behavior differs.

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
5. Any backend-specific motion-history construction must be fed by v4 one-step payloads, not by hidden long-horizon reference targets.

## 4. Coordinate Frames And Timebase

Use explicit frame names:

```text
W      Isaac world frame
B_t    current robot root/base frame at control step t
H_t    yaw-only heading frame of B_t, used by the observation state only
S_t    source/SONIC body target frame at t
T_t    one-step target frame produced by a v4 runtime adapter at t
```

### 4.1 Root Linear Command Frame

V4 root linear velocity is expressed in the **full current robot base frame `B_t`**, not in heading-only frame `H_t`.

```text
v_B_t = inverse(Q_robot_current_t) * v_W_t
action.root_base_local_vx = v_B_t.x
action.root_base_local_vy = v_B_t.y
```

This choice is intentional. Current TWIST2 mimic extraction uses the full root quaternion:

```python
base_vel_local = quat_rotate_inverse(root_quat, base_vel)
mimic_obs[0:2] = base_vel_local[:2]
```

Using heading-local xy would be only approximately equivalent when roll/pitch are small. V4 should not rely on that approximation.

For SONIC pose trajectories, the converter must also compute full-base-local velocity by applying `inverse(Q_robot_current_t)` to the target world velocity. This keeps SONIC and TWIST2 in the same physical frame.

### 4.2 Root Angular Command

V4 stores one scalar angular command:

```text
action.root_yaw_rate = omega_B_t.z
```

This is the z component of desired short-horizon angular velocity expressed in the current full robot base frame `B_t`, in rad/s. For an upright robot this is numerically close to Euler yaw-rate, but the protocol definition is base-local angular-z, not world-yaw finite difference.

This choice matches TWIST2 native mimic extraction, which computes local angular velocity using the full root quaternion and stores the local z component.

Runtime can use the command as a one-step heading increment:

```text
yaw_delta ~= root_yaw_rate * runtime_control_dt
```

The approximation is acceptable only for the initial v4 schema, which stores roll/pitch targets plus a single angular-z command. If future tasks require large roll/pitch angular motion, use the reserved dims for full base-local angular velocity instead of overloading `root_yaw_rate`.

The stored value is a rate, so its unit is not tied to a particular fps. However, conversion and runtime still depend on the correct control period. See the metadata rule below.

### 4.3 Mandatory Timebase Metadata

Every v4 dataset must record:

```text
schema:             unitree_g1_gmt_cmd_v4
fps:                nominal control frequency
control_dt:         1 / fps, or measured median control dt
control_dt_source:  metadata key or measurement method used by converter
```

If a recording has variable control dt, the converter must either:

1. resample to a fixed control dt before writing v4, or
2. store per-frame dt and make every runtime/training consumer explicitly handle it.

For the first v4 implementation, prefer fixed-dt datasets. The verifier must assert that offline conversion dt and runtime adapter dt match in round-trip tests. If a model is trained on mixed fps datasets, the data must be resampled or the policy must be explicitly conditioned on dt.

## 5. Data Format

Keep the first implementation storage-compatible with the existing 40D LeRobot/VLA plumbing:

```text
observation.state: float32[64]
action:            float32[40]
fps:               Isaac control fps, normally 50
control_dt:        required metadata
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

Use new feature names. Do not call `action[3:9]` rot6d in v4.

```text
action[0]     action.root_base_local_vx           # m/s, full B_t frame
action[1]     action.root_base_local_vy           # m/s, full B_t frame
action[2]     action.root_z                       # m
action[3]     action.root_roll                    # rad, short-horizon target roll
action[4]     action.root_pitch                   # rad, short-horizon target pitch
action[5]     action.root_yaw_rate                # rad/s, base-local angular-z command
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

Rationale for base-local velocity/rate command:

- TWIST2 already consumes base-local `xy_vel` and `yaw_vel` natively.
- SONIC target-pose data can be converted into the same command semantics through finite differences and current-root-frame projection.
- Velocity/rate commands do not accumulate backend tracking error into the supervised label.

## 6. Offline Conversion Rules

### 6.1 Shared Rules

For every backend:

1. Use the actual Isaac control `dt` from recording metadata.
2. Store `control_dt` and `control_dt_source` in dataset metadata.
3. Express root xy velocity in the current full robot base frame `B_t`.
4. Store `root_yaw_rate` as a short-horizon rate, not as cumulative heading residual.
5. Store joints in canonical 29-DOF order.
6. Set `action[6:9] = 0.0` exactly.
7. Reset all finite differences at episode boundaries.
8. Compute angular command from quaternion finite differences, then project into the current robot base frame:

```text
omega_world_t = angular_velocity_from_quat_pair(Q_target_{t-1}, Q_target_t, dt)
omega_base_t = inverse(Q_robot_current_t) * omega_world_t
root_yaw_rate_t = omega_base_t.z
```

A simple wrapped Euler yaw difference may be logged for diagnostics, but it is not the protocol definition when roll/pitch are nonzero.

Do not compute or supervise:

```text
wrap_to_pi(yaw_target_integrated_t - yaw_robot_current_t)
```

### 6.2 TWIST2 To V4

TWIST2 should convert from its raw mimic command without integrating yaw:

```python
mimic = robot_action_mimic[t]

action[0] = mimic[0]                         # base-local vx, m/s
action[1] = mimic[1]                         # base-local vy, m/s
action[2] = mimic[2]                         # root z
action[3] = mimic[3]                         # roll
action[4] = mimic[4]                         # pitch
action[5] = mimic[5]                         # base-local angular-z rate, rad/s
action[6:9] = 0.0
action[9:38] = reorder_twist2_to_canonical_29(mimic[6:35])
action[38:40] = hand_binary
```

This direct copy is correct only because v4 defines xy as full-base-local, matching TWIST2's native mimic extraction. Do not reinterpret these values as heading-local.

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
v_world_t = (P_target_t - P_target_{t-1}) / dt
v_base_t = inverse(Q_robot_current_t) * v_world_t

action[0:2] = v_base_t[:2]
action[2] = P_target_t.z
action[3:5] = roll_pitch(Q_target_t)
omega_world_t = angular_velocity_from_quat_pair(Q_target_{t-1}, Q_target_t, dt)
omega_base_t = inverse(Q_robot_current_t) * omega_world_t
action[5] = omega_base_t.z
action[6:9] = 0.0
action[9:38] = joint_targets_canonical_29[t]
action[38:40] = hand_binary[t]
```

For the first frame of an episode:

```text
action[0] = 0
action[1] = 0
action[5] = 0
```

This is a deliberate semantic change for SONIC. The old SONIC target-pose trajectory is converted into one-step command labels. The original target pose may still be stored as debug metadata, but it is not the v4 supervised action.

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
  vx_base,
  vy_base,
  root_z,
  root_roll,
  root_pitch,
  root_yaw_rate,
  joint_pos_canonical_29,
  hand_binary,
)
```

It should not maintain an accumulated root target pose that is independent of the current robot. Hidden accumulation is the main failure mode v4 is designed to remove.

A backend may still maintain short rolling histories required by its own low-level model, but those histories must be derived from executed v4 one-step payloads and current robot state. They must not be derived from a hidden long-horizon reference target.

### 7.1 TWIST2 Adapter

TWIST2 receives the native mimic observation:

```python
mimic_obs = [
    vx_base,
    vy_base,
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

SONIC needs a body target payload, but v4 changes how that payload is produced. This is a core semantic change, not a simple file-format migration.

Current SONIC provider logic builds relative root/reference features and rolling histories from `body_quat_w`, for example the local provider computes `base_to_ref = base^-1 * ref` and updates motion anchor/history windows. V4 must feed that machinery with one-step payloads generated from the current robot pose and v4 command. It should not feed an accumulated external pose target that can drift away from the robot.

Build the one-step SONIC payload as follows:

```python
R = rotation_matrix(Q_robot_current_t)
dt = runtime_control_dt

# Choose the local z velocity that makes the one-step world target land at root_z.
world_z_vel = (root_z - robot_z_current) / dt
vz_base = (world_z_vel - R[2, 0] * vx_base - R[2, 1] * vy_base) / max(R[2, 2], eps)

v_world = R @ [vx_base, vy_base, vz_base]
body_pos = robot_pos_current + v_world * dt
body_pos.z = root_z

# Initial v4 uses the angular-z command as a one-step heading increment.
# This is exact only near the upright regime; large roll/pitch angular tasks need a full angular command extension.
target_yaw = yaw(Q_robot_current_t) + root_yaw_rate * dt
body_quat_w = quat_from_roll_pitch_yaw(root_roll, root_pitch, target_yaw)

payload = {
    "body_pos": body_pos,
    "body_quat_w": body_quat_w,
    "joint_pos": joint_pos_canonical_29,
    "joint_vel": finite_difference(joint_pos_canonical_29),
}
```

If `abs(R[2,2])` is too small, the robot is far from upright and the adapter should fail loudly or enter a documented safety fallback. Silent clipping would hide a protocol error.

This adapter no longer asks SONIC to track a long source/reference trajectory directly. It gives SONIC a one-step body target close to the current robot. That may change SONIC stability and task success because SONIC's existing reference/history pathway was tuned for body pose streams. V4 implementation therefore requires explicit SONIC-only regression tests before merged training.

Required SONIC v4 regression checks:

```text
- Compare v4 SONIC success/fall rates against v3 SONIC on the same tasks and checkpoints.
- Log and compare motion-anchor/root-reference feature distributions before and after v4.
- Check body_quat_w step delta p90/p99 and reject unexplained spikes.
- Check that provider history windows remain valid and do not collapse to constants.
- If SONIC needs longer lookahead to preserve performance, add explicit short-horizon command history or horizon fields to v4; do not reintroduce hidden target residual.
```

## 8. Statistical Compatibility Requirements

A v4 verifier must compare SONIC and TWIST2 per task before training. The verifier should fail loudly when one backend has a much larger distribution than the other, but it must also avoid false positives for tasks with real phase changes.

Recommended verifier:

```text
tools/data_tools/verify_lerobot_gmt_cmd_v4.py
```

Required per-task, per-backend report:

```text
rows, episodes, fps, control_dt
abs(root_base_local_vx), abs(root_base_local_vy), xy_speed
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
No root command statistic should grow through the episode because of hidden target accumulation.
```

However, some tasks naturally have phase changes. For example, a demonstration can stand still early and turn later. The verifier must therefore combine ratio gates with absolute floors and require human/task-level explanation for phase-driven trends.

Suggested same-task SONIC vs TWIST2 hard gates:

```text
reserved max_abs == 0.0

backend_ratio(field) = max(p99_sonic, p99_twist2) / max(min(p99_sonic, p99_twist2), floor(field))

floor(abs(root_yaw_rate)) = 0.25 rad/s
floor(xy_speed)           = 0.05 m/s
floor(abs(root_roll))     = 0.05 rad
floor(abs(root_pitch))    = 0.05 rad

backend_ratio(abs(root_yaw_rate)) <= 3.0
backend_ratio(xy_speed)           <= 3.0
backend_ratio(abs(root_roll))     <= 3.0
backend_ratio(abs(root_pitch))    <= 3.0
```

Suggested trend gates:

```text
trend_ratio(field) = last_bin_p90(field) / max(first_bin_p90(field), floor(field))

Warn if trend_ratio(abs(root_yaw_rate)) > 2.5 and last_bin_p90 > 0.5 rad/s.
Warn if trend_ratio(xy_speed)           > 2.5 and last_bin_p90 > 0.2 m/s.

Hard-fail a trend only if:
  1. it appears in one backend but not the other for the same task,
  2. it is not explained by task phase or scripted curriculum,
  3. and the absolute last-bin value is physically large.
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

These thresholds are guardrails, not physics laws. The important rule is that any large SONIC/TWIST2 discrepancy must be explained and signed off before training. Normalization must not be used to hide an unexplained raw-stat mismatch.

## 9. Round-Trip And Adapter Tests

### 9.1 TWIST2 Native Round-Trip

For each converted TWIST2 episode, reconstruct mimic obs from v4 action and compare to the original raw mimic command:

```text
max_abs(vx/vy/z/roll/pitch/yaw_rate) < 1e-5
max_abs(joint_pos after order conversion) < 1e-5
hand binary exact
```

This verifies that v4 did not accidentally integrate yaw, change units, or reinterpret base-local xy as heading-local xy.

### 9.2 SONIC Command Reconstruction

For SONIC, reconstruct one-step pose targets from v4 using the current robot pose and the recorded `control_dt`. Then recover the implied command from that one-step target and compare it to the stored action:

```text
reconstructed base-local vx/vy matches action[0:2]
reconstructed yaw_rate matches action[5]
roll/pitch match action[3:5]
joint_pos match action[9:38]
```

Do not require the reconstructed one-step target pose to equal the original accumulated SONIC target pose exactly. V4 intentionally converts pose-target trajectories into command semantics.

### 9.3 Timebase Round-Trip

Every verifier and runtime smoke test must check:

```text
abs(dataset_control_dt - runtime_control_dt) < tolerance
```

For fixed-fps datasets, use a strict tolerance such as `1e-6`. If runtime intentionally uses a different control period, the test must explicitly show that velocities/rates are applied with the runtime dt and that success/stability does not regress.

### 9.4 No Hidden Residual Test

For every backend, compute diagnostics:

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
6. Store and validate `control_dt`; do not train/evaluate mixed-fps v4 data unless resampling or dt conditioning is explicitly implemented.

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
  - Treat this as a SONIC semantic change and run SONIC-only regression.

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

### Risk 2: SONIC semantic regression

V4 changes SONIC from following an external/source pose trajectory to receiving one-step body targets reconstructed from current robot pose and v4 command. This can affect SONIC's reference/history machinery. A v4 implementation is incomplete until SONIC-only success/fall rate, motion-anchor statistics, root-reference jumps, and history-window validity are checked against v3.

### Risk 3: SONIC and TWIST2 still using different root semantics

If SONIC keeps target-pose residual while TWIST2 uses velocity command, the merged dataset will still be semantically inconsistent. V4 requires both to use short-horizon command fields.

### Risk 4: Velocity frame mismatch

TWIST2 native `xy_vel` is full root/base local. V4 also uses full base-local xy. Do not convert TWIST2 values to heading-local unless the protocol is explicitly changed and all adapters are updated. Do not rotate this field twice.

### Risk 5: Finite-difference noise in SONIC

SONIC pose targets converted to velocities may introduce spikes. Use quaternion angular velocity, episode-boundary resets, optional documented smoothing, and raw p99/max verification before training.

### Risk 6: Angular-rate frame mismatch

TWIST2 native `yaw_vel` is the z component of base-local angular velocity, not necessarily Euler world yaw-rate. SONIC conversion must project quaternion-derived angular velocity into the current robot base frame before taking z. If this is approximated with Euler yaw difference, the verifier must report the approximation error and roll/pitch distribution.

### Risk 7: FPS and dt mismatch

Because v4 stores velocities/rates, every converter and runtime must use the correct `dt` when deriving or applying short-horizon targets. Metadata must preserve fps/control_dt, and round-trip tests must check converter dt against runtime dt.

### Risk 8: Reserved dims leaking values

`action[6:9]` are reserved to preserve 40D width. They must be zero in data and ignored by runtime. Nonzero values should fail verification.

### Risk 9: Statistical gates causing false positives

Ratio-only trend gates can misfire when a task is intentionally static early and moves later. Use absolute floors, absolute-value warnings, and task-phase explanations. Hard-fail only unexplained backend-specific drift.

### Risk 10: Normalization hiding bad semantics

Merged normalization can make two incompatible distributions look numerically trainable. Always inspect raw physical-unit stats first.

### Risk 11: Backend-specific duplicate demonstrations

If SONIC and TWIST2 recordings of the same task encode different user behavior, a merged policy may see multi-modal labels. The first fix should be converter semantics and stats. If true backend-specific behavior remains, add explicit backend conditioning rather than relying on hidden distribution differences.

## 13. Acceptance Checklist

Before training v4 models, require:

```text
[ ] All datasets declare schema=unitree_g1_gmt_cmd_v4.
[ ] Dataset metadata includes fps, control_dt, and control_dt_source.
[ ] Feature names match the v4 action layout exactly.
[ ] Reserved dims max_abs == 0.0.
[ ] TWIST2 v4->adapter round-trip to robot_action_mimic passes.
[ ] TWIST2 xy is verified as full base-local, not heading-local.
[ ] SONIC finite-difference command reconstruction passes, using quaternion angular velocity projected to current base frame.
[ ] SONIC v4 adapter success/fall regression against v3 is completed.
[ ] SONIC motion-anchor/root-reference feature stats are checked for spikes/collapse.
[ ] Per-task SONIC/TWIST2 raw stats report is generated.
[ ] No same-task root field has unexplained p99 backend ratio > 3x after floors.
[ ] No episode-bin yaw_rate or xy_speed drift indicates hidden accumulation.
[ ] Any trend-gate warning is explained by task phase or fixed before training.
[ ] Eval scripts use ROBOT_TYPE=unitree_g1_gmt_cmd_v4.
[ ] V3 models are not evaluated with v4 runtime adapters.
```

## 14. Summary

V4 should make the common VLA-GMT interface a short-horizon base-local command protocol:

```text
[vx_base, vy_base, z, roll, pitch, yaw_rate, reserved(3), joint_pos_29, hand_binary_2]
```

TWIST2 maps to this directly from native mimic commands. SONIC maps to it by finite-differencing aligned pose targets into current-full-base-frame velocities and yaw rates, then using a runtime adapter to create one-step body target payloads.

This removes the v3 failure mode where backend tracking residual, especially TWIST2 integrated yaw residual, becomes a supervised VLA action. It also makes the SONIC migration risk explicit: v4 changes SONIC's runtime reference semantics and must be validated with SONIC-specific regression tests. The protocol is only accepted if raw SONIC/TWIST2 action statistics are comparable per task before normalization, with robust gates that distinguish hidden accumulation from real task phases.
