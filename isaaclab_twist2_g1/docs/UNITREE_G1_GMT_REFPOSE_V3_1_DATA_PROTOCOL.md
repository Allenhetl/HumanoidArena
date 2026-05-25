# Unitree G1 GMT Reference Pose V3.1 Data Protocol

## 1. Goal

V3.1 defines a reference-pose action protocol for a VLA policy that should drive multiple Unitree G1 GMT backends while keeping the VLA layer separated from backend tracking error.

The central rule is:

```text
VLA action = reference motion target.
VLA action != inverse(Q_robot_current) * Q_target.
```

The schema name should be:

```text
unitree_g1_gmt_refpose_v3_1
```

V3.1 is not the same as V4. V4 is a backend-agnostic command protocol. V3.1 keeps the existing 40D pose/reference action shape, but changes the semantic contract so the action no longer stores a robot-current residual.

```text
v3:   action_rot = rot6d(inverse(Q_robot_current_t) * Q_target_t)
v3.1: action_rot = rot6d(Q_ref_t in the episode reference frame)
v4:   action_root = short-horizon command in current robot/base frame
```

V3.1 should be treated as a data repair and semantic clarification for the current pose-target training path, especially for TWIST2.

## 2. Key Terms

```text
Q_robot_current_t
    The simulated robot root orientation at frame t. It is observation/state.
    It must not be subtracted from the action label.

Q_ref_t / Q_target_t
    The root reference orientation at frame t. In this document these two names
    refer to the same object: the root quaternion of the reference trajectory.

P_ref_t
    The root reference position at frame t.

J_ref_t
    The 29D canonical joint reference target at frame t.

Q_residual_t
    inverse(Q_robot_current_t) * Q_ref_t. This is a backend tracking error.
    It is not a V3.1 action label.

Episode reference frame E
    A per-episode gauge frame used to remove arbitrary global yaw/position.
    The action stores reference motion in E, not in the robot-current frame.
```

A full reference pose is not only `Q_ref_t`. It includes:

```text
P_ref_t        root position reference
Q_ref_t        root orientation reference quaternion
J_ref_t        29D joint reference
hand_ref_t     hand or gripper target
```

## 3. Why V3 Needed Repair

The old v3 conversion used the same 40D layout for SONIC and TWIST2, but the root rotation label was built as a robot-current residual:

```text
action_rot_t = inverse(Q_robot_current_t) * Q_target_t
```

This looked numerically acceptable for SONIC because SONIC is closer to a pose-tracking backend and its reference pose usually stays close to the robot root.

For TWIST2, the native low-level input is not an absolute root pose target. It is a mimic command:

```text
robot_action_mimic = [vx_base, vy_base, z, roll, pitch, yaw_vel, joint_pos_29]
```

The online TWIST2 server first has a GMR qpos, but it immediately converts it into local velocities and setpoints:

```text
qpos_t, qpos_{t-1}
  -> finite difference
  -> [vx_base, vy_base, z, roll, pitch, yaw_vel, joint_pos_29]
```

The TWIST2 low-level policy consumes this command. It does not directly consume:

```text
inverse(Q_robot_current_t) * Q_gmr_t
```

Therefore, if the operator/GMR yaw turns 90 degrees but the simulated G1 only turns 45 degrees, the native TWIST2 command can still be smooth and stable, while the v3 residual label contains a large 45 degree yaw error. This contaminates the VLA action with backend tracking lag.

## 4. V3.1 40D Action Layout

V3.1 keeps the same 40D action size so existing model heads can remain compatible.

```text
action[0:2]   root_ref_local_xy_delta
              Reference root XY displacement from frame t-1 to t,
              expressed in the current reference root frame Q_ref_t.
              Unit: meter per control step.

action[2]     root_ref_z
              Reference root height at frame t.
              Unit: meter.

action[3:9]   root_ref_rot6d
              6D rotation representation of Q_ref_t in the episode reference frame E.
              This is an absolute reference orientation in E, not a residual to Q_robot_current_t.

action[9:38]  joint_ref_canonical_29
              29D canonical Unitree G1 joint position target.
              Unit: rad.

action[38:40] hand_binary
              [left_hand, right_hand] binary target.
```

Important constraints:

```text
- Q_robot_current_t is allowed in observation/state.
- Q_robot_current_t is not allowed in action label construction.
- action[3:9] must encode Q_ref_t, not inverse(Q_robot_current_t) * Q_ref_t.
- action[0:2] is a reference-trajectory displacement, not a robot-current correction.
```

## 5. Episode Reference Frame

The absolute global yaw of a pose target is a gauge. V3.1 stores reference motion in an episode reference frame E.

Recommended anchor:

```text
P_anchor = P_ref_0
Q_anchor = heading(Q_ref_0)
```

Then:

```text
P_ref_E_t = R(Q_anchor)^-1 * (P_ref_world_t - P_anchor)
Q_ref_E_t = inverse(Q_anchor) * Q_ref_world_t
```

For TWIST2 mimic-integrated reference, there may be no meaningful original global world yaw. In that case use:

```text
P_ref_E_0 = [0, 0, z_0]
yaw_ref_E_0 = 0
Q_anchor = identity heading
```

This is intentional. The target trajectory's relative motion matters; its absolute yaw origin should not encode robot tracking error.

Metadata must record the anchor convention.

## 6. SONIC Conversion

SONIC already has a reference-pose style input, such as:

```text
body_pos_t
body_quat_w_t
joint_target_t
```

For V3.1, SONIC conversion should treat this as the reference trajectory:

```text
P_ref_t = body_pos_t
Q_ref_t = body_quat_w_t
J_ref_t = sonic joint target in canonical 29D order
```

Then convert to the 40D V3.1 action:

```text
Q_ref_E_t = inverse(Q_anchor) * Q_ref_t
action[3:9] = rot6d(Q_ref_E_t)
```

The root XY displacement should be computed from the reference trajectory, not from robot-current residual:

```text
dP_world_t = P_ref_t - P_ref_{t-1}
dP_E_t = R(Q_anchor)^-1 * dP_world_t
dP_local_t = R(Q_ref_E_t)^-1 * dP_E_t
action[0:2] = dP_local_t[0:2]
```

For `t = 0`, use:

```text
action[0:2] = [0, 0]
```

SONIC runtime may still compute internal relative features such as:

```text
inverse(Q_robot_current_t) * aligned_Q_ref_t
```

That is acceptable inside the SONIC adapter/provider. It is not acceptable as the VLA action label.

## 7. TWIST2 Conversion: Main Path

For old TWIST2 data, the recommended main path is not offline GMR reconstruction. The recommended main path is to construct a command-consistent reference trajectory from the recorded native command:

```text
mimic_t = robot_action_mimic_t
        = [vx_base, vy_base, z, roll, pitch, yaw_vel, joint_pos_29]
```

This is preferred because `robot_action_mimic` is exactly what the TWIST2 low-level backend consumed during recording.

### 7.1 TWIST2 Reference Reconstruction From Mimic

Initialize:

```text
yaw_ref_0 = 0
P_ref_0 = [0, 0, z_0]
Q_ref_0 = quat_from_rpy(roll_0, pitch_0, yaw_ref_0)
```

For `t > 0`, let:

```text
dt_t = control time step for transition t-1 -> t
vx_t, vy_t = mimic_t[0:2]
z_t = mimic_t[2]
roll_t = mimic_t[3]
pitch_t = mimic_t[4]
yaw_vel_t = mimic_t[5]
```

Then:

```text
yaw_ref_t = yaw_ref_{t-1} + yaw_vel_t * dt_t
Q_ref_t = quat_from_rpy(roll_t, pitch_t, yaw_ref_t)
```

To reconstruct root position with the same frame convention as TWIST2, use the current reference orientation `Q_ref_t`, because the online extractor computes local velocity with the current root quaternion:

```text
base_vel_local_t = R(Q_ref_t)^-1 * ((P_ref_t - P_ref_{t-1}) / dt_t)
```

Since only local `vx, vy` and world `z` are stored, solve the missing local z velocity from the desired world z change:

```text
R_t = R(Q_ref_t)
dz_world_t = z_t - z_{t-1}

vz_base_t = (dz_world_t / dt_t - R_t[2,0] * vx_t - R_t[2,1] * vy_t) / max(R_t[2,2], eps)

v_world_t = R_t * [vx_t, vy_t, vz_base_t]
P_ref_t = P_ref_{t-1} + v_world_t * dt_t
P_ref_t.z = z_t
```

This makes the finite-difference round trip back to TWIST2 mimic as close as possible under the available 35D command representation.

Then build the 40D action:

```text
for t = 0:
  action[0:2] = [0, 0]
  mark first transition as invalid or warmup if needed

for t > 0:
  action[0:2] = [vx_t * dt_t, vy_t * dt_t]

action[2] = z_t
action[3:9] = rot6d(Q_ref_t in episode frame)
action[9:38] = mimic_t[6:35]
action[38:40] = hand_binary_t
```

This `Q_ref_t` is a pose target for the upper layer, but it is not a robot-current residual.

### 7.2 Why GMR Reconstruction Is Not the Main Label

Offline GMR reconstruction from `human_smplx_data` is useful and can be accurate after warmup. Existing checks showed last-half reconstruction errors around:

```text
joint mean error:       0.0007 - 0.0018 rad
z/roll/pitch mean err:  0.0005 - 0.0017
```

However, GMR reconstruction has risks as the main training label:

```text
- GMR has iterative IK state and warm-start behavior.
- Recording may start after teleop has already been running, but the old npz does not store GMR internal state.
- The first 50-100 frames can differ from the online qpos.
- The absolute yaw of GMR qpos has frame/gauge ambiguity.
- TWIST2 low-level consumed robot_action_mimic, not GMR qpos directly.
```

Therefore, for old TWIST2 data:

```text
main training label:      mimic-integrated reference trajectory
auxiliary/debug field:    offline GMR reconstructed qpos/root quat
not recommended:          inverse(Q_robot_current) * Q_gmr
```

## 8. Optional TWIST2 GMR Reconstruction Fields

If offline GMR is available, the converter may add diagnostic fields:

```text
twist2_gmr_reference_qpos              shape [T, 36]
twist2_gmr_reference_root_pos          shape [T, 3]
twist2_gmr_reference_root_quat_wxyz    shape [T, 4]
twist2_gmr_reconstruction_valid_mask   shape [T]
twist2_gmr_warmup_mask                 shape [T]
twist2_gmr_reconstruction_stats        json/dict
```

These fields are for analysis, quality control, and future protocol experiments. They should not silently replace the main V3.1 action target unless an experiment explicitly opts into that source.

If GMR fields are used, the converter must verify that reconstructed qpos can regenerate the recorded `robot_action_mimic`:

```text
qpos_t, qpos_{t-1}, dt_t
  -> extract_mimic_obs_whole_body
  -> compare with original robot_action_mimic_t
```

Recommended checks:

```text
- joint reconstruction mean error
- z/roll/pitch reconstruction error
- vx/vy/yaw_vel reconstruction p95/p99 error
- first-frame and warmup-frame mask
- dt anomaly mask
```

## 9. Runtime Adapters

V3.1 standardizes the VLA output as reference motion. Each GMT backend owns its adapter.

### 9.1 SONIC Adapter

Input:

```text
P_ref_t, Q_ref_t, J_ref_t
```

Adapter behavior:

```text
- reconstruct body_pos/body_quat or root/body reference payloads expected by SONIC
- preserve SONIC history/window requirements
- allow SONIC provider to compute internal base-relative anchor features
```

The adapter may use `Q_robot_current_t` internally because SONIC expects relative tracking features, but this must happen after VLA output, not in the supervised action label.

### 9.2 TWIST2 Adapter

Input:

```text
P_ref_t, Q_ref_t, J_ref_t
previous reference pose P_ref_{t-1}, Q_ref_{t-1}
dt_t
```

Output:

```text
mimic_t = [vx_base, vy_base, z, roll, pitch, yaw_vel, joint_pos_29]
```

For root linear velocity:

```text
v_world_t = (P_ref_t - P_ref_{t-1}) / dt_t
v_base_t = R(Q_ref_t)^-1 * v_world_t
vx_base_t, vy_base_t = v_base_t[0:2]
```

For roll/pitch:

```text
roll_t, pitch_t = rpy(Q_ref_t)[0:2]
```

For yaw velocity in the normal upright regime:

```text
yaw_vel_t = wrap(yaw(Q_ref_t) - yaw(Q_ref_{t-1})) / dt_t
```

For non-upright tasks, use quaternion finite difference and project angular velocity into the current reference base frame:

```text
omega_world_t = quat_diff(Q_ref_{t-1}, Q_ref_t) / dt_t
omega_base_t = R(Q_ref_t)^-1 * omega_world_t
yaw_vel_t = omega_base_t.z
```

Then:

```text
mimic_t = [vx_base_t, vy_base_t, z_t, roll_t, pitch_t, yaw_vel_t, J_ref_t]
```

This adapter is stateful because it needs the previous reference pose.

## 10. Metadata Contract

Every converted dataset should record the protocol metadata.

Dataset-level metadata:

```json
{
  "vla_protocol": {
    "schema": "unitree_g1_gmt_refpose_v3_1",
    "version": "3.1",
    "action_dim": 40,
    "action_layout": "root_xy_delta_z_rot6d_joints29_hands2",
    "rotation_6d_layout": "row",
    "action_semantics": "reference_pose_not_robot_current_residual",
    "root_xy_delta_frame": "current_reference_root_frame",
    "root_rotation_frame": "episode_reference_frame",
    "control_dt": 0.0333333333,
    "fps": 30.0,
    "dt_source": "system_control_frequency_or_timestamp",
    "backend_source": "sonic|twist2",
    "twist2_reference_source": "mimic_integrated",
    "sonic_reference_source": "body_pos_body_quat"
  }
}
```

Episode-level metadata:

```json
{
  "reference_anchor": {
    "position_xyz": [0.0, 0.0, 0.0],
    "heading_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "convention": "first_reference_heading_or_identity_for_mimic_integrated_twist2"
  },
  "quality": {
    "valid_mask_field": "vla_action_valid_mask",
    "dt_anomaly_mask_field": "dt_anomaly_mask",
    "twist2_cmd_spike_mask_field": "twist2_cmd_spike_mask",
    "gmr_warmup_mask_field": "twist2_gmr_warmup_mask"
  }
}
```

## 11. Quality Gates

V3.1 must include checks that prevent the old v3 error from returning.

### 11.1 No Robot-Current Residual Gate

The converter must not construct action rotation as:

```text
inverse(Q_robot_current_t) * Q_ref_t
```

Allowed uses of `Q_robot_current_t`:

```text
- observation/state construction
- diagnostics
- backend runtime adapter internal features
```

Forbidden uses:

```text
- supervised VLA action label construction
- stored action[3:9]
```

### 11.2 TWIST2 Round-Trip Gate

For TWIST2, the converted V3.1 reference trajectory should round-trip back to the native command:

```text
V3.1 reference pose trajectory
  -> TWIST2 adapter finite difference
  -> reconstructed robot_action_mimic
```

Compare reconstructed command against original `robot_action_mimic`:

```text
vx/vy error
z error
roll/pitch error
yaw_vel error
joint_pos_29 error
```

Frames with dt anomalies, command spikes, or explicit recording transitions may be masked, but the mask must be stored.

### 11.3 Raw Command Spike Gate

TWIST2 raw commands can contain finite-difference spikes. The converter must report at least:

```text
max_abs_yaw_vel
p99_abs_yaw_vel
max_xy_speed
p99_xy_speed
num_dt_anomalies
num_spike_masked_frames
```

Do not silently integrate severe spikes into `Q_ref_t`. The experiment must choose one of:

```text
- keep with warning
- mask frame
- clip with explicit metadata
- fail conversion
```

### 11.4 SONIC Regression Gate

For SONIC, verify that V3.1 reference reconstruction preserves the original SONIC reference behavior:

```text
- body_pos/body_quat adapter output close to original reference
- motion history/window still valid
- eval success/fall rate not regressed on a small validation set
```

### 11.5 Cross-Backend Statistics

Cross-backend statistics should be computed on reference-motion quantities, not robot residuals:

```text
- step yaw delta of Q_ref_t
- step xy displacement of P_ref_t
- root_z, roll, pitch distribution
- joint target distribution
- hand target distribution
```

Do not use `inverse(Q_robot_current) * Q_ref` as the primary cross-backend action statistic.

## 12. Main Risks

### Risk 1: Calling a residual a target

The biggest risk is accidentally reintroducing the v3 label:

```text
rot6d(inverse(Q_robot_current_t) * Q_ref_t)
```

This must be treated as a protocol violation.

### Risk 2: TWIST2 integrated pose has gauge freedom

TWIST2 mimic-integrated `Q_ref_t` has arbitrary initial yaw. This is acceptable because the protocol stores reference motion in an episode frame. Do not interpret the absolute yaw as world truth.

### Risk 3: Raw TWIST2 command spikes

Mimic integration is command-consistent, but raw finite-difference spikes can still pollute the reference trajectory. Conversion must include spike reporting and an explicit policy.

### Risk 4: SONIC and TWIST2 references are not identical sources

SONIC reference comes from pose/body target fields. TWIST2 main reference comes from the command actually consumed by the backend. This is intentional: V3.1 unifies the VLA-level semantic as reference motion, not the data provenance.

### Risk 5: Runtime dt mismatch

TWIST2 adapter must use the same `dt` convention used during conversion. Store `control_dt`, `fps`, and `dt_source`.

### Risk 6: GMR reconstruction over-trust

Offline GMR qpos is valuable for debug, but old recordings do not store online IK state. Do not use GMR reconstruction as the default main label without explicit opt-in and validation.

## 13. Recommended Implementation Plan

1. Add a V3.1 converter mode:

```text
--protocol unitree_g1_gmt_refpose_v3_1
```

2. For SONIC:

```text
body_pos/body_quat -> anchored P_ref/Q_ref -> V3.1 40D action
```

3. For TWIST2:

```text
robot_action_mimic -> mimic-integrated P_ref/Q_ref -> V3.1 40D action
```

4. Add optional GMR reconstruction output:

```text
--twist2-write-gmr-debug-fields
```

5. Add verification scripts:

```text
- verify no robot-current residual in labels
- verify TWIST2 pose-to-mimic round trip
- verify SONIC reference adapter reconstruction
- report command spikes and dt anomalies
```

6. Keep V4 separate:

```text
V3.1 = reference pose protocol, 40D, fixes residual semantics.
V4   = command protocol, backend-agnostic command interface.
```

## 14. One-Line Summary

V3.1 defines `Q_target` as the root quaternion of an upper-layer reference trajectory in an episode reference frame. It is not the quaternion residual from the current robot to that target. For old TWIST2 data, the main target should be built from the recorded native `robot_action_mimic` so that the pose target round-trips to the command TWIST2 actually consumed; offline GMR reconstruction is useful as a debug/reference field, not the default training label.
