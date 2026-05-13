# VLA V3 Robot-Current-Local Rotation Migration Plan

## 1. Goal

V3 defines a new canonical VLA data and runtime protocol for mixed SONIC/TWIST2 training.

The core change is:

```text
action.root_rot6d_v3 = rot6d(inverse(Q_robot_world_t) * Q_target_world_t)
```

This is a **robot-current-local target rotation**. It is not a world/global root orientation and it is not a frame-to-frame target delta. It is the relative orientation from the current Isaac robot root frame to the target root/body frame for the same control step.

The existing v2 protocol already made the xy translation target local, but it did not unify the root 6D rotation reference frame. V3 keeps the local translation idea and fixes the root rotation semantics.

## 2. Requirements

V3 must satisfy four constraints:

1. Make the action target easier for VLA training.
2. Support mixing tasks whose robots start with different world headings.
3. Convert from already recorded SONIC/TWIST2 data without requiring recollection.
4. Allow the same VLA model output to switch between SONIC and TWIST2 GMT backends through runtime adapters.

## 3. Current V2 Problem

V2 action layout is:

```text
action[0:2]   root_local_xy_delta
action[2]     root_z
action[3:9]   root_rot6d
action[9:38]  canonical G1 joint positions
action[38:40] hand binary
```

The problem is only partially visible in the field name `root_rot6d`. The same field currently carries different reference-frame semantics depending on backend.

### SONIC V2

The v2 SONIC converter uses:

```python
body_quat = data["human_body_quat_w"]
root_rot6d = quat_to_rot6d_wxyz(body_quat[i])
```

This exactly matches raw `human_body_quat_w`, not `human_body_quat_w_aligned`.

Observed check on a current recording:

```text
max |v2_action6d - rot6d(human_body_quat_w)|         = 0.0
max |v2_action6d - rot6d(human_body_quat_w_aligned)| = 1.9239865
max |v2_action6d - original vla_action[3:9]|         = 1.9239869
max |v2_action6d - original vla_action_raw[3:9]|     = 0.0
```

Therefore SONIC v2 action rotation is an unaligned external reference/body root orientation.

### TWIST2 V2

The v2 TWIST2 converter uses:

```python
yaw_world = 0.0
yaw_world += yaw_vel * control_dt
root_quat = quat_from_roll_pitch_yaw_wxyz(roll, pitch, yaw_world)
root_rot6d = quat_to_rot6d_wxyz(root_quat)
```

Observed check on a current recording:

```text
max |v2_action6d - manual yaw0 integration|      = 0.0
median |v2_action6d - robot_root_orientation6d| = 1.1998307
max |v2_action6d - robot_root_orientation6d|    = 1.9457471
```

Therefore TWIST2 v2 action rotation is an episode-yaw-zero target orientation, not Isaac world root orientation.

### Consequence

In v2:

```text
SONIC action.root_rot6d  ~= external source/root orientation
TWIST2 action.root_rot6d ~= yaw-integrated-from-zero target orientation
state.root_rot6d         = Isaac robot world root orientation
```

This is not a single physical quantity. Mixed-backend training can learn backend-specific heading shortcuts, and runtime switching can rotate `root_local_xy_delta` through the wrong target heading.

## 4. V3 Coordinate Frames

Use these names consistently in converter and runtime code:

```text
W_I    Isaac world frame
W_S    SONIC/source reference frame
B_t    current Isaac robot root frame at control step t
S_t    SONIC/source body root frame at step t
T_t    canonical target root frame in Isaac world at step t
A      episode source-heading-to-Isaac-heading alignment rotation
```

Quaternion notation:

```text
Q_robot_t        = orientation of B_t in W_I
Q_source_t       = orientation of S_t in W_S
Q_target_t       = orientation of T_t in W_I
Q_action_rel_t   = inverse(Q_robot_t) * Q_target_t
```

V3 action stores `Q_action_rel_t` as 6D.

## 5. V3 Data Format

Use a new schema version:

```text
robot_current_local_rot_isaac_time_v3
```

Keep dimensions unchanged to minimize model and LeRobot plumbing:

```text
observation.state: float32[64]
action:            float32[40]
fps:               Isaac control fps, normally 50
```

### 5.1 Observation State Names

V3 should rename the first six state dimensions to make the heading normalization explicit:

```text
state.root_heading_canonical_rot6d.0
state.root_heading_canonical_rot6d.1
state.root_heading_canonical_rot6d.2
state.root_heading_canonical_rot6d.3
state.root_heading_canonical_rot6d.4
state.root_heading_canonical_rot6d.5
state.dof_pos.<29 canonical joints>
state.dof_vel.<29 canonical joints>
```

Recommended state root formula:

```text
Q_state_root_t = inverse(heading(Q_robot_0)) * Q_robot_t
```

This removes the arbitrary episode initial yaw while preserving current robot roll, pitch, and yaw progress relative to the episode heading.

Rationale:

- VLA should not have to memorize that the same task was initialized at world yaw 0, pi/2, or -pi/2.
- The state still tells the model whether the robot has rotated relative to its episode start.
- This can be computed from already recorded `robot_root_orientation`.

### 5.2 Action Names

V3 action keeps the 40D layout but changes the root rotation name and semantics:

```text
action[0]     action.root_target_heading_local_xy_delta.x
action[1]     action.root_target_heading_local_xy_delta.y
action[2]     action.root_z
action[3:9]   action.root_current_local_target_rot6d.0..5
action[9:38]  action.joint_pos.<29 canonical joints>
action[38]    action.hand_binary.left
action[39]    action.hand_binary.right
```

The xy delta remains a 2D planar local translation delta. Define it as target-heading-local, not full target-root-local:

```text
D_world_t = target_xy_world_t - target_xy_world_{t-1}
H_target_t = heading(Q_target_t)
D_local_t = inverse(H_target_t) * [D_world_t.x, D_world_t.y, 0]
action.root_target_heading_local_xy_delta = D_local_t.xy
```

Do not rotate this xy delta again with episode heading alignment during runtime.

Rationale: `action.xy` stores only 2D planar displacement. Using the full `Q_target_t`
would create and then discard a local z component whenever the target root has roll or
pitch, making encode/decode non-invertible on the world xy plane.

The root rotation field is:

```text
action.root_current_local_target_rot6d = rot6d(inverse(Q_robot_t) * Q_target_t)
```

This field means "from the robot's current root frame, what target root orientation should be used for this control step?"

## 6. Offline Conversion

Create independent v3 files. Do not modify v1 or v2 converters in place.

Recommended new files:

```text
isaaclab_twist2_g1/action_provider/vla_robot_current_local_runtime_v3.py
isaaclab_twist2_g1/tools/data_tools/smpl_lerobot_v3_common.py
isaaclab_twist2_g1/tools/data_tools/sonic2lerobot_rotlocal_v3.py
isaaclab_twist2_g1/tools/data_tools/twist2lerobot_rotlocal_v3.py
isaaclab_twist2_g1/tools/data_tools/verify_lerobot_rotlocal_v3.py
isaaclab_twist2_g1/tools/data_tools/batch_convert_rotlocal_v3.sh
isaaclab_twist2_g1/tools/data_tools/visualize_rotlocal_v3_mujoco.py
```

### 6.1 Shared Converter Helpers

Add helper functions in `smpl_lerobot_v3_common.py`:

```text
quat_heading_wxyz(q) -> yaw-only quaternion
align_source_heading_to_robot(Q_robot_0, Q_source_0) -> A
heading_canonical_robot_quat(Q_robot_0, Q_robot_t) -> Q_state_root_t
target_heading_local_xy_delta(Q_target_t, D_world_t) -> xy
robot_current_local_target_quat(Q_robot_t, Q_target_t) -> Q_action_rel_t
build_vla_rotlocal_v3_action(...)
build_vla_rotlocal_v3_state(...)
```

Use structured quaternion utilities, not ad hoc Euler/string manipulation.

### 6.2 SONIC V3 Conversion

Required source fields from current recordings:

```text
robot_qpos_before_decimation
robot_qvel_before_decimation
robot_root_orientation
human_body_pos
human_body_quat_w
human_joint_pos or vla_action_joint_pos_29 or vla_action_raw[:, 9:38]
pico grip or vla hand fields
episode_step
vision_rgb / vision_rgb_video_path
vision_frame_indices
```

The current data has enough information for v3 if these fields are present.

For each episode:

```text
Q_robot_0 = robot_root_orientation[0]
Q_source_0 = human_body_quat_w[0]
A = heading(Q_robot_0) * inverse(heading(Q_source_0))
```

For each Isaac control row `t`:

```text
Q_robot_t = robot_root_orientation[t]
Q_source_t = human_body_quat_w[t]
Q_target_t = A * Q_source_t

P_source_t = human_body_pos[t]
D_source_world_t = P_source_t.xy - P_source_{t-1}.xy
D_target_world_t = A * [D_source_world_t.x, D_source_world_t.y, 0]

action.xy = inverse(heading(Q_target_t)) * D_target_world_t
action.z = P_source_t.z
action.rot6d = rot6d(inverse(Q_robot_t) * Q_target_t)
action.joint_pos = human_joint_pos[t] or canonical joint target
action.hand_binary = hand target at t
```

For `t == 0`, set `action.xy = [0, 0]`.

Observation state:

```text
state.root = rot6d(inverse(heading(Q_robot_0)) * Q_robot_t)
state.dof_pos = robot_qpos_before_decimation[t]
state.dof_vel = robot_qvel_before_decimation[t]
```

Important:

- Do not use raw `human_body_quat_w` directly as action 6D.
- Do not use `human_body_quat_w_aligned` blindly unless it is verified to equal `A * human_body_quat_w` for that episode.
- Do not use v2 `vla_action` as the v3 target; current v2 `vla_action` is a different historical canonical field.

### 6.3 TWIST2 V3 Conversion

Required source fields from current recordings:

```text
robot_qpos_before_decimation
robot_qvel_before_decimation
robot_root_orientation
robot_action_mimic or robot_obs_buf action_mimic slice
system_control_frequency or fps override
gripper fields
episode_step
vision_rgb / vision_rgb_video_path
vision_frame_indices
```

For each episode:

```text
Q_robot_0 = robot_root_orientation[0]
yaw_target_0 = yaw(Q_robot_0)
yaw_target = yaw_target_0
```

For each row `t`:

```text
xy_vel_local = action_mimic[t, 0:2]
root_z = action_mimic[t, 2]
roll = action_mimic[t, 3]
pitch = action_mimic[t, 4]
yaw_vel = action_mimic[t, 5]

yaw_target += yaw_vel * control_dt
Q_target_t = quat_from_roll_pitch_yaw(roll, pitch, yaw_target)
Q_robot_t = robot_root_orientation[t]

action.xy = target-heading-local planar delta derived from xy_vel_local * control_dt
action.z = root_z
action.rot6d = rot6d(inverse(Q_robot_t) * Q_target_t)
action.joint_pos = reorder_twist2_to_canonical_29(action_mimic[t, 6:35])
action.hand_binary = hand target at t
```

Observation state:

```text
state.root = rot6d(inverse(heading(Q_robot_0)) * Q_robot_t)
state.dof_pos = reorder_twist2_to_canonical_29(robot_qpos_before_decimation[t])
state.dof_vel = reorder_twist2_to_canonical_29(robot_qvel_before_decimation[t])
```

This fixes the v2 issue where TWIST2 action yaw always started from zero even when Isaac robot initial heading was not zero.

## 7. Runtime Control Chain

Create `UnifiedRobotCurrentLocalActionRuntimeV3` and make it the single active VLA runtime control chain.

The data conversion path is side-by-side with v1/v2, but the runtime path should stay clean: after v3 migration, active VLA inference should call the v3 runtime directly. Do not add broad v2/v3 branching through the provider hot path. Keep v2 runtime code only for historical tests, old datasets, and explicit offline verification.

The runtime step must receive the current robot root state from Isaac:

```python
runtime.step(
    action,
    current_robot_quat_wxyz=robot.root_state_w[0, 3:7],
    current_robot_xy_world=robot.root_state_w[0, 0:2],
)
```

Do not decode v3 action only from internal previous state. The v3 rotation is defined relative to the current robot root.

### 7.1 Runtime Decode

For each VLA action:

```text
Q_action_rel_t = quat_from_rot6d(action[3:9])
Q_robot_t = current Isaac robot root orientation
Q_target_t = Q_robot_t * Q_action_rel_t

D_local_t = [action[0], action[1], 0]
D_world_t = Q_target_t * D_local_t

body_xy_world_t = previous_body_xy_world + D_world_t.xy
body_z_t = action[2]
```

Reset behavior:

```text
previous_body_xy_world = current Isaac robot root xy at episode reset
previous_target_quat_world = current Isaac robot root orientation at episode reset
```

The previous v2 default of starting accumulated xy from `[0, 0]` is not robust for scenes with nonzero initial robot positions.

### 7.2 SONIC Adapter

V3 runtime output to SONIC:

```text
body_pos = [body_xy_world_t.x, body_xy_world_t.y, action.root_z]
body_quat_w = Q_target_t
joint_pos = action.joint_pos
joint_vel = finite_difference(action.joint_pos, previous_action.joint_pos) / control_dt
```

When the source is VLA v3, `_apply_pose_data(..., "lerobot_vla_joint29")` must not apply an additional episode heading alignment to `body_quat_w`.

Required behavior:

```text
source == "lerobot_vla_joint29_v3" -> anchor heading align disabled / identity
```

Rationale:

`Q_target_t` is already in Isaac world, reconstructed from current robot root and local target rotation. Applying the old SONIC heading align again would double-transform the target.

### 7.3 TWIST2 Adapter

V3 runtime output to TWIST2 mimic obs:

```text
xy_vel_heading_local = action.root_target_heading_local_xy_delta / control_dt
roll_pitch = roll_pitch(Q_target_t)
yaw_vel = yaw_delta(previous_Q_target_t, Q_target_t) / control_dt
joint_pos_twist2 = reorder_canonical_to_twist2_29(action.joint_pos)
```

Return:

```text
[xy_vel_local(2), root_z(1), roll_pitch(2), yaw_vel(1), joint_pos_twist2(29)]
```

This keeps the TWIST2 backend interface unchanged while giving VLA one backend-neutral action schema.

## 8. Feature and Name Changes

Do not reuse v2 robot type names for v3.

Recommended LeRobot metadata:

```text
robot_type = unitree_g1_rotlocal_v3
repo_id suffix = *_rotlocal_v3
schema_version = robot_current_local_rot_isaac_time_v3
```

Feature names should be explicit:

```text
state.root_heading_canonical_rot6d.<0..5>
action.root_target_heading_local_xy_delta.x
action.root_target_heading_local_xy_delta.y
action.root_z
action.root_current_local_target_rot6d.<0..5>
action.joint_pos.<joint_name>
action.hand_binary.left
action.hand_binary.right
```

Avoid the ambiguous v2 name `action.root_rot6d`.

## 9. Tests and Verification

Add unit tests before bulk conversion.

### 9.1 Geometry Tests

Test identity case:

```text
Q_robot = yaw 90 deg
Q_target = yaw 90 deg
Q_rel = inverse(Q_robot) * Q_target = identity
```

Test relative yaw:

```text
Q_robot = yaw 90 deg
Q_target = yaw 120 deg
Q_rel = yaw 30 deg
```

Test runtime reconstruction:

```text
Q_target_reconstructed = Q_robot * Q_rel
```

Test xy delta:

```text
D_local -> Q_target -> D_world
```

### 9.2 Converter Tests

SONIC:

- Synthetic source frame with nonzero initial source yaw and nonzero robot initial yaw.
- Verify `action.rot6d == rot6d(inverse(Q_robot_t) * A * Q_source_t)`.
- Verify `action.xy` is unchanged by pure scene initial heading changes.

TWIST2:

- Synthetic `robot_root_orientation[0] = yaw pi/2`.
- Synthetic `yaw_vel = 0`.
- Verify v3 action rotation is identity relative to robot when target follows robot heading.
- Verify v2 behavior would not pass this test.

### 9.3 Dataset Validation

For each converted dataset:

- `info.total_frames == sum(meta/episodes length)`.
- video metadata frame counts match `total_frames`.
- `observation.images.front` exists for every row.
- no episode length is abnormally long.
- action/state dims are exactly 40/64.
- no NaN/inf in state/action.
- hand fields remain binary in raw action values.
- root relative rotation angular magnitude distribution is bounded and interpretable.

Recommended reports:

```text
per dataset:
  action.root_current_local_target_rot angle mean/p50/p95/p99/max
  action.root_target_heading_local_xy_delta norm mean/p95/p99/max
  state.root_heading_canonical yaw p50/p95/max
  hand active ratios
  sonic/twist2 frame counts
```

### 9.4 Lightweight Visualization

Update the Mujoco visualizer for v3:

```text
left:  Isaac first-person video
mid:   Mujoco replay reconstructed from v3 runtime
right: 40D action strip plot
```

The visualizer must reconstruct:

```text
Q_target_t = Q_robot_t * Q_action_rel_t
D_world_t = Q_target_t * D_local_t
```

It should not use the v2 runtime.

## 10. Training and Inference Risks

### Risk: model trained on v3 but run through v2 runtime

This will be wrong. V2 expects `action[3:9]` to be a full target root orientation. V3 stores a relative target orientation.

Mitigation:

- Use distinct `robot_type`.
- Use distinct schema string.
- Active VLA runtime should be v3-only after migration.
- Reject non-v3 policy metadata at load time instead of adding a v2/v3 runtime switch.

### Risk: SONIC old heading align double-applies

V3 reconstructs `Q_target_t` in Isaac world. Old SONIC heading align should not be applied to VLA v3 payloads.

Mitigation:

- Use a new source tag such as `lerobot_vla_joint29_v3`.
- Force anchor heading align identity for that source.
- Add a debug assertion that first-step `anchor_rot6d` equals VLA relative target rotation within tolerance.

### Risk: xy local frame ambiguity

V3 should explicitly name xy as `root_target_heading_local_xy_delta`, not generic `root_local_xy_delta`.

Mitigation:

- Converter and runtime tests must check whether xy is rotated by `heading(Q_target_t)`, not full `Q_target_t` or `Q_robot_t`.
- If future experiments prefer robot-current-local xy, that should be a v3.1 schema, not a silent change.

### Risk: current robot orientation noise

Because action rotation is relative to current robot root, noisy robot orientation can change the reconstructed world target.

Mitigation:

- Use Isaac root orientation before decimation consistently for training.
- Use the same root state timing in runtime before applying the VLA action.
- Optionally clamp relative rotation angle per step for online safety, but do not clamp during dataset conversion.

## 11. Implementation Checklist

1. Add v3 runtime constants and names.
2. Add quaternion/frame helper functions in a new v3 common module.
3. Add `sonic2lerobot_rotlocal_v3.py`.
4. Add `twist2lerobot_rotlocal_v3.py`.
5. Add v3 batch conversion script.
6. Add v3 verifier.
7. Add v3 Mujoco visualizer.
8. Add unit tests for quaternion frame math and converter formulas.
9. Integrate v3 runtime into `action_provider_sonic.py`.
10. Integrate v3 runtime into `action_provider_wh_twist2.py`.
11. Remove active v2 runtime usage from VLA inference provider paths; keep v2 tests/utilities only as historical/offline tools.
12. Update eval scripts and defaults to use `unitree_g1_rotlocal_v3` for active VLA runs.
13. Run batch conversion on current recording data into a new output root.
14. Run dataset stats and visual QA before training.

## 12. Acceptance Criteria

V3 is complete only when all of these pass:

1. Unit tests prove `Q_target = Q_robot * Q_action_rel`.
2. SONIC converter test proves action rotation uses heading-aligned source target relative to current robot root.
3. TWIST2 converter test proves action rotation is no longer yaw-integrated from zero in dataset coordinates.
4. Runtime tests prove the same 40D v3 action can produce:
   - SONIC `body_pos/body_quat_w/joint_pos/joint_vel`
   - TWIST2 mimic obs
5. Converted LeRobot datasets have correct 64D state and 40D action names.
6. Converted datasets pass frame/video/image count validation.
7. Mujoco v3 visualization produces mp4 previews for random samples without Isaac.
8. Online VLA v3 path disables old SONIC heading align for VLA v3 payloads.
9. Active VLA inference provider paths contain no broad v2/v3 runtime selection logic; they call v3 directly and reject incompatible policy metadata.

## 13. Summary

V3 should not try to make every field global. It should make the VLA action local to the robot state that the policy observes.

The final intended semantics are:

```text
state.root = robot root orientation in episode-heading-canonical frame
action.xy = target-heading-local planar xy delta
action.rot6d = current-robot-local target rotation
action.joint_pos = canonical 29D target joints
action.hand_binary = semantic open/close targets
```

This gives the model a stable training target across different scene headings and lets runtime adapters reconstruct backend-specific control signals for SONIC or TWIST2.
