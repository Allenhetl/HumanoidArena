# Canonical G1 Action For SONIC And TWIST2

## 1. Goal

This document defines a single robot-level canonical G1 action that can be converted to:

- SONIC `joint29` live input
- TWIST2 `mimic_obs35`

The design target is:

- no dependency on SMPL body shape
- no lossy downstream conversion for the body path
- direct compatibility with current code paths
- easier VLA learning than absolute root-position targets

One practical motivation is:

- `TWIST2` uses GMR with human joints as IK targets
- different operators have different SMPL body shapes
- forcing all data to `SMPL neutral` introduces visible retarget error
- measured error can reach about `0.5 rad` on some joints

So the canonical action should live in robot space, not SMPL space.

## 2. Final Decision

### 2.1 Canonical body action

The canonical body action should be:

- `root_xy_delta`: `float32[2]`
- `root_z`: `float32[1]`
- `root_rot6d`: `float32[6]`
- `joint_pos_29`: `float32[29]`, in canonical IsaacLab order

Total body dimension:

- `38D`

### 2.2 Canonical full action

For hands, use a semantic binary open-close signal:

- `hand_binary`: `float32[2]`

where:

- `0.0` = open
- `1.0` = close

Then the full canonical action is:

- `root_xy_delta`: `2`
- `root_z`: `1`
- `root_rot6d`: `6`
- `joint_pos_29`: `29`
- `hand_binary`: `2`

Total full dimension:

- `40D`

### 2.3 Why this parametrization

- `root_rot6d` is a better VLA regression target than quaternion
- `root_xy_delta` is easier for VLA to learn than absolute `root_xy`
- `root_z` should stay explicit because both backends consume height directly or through `body_pos.z`
- `hand_binary_2` matches the current semantic open-close execution path used by VLA runtime

### 2.4 Hand tradeoff

Current teleop servers internally keep a continuous hand-open ratio.

However, for the unified VLA control signal we intentionally choose:

- `hand_binary_2`

Reason:

- both current VLA runtime paths ultimately map hand intent to open/close target poses
- binary open-close is easier for VLA to learn and debug
- current unified recordings already expose clean semantic binary hand state

This is intentionally coarser than `hand_open_ratio_2`.

If future tasks need graded grasp closure, `hand_open_ratio_2` can be added later as an optional auxiliary field, but it is not part of the current canonical action.

## 3. Field Semantics

### 3.1 `joint_pos_29`

Use absolute joint position, not delta-from-default.

Reason:

- SONIC joint29 input expects joint positions directly
- TWIST2 `mimic_obs35` also uses absolute 29 joint targets
- delta-from-default would add one more backend-specific assumption

### 3.2 `root_xy_delta`

Use per-step planar root displacement in an episode-local world frame:

- `z-up`
- metric units
- `x/y` origin placed at the robot root projection on the ground at episode reset
- stored as `[delta_x_world, delta_y_world]`

Reason:

- `TWIST2` only needs planar velocity, which is directly derived from delta and `dt`
- `SONIC` can reconstruct `body_pos.xy` by accumulating delta in adapter state
- delta is a more stable and more actionable learning target than absolute `root_xy`

Do not use:

- absolute `root_pos_xy` as canonical action
- `root_xy_vel` as canonical action

`root_xy_vel` is derived from delta and `dt`.

Absolute `root_pos_xy` is reconstructable in the adapter by accumulation.

### 3.3 `root_z`

Use absolute root height in the same episode-local world frame.

Reason:

- `TWIST2` directly consumes `z_pos`
- `SONIC` currently derives `root_z` from `body_pos[2]`
- height should remain explicit instead of being hidden inside a reconstructed `body_pos`

### 3.4 `root_rot6d`

Use world-frame root orientation in continuous 6D rotation representation.

More precisely:

- use episode-local global orientation
- `z-up`
- heading zero is defined at episode reset
- do not use base-relative local orientation
- do not use frame-to-frame rotational delta as the canonical field

Reason:

- it is easier for VLA to learn than quaternion
- SONIC and TWIST2 downstream logic can recover quaternion from `root_rot6d`
- root orientation cannot be recovered from `joint_pos_29`
- `roll_pitch`, `yaw_vel`, `body_quat_w`, and `anchor_rot6d` all depend on root orientation
- the same semantic motion should not change target definition just because the robot base is currently tilted or rotated

Why not use local rotation:

- if `root_rot6d` is defined relative to the current robot base, the label depends on downstream robot state
- that makes the canonical action backend-dependent
- it also makes VLA learning harder because identical high-level intent can map to different labels

So:

- canonical action stores global root orientation
- SONIC adapter derives local `anchor_rot6d` internally from current base quaternion and canonical root orientation

### 3.5 `hand_binary`

Use semantic manipulation-state hand signal:

- `left = 1.0` means close left hand
- `left = 0.0` means open left hand
- `right = 1.0` means close right hand
- `right = 0.0` means open right hand

Important:

- this is not measured robot hand joint state
- this is not the continuous latent hand-open ratio
- this is the canonical semantic hand command used by the VLA control signal

## 4. Canonical 29-Joint Order

The canonical order is SONIC IsaacLab order:

```text
0  left_hip_pitch_joint
1  right_hip_pitch_joint
2  waist_yaw_joint
3  left_hip_roll_joint
4  right_hip_roll_joint
5  waist_roll_joint
6  left_hip_yaw_joint
7  right_hip_yaw_joint
8  waist_pitch_joint
9  left_knee_joint
10 right_knee_joint
11 left_shoulder_pitch_joint
12 right_shoulder_pitch_joint
13 left_ankle_pitch_joint
14 right_ankle_pitch_joint
15 left_shoulder_roll_joint
16 right_shoulder_roll_joint
17 left_ankle_roll_joint
18 right_ankle_roll_joint
19 left_shoulder_yaw_joint
20 right_shoulder_yaw_joint
21 left_elbow_joint
22 right_elbow_joint
23 left_wrist_roll_joint
24 right_wrist_roll_joint
25 left_wrist_pitch_joint
26 right_wrist_pitch_joint
27 left_wrist_yaw_joint
28 right_wrist_yaw_joint
```

This is already SONIC native order.

## 5. Why This Action Is Sufficient

### 5.1 Enough for SONIC

Current SONIC joint29 path can consume:

- `joint_pos`
- `joint_vel`
- `body_pos`
- `body_quat_w`

from the Redis bridge.

Every one of these is derivable from canonical action:

- `joint_vel` is finite-differenced from `joint_pos_29`
- `body_quat_w` is recovered from `root_rot6d`
- `body_pos.xy` is reconstructed by accumulating `root_xy_delta`
- `body_pos.z` is `root_z`

Current SONIC joint29 encoder still mainly relies on:

- `joint_pos`
- `joint_vel`
- `anchor_rot6d`

So this canonical action covers the active body-conditioning path cleanly.

### 5.2 Enough for TWIST2

Current TWIST2 teleop body action is:

- `xy_vel`
- `root_z`
- `roll_pitch`
- `yaw_vel`
- `joint_pos_29`

Every one of these can be derived from:

- `root_xy_delta`
- `root_z`
- `root_rot6d`
- `joint_pos_29`

So storing the already-derived TWIST2 action as canonical would be unnecessary and less general.

### 5.3 Why `joint_pos_29 + root_xy_delta + root_z` is not enough

It is missing `root_rot6d`.

Without root orientation, you cannot recover:

- SONIC `body_quat_w`
- SONIC `anchor_rot6d`
- TWIST2 `roll_pitch`
- TWIST2 `yaw_vel`
- TWIST2 local-frame `xy_vel`

Therefore a valid canonical action must contain root orientation, not just translation and joints.

### 5.4 Why absolute `root_pos_xyz` is not chosen

Absolute root position is valid as a logging or exchange format, but it is not the best VLA control target.

Reason:

- it bakes episode origin into the target
- it accumulates long-horizon trajectory state into every frame label
- the same local maneuver can correspond to many different absolute `x/y` labels
- `root_xy_delta` is closer to actionable control and still reconstructs backend inputs cleanly

## 6. Conversion To SONIC

### 6.1 SONIC target interface

Current SONIC joint29 bridge expects:

- `joint_pos`: `float32[29]`, canonical order
- `joint_vel`: `float32[29]`, canonical order
- `body_pos`: `float32[3]`
- `body_quat_w`: `float32[4]`
- left/right hand joint targets: `float32[7] + float32[7]`

The provider then computes:

- `anchor_rot6d` from current robot base quaternion and incoming `body_quat_w`
- `root_z` from `body_pos[2]`

Important:

- `anchor_rot6d` is not part of canonical action
- it is a SONIC-specific derived quantity
- it must be computed inside the SONIC adapter from current base state and canonical global root orientation

### 6.2 Adapter state

To convert canonical action to SONIC without losing body information, the adapter must maintain:

- accumulated `body_xy_world`
- previous `joint_pos_29`
- previous smoothing state if exact parity is needed

`body_xy_world` must be reset to zero on episode reset.

### 6.3 Conversion formula

Given canonical actions at time `t` and `t-1` with interval `dt`:

```python
joint_pos_sonic = joint_pos_29_canonical_t
body_quat_w_sonic = rot6d_to_quat_wxyz(root_rot6d_t)

if t == 0:
    joint_vel_sonic = zeros(29)
    body_xy_world_t = zeros(2)
else:
    joint_vel_sonic = (joint_pos_29_canonical_t - joint_pos_29_canonical_prev) / dt
    body_xy_world_t = body_xy_world_prev + root_xy_delta_t

body_pos_sonic = np.array(
    [body_xy_world_t[0], body_xy_world_t[1], root_z_t],
    dtype=np.float32,
)
```

Hand conversion:

```python
left_hand_joints  = left_close_pose  if left_binary_t  >= 0.5 else left_open_pose
right_hand_joints = right_close_pose if right_binary_t >= 0.5 else right_open_pose
```

### 6.4 Exact-compatibility note

If the goal is to match the current `twist2_teleop_server.py -> SONIC` bridge more closely, optional adapter details can be kept:

- apply EMA on `joint_vel`
- clip `joint_vel` to the same range as the current bridge
- optionally smooth `body_quat_w` after converting from `root_rot6d`

These are adapter details, not canonical-action fields.

### 6.5 Optional `full_qpos`

For replay and logging compatibility, an adapter may also publish:

```python
joint_pos_29_twist2_order_t = joint_pos_29_canonical_t[CANONICAL_TO_TWIST2_INDEX_29]

full_qpos = concat(
    body_pos_sonic,                  # 3
    body_quat_w_sonic,               # 4
    joint_pos_29_twist2_order_t,     # 29
)
```

`full_qpos` is optional for canonical action definition, but useful for compatibility with existing SONIC recording and replay tools.

## 7. Conversion To TWIST2

### 7.1 TWIST2 target interface

Current TWIST2 teleop body action is:

```text
[xy_vel(2), z_pos(1), roll_pitch(2), yaw_vel(1), joint_pos_29(29)]
```

Total:

- `35D`

The 29 joint dimensions are in TWIST2 action order, not canonical order.

### 7.2 Canonical-to-TWIST2 reorder

Use:

```python
CANONICAL_TO_TWIST2_INDEX_29 = [
    0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8,
    11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
]
```

Then:

```python
joint_pos_29_twist2 = joint_pos_29_canonical[CANONICAL_TO_TWIST2_INDEX_29]
```

For the reverse direction:

```python
TWIST2_TO_CANONICAL_INDEX_29 = [
    0, 6, 12, 1, 7, 13,
    2, 8, 14, 3, 9, 15,
    22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
```

### 7.3 Conversion formula

Given canonical action at time `t` and previous root orientation at `t-1`:

```python
root_quat = rot6d_to_quat_wxyz(root_rot6d_t)
joint_pos_twist2 = joint_pos_29_canonical_t[CANONICAL_TO_TWIST2_INDEX_29]
```

World linear velocity:

```python
root_lin_vel_world = np.array(
    [root_xy_delta_t[0] / dt, root_xy_delta_t[1] / dt, 0.0],
    dtype=np.float32,
)
```

Convert to root-local planar velocity:

```python
xy_vel = quat_rotate_inverse(root_quat, root_lin_vel_world)[:2]
```

Height:

```python
z_pos = np.array([root_z_t], dtype=np.float32)
```

Roll and pitch:

```python
roll_pitch = quat_to_roll_pitch(root_quat)
```

Yaw velocity:

```python
if t == 0:
    yaw_vel = np.zeros(1, dtype=np.float32)
else:
    yaw_prev = yaw_from_rot6d(root_rot6d_prev)
    yaw_curr = yaw_from_rot6d(root_rot6d_t)
    yaw_vel = np.array([wrap_to_pi(yaw_curr - yaw_prev) / dt], dtype=np.float32)
```

Final TWIST2 body action:

```python
mimic_obs35 = concat(
    xy_vel,              # 2
    z_pos,               # 1
    roll_pitch,          # 2
    yaw_vel,             # 1
    joint_pos_twist2,    # 29
)
```

This exactly matches the structure already used in current TWIST2 code.

### 7.4 Hand conversion

Use semantic binary open-close mapping:

```python
left_hand_joints  = left_close_pose  if left_binary_t  >= 0.5 else left_open_pose
right_hand_joints = right_close_pose if right_binary_t >= 0.5 else right_open_pose
```

### 7.5 Neck

`action_neck` is not part of the canonical body action.

Reason:

- it is an auxiliary side channel
- it is not part of `mimic_obs35`
- SONIC body control path does not depend on it

If needed, neck can be handled by a separate head or a separate adapter.

## 8. Reference Adapter Pseudocode

```python
def canonical_to_sonic(curr, prev, dt, body_xy_world_prev):
    joint_pos = curr.joint_pos_29
    body_quat = rot6d_to_quat_wxyz(curr.root_rot6d)

    if prev is None:
        joint_vel = np.zeros(29, dtype=np.float32)
        body_xy_world = np.zeros(2, dtype=np.float32)
    else:
        joint_vel = (curr.joint_pos_29 - prev.joint_pos_29) / dt
        body_xy_world = body_xy_world_prev + curr.root_xy_delta

    body_pos = np.array([body_xy_world[0], body_xy_world[1], curr.root_z], dtype=np.float32)
    left_hand = left_close_pose if curr.hand_binary[0] >= 0.5 else left_open_pose
    right_hand = right_close_pose if curr.hand_binary[1] >= 0.5 else right_open_pose
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos": body_pos,
        "body_quat_w": body_quat,
        "left_hand_joints": left_hand,
        "right_hand_joints": right_hand,
        "body_xy_world": body_xy_world,
    }


def canonical_to_twist2(curr, prev, dt):
    root_quat = rot6d_to_quat_wxyz(curr.root_rot6d)
    joint_pos_twist2 = curr.joint_pos_29[CANONICAL_TO_TWIST2_INDEX_29]
    root_lin_vel_world = np.array(
        [curr.root_xy_delta[0] / dt, curr.root_xy_delta[1] / dt, 0.0],
        dtype=np.float32,
    )

    if prev is None:
        yaw_vel = np.zeros(1, dtype=np.float32)
    else:
        yaw_vel = np.array(
            [wrap_to_pi(yaw_from_rot6d(curr.root_rot6d) - yaw_from_rot6d(prev.root_rot6d)) / dt],
            dtype=np.float32,
        )

    xy_vel = quat_rotate_inverse(root_quat, root_lin_vel_world)[:2]
    roll_pitch = quat_to_roll_pitch(root_quat)
    z_pos = np.array([curr.root_z], dtype=np.float32)

    mimic_obs35 = np.concatenate([xy_vel, z_pos, roll_pitch, yaw_vel, joint_pos_twist2], axis=0)
    left_hand = left_close_pose if curr.hand_binary[0] >= 0.5 else left_open_pose
    right_hand = right_close_pose if curr.hand_binary[1] >= 0.5 else right_open_pose
    return {
        "action_body": mimic_obs35,
        "action_hand_left": left_hand,
        "action_hand_right": right_hand,
    }
```

## 9. Practical Notes

### 9.1 Publish rate

The finite-difference quantities depend on `dt`.

So:

- either publish canonical action at the same control cadence as the target backend
- or resample canonical action first, then finite-difference after resampling

For exact SONIC live parity, using the current `50 Hz` joint29 cadence is the safest choice.

### 9.2 Reset behavior

On episode reset, the adapter must reset:

- previous root rotation
- previous joint pose
- accumulated `body_xy_world`
- previous velocity state used for smoothing

Otherwise the first finite-difference frame after reset will be wrong.

### 9.3 Training-time representation

`root_rot6d` is already the canonical orientation representation.

Downstream adapters can recover:

- `root_quat_wxyz`

whenever SONIC or TWIST2 needs quaternion-based logic.

If future manipulation tasks need graded finger closure, add:

- `hand_open_ratio_2`

as an optional extension, not as the current canonical control action.

## 10. One-Line Summary

The right robot-level canonical G1 action for unified VLA control is:

- `root_xy_delta + root_z + root_rot6d + canonical_joint_pos_29 + hand_binary_2`

and not:

- `root_pos_xyz + root_quat_wxyz + hand_open_ratio_2`

because the VLA head should predict stable, actionable robot-level targets, while adapters reconstruct backend-specific absolute pose and finite-difference quantities.

Paired unified VLA proprioception is:

- `root_rot6d + dof_pos_29 + dof_vel_29`

Total:

- `6 + 29 + 29 = 64D`
