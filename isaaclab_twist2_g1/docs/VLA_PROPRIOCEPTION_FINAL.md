# VLA Proprioception Final Definition

## 1. Purpose

This document records the final decision for the unified robot proprioception used as VLA input.

This proprioception definition is intentionally:

- robot-centric
- backend-agnostic
- independent from SONIC/TWIST2 internal network tensors
- limited to the robot's current body state
- minimal, but still self-consistent

It is not meant to reproduce:

- SONIC `decoder_obs(994)`
- SONIC `95D state`
- TWIST2 `92D obs_proprio`

Those are backend-specific policy inputs. The unified VLA proprioception should stay simpler and semantically cleaner.

## 2. Final Definition

The final unified proprioception is:

- `root_rot6d`: `6D`
- `dof_pos_29`: `29D`
- `dof_vel_29`: `29D`

Total:

- `64D`

Recommended field layout:

```text
state.root_rot6d.<6>
state.dof_pos.<29 joints in canonical IsaacLab order>
state.dof_vel.<29 joints in canonical IsaacLab order>
```

## 3. Why This Definition

### 3.1 Pure current robot body state

This definition only describes the robot's current body state.

It deliberately does not include:

- `prev_action`
- `last_body_raw_action`
- `last_body_target`
- decoder or encoder history buffers
- hand / gripper semantic state

Those quantities are either control context or side-channel task semantics, not pure body proprioception.

### 3.2 Why use `root_rot6d`

The final choice is:

- `root_rot6d`

instead of:

- `roll_pitch`
- `gravity_dir`
- quaternion directly

Reason:

- `roll_pitch` is only a compressed orientation view, not a complete root orientation state
- `root_rot6d` is a complete orientation representation without quaternion sign ambiguity
- it is easier for VLA to regress than quaternion
- it is still easy to convert back to quaternion when downstream logic needs it

The intended semantics are:

- episode-local global root orientation
- `z-up`
- heading zero defined at episode reset

Do not use:

- base-relative local orientation
- frame-to-frame rotational delta

Those are downstream- or controller-dependent quantities, not canonical current robot state.

### 3.3 Why use `dof_pos`, not `dof_pos_delta`

The final choice is absolute joint position:

- `dof_pos_29`

instead of:

- `dof_pos_delta_29 = dof_pos_29 - default_pos_29`

Reason:

- `dof_pos` is the most direct physical state
- it does not depend on a chosen default pose
- it is easier to interpret and debug
- if needed, `dof_pos_delta` can always be derived later from `dof_pos`

### 3.4 Why keep `dof_vel_29`

`dof_vel_29` is the minimal dynamic term that should remain in the unified proprioception.

Reason:

- without joint velocity, the state loses first-order motion information
- many actions that look identical in pose differ in continuation because of velocity
- `dof_vel_29` is already available in both SONIC and TWIST2 recordings and runtimes

### 3.5 Why not complete dynamics state

The final unified proprioception is not intended to be a complete rigid-body dynamics state.

It does not include:

- root linear velocity
- root angular velocity
- contact state
- torque / effort
- controller history

Reason:

- VLA is not replacing the low-level controller
- VLA needs a minimal, learnable robot-state summary
- complete dynamics state would add complexity and backend coupling too early

If later experiments show that `64D` is too weak for dynamic tasks, the first recommended extension is:

- `root_ang_vel_3`

not a full jump to complete dynamics state.

## 4. Availability In Current Recordings

The current dataset root is:

- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/HOI_football_v2`

The required raw fields for reconstructing this `64D` proprioception are:

- `robot_root_orientation`
- `robot_qpos_before_decimation`
- `robot_qvel_before_decimation`

Required shapes:

- `robot_root_orientation`: `(T, 4)` in `wxyz`
- `robot_qpos_before_decimation`: `(T, 29)`
- `robot_qvel_before_decimation`: `(T, 29)`

So the final `64D` proprioception is extractable from both SONIC and TWIST2 recordings.

## 5. Canonical 29-Joint Order

The unified 29-DoF order is defined to be the IsaacLab order used by SONIC.

This is the canonical order for all unified VLA state and action representations.

### 5.1 Why choose IsaacLab order

Reason:

- SONIC already uses this order natively
- SONIC joint-based tracking also expects IsaacLab order
- it is the cleaner common reference for G1 in the current project
- only TWIST2 needs reordering

### 5.2 Canonical joint names

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

## 6. Backend Extraction Rules

### 6.1 SONIC

SONIC already stores the 29-DoF robot body state in canonical IsaacLab order.

Per frame `t`:

- `root_quat_wxyz = robot_root_orientation[t]`
- `root_rot6d = quaternion_to_rot6d(root_quat_wxyz)`
- `dof_pos_29 = robot_qpos_before_decimation[t]`
- `dof_vel_29 = robot_qvel_before_decimation[t]`

### 6.2 TWIST2

TWIST2 stores the 29-DoF robot body state in its own action order, not the canonical IsaacLab order.

Per frame `t`:

- `root_quat_wxyz = robot_root_orientation[t]`
- `root_rot6d = quaternion_to_rot6d(root_quat_wxyz)`
- `dof_pos_twist2 = robot_qpos_before_decimation[t]`
- `dof_vel_twist2 = robot_qvel_before_decimation[t]`

Then reorder `dof_pos_twist2` and `dof_vel_twist2` into canonical IsaacLab order.

### 6.3 TWIST2 to canonical reorder index

For an array in TWIST2 order:

```python
canonical = twist2_array[twist2_to_canonical]
```

with:

```text
[0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
```

This mapping converts:

- TWIST2 29-joint order

into:

- canonical IsaacLab 29-joint order

## 7. What Was Explicitly Rejected

The following were considered but are not part of the final proprioception:

- `roll_pitch`
- `gravity_dir`
- `dof_pos_delta`
- `gripper_state_2`
- `prev_action`
- `last_body_raw_action`
- `last_body_target`
- SONIC `decoder_obs`
- TWIST2 `obs_proprio`
- complete dynamics state

Reasons:

- they are either compressed views rather than complete current orientation
- or they are not part of pure current robot body state
- or they are backend-specific
- or they add controller context that VLA should not have to reconstruct

## 8. Final Decision Summary

Final unified VLA proprioception:

```text
root_rot6d_6
dof_pos_29
dof_vel_29
```

Total:

```text
6 + 29 + 29 = 64D
```
