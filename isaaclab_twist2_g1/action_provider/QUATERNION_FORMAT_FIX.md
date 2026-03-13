# Quaternion Format Fix

## Problem

When replaying recorded data, the robot was lying on the ground instead of standing upright.

## Root Cause

**Quaternion format mismatch between recording and replay:**

- **Recording script** (`action_provider_wh_twist2.py`):
  - Reads `root_state_w[:, 3:7]` from Isaac Lab
  - Saves as (w,x,y,z) format
  - Comment at line 1866: `# [4] quaternion (w,x,y,z)`
  - Uses `_twist2_roll_pitch_from_quaternion` which expects (w,x,y,z)

- **Isaac Lab's `write_root_pose_to_sim`**:
  - Expects quaternion in (x,y,z,w) format
  - This is the standard format used by PyTorch, Isaac Gym, and ROS

## Evidence

Recorded quaternion: `[0.70253754, -0.01670154, 0.01430531, 0.71130675]`

**If interpreted as (w,x,y,z):**
- Roll = -0.18°, Pitch = 2.51°, Yaw = 90.71°
- Robot is **standing upright** ✓

**If interpreted as (x,y,z,w):**
- Roll = 89.29°, Pitch = -2.51°, Yaw = -0.18°
- Robot is **lying on the ground** ✗

## Solution

Convert quaternion from (w,x,y,z) to (x,y,z,w) before passing to `write_root_pose_to_sim`:

```python
# Recorded as (w,x,y,z)
root_quat_wxyz = self.replay_data_root_quat[0]

# Convert to (x,y,z,w) for Isaac Lab
root_quat_xyzw = np.array([
    root_quat_wxyz[1],  # x
    root_quat_wxyz[2],  # y
    root_quat_wxyz[3],  # z
    root_quat_wxyz[0]   # w
])

# Pass to Isaac Lab
root_quat_tensor = torch.from_numpy(root_quat_xyzw).to(device).unsqueeze(0)
env.scene["robot"].write_root_pose_to_sim(
    root_pose=torch.cat([root_pos_tensor, root_quat_tensor], dim=-1)
)
```

## Fix Applied

Updated `action_provider_wh_twist2_replay.py` at line 237-251 to convert quaternion format.

## Note

The recording script's comment saying "(w,x,y,z)" is **misleading**. While the recording script internally uses (w,x,y,z) format for its own calculations, Isaac Lab's underlying data structure (`root_state_w`) actually stores quaternions in (x,y,z,w) format. The recording script reads it as (w,x,y,z) and saves it that way, but when we write it back to Isaac Lab, we need to convert it to (x,y,z,w).

## Verification

After the fix:
- Robot should spawn in standing position
- Initial orientation should match recorded data
- Roll ≈ -0.18°, Pitch ≈ 2.51°, Yaw ≈ 90.71°
