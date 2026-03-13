#!/usr/bin/env python3
"""
Diagnostic tool to analyze replay consistency issues.
Compares recorded data with replay execution to identify discrepancies.
"""

import numpy as np
import sys
from pathlib import Path

def load_recording(npz_path):
    """Load recording data from npz file"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"❌ Failed to load recording: {e}")
        return None

def check_data_completeness(data):
    """Check if recording contains all necessary data for replay"""
    print("\n" + "="*60)
    print("📋 DATA COMPLETENESS CHECK")
    print("="*60)

    required_fields = {
        'robot_qpos_before_decimation': 'Joint positions (29 DOF)',
        'robot_qvel_before_decimation': 'Joint velocities (29 DOF)',
        'robot_root_position': 'Root position (3D)',
        'robot_root_orientation': 'Root orientation (quaternion)',
        'robot_twist2_inference_qpos': 'ONNX inference output (29 DOF)',
        'robot_obs_buf': 'Observation buffer (1432D)',
    }

    velocity_fields = {
        'robot_root_lin_vel_world': 'Root linear velocity (world frame)',
        'robot_root_ang_vel_world': 'Root angular velocity (world frame)',
        'robot_root_lin_vel_local': 'Root linear velocity (local frame)',
        'robot_root_ang_vel_local': 'Root angular velocity (local frame)',
    }

    football_fields = {
        'env_obj_football_position': 'Football position',
        'env_obj_football_linear_velocity': 'Football linear velocity',
        'env_obj_football_angular_velocity': 'Football angular velocity',
    }

    all_complete = True

    # Check required fields
    print("\n✅ Required fields:")
    for field, desc in required_fields.items():
        if field in data:
            shape = data[field].shape
            print(f"  ✓ {field}: {desc} - shape {shape}")
        else:
            print(f"  ❌ {field}: {desc} - MISSING")
            all_complete = False

    # Check velocity fields
    print("\n🔄 Velocity fields (at least one set required):")
    has_world_vel = 'robot_root_lin_vel_world' in data and 'robot_root_ang_vel_world' in data
    has_local_vel = 'robot_root_lin_vel_local' in data and 'robot_root_ang_vel_local' in data

    for field, desc in velocity_fields.items():
        if field in data:
            shape = data[field].shape
            print(f"  ✓ {field}: {desc} - shape {shape}")
        else:
            print(f"  ⚠️  {field}: {desc} - missing")

    if not (has_world_vel or has_local_vel):
        print("  ❌ WARNING: No complete velocity data found!")
        all_complete = False

    # Check football fields
    print("\n⚽ Football fields (optional):")
    for field, desc in football_fields.items():
        if field in data:
            shape = data[field].shape
            print(f"  ✓ {field}: {desc} - shape {shape}")
        else:
            print(f"  ⚠️  {field}: {desc} - missing")

    return all_complete

def analyze_initial_state(data):
    """Analyze the initial state (frame 0) of the recording"""
    print("\n" + "="*60)
    print("🔍 INITIAL STATE ANALYSIS (Frame 0)")
    print("="*60)

    # Robot root state
    if 'robot_root_position' in data:
        root_pos = data['robot_root_position'][0]
        print(f"\n🤖 Robot root position: {root_pos}")
        print(f"   Distance from origin: {np.linalg.norm(root_pos):.4f} m")

    if 'robot_root_orientation' in data:
        root_quat = data['robot_root_orientation'][0]
        print(f"   Root orientation (w,x,y,z): {root_quat}")
        # Check if it's identity quaternion (1,0,0,0)
        is_identity = np.allclose(root_quat, [1, 0, 0, 0], atol=0.01)
        print(f"   Is identity quaternion: {is_identity}")

    # Root velocities
    if 'robot_root_lin_vel_world' in data:
        lin_vel = data['robot_root_lin_vel_world'][0]
        print(f"   Root linear velocity (world): {lin_vel}")
        print(f"   Speed: {np.linalg.norm(lin_vel):.4f} m/s")

    if 'robot_root_ang_vel_world' in data:
        ang_vel = data['robot_root_ang_vel_world'][0]
        print(f"   Root angular velocity (world): {ang_vel}")
        print(f"   Angular speed: {np.linalg.norm(ang_vel):.4f} rad/s")

    # Joint positions
    if 'robot_qpos_before_decimation' in data:
        qpos = data['robot_qpos_before_decimation'][0]
        print(f"\n🦾 Joint positions (29 DOF):")
        print(f"   Range: [{qpos.min():.4f}, {qpos.max():.4f}]")
        print(f"   Mean: {qpos.mean():.4f}, Std: {qpos.std():.4f}")
        print(f"   First 5 joints: {qpos[:5]}")

    # Joint velocities
    if 'robot_qvel_before_decimation' in data:
        qvel = data['robot_qvel_before_decimation'][0]
        print(f"\n🦾 Joint velocities (29 DOF):")
        print(f"   Range: [{qvel.min():.4f}, {qvel.max():.4f}]")
        print(f"   Mean: {qvel.mean():.4f}, Std: {qvel.std():.4f}")
        print(f"   Max absolute velocity: {np.abs(qvel).max():.4f} rad/s")

    # Football state
    if 'env_obj_football_position' in data:
        football_pos = data['env_obj_football_position'][0]
        print(f"\n⚽ Football position: {football_pos}")
        print(f"   Distance from origin: {np.linalg.norm(football_pos):.4f} m")

        if 'env_obj_football_linear_velocity' in data:
            football_vel = data['env_obj_football_linear_velocity'][0]
            print(f"   Football velocity: {football_vel}")
            print(f"   Speed: {np.linalg.norm(football_vel):.4f} m/s")

def analyze_trajectory(data):
    """Analyze the trajectory over time"""
    print("\n" + "="*60)
    print("📈 TRAJECTORY ANALYSIS")
    print("="*60)

    num_frames = len(data['robot_qpos_before_decimation'])
    print(f"\nTotal frames: {num_frames}")

    # Root position trajectory
    if 'robot_root_position' in data:
        root_pos = data['robot_root_position']
        displacement = np.linalg.norm(root_pos[-1] - root_pos[0])
        max_displacement = np.max([np.linalg.norm(root_pos[i] - root_pos[0])
                                   for i in range(num_frames)])
        print(f"\n🤖 Root position trajectory:")
        print(f"   Initial: {root_pos[0]}")
        print(f"   Final: {root_pos[-1]}")
        print(f"   Total displacement: {displacement:.4f} m")
        print(f"   Max displacement: {max_displacement:.4f} m")

    # Joint position changes
    if 'robot_qpos_before_decimation' in data:
        qpos = data['robot_qpos_before_decimation']
        qpos_change = np.abs(qpos[-1] - qpos[0])
        print(f"\n🦾 Joint position changes:")
        print(f"   Max change: {qpos_change.max():.4f} rad")
        print(f"   Mean change: {qpos_change.mean():.4f} rad")
        print(f"   Joints with >0.5 rad change: {np.sum(qpos_change > 0.5)}/29")

    # Football trajectory
    if 'env_obj_football_position' in data:
        football_pos = data['env_obj_football_position']
        football_displacement = np.linalg.norm(football_pos[-1] - football_pos[0])
        print(f"\n⚽ Football trajectory:")
        print(f"   Initial: {football_pos[0]}")
        print(f"   Final: {football_pos[-1]}")
        print(f"   Total displacement: {football_displacement:.4f} m")

def check_determinism_requirements(data):
    """Check if data meets requirements for deterministic replay"""
    print("\n" + "="*60)
    print("🎯 DETERMINISM REQUIREMENTS CHECK")
    print("="*60)

    issues = []

    # Check if velocities are recorded
    has_root_vel = ('robot_root_lin_vel_world' in data or 'robot_root_lin_vel_local' in data)
    if not has_root_vel:
        issues.append("❌ Root velocities not recorded - physics will diverge")
    else:
        print("✅ Root velocities recorded")

    has_joint_vel = 'robot_qvel_before_decimation' in data
    if not has_joint_vel:
        issues.append("❌ Joint velocities not recorded - physics will diverge")
    else:
        print("✅ Joint velocities recorded")

    # Check if observation buffer is complete
    if 'robot_obs_buf' in data:
        obs_shape = data['robot_obs_buf'].shape
        expected_obs_dim = 1432  # 10 frames * (127 single obs + 29 action)
        if obs_shape[1] != expected_obs_dim:
            issues.append(f"⚠️  Observation dimension mismatch: {obs_shape[1]} vs expected {expected_obs_dim}")
        else:
            print(f"✅ Observation buffer complete: {obs_shape}")

    # Check if ONNX inference output is recorded
    if 'robot_twist2_inference_qpos' not in data:
        issues.append("⚠️  ONNX inference output not recorded - direct replay not possible")
    else:
        print("✅ ONNX inference output recorded")

    # Check for NaN or Inf values
    for field in ['robot_qpos_before_decimation', 'robot_root_position']:
        if field in data:
            if np.any(np.isnan(data[field])) or np.any(np.isinf(data[field])):
                issues.append(f"❌ {field} contains NaN or Inf values")

    if not issues:
        print("\n✅ All determinism requirements met!")
    else:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")

    return len(issues) == 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_replay_consistency.py <recording.npz>")
        print("   or: python diagnose_replay_consistency.py latest")
        sys.exit(1)

    # Handle 'latest' shortcut
    if sys.argv[1] == 'latest':
        recording_dir = Path(__file__).parent.parent / "recording_data"
        npz_files = sorted(recording_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not npz_files:
            print("❌ No recording files found in recording_data/")
            sys.exit(1)
        npz_path = npz_files[-1]
        print(f"📁 Using latest recording: {npz_path.name}")
    else:
        npz_path = Path(sys.argv[1])
        if not npz_path.exists():
            # Try relative to recording_data/
            npz_path = Path(__file__).parent.parent / "recording_data" / sys.argv[1]
            if not npz_path.exists():
                print(f"❌ Recording file not found: {sys.argv[1]}")
                sys.exit(1)

    print("="*60)
    print("🔬 REPLAY CONSISTENCY DIAGNOSTIC TOOL")
    print("="*60)
    print(f"📁 Recording file: {npz_path}")
    print(f"📊 File size: {npz_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Load data
    data = load_recording(npz_path)
    if data is None:
        sys.exit(1)

    # Run diagnostics
    is_complete = check_data_completeness(data)
    analyze_initial_state(data)
    analyze_trajectory(data)
    is_deterministic = check_determinism_requirements(data)

    # Summary
    print("\n" + "="*60)
    print("📝 SUMMARY")
    print("="*60)

    if is_complete and is_deterministic:
        print("✅ Recording is complete and ready for deterministic replay")
        print("\nRecommended replay command:")
        print(f"  ./run_replay.sh {npz_path.name} inference")
    else:
        print("⚠️  Recording has issues that may affect replay consistency")
        print("\nRecommended actions:")
        if not is_complete:
            print("  1. Re-record with updated action_provider_wh_twist2.py")
        if not is_deterministic:
            print("  2. Ensure random seed is set consistently")
            print("  3. Check for NaN/Inf values in physics simulation")

    print("="*60)

if __name__ == "__main__":
    main()
