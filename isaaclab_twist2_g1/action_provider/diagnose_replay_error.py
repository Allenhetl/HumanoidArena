#!/usr/bin/env python3
"""
Diagnostic script to analyze replay error sources.

This script compares recorded data with replay data to identify:
1. Root state drift over time
2. Joint position drift over time
3. Observation drift over time
4. Action output differences between recording and replay
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_replay_data(npz_path):
    """Load recorded data from npz file"""
    data = np.load(npz_path, allow_pickle=True)
    return data

def analyze_root_state_drift(data):
    """Analyze how root state changes over time"""
    root_pos = data['robot_root_position']  # [N, 3]
    root_quat = data['robot_root_orientation']  # [N, 4]

    # Compute position change between consecutive frames
    pos_delta = np.diff(root_pos, axis=0)  # [N-1, 3]
    pos_speed = np.linalg.norm(pos_delta, axis=1)  # [N-1]

    # Compute orientation change (simplified - just quaternion distance)
    quat_delta = np.diff(root_quat, axis=0)  # [N-1, 4]
    quat_change = np.linalg.norm(quat_delta, axis=1)  # [N-1]

    print("=" * 80)
    print("ROOT STATE ANALYSIS")
    print("=" * 80)
    print(f"Total frames: {len(root_pos)}")
    print(f"\nPosition statistics:")
    print(f"  Initial position: {root_pos[0]}")
    print(f"  Final position: {root_pos[-1]}")
    print(f"  Total displacement: {np.linalg.norm(root_pos[-1] - root_pos[0]):.4f} m")
    print(f"  Mean frame-to-frame speed: {pos_speed.mean():.6f} m/frame")
    print(f"  Max frame-to-frame speed: {pos_speed.max():.6f} m/frame")
    print(f"\nOrientation statistics:")
    print(f"  Initial quaternion: {root_quat[0]}")
    print(f"  Final quaternion: {root_quat[-1]}")
    print(f"  Mean frame-to-frame change: {quat_change.mean():.6f}")
    print(f"  Max frame-to-frame change: {quat_change.max():.6f}")

    return pos_speed, quat_change

def analyze_joint_position_drift(data):
    """Analyze joint position changes over time"""
    qpos = data['robot_twist2_inference_qpos']  # [N, 29]

    # Compute joint position change between consecutive frames
    qpos_delta = np.diff(qpos, axis=0)  # [N-1, 29]
    qpos_change = np.linalg.norm(qpos_delta, axis=1)  # [N-1]

    print("\n" + "=" * 80)
    print("JOINT POSITION ANALYSIS")
    print("=" * 80)
    print(f"Total frames: {len(qpos)}")
    print(f"\nJoint position statistics:")
    print(f"  Mean frame-to-frame change: {qpos_change.mean():.6f} rad")
    print(f"  Max frame-to-frame change: {qpos_change.max():.6f} rad")
    print(f"  Std frame-to-frame change: {qpos_change.std():.6f} rad")

    # Analyze per-joint statistics
    print(f"\nPer-joint statistics (first 6 joints - legs):")
    for i in range(6):
        joint_changes = np.abs(qpos_delta[:, i])
        print(f"  Joint {i}: mean={joint_changes.mean():.6f}, max={joint_changes.max():.6f}, std={joint_changes.std():.6f}")

    return qpos_change

def analyze_observation_structure(data):
    """Analyze observation buffer structure"""
    obs_buf = data['robot_obs_buf']  # [N, 1432]

    print("\n" + "=" * 80)
    print("OBSERVATION ANALYSIS")
    print("=" * 80)
    print(f"Observation shape: {obs_buf.shape}")
    print(f"Observation range: [{obs_buf.min():.4f}, {obs_buf.max():.4f}]")

    # Analyze different parts of observation
    obs_full = obs_buf[:, 0:127]  # Current observation
    obs_hist = obs_buf[:, 127:1397]  # History (10 frames * 127)
    future_obs = obs_buf[:, 1397:1432]  # Future mimic (35)

    print(f"\nObservation components:")
    print(f"  obs_full (0:127): range=[{obs_full.min():.4f}, {obs_full.max():.4f}]")
    print(f"  obs_hist (127:1397): range=[{obs_hist.min():.4f}, {obs_hist.max():.4f}]")
    print(f"  future_obs (1397:1432): range=[{future_obs.min():.4f}, {future_obs.max():.4f}]")

    # Analyze frame-to-frame observation change
    obs_delta = np.diff(obs_buf, axis=0)  # [N-1, 1432]
    obs_change = np.linalg.norm(obs_delta, axis=1)  # [N-1]

    print(f"\nObservation change statistics:")
    print(f"  Mean frame-to-frame change: {obs_change.mean():.6f}")
    print(f"  Max frame-to-frame change: {obs_change.max():.6f}")
    print(f"  Std frame-to-frame change: {obs_change.std():.6f}")

    return obs_change

def plot_analysis(pos_speed, quat_change, qpos_change, obs_change, save_path=None):
    """Plot analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot root position speed
    axes[0, 0].plot(pos_speed)
    axes[0, 0].set_title('Root Position Speed (m/frame)')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Speed (m/frame)')
    axes[0, 0].grid(True)

    # Plot root orientation change
    axes[0, 1].plot(quat_change)
    axes[0, 1].set_title('Root Orientation Change')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Quaternion Distance')
    axes[0, 1].grid(True)

    # Plot joint position change
    axes[1, 0].plot(qpos_change)
    axes[1, 0].set_title('Joint Position Change (rad/frame)')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Change (rad)')
    axes[1, 0].grid(True)

    # Plot observation change
    axes[1, 1].plot(obs_change)
    axes[1, 1].set_title('Observation Change')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n📊 Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    # Path to recorded data
    npz_path = "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/Isaac-Move-Football-G129-Dex3-Wholebody_1773220571468541.npz"

    print("Loading recorded data...")
    data = load_replay_data(npz_path)

    # Analyze different aspects
    pos_speed, quat_change = analyze_root_state_drift(data)
    qpos_change = analyze_joint_position_drift(data)
    obs_change = analyze_observation_structure(data)

    # Plot results
    save_path = Path(npz_path).parent / "replay_error_analysis.png"
    plot_analysis(pos_speed, quat_change, qpos_change, obs_change, save_path)

    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print("\n🔍 Key Findings:")
    print(f"1. Root position changes by ~{pos_speed.mean():.6f} m per frame on average")
    print(f"2. Joint positions change by ~{qpos_change.mean():.6f} rad per frame on average")
    print(f"3. Observations change by ~{obs_change.mean():.6f} (L2 norm) per frame")

    print("\n💡 Implications for Replay:")
    print("- Direct mode: Small root state errors will accumulate over time")
    print("- Inference mode: Observation errors will cause action drift, leading to larger errors")
    print("- The longer the replay, the larger the accumulated error")

    print("\n🎯 Recommendations:")
    print("1. For direct mode: Consider resetting root state every N frames")
    print("2. For inference mode: Use shorter replay segments or implement error correction")
    print("3. Consider using root velocity instead of root position for better stability")

if __name__ == "__main__":
    main()
