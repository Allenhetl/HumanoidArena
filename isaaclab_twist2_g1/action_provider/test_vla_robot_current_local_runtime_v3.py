from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(_PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from action_provider.vla_robot_current_local_runtime_v3 import (
    ACTION_NAMES,
    STATE_NAMES,
    UnifiedRobotCurrentLocalActionRuntimeV3,
    build_sonic_joint29_payload_v3,
    build_twist2_mimic_obs_v3,
    build_vla_rotlocal_v3_action,
    build_vla_rotlocal_v3_observation_state,
    heading_canonical_robot_quat_wxyz,
    robot_current_local_target_quat_wxyz,
    rotate_target_heading_local_delta_to_world,
    rotate_world_delta_to_target_heading_local,
)
from action_provider.vla_smpl_runtime import (
    quat_angle_between_wxyz,
    quat_from_roll_pitch_yaw_wxyz,
    quat_to_rot6d_wxyz,
    reorder_twist2_to_canonical_29,
    yaw_from_quat_wxyz,
)


def test_v3_feature_names_are_explicit() -> None:
    assert STATE_NAMES[0] == "state.root_heading_canonical_rot6d.0"
    assert ACTION_NAMES[0] == "action.root_target_heading_local_xy_delta.x"
    assert ACTION_NAMES[3] == "action.root_current_local_target_rot6d.0"
    assert len(STATE_NAMES) == 64
    assert len(ACTION_NAMES) == 40


def test_robot_current_local_target_rotation_identity_for_matching_heading() -> None:
    robot_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)
    target_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)

    rel_quat = robot_current_local_target_quat_wxyz(robot_quat, target_quat)

    np.testing.assert_allclose(rel_quat, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_robot_current_local_target_rotation_keeps_relative_yaw() -> None:
    robot_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)
    target_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=3.0 * np.pi / 4.0)

    rel_quat = robot_current_local_target_quat_wxyz(robot_quat, target_quat)

    np.testing.assert_allclose(yaw_from_quat_wxyz(rel_quat), np.pi / 4.0, atol=1e-6)


def test_heading_canonical_state_removes_episode_initial_heading() -> None:
    initial = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)
    current = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)

    canonical = heading_canonical_robot_quat_wxyz(initial, current)
    state = build_vla_rotlocal_v3_observation_state(
        initial_robot_orientation_wxyz=initial,
        root_orientation_wxyz=current,
        joint_pos_canonical_29=np.zeros(29, dtype=np.float32),
        joint_vel_canonical_29=np.zeros(29, dtype=np.float32),
    )

    np.testing.assert_allclose(canonical, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
    assert state.shape == (64,)


def test_runtime_reconstructs_target_from_current_robot_and_relative_action() -> None:
    current_robot = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)
    target = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=3.0 * np.pi / 4.0)
    rel = robot_current_local_target_quat_wxyz(current_robot, target)
    joint_pos_canonical = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    action = build_vla_rotlocal_v3_action(
        root_target_heading_local_xy_delta=np.array([0.1, 0.0], dtype=np.float32),
        root_z=0.82,
        root_current_local_target_rot6d=quat_to_rot6d_wxyz(rel).reshape(6),
        joint_pos_canonical_29=joint_pos_canonical,
        hand_binary=np.array([1.0, 0.0], dtype=np.float32),
    )

    runtime = UnifiedRobotCurrentLocalActionRuntimeV3()
    frame = runtime.step(
        action,
        current_robot_quat_wxyz=current_robot,
        current_robot_xy_world=np.array([2.0, -1.0], dtype=np.float32),
    )
    expected_world_delta = rotate_target_heading_local_delta_to_world(target, np.array([0.1, 0.0], dtype=np.float32))
    mimic_obs = build_twist2_mimic_obs_v3(runtime_frame=frame, control_dt=0.1)
    sonic_payload = build_sonic_joint29_payload_v3(runtime_frame=frame, control_dt=0.1)

    assert quat_angle_between_wxyz(frame.target_root_quat_wxyz, target) < 1e-5
    np.testing.assert_allclose(frame.root_xy_delta_world, expected_world_delta, atol=1e-6)
    np.testing.assert_allclose(frame.body_pos_world[:2], np.array([2.0, -1.0], dtype=np.float32) + expected_world_delta, atol=1e-6)
    np.testing.assert_allclose(mimic_obs[0:2], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(sonic_payload["body_quat_w"], target, atol=1e-6)
    np.testing.assert_allclose(reorder_twist2_to_canonical_29(mimic_obs[6:35]), joint_pos_canonical)


def test_heading_local_xy_roundtrip_preserves_world_delta_with_pitch() -> None:
    target = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=np.pi / 6.0, yaw=0.0)
    world_delta = np.array([1.0, 0.0], dtype=np.float32)

    heading_local = rotate_world_delta_to_target_heading_local(target, world_delta)
    reconstructed = rotate_target_heading_local_delta_to_world(target, heading_local)

    np.testing.assert_allclose(heading_local, world_delta, atol=1e-6)
    np.testing.assert_allclose(reconstructed, world_delta, atol=1e-6)
