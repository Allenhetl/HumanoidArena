from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(_PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from action_provider.vla_local_delta_runtime_v2 import (
    UnifiedLocalDeltaActionRuntimeV2,
    build_sonic_joint29_payload_v2,
    build_twist2_mimic_obs_v2,
    build_vla_local_delta_action,
    rotate_root_local_delta_to_world,
    rotate_world_delta_to_root_local,
)
from action_provider.vla_smpl_runtime import (
    quat_from_roll_pitch_yaw_wxyz,
    quat_to_rot6d_wxyz,
    reorder_twist2_to_canonical_29,
)


def test_root_local_delta_rotation_roundtrip() -> None:
    root_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)

    world_delta = rotate_root_local_delta_to_world(
        root_quat_wxyz=root_quat,
        root_local_xy_delta=np.array([1.0, 0.0], dtype=np.float32),
    )
    local_delta = rotate_world_delta_to_root_local(root_quat, world_delta)

    np.testing.assert_allclose(world_delta, np.array([0.0, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(local_delta, np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_runtime_uses_local_delta_for_twist2_and_world_payload_for_sonic() -> None:
    joint_pos_canonical = np.linspace(-0.4, 0.4, 29, dtype=np.float32)
    root_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0)
    action = build_vla_local_delta_action(
        root_local_xy_delta=np.array([0.2, 0.0], dtype=np.float32),
        root_z=0.81,
        root_rot6d=quat_to_rot6d_wxyz(root_quat).reshape(6),
        joint_pos_canonical_29=joint_pos_canonical,
        hand_binary=np.array([0.0, 1.0], dtype=np.float32),
    )

    runtime = UnifiedLocalDeltaActionRuntimeV2()
    frame = runtime.step(action)
    mimic_obs = build_twist2_mimic_obs_v2(runtime_frame=frame, control_dt=0.1)
    sonic_payload = build_sonic_joint29_payload_v2(runtime_frame=frame, control_dt=0.1)

    np.testing.assert_allclose(frame.root_local_xy_delta, np.array([0.2, 0.0], dtype=np.float32))
    np.testing.assert_allclose(frame.root_xy_delta_world, np.array([0.0, 0.2], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(frame.body_pos_world, np.array([0.0, 0.2, 0.81], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(mimic_obs[0:2], np.array([2.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(sonic_payload["body_pos"], frame.body_pos_world)
    np.testing.assert_allclose(sonic_payload["joint_pos"], joint_pos_canonical)
    np.testing.assert_allclose(reorder_twist2_to_canonical_29(mimic_obs[6:35]), joint_pos_canonical)


def test_sonic_payload_joint_velocity_uses_isaac_control_dt() -> None:
    runtime = UnifiedLocalDeltaActionRuntimeV2()
    root_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=0.0)
    root_rot6d = quat_to_rot6d_wxyz(root_quat).reshape(6)

    runtime.step(
        build_vla_local_delta_action(
            root_local_xy_delta=np.zeros(2, dtype=np.float32),
            root_z=0.8,
            root_rot6d=root_rot6d,
            joint_pos_canonical_29=np.zeros(29, dtype=np.float32),
            hand_binary=np.zeros(2, dtype=np.float32),
        )
    )
    second = runtime.step(
        build_vla_local_delta_action(
            root_local_xy_delta=np.zeros(2, dtype=np.float32),
            root_z=0.8,
            root_rot6d=root_rot6d,
            joint_pos_canonical_29=np.ones(29, dtype=np.float32) * 0.1,
            hand_binary=np.zeros(2, dtype=np.float32),
        )
    )
    payload = build_sonic_joint29_payload_v2(runtime_frame=second, control_dt=0.02)

    np.testing.assert_allclose(payload["joint_vel"], np.ones(29, dtype=np.float32) * 5.0)
