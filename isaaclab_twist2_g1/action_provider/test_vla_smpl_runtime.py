from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(_PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from action_provider.vla_smpl_runtime import (
    UnifiedSMPLActionRuntime,
    build_sonic_joint29_payload,
    build_twist2_mimic_obs,
    build_vla_action,
    quat_from_roll_pitch_yaw_wxyz,
    quat_to_rot6d_wxyz,
    reorder_twist2_to_canonical_29,
)


def test_unified_runtime_roundtrip_shapes_and_joint_order() -> None:
    joint_pos_canonical = np.linspace(-0.5, 0.5, 29, dtype=np.float32)
    root_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.1, pitch=-0.05, yaw=0.3)
    canonical_action = build_vla_action(
        root_xy_delta_world=np.array([0.12, -0.08], dtype=np.float32),
        root_z=0.84,
        root_rot6d=quat_to_rot6d_wxyz(root_quat).reshape(6),
        joint_pos_canonical_29=joint_pos_canonical,
        hand_binary=np.array([0.25, 0.9], dtype=np.float32),
    )

    runtime = UnifiedSMPLActionRuntime()
    frame = runtime.step(canonical_action)
    mimic_obs = build_twist2_mimic_obs(runtime_frame=frame, control_dt=0.2)
    sonic_payload = build_sonic_joint29_payload(runtime_frame=frame, control_dt=0.2)

    assert mimic_obs.shape == (35,)
    assert sonic_payload["body_pos"].shape == (3,)
    assert sonic_payload["body_quat_w"].shape == (4,)
    assert sonic_payload["joint_pos"].shape == (29,)
    assert sonic_payload["joint_vel"].shape == (29,)
    np.testing.assert_allclose(frame.hand_binary, np.array([0.25, 0.9], dtype=np.float32))
    np.testing.assert_allclose(sonic_payload["joint_pos"], joint_pos_canonical)
    np.testing.assert_allclose(reorder_twist2_to_canonical_29(mimic_obs[6:35]), joint_pos_canonical)


def test_unified_runtime_accumulates_xy_and_derives_joint_velocity() -> None:
    runtime = UnifiedSMPLActionRuntime()
    root_quat = quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=0.0)

    first = runtime.step(
        build_vla_action(
            root_xy_delta_world=np.array([0.1, 0.0], dtype=np.float32),
            root_z=0.8,
            root_rot6d=quat_to_rot6d_wxyz(root_quat).reshape(6),
            joint_pos_canonical_29=np.zeros(29, dtype=np.float32),
            hand_binary=np.zeros(2, dtype=np.float32),
        )
    )
    second = runtime.step(
        build_vla_action(
            root_xy_delta_world=np.array([0.05, -0.02], dtype=np.float32),
            root_z=0.8,
            root_rot6d=quat_to_rot6d_wxyz(root_quat).reshape(6),
            joint_pos_canonical_29=np.ones(29, dtype=np.float32) * 0.2,
            hand_binary=np.array([1.0, 0.0], dtype=np.float32),
        )
    )

    payload = build_sonic_joint29_payload(runtime_frame=second, control_dt=0.1)

    np.testing.assert_allclose(first.body_pos_world, np.array([0.1, 0.0, 0.8], dtype=np.float32))
    np.testing.assert_allclose(second.body_pos_world, np.array([0.15, -0.02, 0.8], dtype=np.float32))
    np.testing.assert_allclose(payload["joint_vel"], np.ones(29, dtype=np.float32) * 2.0)
