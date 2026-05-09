from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (_THIS_DIR, _PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from action_provider.vla_smpl_runtime import quat_from_roll_pitch_yaw_wxyz
from smpl_lerobot_v2_common import (
    build_features,
    build_sonic_localdelta_v2_actions,
    build_twist2_localdelta_v2_actions,
    reorder_twist2_to_canonical_29,
)


def test_sonic_v2_action_uses_current_root_frame_for_xy_delta(tmp_path: Path) -> None:
    path = tmp_path / "sonic.npz"
    body_quat = np.stack(
        [
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0),
            quat_from_roll_pitch_yaw_wxyz(roll=0.0, pitch=0.0, yaw=np.pi / 2.0),
        ],
        axis=0,
    )
    vla_action_raw = np.zeros((2, 40), dtype=np.float32)
    vla_action_raw[:, 38:40] = np.array([[0.25, 0.75], [0.5, 1.0]], dtype=np.float32)
    np.savez(
        path,
        human_body_pos=np.array([[0.0, 0.0, 0.8], [0.0, 1.0, 0.82]], dtype=np.float32),
        human_body_quat_w=body_quat.astype(np.float32),
        human_joint_pos=np.zeros((2, 29), dtype=np.float32),
        vla_action_raw=vla_action_raw,
        pico_left_grip_binary=np.ones(2, dtype=np.float32),
        pico_right_grip_binary=np.zeros(2, dtype=np.float32),
    )

    with np.load(path, allow_pickle=True) as data:
        actions = build_sonic_localdelta_v2_actions(data, num_frames=2)

    np.testing.assert_allclose(actions[0, 0:2], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(actions[1, 0:2], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(actions[:, 38:40], vla_action_raw[:, 38:40])


def test_twist2_v2_action_keeps_mimic_xy_as_local_delta(tmp_path: Path) -> None:
    path = tmp_path / "twist2.npz"
    mimic = np.zeros((2, 35), dtype=np.float32)
    mimic[:, 0:2] = np.array([[0.5, -0.25], [1.0, 0.75]], dtype=np.float32)
    mimic[:, 2] = 0.8
    mimic[:, 6:35] = np.arange(58, dtype=np.float32).reshape(2, 29) * 0.01
    hand = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    np.savez(path, robot_action_mimic=mimic, vla_action_hand_binary=hand)

    with np.load(path, allow_pickle=True) as data:
        actions = build_twist2_localdelta_v2_actions(data, num_frames=2, control_dt=0.02)

    np.testing.assert_allclose(actions[:, 0:2], mimic[:, 0:2] * 0.02)
    np.testing.assert_allclose(actions[:, 9:38], reorder_twist2_to_canonical_29(mimic[:, 6:35]))
    np.testing.assert_allclose(actions[:, 38:40], hand)


def test_v2_feature_names_are_local_delta() -> None:
    features = build_features((64, 64, 3), use_videos=False)

    assert features["action"]["names"][0] == "action.root_local_xy_delta.x"
    assert features["action"]["names"][1] == "action.root_local_xy_delta.y"
    assert features["action"]["shape"] == (40,)
    assert features["observation.state"]["shape"] == (64,)
