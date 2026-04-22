#!/usr/bin/env python3
import pathlib
import sys

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "isaaclab_twist2_g1"))

from action_provider.action_provider_sonic import (  # noqa: E402
    StreamedMotionPacket,
    advance_stream_playback,
    gather_stream_playback_window,
    merge_streamed_motion_packet,
)


def make_packet(start: int, count: int) -> StreamedMotionPacket:
    frame_indices = np.arange(start, start + count, dtype=np.int64)
    smpl_joints = np.repeat(frame_indices[:, None, None], 24 * 3, axis=1).reshape(count, 24, 3).astype(np.float32)
    smpl_pose = np.repeat(frame_indices[:, None, None], 21 * 3, axis=1).reshape(count, 21, 3).astype(np.float32)
    body_quat_w = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1))
    joint_pos = np.repeat(frame_indices[:, None], 29, axis=1).astype(np.float32)
    joint_vel = (joint_pos + 0.5).astype(np.float32)
    root_z = (frame_indices.astype(np.float32) * 0.01).astype(np.float32)
    return StreamedMotionPacket(
        frame_indices=frame_indices,
        smpl_joints=smpl_joints,
        smpl_pose=smpl_pose,
        body_quat_w=body_quat_w,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_z=root_z,
    )


def test_overlap_merge_keeps_continuous_timeline():
    state, info = merge_streamed_motion_packet(None, make_packet(100, 5))
    assert info["did_catchup_reset"] is True

    state, info = merge_streamed_motion_packet(state, make_packet(101, 5))
    assert info["did_catchup_reset"] is False

    state, info = merge_streamed_motion_packet(state, make_packet(102, 5))
    assert info["did_catchup_reset"] is False
    assert np.array_equal(state.frame_indices, np.arange(100, 107, dtype=np.int64))


def test_gap_forces_catchup_reset():
    state, _ = merge_streamed_motion_packet(None, make_packet(100, 5))
    state, info = merge_streamed_motion_packet(state, make_packet(106, 5))
    assert info["did_catchup_reset"] is True
    assert np.array_equal(state.frame_indices, np.arange(106, 111, dtype=np.int64))
    assert state.playback_idx == 0


def test_delayed_playback_uses_latest_complete_window():
    state, _ = merge_streamed_motion_packet(None, make_packet(10, 3))
    state = advance_stream_playback(state, lookahead_frames=3)
    assert state.playback_idx == 0

    state, _ = merge_streamed_motion_packet(state, make_packet(11, 3))
    state = advance_stream_playback(state, lookahead_frames=3)
    assert state.playback_idx == 1

    window = gather_stream_playback_window(state, num_frames=3)
    assert np.array_equal(window["frame_indices"], np.array([11, 12, 13], dtype=np.int64))
    assert np.all(window["joint_pos"][:, 0] == np.array([11.0, 12.0, 13.0], dtype=np.float32))


if __name__ == "__main__":
    test_overlap_merge_keeps_continuous_timeline()
    test_gap_forces_catchup_reset()
    test_delayed_playback_uses_latest_complete_window()
    print("streamed motion window verifier passed")
