#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smpl_lerobot_v2_common import (
    build_sonic_localdelta_v2_actions,
    build_twist2_localdelta_v2_actions,
    extract_twist2_action_mimic,
    find_npz_files,
    infer_twist2_control_dt,
    load_vision_rgb_and_indices,
)


def load_lerobot_fps(dataset_root: Path) -> int | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return None
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps")
    return int(fps) if fps is not None else None


def load_lerobot_rows(dataset_root: Path) -> dict[int, dict[str, np.ndarray]]:
    data_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No LeRobot parquet files found under {dataset_root / 'data'}")

    df = pd.concat([pd.read_parquet(path) for path in data_paths], ignore_index=True)
    if "episode_index" not in df or "frame_index" not in df or "timestamp" not in df or "action" not in df:
        raise ValueError("LeRobot parquet must contain episode_index, frame_index, timestamp, and action columns")

    episodes: dict[int, dict[str, np.ndarray]] = {}
    for episode_index, ep_df in df.groupby("episode_index", sort=True):
        ep_df = ep_df.sort_values("frame_index")
        actions = np.stack(ep_df["action"].to_numpy()).astype(np.float32)
        episodes[int(episode_index)] = {
            "frame_index": ep_df["frame_index"].to_numpy(dtype=np.int64),
            "timestamp": ep_df["timestamp"].to_numpy(dtype=np.float64),
            "action": actions,
        }
    return episodes


def verify_lerobot_clock(
    episode_rows: dict[int, dict[str, np.ndarray]],
    *,
    fps: int | None,
    atol: float,
) -> None:
    for episode_index, rows in episode_rows.items():
        frame_index = rows["frame_index"]
        expected_frame_index = np.arange(frame_index.shape[0], dtype=np.int64)
        if not np.array_equal(frame_index, expected_frame_index):
            raise AssertionError(f"episode {episode_index} frame_index is not 0..N-1")
        if fps is not None:
            expected_timestamp = expected_frame_index.astype(np.float64) / float(fps)
            if not np.allclose(rows["timestamp"], expected_timestamp, atol=atol, rtol=0.0):
                raise AssertionError(f"episode {episode_index} timestamp is not frame_index / fps")


def verify_sonic_episode(
    *,
    npz_path: Path,
    dataset_rows: dict[str, np.ndarray],
    atol: float,
) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
        _, vision_frame_indices = load_vision_rgb_and_indices(data, npz_path)
        expected = build_sonic_localdelta_v2_actions(data, num_frames=qpos.shape[0])

    if dataset_rows["action"].shape[0] != len(vision_frame_indices):
        raise AssertionError(
            f"{npz_path} frame count mismatch: dataset={dataset_rows['action'].shape[0]} "
            f"source vision frames={len(vision_frame_indices)}"
        )
    expected = expected[np.asarray(vision_frame_indices, dtype=np.int64)]
    np.testing.assert_allclose(dataset_rows["action"], expected, atol=atol, rtol=0.0)
    return int(expected.shape[0])


def verify_twist2_episode(
    *,
    npz_path: Path,
    dataset_rows: dict[str, np.ndarray],
    fps: int | None,
    atol: float,
) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
        _, vision_frame_indices = load_vision_rgb_and_indices(data, npz_path)
        control_dt = infer_twist2_control_dt(data, fps or 50)
        expected = build_twist2_localdelta_v2_actions(
            data,
            num_frames=qpos.shape[0],
            control_dt=control_dt,
        )
        mimic = extract_twist2_action_mimic(data, qpos.shape[0])

    if dataset_rows["action"].shape[0] != len(vision_frame_indices):
        raise AssertionError(
            f"{npz_path} frame count mismatch: dataset={dataset_rows['action'].shape[0]} "
            f"source vision frames={len(vision_frame_indices)}"
        )
    indices = np.asarray(vision_frame_indices, dtype=np.int64)
    expected = expected[indices]
    np.testing.assert_allclose(dataset_rows["action"], expected, atol=atol, rtol=0.0)
    local_vel = dataset_rows["action"][:, 0:2] / max(float(control_dt), 1e-6)
    np.testing.assert_allclose(local_vel, mimic[indices, 0:2], atol=atol, rtol=0.0)
    return int(expected.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local-delta v2 LeRobot data against source npz files.")
    parser.add_argument("--backend", choices=("sonic", "twist2"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_root = args.input_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    npz_paths = find_npz_files(input_root)
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found under {input_root}")

    rows_by_episode = load_lerobot_rows(dataset_root)
    fps = load_lerobot_fps(dataset_root)
    verify_lerobot_clock(rows_by_episode, fps=fps, atol=args.atol)

    if len(rows_by_episode) < len(npz_paths):
        raise AssertionError(
            f"Dataset has fewer episodes than source paths: {len(rows_by_episode)} < {len(npz_paths)}"
        )

    total_frames = 0
    for episode_index, npz_path in enumerate(npz_paths):
        rows = rows_by_episode[episode_index]
        if args.backend == "sonic":
            frames = verify_sonic_episode(npz_path=npz_path, dataset_rows=rows, atol=args.atol)
        else:
            frames = verify_twist2_episode(npz_path=npz_path, dataset_rows=rows, fps=fps, atol=args.atol)
        total_frames += frames
        logging.info("verified episode %d: %s (%d frames)", episode_index, npz_path.relative_to(input_root), frames)

    logging.info("Verified %d %s episodes, %d frames", len(npz_paths), args.backend, total_frames)


if __name__ == "__main__":
    main()
