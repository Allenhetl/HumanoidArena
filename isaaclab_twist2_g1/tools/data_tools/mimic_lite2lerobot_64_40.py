#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from smpl_lerobot_common import (
    _apply_heading_align_to_action,
    _resolve_sonic_heading_align_quat,
    build_action,
    build_features,
    extract_canonical_state,
    find_npz_files,
    inspect_image_shape,
    load_vision_rgb_and_indices,
    normalize_task,
    prepare_output_root,
    quat_to_rot6d_wxyz,
)


DEFAULT_INPUT_ROOT = Path(
    "./HumanoidArena/isaaclab_twist2_g1/recording_data/HSI_open_door/mimic_lite"
)
DEFAULT_OUTPUT_ROOT = Path(
    "./HumanoidArena/lerobot/outputs/HumanoidArena_datasets/HSI_open_door/mimic_lite_64_40"
)
DEFAULT_REPO_ID = "local/mimic_lite_hsi_open_door_64_40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MimicLite IsaacLab recordings into a LeRobot v3.1-style "
            "dataset with observation.state=64D and action=40D. The action uses "
            "human_raw_body_pos/body_quat_w for the root target, action_body_29dof "
            "for the 29-DoF joint target, and SONIC-compatible Pico grip_binary "
            "for hand binary labels. By default the action root rotation/xy delta "
            "is heading-aligned so each episode's initial human reference yaw is "
            "rotated into the robot's initial yaw frame (matching SONIC's "
            "canonical_action convention); use --no-align-heading-targets to keep "
            "raw world-frame yaw."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--robot-type", type=str, default="unitree_g1_smpl_pose6d_vla")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-images", action="store_true")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=10,
        help="Skip tiny/accidental episodes with fewer than this many recorded frames.",
    )
    parser.add_argument(
        "--strict-hand-binary",
        action="store_true",
        help="Fail if Pico grip_binary cannot be decoded from human_raw_controller_data_json.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help=(
            "Language instruction to write into meta/tasks.parquet after conversion, "
            "e.g. 'Open the door.'. When omitted the raw task name from the npz is kept."
        ),
    )
    parser.set_defaults(fix_gripper=True)
    parser.add_argument(
        "--fix-gripper",
        dest="fix_gripper",
        action="store_true",
        help=(
            "Patch meta/stats.json so action.hand_binary.left/right normalize to -1/+1 "
            "(q01=0, q99=1, mean=0.5, std=0.5). Matches SONIC's "
            "fix_gripper_quantile_stats.py postprocess. (default)"
        ),
    )
    parser.add_argument(
        "--no-fix-gripper",
        dest="fix_gripper",
        action="store_false",
        help="Do not patch hand-binary stats in meta/stats.json.",
    )
    parser.set_defaults(align_heading_targets=True)
    parser.add_argument(
        "--align-heading-targets",
        dest="align_heading_targets",
        action="store_true",
        help=(
            "Rotate action root orientation/xy delta into a heading-aligned frame so "
            "each episode's initial human reference yaw is aligned to the robot's "
            "initial yaw. Matches SONIC's canonical_action convention. (default)"
        ),
    )
    parser.add_argument(
        "--no-align-heading-targets",
        dest="align_heading_targets",
        action="store_false",
        help="Keep raw world-frame heading in action targets (absolute yaw).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print episode counts without writing a LeRobot dataset.",
    )
    return parser.parse_args()


def infer_fps(npz_paths: list[Path], fps_override: int | None) -> int:
    if fps_override is not None:
        return fps_override
    if not npz_paths:
        raise ValueError("No npz files found.")

    freqs = []
    for path in npz_paths[: min(len(npz_paths), 32)]:
        with np.load(path, allow_pickle=True) as data:
            if "meta_control_dt" not in data:
                continue
            control_dt = float(np.asarray(data["meta_control_dt"]).reshape(-1)[0])
            if control_dt > 0:
                freqs.append(1.0 / control_dt)

    if not freqs:
        raise ValueError("Could not infer MimicLite fps from meta_control_dt.")

    fps = int(round(float(np.median(freqs))))
    if fps <= 0:
        raise ValueError(f"Invalid inferred fps: {fps}")
    return fps


def _decode_controller_json(value) -> dict | None:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None:
        return None
    text = str(value)
    if not text or text == "null":
        return None
    try:
        decoded = json.loads(text)
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


def extract_sonic_compatible_hand_binary(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
    strict: bool,
) -> np.ndarray:
    """Return SONIC-compatible [left_grip_binary, right_grip_binary] labels.

    SONIC records Pico controller binary signals as pico_left/right_grip_binary,
    which come from LeftController/RightController.grip_binary. MimicLite stores
    the same raw controller payload in human_raw_controller_data_json, so decode
    the same fields here to keep the convention aligned.
    """

    if "vla_action_hand_binary_2" in data:
        hand = np.asarray(data["vla_action_hand_binary_2"], dtype=np.float32)
        if hand.shape == (num_frames, 2):
            return hand
    if "vla_action" in data:
        action = np.asarray(data["vla_action"], dtype=np.float32)
        if action.ndim == 2 and action.shape == (num_frames, 40):
            return action[:, 38:40].astype(np.float32)

    left = np.zeros(num_frames, dtype=np.float32)
    right = np.zeros(num_frames, dtype=np.float32)
    valid = np.zeros(num_frames, dtype=np.bool_)

    if "human_raw_controller_data_json" in data:
        raw = np.asarray(data["human_raw_controller_data_json"])
        if raw.shape[0] != num_frames:
            raise ValueError(
                "Unexpected human_raw_controller_data_json length: "
                f"{raw.shape[0]}, expected {num_frames}"
            )
        for i, value in enumerate(raw):
            payload = _decode_controller_json(value)
            if not payload:
                continue
            left_controller = payload.get("LeftController") or {}
            right_controller = payload.get("RightController") or {}
            left[i] = float(bool(left_controller.get("grip_binary", False)))
            right[i] = float(bool(right_controller.get("grip_binary", False)))
            valid[i] = True

    if strict and not np.all(valid):
        missing = int(np.size(valid) - np.count_nonzero(valid))
        raise ValueError(f"Missing Pico grip_binary controller data for {missing}/{num_frames} frames")

    return np.stack([left, right], axis=1).astype(np.float32)


def build_mimic_lite_actions_from_recording(
    data: np.lib.npyio.NpzFile,
    *,
    num_frames: int,
    strict_hand_binary: bool,
    align_heading_targets: bool = True,
) -> np.ndarray:
    if "vla_action" in data:
        action = np.asarray(data["vla_action"], dtype=np.float32)
        if action.ndim == 2 and action.shape == (num_frames, 40):
            if align_heading_targets and "vla_action_heading_aligned" not in data:
                align_quat = _resolve_sonic_heading_align_quat(
                    data,
                    num_frames=num_frames,
                    ref_quat_keys=("human_raw_body_quat_w", "human_body_quat_w"),
                )
                if align_quat is not None:
                    return _apply_heading_align_to_action(action, align_quat_wxyz=align_quat)
            return action

    required = ["human_raw_body_pos", "human_raw_body_quat_w", "action_body_29dof"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"MimicLite recording missing required action fields: {missing}")

    body_pos = np.asarray(data["human_raw_body_pos"], dtype=np.float32)
    body_quat = np.asarray(data["human_raw_body_quat_w"], dtype=np.float32)
    joint_targets = np.asarray(data["action_body_29dof"], dtype=np.float32)
    hand_binary = extract_sonic_compatible_hand_binary(
        data,
        num_frames=num_frames,
        strict=strict_hand_binary,
    )

    if body_pos.shape != (num_frames, 3):
        raise ValueError(f"Unexpected human_raw_body_pos shape: {body_pos.shape}")
    if body_quat.shape != (num_frames, 4):
        raise ValueError(f"Unexpected human_raw_body_quat_w shape: {body_quat.shape}")
    if joint_targets.shape != (num_frames, 29):
        raise ValueError(f"Unexpected action_body_29dof shape: {joint_targets.shape}")

    actions = np.zeros((num_frames, 40), dtype=np.float32)
    prev_body_pos = None
    for i in range(num_frames):
        if prev_body_pos is None:
            root_xy_delta = np.zeros((2,), dtype=np.float32)
        else:
            root_xy_delta = (body_pos[i, :2] - prev_body_pos[:2]).astype(np.float32)
        prev_body_pos = body_pos[i].copy()
        actions[i] = build_action(
            root_xy_delta=root_xy_delta,
            root_z=body_pos[i, 2],
            root_rot6d=quat_to_rot6d_wxyz(body_quat[i]).reshape(6),
            joint_pos=joint_targets[i],
            hand_binary=hand_binary[i],
        )

    if align_heading_targets:
        align_quat = _resolve_sonic_heading_align_quat(
            data,
            num_frames=num_frames,
            ref_quat_keys=("human_raw_body_quat_w", "human_body_quat_w"),
        )
        if align_quat is not None:
            actions = _apply_heading_align_to_action(actions, align_quat_wxyz=align_quat)
    return actions.astype(np.float32)


def _write_language_instruction(output_root: Path, instruction: str) -> None:
    """Overwrite meta/tasks.parquet with a single language instruction row.

    This mirrors what SONIC's write_dataset_instructions.py does so that the
    task text becomes a natural-language instruction (e.g. "Open the door.")
    instead of the raw IsaacLab task id stored in the npz.
    """
    import pandas as pd

    tasks_path = output_root / "meta" / "tasks.parquet"
    df = pd.DataFrame({"task_index": [0]}, index=pd.Index([instruction], name="task"))
    df.to_parquet(tasks_path)
    logging.info("Wrote language instruction %r to %s", instruction, tasks_path)


# --- hand-binary stats patching -------------------------------------------------
#
# Mirrors lerobot/scripts/fix_gripper_quantile_stats.py so that
# action.hand_binary.left/right (indices 38/39 in the 40-D action) normalize
# to -1/+1 regardless of the actual 0/1 ratio in the dataset.  This keeps
# normalization consistent with SONIC datasets across all training policies.
_HAND_BINARY_FEATURES = ("action.hand_binary.left", "action.hand_binary.right")
_GRIPPER_TARGET_VALUES = {
    "q01": 0.0,
    "q99": 1.0,
    "min": 0.0,
    "max": 1.0,
    "mean": 0.5,
    "std": 0.5,
}


def _patch_gripper_stats(output_root: Path) -> bool:
    """Patch meta/stats.json in-place for hand-binary dimensions.

    Returns True if the stats file was modified.
    """
    import shutil

    info_path = output_root / "meta" / "info.json"
    stats_path = output_root / "meta" / "stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        logging.warning("Cannot patch gripper stats: missing info.json or stats.json under %s", output_root)
        return False

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    action_names = info.get("features", {}).get("action", {}).get("names")
    if not isinstance(action_names, list):
        logging.warning("Cannot patch gripper stats: action.names missing in info.json")
        return False

    try:
        left_idx = action_names.index(_HAND_BINARY_FEATURES[0])
        right_idx = action_names.index(_HAND_BINARY_FEATURES[1])
    except ValueError:
        logging.warning("Cannot patch gripper stats: hand_binary feature names not found in action.names")
        return False

    action_stats = stats.get("action")
    if not isinstance(action_stats, dict):
        logging.warning("Cannot patch gripper stats: stats.json has no action section")
        return False

    required_index = max(left_idx, right_idx)
    changes: list[str] = []
    for stat_key, new_value in _GRIPPER_TARGET_VALUES.items():
        values = action_stats.get(stat_key)
        if not isinstance(values, list) or len(values) <= required_index:
            logging.warning("Cannot patch gripper stats: stats.action.%s missing or too short", stat_key)
            return False
        for idx in (left_idx, right_idx):
            old_value = values[idx]
            if old_value == new_value:
                continue
            values[idx] = new_value
            changes.append(f"action.{stat_key}[{idx}] {old_value!r} -> {new_value!r}")

    if not changes:
        logging.info("Gripper stats already patched, no changes needed")
        return False

    backup_path = stats_path.with_name(stats_path.name + ".gripper_fix.bak")
    if not backup_path.exists():
        shutil.copy2(stats_path, backup_path)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
        f.write("\n")

    for change in changes:
        logging.info("  %s", change)
    logging.info("Patched gripper stats in %s", stats_path)
    return True


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    npz_paths = find_npz_files(input_root)
    if args.limit is not None:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found under {input_root}")

    fps = infer_fps(npz_paths, args.fps)
    image_shape = inspect_image_shape(npz_paths)
    features = build_features(image_shape=image_shape, use_videos=not args.use_images)

    dataset = None
    if not args.dry_run:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        prepare_output_root(output_root, overwrite=args.overwrite)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=output_root,
            robot_type=args.robot_type,
            fps=fps,
            features=features,
            use_videos=not args.use_images,
        )

    logging.info("Found %d MimicLite episodes under %s", len(npz_paths), input_root)
    logging.info(
        "Using fps=%d image_shape=%s min_frames=%d align_heading_targets=%s",
        fps,
        image_shape,
        args.min_frames,
        args.align_heading_targets,
    )
    if not args.dry_run:
        logging.info("Writing LeRobot dataset to %s", output_root)

    total_frames = 0
    converted_episodes = 0
    skipped_tiny = 0
    skipped_empty = 0
    try:
        for episode_index, npz_path in enumerate(npz_paths, start=1):
            frames_in_episode = 0
            skipped_invalid_indices = 0

            with np.load(npz_path, allow_pickle=True) as data:
                qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
                qvel = np.asarray(data["robot_qvel_before_decimation"], dtype=np.float32)
                root_orientation = np.asarray(data["robot_root_orientation"], dtype=np.float32)
                num_frames = int(qpos.shape[0])

                if num_frames < int(args.min_frames):
                    skipped_tiny += 1
                    logging.info(
                        "[%d/%d] skip tiny episode %s frames=%d",
                        episode_index,
                        len(npz_paths),
                        npz_path.relative_to(input_root),
                        num_frames,
                    )
                    continue

                vision_rgb, vision_frame_indices = load_vision_rgb_and_indices(data, npz_path)
                obs_state = extract_canonical_state(
                    data=data,
                    root_orientation=root_orientation,
                    joint_pos=qpos,
                    joint_vel=qvel,
                )
                action = build_mimic_lite_actions_from_recording(
                    data,
                    num_frames=num_frames,
                    strict_hand_binary=args.strict_hand_binary,
                    align_heading_targets=args.align_heading_targets,
                )

                if qpos.ndim != 2 or qpos.shape[1] != 29:
                    raise ValueError(f"{npz_path} has unexpected robot_qpos_before_decimation shape: {qpos.shape}")
                if qvel.shape != qpos.shape:
                    raise ValueError(f"{npz_path} has unexpected robot_qvel_before_decimation shape: {qvel.shape}")
                if root_orientation.shape != (num_frames, 4):
                    raise ValueError(f"{npz_path} has unexpected robot_root_orientation shape: {root_orientation.shape}")
                if obs_state.shape != (num_frames, 64):
                    raise ValueError(f"{npz_path} has unexpected observation.state shape: {obs_state.shape}")
                if action.shape != (num_frames, 40):
                    raise ValueError(f"{npz_path} has unexpected action shape: {action.shape}")
                if vision_rgb.ndim != 4:
                    raise ValueError(f"{npz_path} has unexpected vision_rgb shape: {vision_rgb.shape}")
                if vision_rgb.shape[0] != len(vision_frame_indices):
                    raise ValueError(
                        f"{npz_path} has mismatched vision buffers: "
                        f"{vision_rgb.shape[0]} rgb frames vs {len(vision_frame_indices)} indices"
                    )

                task = normalize_task(data.get("task"), npz_path, input_root)
                for rgb_index, frame_index_raw in enumerate(vision_frame_indices):
                    frame_index = int(frame_index_raw)
                    if frame_index < 0 or frame_index >= num_frames:
                        skipped_invalid_indices += 1
                        continue
                    frames_in_episode += 1
                    if dataset is not None:
                        dataset.add_frame(
                            {
                                "task": task,
                                "observation.images.front": np.ascontiguousarray(vision_rgb[rgb_index]),
                                "observation.state": np.ascontiguousarray(obs_state[frame_index]),
                                "action": np.ascontiguousarray(action[frame_index]),
                            }
                        )

            if frames_in_episode == 0:
                skipped_empty += 1
                logging.warning("Skip empty episode: %s", npz_path)
                if dataset is not None:
                    dataset.clear_episode_buffer(delete_images=True)
                continue

            if skipped_invalid_indices:
                logging.warning(
                    "%s skipped %d out-of-range vision frame indices",
                    npz_path.relative_to(input_root),
                    skipped_invalid_indices,
                )

            if dataset is not None:
                dataset.save_episode()
            total_frames += frames_in_episode
            converted_episodes += 1
            logging.info(
                "[%d/%d] %s %s with %d frames",
                episode_index,
                len(npz_paths),
                "validated" if args.dry_run else "saved",
                npz_path.relative_to(input_root),
                frames_in_episode,
            )
    finally:
        if dataset is not None:
            dataset.finalize()

    logging.info(
        "Finished conversion: %d episodes, %d frames, skipped_tiny=%d, skipped_empty=%d",
        converted_episodes,
        total_frames,
        skipped_tiny,
        skipped_empty,
    )

    if args.dry_run:
        return

    if args.instruction:
        _write_language_instruction(output_root, args.instruction)

    if args.fix_gripper:
        _patch_gripper_stats(output_root)


if __name__ == "__main__":
    main()
