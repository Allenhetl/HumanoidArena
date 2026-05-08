#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
for path in (_THIS_DIR, _PROJECT_ROOT / "isaaclab_twist2_g1", _PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from action_provider.vla_smpl_runtime import (  # noqa: E402
    UnifiedSMPLActionRuntime,
    build_sonic_joint29_payload,
    build_twist2_mimic_obs,
    quat_from_roll_pitch_yaw_wxyz,
    quat_to_rot6d_wxyz,
)
from smpl_lerobot_common import (  # noqa: E402
    build_sonic_actions_from_recording,
    build_twist2_actions_from_recording,
    extract_canonical_state,
    find_npz_files,
)


DEFAULT_TWIST2_INPUT_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/test/twist2"
)
DEFAULT_SONIC_INPUT_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/test/sonic"
)
DEFAULT_TWIST2_DATASET_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/test_twist2"
)
DEFAULT_SONIC_DATASET_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/test_sonic"
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify SMPL-pose VLA conversion outputs and runtime postprocessing "
            "for TWIST2 / SONIC recordings."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("twist2", "sonic", "both"),
        default="both",
        help="Which backend to verify.",
    )
    parser.add_argument("--twist2-input-root", type=Path, default=DEFAULT_TWIST2_INPUT_ROOT)
    parser.add_argument("--sonic-input-root", type=Path, default=DEFAULT_SONIC_INPUT_ROOT)
    parser.add_argument("--twist2-dataset-root", type=Path, default=DEFAULT_TWIST2_DATASET_ROOT)
    parser.add_argument("--sonic-dataset-root", type=Path, default=DEFAULT_SONIC_DATASET_ROOT)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for checks that are expected to be exact.",
    )
    parser.add_argument(
        "--twist2-xy-warn-threshold",
        type=float,
        default=0.1,
        help="Warn if TWIST2 local xy reconstruction error exceeds this value.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary in addition to human-readable output.",
    )
    return parser.parse_args()


def _load_dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    return json.loads(info_path.read_text())


def _dataset_parquet_paths(dataset_root: Path) -> list[Path]:
    paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return paths


def _column_to_ndarray(column, *, width: int) -> np.ndarray:
    if len(column) == 0:
        return np.zeros((0, width), dtype=np.float32)
    values = np.asarray(column.to_pylist(), dtype=np.float32)
    return values.reshape(-1, width)


def _load_dataset_arrays(dataset_root: Path) -> dict[str, np.ndarray]:
    episode_index_parts: list[np.ndarray] = []
    frame_index_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []

    for parquet_path in _dataset_parquet_paths(dataset_root):
        table = pq.read_table(
            parquet_path,
            columns=["episode_index", "frame_index", "observation.state", "action"],
        )
        episode_index_parts.append(table["episode_index"].to_numpy(zero_copy_only=False))
        frame_index_parts.append(table["frame_index"].to_numpy(zero_copy_only=False))
        state_width = table.schema.field("observation.state").type.list_size
        action_width = table.schema.field("action").type.list_size
        state_parts.append(_column_to_ndarray(table["observation.state"], width=state_width))
        action_parts.append(_column_to_ndarray(table["action"], width=action_width))

    return {
        "episode_index": np.concatenate(episode_index_parts, axis=0).astype(np.int64),
        "frame_index": np.concatenate(frame_index_parts, axis=0).astype(np.int64),
        "state": np.concatenate(state_parts, axis=0).astype(np.float32),
        "action": np.concatenate(action_parts, axis=0).astype(np.float32),
    }


def _valid_frame_indices(indices: np.ndarray, num_frames: int) -> tuple[np.ndarray, int]:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    valid_mask = (indices >= 0) & (indices < int(num_frames))
    return indices[valid_mask], int((~valid_mask).sum())


def _build_expected_canonical_state(data: np.lib.npyio.NpzFile, backend: str) -> np.ndarray:
    if backend == "twist2":
        qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
        qvel = np.asarray(data["robot_qvel_before_decimation"], dtype=np.float32)
    elif backend == "sonic":
        qpos = np.asarray(data["robot_qpos_before_decimation"], dtype=np.float32)
        qvel = np.asarray(data["robot_qvel_before_decimation"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    root_orientation = np.asarray(data["robot_root_orientation"], dtype=np.float32)
    return extract_canonical_state(
        data=data,
        root_orientation=root_orientation,
        joint_pos=qpos,
        joint_vel=qvel,
    ).astype(np.float32)


def _build_expected_canonical_action(data: np.lib.npyio.NpzFile, backend: str) -> np.ndarray:
    num_frames = int(np.asarray(data["num_frames"]).reshape(-1)[0]) if "num_frames" in data else None
    if num_frames is None:
        if backend == "twist2":
            num_frames = int(np.asarray(data["robot_qpos_before_decimation"]).shape[0])
        else:
            num_frames = int(np.asarray(data["robot_qpos_before_decimation"]).shape[0])

    if backend == "twist2":
        control_dt = 1.0 / float(np.median(np.asarray(data["system_control_frequency"], dtype=np.float32)))
        return build_twist2_actions_from_recording(
            data,
            num_frames=num_frames,
            control_dt=control_dt,
        ).astype(np.float32)
    if backend == "sonic":
        return build_sonic_actions_from_recording(
            data,
            num_frames=num_frames,
        ).astype(np.float32)
    raise ValueError(f"Unsupported backend: {backend}")


def verify_dataset_conversion(
    *,
    backend: str,
    input_root: Path,
    dataset_root: Path,
    atol: float,
) -> CheckResult:
    npz_paths = find_npz_files(input_root)
    if not npz_paths:
        raise FileNotFoundError(f"No npz files found under {input_root}")

    info = _load_dataset_info(dataset_root)
    dataset = _load_dataset_arrays(dataset_root)
    details = [
        f"dataset_root={dataset_root}",
        f"episodes={len(npz_paths)} total_rows={dataset['state'].shape[0]} fps={info.get('fps')}",
        (
            f"state_shape={tuple(info['features']['observation.state']['shape'])} "
            f"action_shape={tuple(info['features']['action']['shape'])}"
        ),
    ]
    warnings: list[str] = []

    max_state_diff = 0.0
    max_action_diff = 0.0

    for episode_idx, npz_path in enumerate(npz_paths):
        with np.load(npz_path, allow_pickle=True) as data:
            episode_rows = dataset["episode_index"] == episode_idx
            ds_state = dataset["state"][episode_rows]
            ds_action = dataset["action"][episode_rows]
            ds_frame_index = dataset["frame_index"][episode_rows]

            expected_state_full = _build_expected_canonical_state(data, backend)
            expected_action_full = _build_expected_canonical_action(data, backend)

            num_frames = expected_state_full.shape[0]
            valid_indices, skipped = _valid_frame_indices(data["vision_frame_indices"], num_frames)
            expected_state = expected_state_full[valid_indices]
            expected_action = expected_action_full[valid_indices]

            if ds_state.shape != expected_state.shape:
                details.append(
                    f"{npz_path.name}: dataset state shape {ds_state.shape} != expected {expected_state.shape}"
                )
                return CheckResult(
                    name=f"{backend}:dataset",
                    ok=False,
                    details=details,
                    warnings=warnings,
                )
            if ds_action.shape != expected_action.shape:
                details.append(
                    f"{npz_path.name}: dataset action shape {ds_action.shape} != expected {expected_action.shape}"
                )
                return CheckResult(
                    name=f"{backend}:dataset",
                    ok=False,
                    details=details,
                    warnings=warnings,
                )

            state_diff = float(np.max(np.abs(ds_state - expected_state))) if ds_state.size else 0.0
            action_diff = float(np.max(np.abs(ds_action - expected_action))) if ds_action.size else 0.0
            max_state_diff = max(max_state_diff, state_diff)
            max_action_diff = max(max_action_diff, action_diff)

            expected_frame_index = np.arange(ds_frame_index.shape[0], dtype=np.int64)
            if not np.array_equal(ds_frame_index, expected_frame_index):
                details.append(
                    f"{npz_path.name}: dataset frame_index is not contiguous from 0 "
                    f"(first={ds_frame_index[:5].tolist()} last={ds_frame_index[-5:].tolist()})"
                )
                return CheckResult(
                    name=f"{backend}:dataset",
                    ok=False,
                    details=details,
                    warnings=warnings,
                )

            if skipped:
                warnings.append(f"{npz_path.name}: skipped {skipped} invalid vision frame indices")

    details.append(f"max_state_abs_diff={max_state_diff:.9f}")
    details.append(f"max_action_abs_diff={max_action_diff:.9f}")
    ok = max_state_diff <= atol and max_action_diff <= atol
    return CheckResult(name=f"{backend}:dataset", ok=ok, details=details, warnings=warnings)


def _twist2_world_z_loss(robot_action_mimic: np.ndarray, control_dt: float) -> np.ndarray:
    yaw_world = 0.0
    z_terms: list[float] = []
    for row in np.asarray(robot_action_mimic, dtype=np.float32):
        yaw_world = ((yaw_world + float(row[5]) * float(control_dt) + np.pi) % (2.0 * np.pi)) - np.pi
        root_quat = quat_from_roll_pitch_yaw_wxyz(
            roll=float(row[3]),
            pitch=float(row[4]),
            yaw=yaw_world,
        )
        local_delta = np.array(
            [float(row[0]) * float(control_dt), float(row[1]) * float(control_dt), 0.0],
            dtype=np.float32,
        )
        world_delta = R.from_quat(root_quat[[1, 2, 3, 0]]).apply(local_delta)
        z_terms.append(float(world_delta[2]))
    return np.asarray(z_terms, dtype=np.float32)


def verify_twist2_postprocess(npz_paths: list[Path], atol: float, xy_warn_threshold: float) -> CheckResult:
    max_full_diff = 0.0
    max_xy_diff = 0.0
    max_joint_diff = 0.0
    max_orientation_diff = 0.0
    max_yaw_rate_diff = 0.0
    max_exact_dims_diff = 0.0
    max_world_z_loss = 0.0
    worst_recording = None

    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=True) as data:
            canonical_action = _build_expected_canonical_action(data, "twist2")
            robot_action_mimic = np.asarray(data["robot_action_mimic"], dtype=np.float32)
            control_dt = 1.0 / float(np.median(np.asarray(data["system_control_frequency"], dtype=np.float32)))

        runtime = UnifiedSMPLActionRuntime()
        rebuilt = np.stack(
            [
                build_twist2_mimic_obs(runtime_frame=runtime.step(action), control_dt=control_dt)
                for action in canonical_action
            ],
            axis=0,
        ).astype(np.float32)

        full_diff = np.abs(rebuilt - robot_action_mimic)
        full_diff_value = float(np.max(full_diff)) if rebuilt.shape[0] else 0.0
        exact_dims_diff = float(np.max(full_diff[:, 2:])) if rebuilt.shape[0] else 0.0
        xy_diff = float(np.max(full_diff[:, :2])) if rebuilt.shape[0] else 0.0
        joint_diff = float(np.max(full_diff[:, 6:35])) if rebuilt.shape[0] else 0.0
        orientation_diff = float(np.max(full_diff[:, 2:6])) if rebuilt.shape[0] else 0.0
        yaw_rate_diff = float(np.max(full_diff[:, 5:6])) if rebuilt.shape[0] else 0.0
        world_z_loss = float(np.max(np.abs(_twist2_world_z_loss(robot_action_mimic, control_dt))))

        if full_diff_value >= max_full_diff:
            worst_recording = npz_path
        max_full_diff = max(max_full_diff, full_diff_value)
        max_xy_diff = max(max_xy_diff, xy_diff)
        max_joint_diff = max(max_joint_diff, joint_diff)
        max_orientation_diff = max(max_orientation_diff, orientation_diff)
        max_yaw_rate_diff = max(max_yaw_rate_diff, yaw_rate_diff)
        max_exact_dims_diff = max(max_exact_dims_diff, exact_dims_diff)
        max_world_z_loss = max(max_world_z_loss, world_z_loss)

    details = [
        f"episodes={len(npz_paths)}",
        f"worst_recording={worst_recording}",
        f"max_abs_diff_full={max_full_diff:.9f}",
        f"max_abs_diff_xy={max_xy_diff:.9f}",
        f"max_abs_diff_root_z_roll_pitch_yaw={max_orientation_diff:.9f}",
        f"max_abs_diff_yaw_rate={max_yaw_rate_diff:.9f}",
        f"max_abs_diff_joint_pos={max_joint_diff:.9f}",
        f"max_abs_diff_dims_2_to_34={max_exact_dims_diff:.9f}",
        f"world_delta_z_loss_max={max_world_z_loss:.9f}",
    ]
    warnings: list[str] = []
    if max_xy_diff > 0.0:
        warnings.append(
            "TWIST2 local xy is not exactly reversible from the 40D canonical action: "
            "the representation stores only root_xy_delta_world, so roll/pitch-induced world z motion is dropped."
        )
    if max_xy_diff > xy_warn_threshold:
        warnings.append(
            f"TWIST2 xy reconstruction error {max_xy_diff:.6f} exceeded warning threshold {xy_warn_threshold:.6f}."
        )

    numerical_tolerance = max(atol, 5e-5)
    ok = max_exact_dims_diff <= numerical_tolerance
    return CheckResult(name="twist2:postprocess", ok=ok, details=details, warnings=warnings)


def _quat_sign_invariant_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    return np.minimum(np.linalg.norm(q1 - q2, axis=1), np.linalg.norm(q1 + q2, axis=1)).astype(np.float32)


def verify_sonic_postprocess(npz_paths: list[Path], atol: float) -> CheckResult:
    max_joint_pos_diff = 0.0
    max_joint_vel_diff = 0.0
    max_hand_diff = 0.0
    max_root_rot6d_diff = 0.0
    max_sign_invariant_quat_error = 0.0
    max_signed_quat_diff = 0.0
    worst_recording = None

    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=True) as data:
            canonical_action = _build_expected_canonical_action(data, "sonic")
            control_dt = float(np.asarray(data["meta_control_dt"]).reshape(-1)[0])
            expected_joint_pos = np.asarray(data["vla_action_joint_pos_29"], dtype=np.float32)
            expected_root_quat = np.asarray(data["human_body_quat_w"], dtype=np.float32)
            expected_hand = np.asarray(data["vla_action_hand_binary_2"], dtype=np.float32)

        runtime = UnifiedSMPLActionRuntime()
        payload_joint_pos: list[np.ndarray] = []
        payload_joint_vel: list[np.ndarray] = []
        payload_root_quat: list[np.ndarray] = []
        runtime_hand: list[np.ndarray] = []

        for action in canonical_action:
            frame = runtime.step(action)
            payload = build_sonic_joint29_payload(runtime_frame=frame, control_dt=control_dt)
            payload_joint_pos.append(payload["joint_pos"])
            payload_joint_vel.append(payload["joint_vel"])
            payload_root_quat.append(payload["body_quat_w"])
            runtime_hand.append(frame.hand_binary)

        payload_joint_pos_np = np.stack(payload_joint_pos, axis=0).astype(np.float32)
        payload_joint_vel_np = np.stack(payload_joint_vel, axis=0).astype(np.float32)
        payload_root_quat_np = np.stack(payload_root_quat, axis=0).astype(np.float32)
        runtime_hand_np = np.stack(runtime_hand, axis=0).astype(np.float32)

        expected_joint_vel = np.zeros_like(expected_joint_pos, dtype=np.float32)
        expected_joint_vel[1:] = (expected_joint_pos[1:] - expected_joint_pos[:-1]) / max(control_dt, 1e-6)

        quat_rot6d_payload = np.stack([quat_to_rot6d_wxyz(q).reshape(6) for q in payload_root_quat_np], axis=0)
        quat_rot6d_expected = np.stack([quat_to_rot6d_wxyz(q).reshape(6) for q in expected_root_quat], axis=0)

        joint_pos_diff = float(np.max(np.abs(payload_joint_pos_np - expected_joint_pos)))
        joint_vel_diff = float(np.max(np.abs(payload_joint_vel_np - expected_joint_vel)))
        hand_diff = float(np.max(np.abs(runtime_hand_np - expected_hand)))
        root_rot6d_diff = float(np.max(np.abs(quat_rot6d_payload - quat_rot6d_expected)))
        sign_invariant_quat_error = float(
            np.max(_quat_sign_invariant_error(payload_root_quat_np, expected_root_quat))
        )
        signed_quat_diff = float(np.max(np.abs(payload_root_quat_np - expected_root_quat)))

        if root_rot6d_diff >= max_root_rot6d_diff:
            worst_recording = npz_path
        max_joint_pos_diff = max(max_joint_pos_diff, joint_pos_diff)
        max_joint_vel_diff = max(max_joint_vel_diff, joint_vel_diff)
        max_hand_diff = max(max_hand_diff, hand_diff)
        max_root_rot6d_diff = max(max_root_rot6d_diff, root_rot6d_diff)
        max_sign_invariant_quat_error = max(max_sign_invariant_quat_error, sign_invariant_quat_error)
        max_signed_quat_diff = max(max_signed_quat_diff, signed_quat_diff)

    details = [
        f"episodes={len(npz_paths)}",
        f"worst_recording={worst_recording}",
        f"max_abs_diff_joint_pos={max_joint_pos_diff:.9f}",
        f"max_abs_diff_joint_vel={max_joint_vel_diff:.9f}",
        f"max_abs_diff_hand_binary={max_hand_diff:.9f}",
        f"max_abs_diff_root_rot6d={max_root_rot6d_diff:.9f}",
        f"max_sign_invariant_quat_error={max_sign_invariant_quat_error:.9f}",
    ]
    warnings: list[str] = []
    if max_signed_quat_diff > atol:
        warnings.append(
            "SONIC root quaternion may differ by sign after rot6d -> quat reconstruction; "
            "use sign-invariant quaternion error or rot6d error to judge correctness."
        )

    ok = (
        max_joint_pos_diff <= atol
        and max_joint_vel_diff <= atol
        and max_hand_diff <= atol
        and max_root_rot6d_diff <= atol
    )
    return CheckResult(name="sonic:postprocess", ok=ok, details=details, warnings=warnings)


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] {result.name}")
    for line in result.details:
        print(f"  - {line}")
    for warning in result.warnings:
        print(f"  - warning: {warning}")


def main() -> int:
    args = parse_args()
    results: list[CheckResult] = []

    if args.backend in ("twist2", "both"):
        twist2_input_root = args.twist2_input_root.expanduser().resolve()
        twist2_dataset_root = args.twist2_dataset_root.expanduser().resolve()
        results.append(
            verify_dataset_conversion(
                backend="twist2",
                input_root=twist2_input_root,
                dataset_root=twist2_dataset_root,
                atol=args.atol,
            )
        )
        twist2_npz_paths = find_npz_files(twist2_input_root)
        if not twist2_npz_paths:
            raise FileNotFoundError(f"No npz files found under {twist2_input_root}")
        results.append(
            verify_twist2_postprocess(
                twist2_npz_paths,
                atol=args.atol,
                xy_warn_threshold=args.twist2_xy_warn_threshold,
            )
        )

    if args.backend in ("sonic", "both"):
        sonic_input_root = args.sonic_input_root.expanduser().resolve()
        sonic_dataset_root = args.sonic_dataset_root.expanduser().resolve()
        results.append(
            verify_dataset_conversion(
                backend="sonic",
                input_root=sonic_input_root,
                dataset_root=sonic_dataset_root,
                atol=args.atol,
            )
        )
        sonic_npz_paths = find_npz_files(sonic_input_root)
        if not sonic_npz_paths:
            raise FileNotFoundError(f"No npz files found under {sonic_input_root}")
        results.append(verify_sonic_postprocess(sonic_npz_paths, atol=args.atol))

    for result in results:
        _print_result(result)

    if args.json:
        payload = {
            result.name: {
                "ok": result.ok,
                "details": result.details,
                "warnings": result.warnings,
            }
            for result in results
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
