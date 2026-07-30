#!/usr/bin/env python3
"""Diagnose and estimate a global Sim(3) from COLMAP to metric Odin poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GL_TO_CV_SIGN = np.array(
    [
        [1, -1, -1, 1],
        [-1, 1, 1, -1],
        [-1, 1, 1, -1],
        [1, 1, 1, 1],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Pose:
    center: np.ndarray
    rotation_wc: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--odin-transforms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--ransac-iterations", type=int, default=10000)
    parser.add_argument("--thresholds-cm", type=float, nargs="+", default=[3, 5, 10, 20])
    parser.add_argument("--holdout-block-size", type=int, default=200)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--window-stride", type=int, default=100)
    return parser.parse_args()


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_colmap_images_binary(path: Path) -> dict[str, Pose]:
    poses: dict[str, Pose] = {}
    with path.open("rb") as file:
        count = struct.unpack("<Q", file.read(8))[0]
        for _ in range(count):
            values = struct.unpack("<i4d3di", file.read(64))
            qvec = np.asarray(values[1:5], dtype=np.float64)
            tvec = np.asarray(values[5:8], dtype=np.float64)
            name_bytes = bytearray()
            while True:
                char = file.read(1)
                if char == b"\x00":
                    break
                if not char:
                    raise EOFError("Unexpected EOF while reading COLMAP image name")
                name_bytes.extend(char)
            points2d = struct.unpack("<Q", file.read(8))[0]
            file.seek(24 * points2d, 1)

            name = name_bytes.decode("utf-8")
            if name in poses:
                raise ValueError(f"Duplicate COLMAP image name: {name}")
            rotation_cw = quaternion_to_rotation(qvec)
            rotation_wc = rotation_cw.T
            poses[name] = Pose(center=-rotation_wc @ tvec, rotation_wc=rotation_wc)
    return poses


def read_colmap_images_text(path: Path) -> dict[str, Pose]:
    poses: dict[str, Pose] = {}
    expect_image = True
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        if not expect_image:
            expect_image = True
            continue
        if not line:
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed COLMAP image record: {line}")
        name = fields[9]
        if name in poses:
            raise ValueError(f"Duplicate COLMAP image name: {name}")
        qvec = np.asarray([float(value) for value in fields[1:5]])
        tvec = np.asarray([float(value) for value in fields[5:8]])
        rotation_cw = quaternion_to_rotation(qvec)
        rotation_wc = rotation_cw.T
        poses[name] = Pose(center=-rotation_wc @ tvec, rotation_wc=rotation_wc)
        expect_image = False
    if not expect_image:
        raise ValueError("COLMAP images.txt is missing the final POINTS2D line")
    return poses


def read_colmap_poses(model: Path) -> dict[str, Pose]:
    binary_path = model / "images.bin"
    text_path = model / "images.txt"
    if binary_path.exists():
        return read_colmap_images_binary(binary_path)
    if text_path.exists():
        return read_colmap_images_text(text_path)
    raise FileNotFoundError(f"No images.bin or images.txt under {model}")


def read_odin_poses(path: Path) -> dict[str, Pose]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses: dict[str, Pose] = {}
    for frame in data["frames"]:
        name = Path(frame["file_path"]).name
        if name in poses:
            raise ValueError(f"Duplicate Odin image basename: {name}")
        nerfstudio_pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if nerfstudio_pose.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 pose for {name}")
        opencv_c2w = nerfstudio_pose * GL_TO_CV_SIGN
        poses[name] = Pose(center=opencv_c2w[:3, 3], rotation_wc=opencv_c2w[:3, :3])
    return poses


def estimate_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(source) < 3:
        raise ValueError("At least three correspondences are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    variance = np.sum(source_centered * source_centered) / len(source)
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate source points")
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = target_mean - scale * rotation @ source_mean
    if scale <= 0 or np.linalg.det(rotation) < 0.999999:
        raise ValueError("Estimated an invalid Sim(3)")
    return scale, rotation, translation


def transform_points(points: np.ndarray, model: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = model
    return (scale * (rotation @ points.T)).T + translation


def residuals(source: np.ndarray, target: np.ndarray, model: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.linalg.norm(transform_points(source, model) - target, axis=1)


def ransac_sim3(
    source: np.ndarray,
    target: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    threshold: float,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[tuple[float, np.ndarray, np.ndarray], np.ndarray]:
    best_model = None
    best_inliers = np.zeros(len(source), dtype=bool)
    best_median = math.inf
    candidate_mask = np.zeros(len(source), dtype=bool)
    candidate_mask[candidate_indices] = True
    for _ in range(iterations):
        sample = rng.choice(candidate_indices, size=4, replace=False)
        centered = source[sample] - source[sample].mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values[1] < 1e-4 * singular_values[0]:
            continue
        try:
            model = estimate_sim3(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = residuals(source, target, model)
        inliers = (errors <= threshold) & candidate_mask
        count = int(inliers.sum())
        median = float(np.median(errors[inliers])) if count else math.inf
        if count > int(best_inliers.sum()) or (count == int(best_inliers.sum()) and median < best_median):
            best_model = model
            best_inliers = inliers
            best_median = median
    if best_model is None or best_inliers.sum() < 4:
        raise RuntimeError(f"RANSAC failed at threshold {threshold} m")

    for _ in range(5):
        best_model = estimate_sim3(source[best_inliers], target[best_inliers])
        errors = residuals(source, target, best_model)
        refined = (errors <= threshold) & candidate_mask
        if np.array_equal(refined, best_inliers) or refined.sum() < 4:
            break
        best_inliers = refined
    return best_model, best_inliers


def error_stats(values: np.ndarray, unit: str = "m") -> dict[str, float | int | None]:
    if len(values) == 0:
        return {
            "count": 0,
            f"median_{unit}": None,
            f"rmse_{unit}": None,
            f"p95_{unit}": None,
            f"max_{unit}": None,
        }
    return {
        "count": int(len(values)),
        f"median_{unit}": float(np.median(values)),
        f"rmse_{unit}": float(np.sqrt(np.mean(values * values))),
        f"p95_{unit}": float(np.percentile(values, 95)),
        f"max_{unit}": float(np.max(values)),
    }


def rotation_errors_degrees(
    source_rotations: np.ndarray,
    target_rotations: np.ndarray,
    world_rotation: np.ndarray,
) -> np.ndarray:
    predicted = np.einsum("ij,njk->nik", world_rotation, source_rotations)
    relative = np.einsum("nij,nkj->nik", target_rotations, predicted)
    cosines = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosines))


def model_dict(model: tuple[float, np.ndarray, np.ndarray]) -> dict[str, object]:
    scale, rotation, translation = model
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "column_vector_matrix": matrix.tolist(),
        "usd_row_vector_matrix": matrix.T.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
    }


def blocked_holdout(count: int, block_size: int, every: int) -> np.ndarray:
    if block_size <= 0 or every <= 1:
        raise ValueError("holdout block size must be positive and holdout-every must exceed one")
    block_ids = np.arange(count) // block_size
    return block_ids % every == every - 1


def rotation_quality(rotations: np.ndarray) -> dict[str, float]:
    orthogonality = np.linalg.norm(
        np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3), axis=(1, 2)
    )
    determinants = np.linalg.det(rotations)
    return {
        "orthogonality_max": float(orthogonality.max()),
        "determinant_min": float(determinants.min()),
        "determinant_max": float(determinants.max()),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    colmap = read_colmap_poses(args.colmap_model)
    odin = read_odin_poses(args.odin_transforms)

    colmap_by_basename: dict[str, tuple[str, Pose]] = {}
    for full_name, pose in colmap.items():
        basename = Path(full_name).name
        if basename in colmap_by_basename:
            raise ValueError(f"Duplicate COLMAP basename: {basename}")
        colmap_by_basename[basename] = (full_name, pose)

    names = sorted(set(colmap_by_basename) & set(odin))
    if len(names) < 4:
        raise ValueError("Fewer than four matching image names")
    source = np.stack([colmap_by_basename[name][1].center for name in names])
    target = np.stack([odin[name].center for name in names])
    source_rotations = np.stack([colmap_by_basename[name][1].rotation_wc for name in names])
    target_rotations = np.stack([odin[name].rotation_wc for name in names])

    holdout = blocked_holdout(len(names), args.holdout_block_size, args.holdout_every)
    train_indices = np.flatnonzero(~holdout)
    rng = np.random.default_rng(args.seed)

    least_squares_model = estimate_sim3(source[train_indices], target[train_indices])
    least_squares_errors = residuals(source, target, least_squares_model)
    trials = []
    for threshold_cm in args.thresholds_cm:
        threshold = threshold_cm / 100.0
        model, train_inliers = ransac_sim3(
            source,
            target,
            train_indices,
            threshold=threshold,
            iterations=args.ransac_iterations,
            rng=rng,
        )
        errors = residuals(source, target, model)
        orientation_errors = rotation_errors_degrees(source_rotations, target_rotations, model[1])
        trials.append(
            {
                "threshold_cm": threshold_cm,
                "model": model,
                "inliers": train_inliers,
                "errors": errors,
                "orientation_errors": orientation_errors,
            }
        )

    # Five centimeters is the maximum global held-out acceptance threshold.
    selected = min(trials, key=lambda trial: abs(float(trial["threshold_cm"]) - 5.0))
    selected_model = selected["model"]
    selected_errors = selected["errors"]
    selected_inliers = selected["inliers"]
    selected_orientation_errors = selected["orientation_errors"]

    windows = []
    for start in range(0, len(names) - args.window_size + 1, args.window_stride):
        stop = start + args.window_size
        window_model = estimate_sim3(source[start:stop], target[start:stop])
        window_errors = residuals(source[start:stop], target[start:stop], window_model)
        windows.append(
            {
                "start_index": start,
                "stop_index_exclusive": stop,
                "start_name": names[start],
                "stop_name": names[stop - 1],
                "model": model_dict(window_model),
                "errors": error_stats(window_errors),
            }
        )

    trial_reports = []
    for trial in trials:
        errors = trial["errors"]
        inliers = trial["inliers"]
        orientation_errors = trial["orientation_errors"]
        trial_reports.append(
            {
                "threshold_cm": trial["threshold_cm"],
                "model": model_dict(trial["model"]),
                "train_inliers": int(inliers.sum()),
                "train_inlier_fraction": float(inliers.sum() / len(train_indices)),
                "train_errors": error_stats(errors[~holdout]),
                "train_inlier_errors": error_stats(errors[inliers]),
                "holdout_errors": error_stats(errors[holdout]),
                "orientation_all": error_stats(orientation_errors, unit="deg"),
                "orientation_inliers": error_stats(orientation_errors[inliers], unit="deg"),
            }
        )

    report = {
        "inputs": {
            "colmap_model": str(args.colmap_model.resolve()),
            "odin_transforms": str(args.odin_transforms.resolve()),
        },
        "pose_conventions": {
            "colmap": "world-to-camera qvec/tvec; center=-R_cw^T*t; qvec=wxyz",
            "odin": "pre-flipped Nerfstudio pose converted with GL_TO_CV_SIGN to metric OpenCV c2w",
            "sim3": "target = scale * rotation @ source + translation",
        },
        "counts": {
            "colmap": len(colmap),
            "odin": len(odin),
            "matched": len(names),
            "missing_from_odin": sorted(set(colmap_by_basename) - set(odin)),
            "missing_from_colmap": sorted(set(odin) - set(colmap_by_basename)),
            "train": int((~holdout).sum()),
            "holdout": int(holdout.sum()),
        },
        "rotation_quality": {
            "colmap": rotation_quality(source_rotations),
            "odin": rotation_quality(target_rotations),
        },
        "trajectory": {
            "colmap_span_native": np.ptp(source, axis=0).tolist(),
            "odin_span_metric": np.ptp(target, axis=0).tolist(),
            "colmap_pca_singular_values": np.linalg.svd(source - source.mean(axis=0), compute_uv=False).tolist(),
            "odin_pca_singular_values": np.linalg.svd(target - target.mean(axis=0), compute_uv=False).tolist(),
        },
        "least_squares_train": {
            "model": model_dict(least_squares_model),
            "train_errors": error_stats(least_squares_errors[~holdout]),
            "holdout_errors": error_stats(least_squares_errors[holdout]),
        },
        "ransac_trials": trial_reports,
        "selected_threshold_cm": selected["threshold_cm"],
        "selected_model": model_dict(selected_model),
        "selected_errors": {
            "train": error_stats(selected_errors[~holdout]),
            "train_inliers": error_stats(selected_errors[selected_inliers]),
            "holdout": error_stats(selected_errors[holdout]),
            "orientation_all": error_stats(selected_orientation_errors, unit="deg"),
            "orientation_train_inliers": error_stats(selected_orientation_errors[selected_inliers], unit="deg"),
        },
        "windows": windows,
        "settings": {
            "seed": args.seed,
            "ransac_iterations": args.ransac_iterations,
            "thresholds_cm": args.thresholds_cm,
            "holdout_block_size": args.holdout_block_size,
            "holdout_every": args.holdout_every,
            "window_size": args.window_size,
            "window_stride": args.window_stride,
        },
    }
    report_path = args.output / "alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output / "correspondences.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "name",
                "holdout",
                "selected_train_inlier",
                "colmap_x",
                "colmap_y",
                "colmap_z",
                "odin_x",
                "odin_y",
                "odin_z",
                "residual_m",
                "orientation_error_deg",
            ]
        )
        for index, name in enumerate(names):
            writer.writerow(
                [
                    index,
                    name,
                    bool(holdout[index]),
                    bool(selected_inliers[index]),
                    *source[index].tolist(),
                    *target[index].tolist(),
                    float(selected_errors[index]),
                    float(selected_orientation_errors[index]),
                ]
            )

    print(json.dumps({"report": str(report_path), "selected": report["selected_model"], "errors": report["selected_errors"]}, indent=2))


if __name__ == "__main__":
    main()
