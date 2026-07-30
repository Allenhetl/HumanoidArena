#!/usr/bin/env python3
"""Refine a camera-seeded COLMAP Sim(3) against a metric LiDAR point cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="COLMAP sparse PLY")
    parser.add_argument("--target", type=Path, required=True, help="Floor-aligned LiDAR PLY")
    parser.add_argument("--camera-alignment", type=Path, required=True)
    parser.add_argument("--collision-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-range", type=float, default=0.03)
    parser.add_argument("--scale-steps", type=int, default=13)
    parser.add_argument("--trim-fraction", type=float, default=0.8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"Invalid XYZ points: {path}")
    return cloud


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def distance_stats(distances: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(distances)),
        "median_m": float(np.median(distances)),
        "rmse_m": float(np.sqrt(np.mean(distances * distances))),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
        "fraction_within_0.02m": float(np.mean(distances <= 0.02)),
        "fraction_within_0.04m": float(np.mean(distances <= 0.04)),
        "fraction_within_0.08m": float(np.mean(distances <= 0.08)),
        "fraction_within_0.15m": float(np.mean(distances <= 0.15)),
    }


def evaluate(
    source_points: np.ndarray,
    target_tree: cKDTree,
    transform: np.ndarray,
    trim_fraction: float,
) -> tuple[float, dict[str, object]]:
    transformed = transform_points(source_points, transform)
    distances, _ = target_tree.query(transformed, workers=-1)
    trim_limit = float(np.quantile(distances, trim_fraction))
    trimmed = distances[distances <= trim_limit]
    score = float(np.sqrt(np.mean(trimmed * trimmed)))
    return score, {
        "trim_fraction": trim_fraction,
        "trim_limit_m": trim_limit,
        "trimmed_rmse_m": score,
        "all": distance_stats(distances),
    }


def multiscale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    transform = initial.copy()
    levels = []
    for voxel, max_distance, iterations in [(0.20, 0.40, 80), (0.10, 0.20, 60), (0.05, 0.10, 50)]:
        source_level = source.voxel_down_sample(voxel)
        target_level = target.voxel_down_sample(voxel)
        target_level.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=50)
        )
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=max_distance)
        )
        result = o3d.pipelines.registration.registration_icp(
            source_level,
            target_level,
            max_distance,
            transform,
            estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        transform = result.transformation
        levels.append(
            {
                "voxel_m": voxel,
                "max_correspondence_m": max_distance,
                "fitness": float(result.fitness),
                "inlier_rmse_m": float(result.inlier_rmse),
            }
        )
    return transform, levels


def main() -> None:
    args = parse_args()
    if not 0 < args.trim_fraction <= 1:
        raise ValueError("trim-fraction must be in (0, 1]")
    if args.scale_steps < 3:
        raise ValueError("scale-steps must be at least three")

    args.output.mkdir(parents=True, exist_ok=True)
    source = load_cloud(args.source)
    target = load_cloud(args.target)
    camera_report = json.loads(args.camera_alignment.read_text(encoding="utf-8"))
    collision_report = json.loads(args.collision_report.read_text(encoding="utf-8"))
    colmap_to_odin = np.asarray(
        camera_report["selected_model"]["column_vector_matrix"], dtype=np.float64
    )
    odin_to_sim = np.asarray(collision_report["odin_to_sim_floor_aligned"], dtype=np.float64)
    if colmap_to_odin.shape != (4, 4) or odin_to_sim.shape != (4, 4):
        raise ValueError("Expected 4x4 alignment matrices")
    initial = odin_to_sim @ colmap_to_odin

    source_eval = source.voxel_down_sample(0.04)
    target_eval = target.voxel_down_sample(0.04)
    source_points = np.asarray(source_eval.points)
    target_tree = cKDTree(np.asarray(target_eval.points))
    initial_score, initial_metrics = evaluate(
        source_points, target_tree, initial, args.trim_fraction
    )

    trials = []
    scale_factors = np.linspace(1 - args.scale_range, 1 + args.scale_range, args.scale_steps)
    for scale_factor in scale_factors:
        trial_initial = initial.copy()
        trial_initial[:3, :3] *= scale_factor
        transform, levels = multiscale_icp(source, target, trial_initial)
        score, metrics = evaluate(source_points, target_tree, transform, args.trim_fraction)
        trials.append(
            {
                "scale_factor": float(scale_factor),
                "score": score,
                "transform": transform,
                "icp_levels": levels,
                "metrics": metrics,
            }
        )

    best = min(trials, key=lambda trial: trial["score"])
    final = best["transform"]
    linear = final[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    final_scale = float(np.mean(singular_values))
    final_rotation = linear / final_scale

    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(final)
    aligned_path = args.output / "repaired_sparse_aligned_to_sim.ply"
    if not o3d.io.write_point_cloud(str(aligned_path), aligned, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write {aligned_path}")

    report = {
        "inputs": {
            "source": {"path": str(args.source.resolve()), "sha256": sha256(args.source)},
            "target": {"path": str(args.target.resolve()), "sha256": sha256(args.target)},
            "camera_alignment": str(args.camera_alignment.resolve()),
            "collision_report": str(args.collision_report.resolve()),
        },
        "convention": "column vectors; p_sim = T_colmap_to_sim @ p_colmap",
        "initial": {
            "colmap_to_odin": colmap_to_odin.tolist(),
            "odin_to_sim": odin_to_sim.tolist(),
            "colmap_to_sim": initial.tolist(),
            "score": initial_score,
            "metrics": initial_metrics,
        },
        "scale_trials": [
            {
                "scale_factor": trial["scale_factor"],
                "score": trial["score"],
                "icp_levels": trial["icp_levels"],
                "metrics": trial["metrics"],
            }
            for trial in trials
        ],
        "selected_scale_factor": best["scale_factor"],
        "final": {
            "colmap_to_sim": final.tolist(),
            "usd_row_vector_matrix": final.T.tolist(),
            "uniform_scale": final_scale,
            "rotation": final_rotation.tolist(),
            "rotation_determinant": float(np.linalg.det(final_rotation)),
            "singular_values": singular_values.tolist(),
            "score": best["score"],
            "metrics": best["metrics"],
            "icp_levels": best["icp_levels"],
        },
        "aligned_source": {"path": str(aligned_path), "sha256": sha256(aligned_path)},
        "settings": {
            "scale_range": args.scale_range,
            "scale_steps": args.scale_steps,
            "trim_fraction": args.trim_fraction,
        },
    }
    report_path = args.output / "structural_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "initial": report["initial"], "final": report["final"]}, indent=2))


if __name__ == "__main__":
    main()
