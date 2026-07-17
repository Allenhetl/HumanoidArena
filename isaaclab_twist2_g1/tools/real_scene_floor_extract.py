#!/usr/bin/env python3
import argparse
import json
import math
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def load_usd_mesh(stage_path: Path, mesh_path: str):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(stage_path))
    if not stage:
        raise RuntimeError(f"Could not open USD stage: {stage_path}")

    prim = stage.GetPrimAtPath(mesh_path)
    if not prim:
        candidates = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
        if not candidates:
            raise RuntimeError(f"No Mesh prims found in {stage_path}")
        prim = max(candidates, key=lambda p: len(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get() or []))
        mesh_path = str(prim.GetPath())

    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    inds = mesh.GetFaceVertexIndicesAttr().Get()
    if pts is None or counts is None or inds is None:
        raise RuntimeError(f"Mesh {mesh_path} is missing points/faces")

    local = np.asarray([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world = np.empty_like(local)
    for i, p in enumerate(local):
        q = mat.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
        world[i] = (q[0], q[1], q[2])

    counts = np.asarray(counts, dtype=np.int64)
    inds = np.asarray(inds, dtype=np.int64)
    tris = []
    cursor = 0
    for count in counts:
        face = inds[cursor : cursor + count]
        cursor += count
        if count < 3:
            continue
        for j in range(1, count - 1):
            tris.append([face[0], face[j], face[j + 1]])
    if not tris:
        raise RuntimeError(f"Mesh {mesh_path} has no triangulatable faces")
    return mesh_path, world, np.asarray(tris, dtype=np.int64)


def face_geometry(vertices, faces):
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normal = np.cross(e1, e2)
    twice_area = np.linalg.norm(normal, axis=1)
    valid = twice_area > 1e-12
    unit = np.zeros_like(normal)
    unit[valid] = normal[valid] / twice_area[valid, None]
    area = 0.5 * twice_area
    centroid = tri.mean(axis=1)
    return tri, centroid, unit, area, valid


def fit_plane(points):
    # z = ax + by + c
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    coef, *_ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    residual = points[:, 2] - A @ coef
    mad = np.median(np.abs(residual - np.median(residual))) if len(residual) else 0.0
    keep = np.abs(residual) <= max(0.02, 3.0 * 1.4826 * mad)
    if keep.sum() >= 8 and keep.sum() < len(points):
        coef, *_ = np.linalg.lstsq(A[keep], points[keep, 2], rcond=None)
        residual = points[:, 2] - A @ coef
    return coef, residual


def plane_z(coef, xy):
    return coef[0] * xy[..., 0] + coef[1] * xy[..., 1] + coef[2]


def flood(mask, seed):
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    if seed is None:
        return out
    sy, sx = seed
    if not (0 <= sy < h and 0 <= sx < w) or not mask[sy, sx]:
        return out
    q = deque([(sy, sx)])
    out[sy, sx] = True
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    q.append((ny, nx))
    return out


def nearest_true_seed(mask, xy_grid, start_xy, max_dist):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    pts = xy_grid[ys, xs]
    d = np.linalg.norm(pts - start_xy[None, :], axis=1)
    i = int(np.argmin(d))
    if d[i] > max_dist:
        return None
    return int(ys[i]), int(xs[i])


def save_mask(mask, title, path, extent, start_xy, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    im = ax.imshow(mask, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
    ax.scatter([start_xy[0]], [start_xy[1]], c="red", s=45, marker="x", label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_heat(arr, title, path, extent, start_xy, label, cmap="magma", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    im = ax.imshow(
        arr,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter([start_xy[0]], [start_xy[1]], c="red", s=45, marker="x", label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_overlay(connected, reachable, residual_grid, title, path, extent, start_xy):
    img = np.zeros((*connected.shape, 4), dtype=np.float32)
    img[connected] = (1.0, 0.85, 0.05, 0.75)
    img[reachable] = (0.05, 0.85, 0.25, 0.90)
    high_res = connected & np.isfinite(residual_grid) & (residual_grid > 0.12)
    img[high_res] = (0.95, 0.05, 0.05, 0.95)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
    ax.scatter([start_xy[0]], [start_xy[1]], c="red", s=45, marker="x", label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--mesh_path", default="/World/mesh")
    parser.add_argument("--start_x", type=float, default=1.5)
    parser.add_argument("--start_y", type=float, default=4.0)
    parser.add_argument("--grid", type=float, default=0.05)
    parser.add_argument("--seed_radius", type=float, default=0.5)
    parser.add_argument("--normal_deg", type=float, default=20.0)
    parser.add_argument("--plane_dist", type=float, default=0.03)
    parser.add_argument("--step_height", type=float, default=0.05)
    parser.add_argument("--robot_clearance_radius", type=float, default=0.30)
    parser.add_argument("--morph_close_radius", type=float, default=0.10)
    parser.add_argument("--bbox_margin", type=float, default=0.25)
    parser.add_argument("--seed_layer", choices=("lowest", "dominant"), default="lowest")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_xy = np.array([args.start_x, args.start_y], dtype=np.float64)

    mesh_path, vertices, faces = load_usd_mesh(args.asset, args.mesh_path)
    tri, centroid, normal, area, valid = face_geometry(vertices, faces)
    normal_cos = math.cos(math.radians(args.normal_deg))
    horizontal = valid & (np.abs(normal[:, 2]) >= normal_cos)

    d_start = np.linalg.norm(centroid[:, :2] - start_xy[None, :], axis=1)
    seed_faces = horizontal & (d_start <= args.seed_radius)
    if seed_faces.sum() < 10:
        raise RuntimeError(f"Too few seed horizontal faces near start: {seed_faces.sum()}")

    # Pick a height layer around the start. In this scene the ceiling often has
    # larger projected area than the floor, so the default is the lowest
    # significant horizontal layer rather than the dominant one.
    seed_z_faces = centroid[seed_faces, 2]
    seed_area_faces = area[seed_faces]
    z_min = math.floor(seed_z_faces.min() / 0.02) * 0.02
    z_max = math.ceil(seed_z_faces.max() / 0.02) * 0.02
    edges = np.arange(z_min, z_max + 0.02, 0.02)
    hist, edges = np.histogram(seed_z_faces, bins=edges, weights=seed_area_faces)
    significant = np.where(hist >= max(float(hist.max()) * 0.05, 0.01))[0]
    if len(significant) == 0:
        layer_idx = int(np.argmax(hist))
    elif args.seed_layer == "lowest":
        layer_idx = int(significant[0])
    else:
        layer_idx = int(np.argmax(hist))
    z0 = 0.5 * (edges[layer_idx] + edges[layer_idx + 1])
    seed_layer_faces = seed_faces & (np.abs(centroid[:, 2] - z0) <= 0.08)
    seed_points = np.concatenate([tri[seed_layer_faces].reshape(-1, 3), centroid[seed_layer_faces]], axis=0)
    seed_points = seed_points[np.abs(seed_points[:, 2] - z0) <= 0.08]
    if len(seed_points) < 12:
        raise RuntimeError("Too few seed points after height-layer filtering")

    coef, seed_residual = fit_plane(seed_points)
    slope_deg = math.degrees(math.atan(math.sqrt(coef[0] * coef[0] + coef[1] * coef[1])))

    centroid_plane_z = plane_z(coef, centroid[:, :2])
    residual = centroid[:, 2] - centroid_plane_z
    floor_face = horizontal & (np.abs(residual) <= args.plane_dist)

    x0 = math.floor((vertices[:, 0].min() - args.bbox_margin) / args.grid) * args.grid
    x1 = math.ceil((vertices[:, 0].max() + args.bbox_margin) / args.grid) * args.grid
    y0 = math.floor((vertices[:, 1].min() - args.bbox_margin) / args.grid) * args.grid
    y1 = math.ceil((vertices[:, 1].max() + args.bbox_margin) / args.grid) * args.grid
    xs = np.arange(x0, x1 + args.grid * 0.5, args.grid)
    ys = np.arange(y0, y1 + args.grid * 0.5, args.grid)
    nx, ny = len(xs), len(ys)
    x_grid, y_grid = np.meshgrid(xs, ys)
    xy_grid = np.stack([x_grid, y_grid], axis=-1)

    valid_floor = np.zeros((ny, nx), dtype=bool)
    residual_grid = np.full((ny, nx), np.nan)
    height_grid = np.full((ny, nx), np.nan)
    count_grid = np.zeros((ny, nx), dtype=np.int32)

    ix = np.floor((centroid[floor_face, 0] - x0) / args.grid).astype(np.int64)
    iy = np.floor((centroid[floor_face, 1] - y0) / args.grid).astype(np.int64)
    cres = np.abs(residual[floor_face])
    cz = centroid[floor_face, 2]
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    for cx, cy, rr, zz in zip(ix[ok], iy[ok], cres[ok], cz[ok]):
        valid_floor[cy, cx] = True
        count_grid[cy, cx] += 1
        residual_grid[cy, cx] = rr if np.isnan(residual_grid[cy, cx]) else min(residual_grid[cy, cx], rr)
        height_grid[cy, cx] = zz if np.isnan(height_grid[cy, cx]) else 0.5 * (height_grid[cy, cx] + zz)

    close_cells = max(0, int(round(args.morph_close_radius / args.grid)))
    if close_cells > 0:
        yy, xx = np.ogrid[-close_cells : close_cells + 1, -close_cells : close_cells + 1]
        disk = (xx * xx + yy * yy) <= close_cells * close_cells
        connect_floor = ndimage.binary_closing(valid_floor, structure=disk)
    else:
        connect_floor = valid_floor.copy()

    start_seed = nearest_true_seed(connect_floor, xy_grid, start_xy, max_dist=0.75)
    connected = flood(connect_floor, start_seed)

    filled = ndimage.binary_fill_holes(connected)
    holes = filled & ~connected
    # Only keep hole candidates that are not huge exterior cavities.
    labels, n_labels = ndimage.label(holes)
    hole_candidates = np.zeros_like(holes)
    hole_areas = []
    for label in range(1, n_labels + 1):
        comp = labels == label
        area_m2 = float(comp.sum() * args.grid * args.grid)
        if area_m2 <= 0.25:
            hole_candidates |= comp
            hole_areas.append(area_m2)

    clearance = ndimage.distance_transform_edt(connected) * args.grid
    reachable_seed_mask = connected & (clearance >= args.robot_clearance_radius)
    reachable_seed = nearest_true_seed(reachable_seed_mask, xy_grid, start_xy, max_dist=1.0)
    reachable = flood(reachable_seed_mask, reachable_seed)

    extent = [x0, x1, y0, y1]
    save_mask(valid_floor.astype(float), "All floor cells near fitted start plane", args.out_dir / "floor_seed_patch.png", extent, start_xy)
    save_mask(connected.astype(float), "Connected floor region from start", args.out_dir / "floor_connected_region_mask.png", extent, start_xy)
    res_mm = residual_grid * 1000.0
    save_heat(
        res_mm,
        "Residual to fitted floor plane (mm)",
        args.out_dir / "floor_plane_residual_heatmap.png",
        extent,
        start_xy,
        "mm",
        vmin=0,
        vmax=np.nanpercentile(res_mm[connected], 95) if np.any(connected & np.isfinite(res_mm)) else None,
    )
    save_mask(hole_candidates.astype(float), "Small hole candidates inside connected floor", args.out_dir / "floor_hole_candidate_map.png", extent, start_xy)
    save_heat(clearance, "Distance to floor boundary / obstacle proxy", args.out_dir / "distance_to_obstacle_map.png", extent, start_xy, "m", cmap="viridis", vmin=0, vmax=1.0)
    save_mask(reachable.astype(float), "G1 reachable floor mask after clearance erosion", args.out_dir / "g1_reachable_mask.png", extent, start_xy)
    save_overlay(
        connected,
        reachable,
        residual_grid,
        "Floor overlay: yellow=connected, green=G1 reachable, red=residual>12cm",
        args.out_dir / "floor_reachable_overlay.png",
        extent,
        start_xy,
    )

    np.savez_compressed(
        args.out_dir / "floor_masks.npz",
        x_grid=x_grid,
        y_grid=y_grid,
        connected=connected,
        reachable=reachable,
        valid_floor=valid_floor,
        connect_floor=connect_floor,
        residual_grid=residual_grid,
        height_grid=height_grid,
        clearance=clearance,
        plane_coef=coef,
        start_xy=start_xy,
        grid=np.array(args.grid, dtype=np.float64),
        extent=np.array(extent, dtype=np.float64),
    )

    connected_res = residual_grid[connected & np.isfinite(residual_grid)]
    report = {
        "asset": str(args.asset),
        "mesh_path": mesh_path,
        "start_xy": [float(args.start_x), float(args.start_y)],
        "mesh": {
            "vertices": int(len(vertices)),
            "triangles": int(len(faces)),
            "bbox_min": vertices.min(axis=0).tolist(),
            "bbox_max": vertices.max(axis=0).tolist(),
        },
        "parameters": {
            "grid_m": args.grid,
            "seed_radius_m": args.seed_radius,
            "seed_layer": args.seed_layer,
            "normal_deg": args.normal_deg,
            "plane_dist_m": args.plane_dist,
            "robot_clearance_radius_m": args.robot_clearance_radius,
            "morph_close_radius_m": args.morph_close_radius,
        },
        "seed": {
            "seed_faces": int(seed_faces.sum()),
            "seed_layer_faces": int(seed_layer_faces.sum()),
            "seed_points": int(len(seed_points)),
            "dominant_seed_z_m": float(z0),
            "height_layer_histogram": [
                {
                    "z_center_m": float(0.5 * (edges[i] + edges[i + 1])),
                    "area": float(hist[i]),
                    "selected": bool(i == layer_idx),
                }
                for i in range(len(hist))
                if hist[i] > 0
            ],
        },
        "plane": {
            "equation": {"z": "a*x + b*y + c", "a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2])},
            "slope_deg": float(slope_deg),
            "seed_residual_abs_p50_m": float(np.percentile(np.abs(seed_residual), 50)),
            "seed_residual_abs_p95_m": float(np.percentile(np.abs(seed_residual), 95)),
        },
        "floor_region": {
            "grid_shape": [int(ny), int(nx)],
            "all_floor_cells": int(valid_floor.sum()),
            "connect_floor_cells_after_closing": int(connect_floor.sum()),
            "connected_floor_cells": int(connected.sum()),
            "connected_area_m2": float(connected.sum() * args.grid * args.grid),
            "residual_abs_p50_m": float(np.percentile(connected_res, 50)) if len(connected_res) else None,
            "residual_abs_p90_m": float(np.percentile(connected_res, 90)) if len(connected_res) else None,
            "residual_abs_p95_m": float(np.percentile(connected_res, 95)) if len(connected_res) else None,
            "residual_abs_max_m": float(np.max(connected_res)) if len(connected_res) else None,
            "hole_candidate_count": int(len(hole_areas)),
            "hole_candidate_area_m2": float(sum(hole_areas)),
            "hole_candidate_areas_m2": hole_areas,
        },
        "reachable": {
            "clearance_radius_m": float(args.robot_clearance_radius),
            "reachable_cells": int(reachable.sum()),
            "reachable_area_m2": float(reachable.sum() * args.grid * args.grid),
            "start_seed_cell": list(start_seed) if start_seed else None,
            "reachable_seed_cell": list(reachable_seed) if reachable_seed else None,
        },
        "outputs": [
            "floor_seed_patch.png",
            "floor_connected_region_mask.png",
            "floor_plane_residual_heatmap.png",
            "floor_hole_candidate_map.png",
            "distance_to_obstacle_map.png",
            "g1_reachable_mask.png",
            "floor_reachable_overlay.png",
            "floor_masks.npz",
            "floor_extract_report.json",
        ],
    }
    (args.out_dir / "floor_extract_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
