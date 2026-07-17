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


def plane_z(coef, xy):
    return coef[0] * xy[..., 0] + coef[1] * xy[..., 1] + coef[2]


def load_world_mesh(stage_path: Path, mesh_path: str):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(stage_path))
    if not stage:
        raise RuntimeError(f"Could not open USD stage: {stage_path}")
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim:
        raise RuntimeError(f"Mesh prim not found: {mesh_path}")
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    inds = mesh.GetFaceVertexIndicesAttr().Get()
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
    return world, np.asarray(tris, dtype=np.int64)


def face_geometry(vertices, faces):
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normal = np.cross(e1, e2)
    norm = np.linalg.norm(normal, axis=1)
    valid = norm > 1e-12
    unit = np.zeros_like(normal)
    unit[valid] = normal[valid] / norm[valid, None]
    area = 0.5 * norm
    centroid = tri.mean(axis=1)
    return centroid, unit, area, valid


def mask_lookup(centroid, grid_mask, x_grid, y_grid, grid):
    x0 = float(x_grid[0, 0])
    y0 = float(y_grid[0, 0])
    ix = np.floor((centroid[:, 0] - x0) / grid).astype(np.int64)
    iy = np.floor((centroid[:, 1] - y0) / grid).astype(np.int64)
    in_grid = (ix >= 0) & (ix < grid_mask.shape[1]) & (iy >= 0) & (iy < grid_mask.shape[0])
    out = np.zeros(len(centroid), dtype=bool)
    out[in_grid] = grid_mask[iy[in_grid], ix[in_grid]]
    return out


def dilate_mask(mask, radius_m, grid):
    cells = max(0, int(round(radius_m / grid)))
    if cells <= 0:
        return mask.copy()
    yy, xx = np.ogrid[-cells : cells + 1, -cells : cells + 1]
    disk = (xx * xx + yy * yy) <= cells * cells
    return ndimage.binary_dilation(mask, structure=disk)


def connected_candidate_faces(faces, candidate, seed_faces):
    cand_idx = np.flatnonzero(candidate)
    if len(cand_idx) == 0:
        return np.zeros_like(candidate)
    cand_set = set(int(i) for i in cand_idx)
    edge_to_faces = {}
    for fi in cand_idx:
        f = faces[fi]
        edges = (
            tuple(sorted((int(f[0]), int(f[1])))),
            tuple(sorted((int(f[1]), int(f[2])))),
            tuple(sorted((int(f[2]), int(f[0])))),
        )
        for edge in edges:
            edge_to_faces.setdefault(edge, []).append(int(fi))
    adj = {int(fi): [] for fi in cand_idx}
    for linked in edge_to_faces.values():
        if len(linked) < 2:
            continue
        for a in linked:
            adj[a].extend(b for b in linked if b != a)
    seeds = [int(i) for i in np.flatnonzero(seed_faces & candidate)]
    if not seeds:
        return np.zeros_like(candidate)
    out = np.zeros_like(candidate)
    q = deque(seeds)
    for s in seeds:
        out[s] = True
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt in cand_set and not out[nxt]:
                out[nxt] = True
                q.append(nxt)
    return out


def save_mask_panel(data, out_path, start_xy, extent):
    connected = data["connected"].astype(bool)
    reachable = data["reachable"].astype(bool)
    valid_floor = data["valid_floor"].astype(bool)
    residual = data["residual_grid"]
    clearance = data["clearance"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=150)
    panels = [
        (valid_floor.astype(float), "Raw floor cells near fitted plane", "Greys", None, None),
        (connected.astype(float) + reachable.astype(float), "Connected floor + G1 reachable", "viridis", None, None),
        (residual * 1000.0, "Original floor abs residual to fitted plane (mm)", "magma", 0, 150),
        (clearance, "Distance to obstacle / non-walkable boundary (m)", "cividis", 0, 1.5),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.contour(reachable.astype(float), levels=[0.5], origin="lower", extent=extent, colors=["#22c55e"], linewidths=1.2)
        ax.scatter([start_xy[0]], [start_xy[1]], c="cyan", s=35, marker="x")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_split_comparison(centroid, current_remove, graph_remove, reachable, x_grid, y_grid, start_xy, out_path):
    rng = np.random.default_rng(13)
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    current_only = current_remove & ~graph_remove
    graph_only = graph_remove & ~current_remove
    both = current_remove & graph_remove
    keep = ~(current_remove | graph_remove)
    samples = []
    for mask, limit in ((keep, 60000), (both, 60000), (current_only, 60000), (graph_only, 60000)):
        idx = np.flatnonzero(mask)
        if len(idx) > limit:
            idx = rng.choice(idx, size=limit, replace=False)
        samples.append(idx)
    fig, ax = plt.subplots(figsize=(11, 10), dpi=180)
    ax.imshow(reachable.astype(float), origin="lower", extent=extent, cmap="Greens", alpha=0.28, interpolation="nearest")
    ax.scatter(centroid[samples[0], 0], centroid[samples[0], 1], s=0.12, c="#6b7280", alpha=0.16, label="kept/other faces")
    ax.scatter(centroid[samples[1], 0], centroid[samples[1], 1], s=0.55, c="#2563eb", alpha=0.80, label="removed by both")
    ax.scatter(centroid[samples[2], 0], centroid[samples[2], 1], s=1.1, c="#ef4444", alpha=0.90, label="current-only removal")
    ax.scatter(centroid[samples[3], 0], centroid[samples[3], 1], s=1.1, c="#f59e0b", alpha=0.90, label="graph-only removal")
    ax.scatter([start_xy[0]], [start_xy[1]], c="cyan", s=55, marker="x", label="start")
    ax.set_aspect("equal")
    ax.set_title("Collision split comparison: current centroid rule vs face-graph connected prototype")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", markerscale=5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_effect_panel(data, current_remove, graph_remove, centroid, area, out_path):
    reachable = data["reachable"].astype(bool)
    residual = data["residual_grid"] * 1000.0
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    start_xy = data["start_xy"]
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    improvement = np.where(reachable, residual, np.nan)
    high_conf = reachable & np.isfinite(residual) & (residual <= 100.0)
    low_conf = reachable & np.isfinite(residual) & (residual > 100.0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=160)
    im0 = axes[0].imshow(improvement, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=150, interpolation="nearest")
    axes[0].contour(reachable.astype(float), levels=[0.5], origin="lower", extent=extent, colors=["#22c55e"], linewidths=1.0)
    axes[0].set_title("Before repair: floor residual in reachable area (mm)")
    fig.colorbar(im0, ax=axes[0], shrink=0.78)
    conf_img = np.zeros((*reachable.shape, 4), dtype=np.float32)
    conf_img[high_conf] = (0.1, 0.75, 0.25, 0.85)
    conf_img[low_conf] = (0.95, 0.1, 0.1, 0.90)
    axes[1].imshow(conf_img, origin="lower", extent=extent, interpolation="nearest")
    axes[1].set_title("Repair confidence: green <=100mm, red >100mm residual")
    rng = np.random.default_rng(17)
    idx_cur = np.flatnonzero(current_remove)
    idx_graph = np.flatnonzero(graph_remove)
    if len(idx_cur) > 50000:
        idx_cur = rng.choice(idx_cur, 50000, replace=False)
    if len(idx_graph) > 50000:
        idx_graph = rng.choice(idx_graph, 50000, replace=False)
    axes[2].scatter(centroid[idx_cur, 0], centroid[idx_cur, 1], s=0.6, c="#ef4444", alpha=0.55, label="current removed")
    axes[2].scatter(centroid[idx_graph, 0], centroid[idx_graph, 1], s=0.35, c="#2563eb", alpha=0.65, label="graph connected")
    axes[2].set_title("Removed original floor faces over scene XY")
    axes[2].legend(loc="upper right", markerscale=5)
    for ax in axes:
        ax.scatter([start_xy[0]], [start_xy[1]], c="cyan", s=40, marker="x")
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    fig.suptitle(f"Repair effect overview: current removed area={area[current_remove].sum():.2f}m^2, graph connected={area[graph_remove].sum():.2f}m^2")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(out_dir, meta):
    md = f"""# Real Scene Collision Repair Visual Report

## Purpose

This report records where the scene collision was modified and what the modification changes geometrically. It is intended to complement first-person standing videos with whole-scene, top-down diagnostics.

## Inputs

- Scene: `{meta['scene']}`
- Mesh path: `{meta['mesh_path']}`
- Masks: `{meta['masks']}`
- Mask key: `{meta['mask_key']}`
- Fitted plane: `z = {meta['plane_coef'][0]:.9f}*x + {meta['plane_coef'][1]:.9f}*y + {meta['plane_coef'][2]:.9f}`

## Current Split vs Face-Graph Prototype

| Metric | Current centroid split | Face-graph connected prototype |
| --- | ---: | ---: |
| Removed triangles | {meta['current_removed_triangles']} | {meta['graph_removed_triangles']} |
| Removed area | {meta['current_removed_area_m2']:.3f} m^2 | {meta['graph_removed_area_m2']:.3f} m^2 |
| Current-only triangles | {meta['current_only_triangles']} | - |
| Graph-only triangles | - | {meta['graph_only_triangles']} |

Interpretation:

- `current centroid split` is the already generated repaired-collision asset rule.
- `face-graph connected prototype` only removes candidate floor faces that are connected to the start-floor component in the original mesh face graph.
- If current-only regions appear around furniture, walls, or disconnected floor islands, graph-connected split is safer.
- If graph-connected removes too little, the original mesh may be disconnected by reconstruction cracks and needs grid-assisted bridging or small-gap stitching.

## Figures

1. `01_floor_masks_and_residuals.png`: floor masks, reachable area, residual heatmap, clearance.
2. `02_split_current_vs_graph.png`: current removed faces versus graph-connected prototype.
3. `03_repair_effect_overview.png`: before-repair residual, high/low confidence repair zones, removed face overview.

## Key Numbers

- Reachable cells: {meta['reachable_cells']}
- Reachable area: {meta['reachable_area_m2']:.3f} m^2
- Reachable residual p50/p90/p95/max: {meta['reachable_residual_mm_p50']:.1f} / {meta['reachable_residual_mm_p90']:.1f} / {meta['reachable_residual_mm_p95']:.1f} / {meta['reachable_residual_mm_max']:.1f} mm
- High-confidence reachable cells (<=100mm residual): {meta['high_conf_cells']}
- Low-confidence reachable cells (>100mm residual): {meta['low_conf_cells']}

## Recommended Next Asset Trial

Generate a second repaired wrapper using the face-graph connected split if visual inspection shows current-only deletion outside the true start-connected floor. Then run the same static-ref A/B and add feet contact / foot height metrics.
"""
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--mesh_path", default="/World/mesh")
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--mask_key", choices=("reachable", "connected"), default="reachable")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--delete_margin", type=float, default=0.10)
    parser.add_argument("--normal_deg", type=float, default=25.0)
    parser.add_argument("--plane_dist", type=float, default=0.16)
    parser.add_argument("--seed_radius", type=float, default=0.75)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.masks)
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    grid = float(data["grid"])
    start_xy = data["start_xy"]
    coef = data["plane_coef"]
    base_mask = data[args.mask_key].astype(bool)
    delete_mask = dilate_mask(base_mask, args.delete_margin, grid)
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]

    vertices, faces = load_world_mesh(args.scene, args.mesh_path)
    centroid, normal, area, valid = face_geometry(vertices, faces)
    in_delete_xy = mask_lookup(centroid, delete_mask, x_grid, y_grid, grid)
    normal_cos = math.cos(math.radians(args.normal_deg))
    horizontal = valid & (np.abs(normal[:, 2]) >= normal_cos)
    residual_face = np.abs(centroid[:, 2] - plane_z(coef, centroid[:, :2]))
    near_floor_plane = residual_face <= args.plane_dist
    current_remove = in_delete_xy & horizontal & near_floor_plane
    d_start = np.linalg.norm(centroid[:, :2] - start_xy[None, :], axis=1)
    seed_faces = current_remove & (d_start <= args.seed_radius)
    if seed_faces.sum() == 0 and current_remove.sum() > 0:
        nearest = np.flatnonzero(current_remove)[np.argmin(d_start[current_remove])]
        seed_faces[nearest] = True
    graph_remove = connected_candidate_faces(faces, current_remove, seed_faces)

    residual = data["residual_grid"]
    reachable = data["reachable"].astype(bool)
    reachable_residual = residual[reachable & np.isfinite(residual)] * 1000.0
    meta = {
        "scene": str(args.scene),
        "mesh_path": args.mesh_path,
        "masks": str(args.masks),
        "mask_key": args.mask_key,
        "plane_coef": coef.tolist(),
        "input_triangles": int(len(faces)),
        "current_removed_triangles": int(current_remove.sum()),
        "graph_removed_triangles": int(graph_remove.sum()),
        "current_removed_area_m2": float(area[current_remove].sum()),
        "graph_removed_area_m2": float(area[graph_remove].sum()),
        "current_only_triangles": int((current_remove & ~graph_remove).sum()),
        "graph_only_triangles": int((graph_remove & ~current_remove).sum()),
        "reachable_cells": int(reachable.sum()),
        "reachable_area_m2": float(reachable.sum() * grid * grid),
        "reachable_residual_mm_p50": float(np.percentile(reachable_residual, 50)),
        "reachable_residual_mm_p90": float(np.percentile(reachable_residual, 90)),
        "reachable_residual_mm_p95": float(np.percentile(reachable_residual, 95)),
        "reachable_residual_mm_max": float(np.max(reachable_residual)),
        "high_conf_cells": int((reachable & np.isfinite(residual) & (residual <= 0.10)).sum()),
        "low_conf_cells": int((reachable & np.isfinite(residual) & (residual > 0.10)).sum()),
        "outputs": [
            "01_floor_masks_and_residuals.png",
            "02_split_current_vs_graph.png",
            "03_repair_effect_overview.png",
            "collision_repair_visual_report.json",
            "REPORT.md",
        ],
    }

    save_mask_panel(data, args.out_dir / "01_floor_masks_and_residuals.png", start_xy, extent)
    save_split_comparison(
        centroid,
        current_remove,
        graph_remove,
        reachable,
        x_grid,
        y_grid,
        start_xy,
        args.out_dir / "02_split_current_vs_graph.png",
    )
    save_effect_panel(data, current_remove, graph_remove, centroid, area, args.out_dir / "03_repair_effect_overview.png")
    (args.out_dir / "collision_repair_visual_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(args.out_dir, meta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
