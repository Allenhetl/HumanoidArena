#!/usr/bin/env python3
"""Select a robot start point on a mesh floor by finding the largest flat open area.

Reads a triangle mesh PLY, identifies horizontal faces, builds a 2D grid of
walkable cells, finds the largest connected open region, and returns its centroid
as the start point candidate.

Outputs:
  - start point xy + floor z
  - a PNG heatmap of walkable density for visual confirmation
"""
import argparse
import json
import math
import os

import numpy as np
from plyfile import PlyData

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_mesh(ply_path):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    x = np.array(v["x"], dtype=np.float64)
    y = np.array(v["y"], dtype=np.float64)
    z = np.array(v["z"], dtype=np.float64)
    verts = np.column_stack([x, y, z])
    faces = np.vstack(ply["face"]["vertex_indices"]).astype(np.int64)
    return verts, faces


def compute_face_data(verts, faces):
    # face vertices
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    # normals
    edge1 = v1 - v0
    edge2 = v2 - v0
    crosses = np.cross(edge1, edge2)
    norms = np.linalg.norm(crosses, axis=1)
    safe = norms > 1e-12
    normals = np.zeros_like(crosses)
    normals[safe] = crosses[safe] / norms[safe, None]
    # area
    areas = 0.5 * norms
    # centroid
    centroids = (v0 + v1 + v2) / 3.0
    return centroids, normals, areas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_ply")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--grid_res", type=float, default=0.10, help="grid cell size in meters")
    parser.add_argument("--horiz_deg", type=float, default=20.0, help="max face tilt from horizontal")
    parser.add_argument("--z_band", type=float, default=0.30, help="height band around floor for walkable")
    parser.add_argument("--open_radius", type=float, default=0.50, help="radius to check openness around cell")
    parser.add_argument("--min_open_frac", type=float, default=0.60, help="min fraction of open cells in radius")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading mesh: {args.mesh_ply}")
    verts, faces = load_mesh(args.mesh_ply)
    print(f"  verts: {len(verts)}, faces: {len(faces)}")

    centroids, normals, areas = compute_face_data(verts, faces)

    # bbox
    xmin, ymin, zmin = verts.min(axis=0)
    xmax, ymax, zmax = verts.max(axis=0)
    print(f"  bbox: x[{xmin:.2f},{xmax:.2f}] y[{ymin:.2f},{ymax:.2f}] z[{zmin:.2f},{zmax:.2f}]")

    # horizontal faces (normal mostly up or down)
    cos_thresh = math.cos(math.radians(args.horiz_deg))
    horiz_mask = np.abs(normals[:, 2]) >= cos_thresh
    print(f"  horizontal faces: {horiz_mask.sum()} / {len(faces)}")

    horiz_z = centroids[horiz_mask, 2]
    if len(horiz_z) == 0:
        print("ERROR: no horizontal faces found")
        return

    # Find floor layer: lowest significant z cluster
    # Use histogram to find the lowest peak
    z_hist, z_edges = np.histogram(horiz_z, bins=200, range=(zmin, zmax))
    # find lowest bin with significant count
    threshold = z_hist.max() * 0.05
    floor_bins = np.where(z_hist > threshold)[0]
    if len(floor_bins) == 0:
        floor_z = float(np.median(horiz_z))
    else:
        floor_z = float((z_edges[floor_bins[0]] + z_edges[floor_bins[0] + 1]) / 2.0)
    print(f"  floor z estimate: {floor_z:.3f}")

    # faces in floor band
    floor_mask = horiz_mask & (np.abs(centroids[:, 2] - floor_z) <= args.z_band)
    print(f"  floor-band faces: {floor_mask.sum()}")

    # Build 2D grid
    gx = math.ceil((xmax - xmin) / args.grid_res)
    gy = math.ceil((ymax - ymin) / args.grid_res)
    # density: area of horizontal floor faces per cell
    density = np.zeros((gy, gx), dtype=np.float64)
    cx = centroids[floor_mask, 0]
    cy = centroids[floor_mask, 1]
    ca = areas[floor_mask]
    gi = ((cx - xmin) / args.grid_res).astype(int)
    gj = ((cy - ymin) / args.grid_res).astype(int)
    gi = np.clip(gi, 0, gx - 1)
    gj = np.clip(gj, 0, gy - 1)
    np.add.at(density, (gj, gi), ca)

    # walkable cells: density above a fraction of a full cell area
    cell_area = args.grid_res ** 2
    walkable = density > (cell_area * 0.15)
    print(f"  walkable cells: {walkable.sum()} / {gx * gy}")

    # Openness: for each walkable cell, count walkable neighbors within radius
    r_cells = max(1, int(args.open_radius / args.grid_res))
    open_count = np.zeros_like(density, dtype=np.int32)
    total_count = np.zeros_like(density, dtype=np.int32)
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            if dx * dx + dy * dy > r_cells * r_cells:
                continue
            shifted = np.roll(np.roll(walkable, -dy, axis=0), -dx, axis=1)
            open_count += shifted.astype(np.int32)
            total_count += 1
    open_frac = np.where(total_count > 0, open_count / total_count, 0.0)
    open_cells = walkable & (open_frac >= args.min_open_frac)
    print(f"  open cells: {open_cells.sum()}")

    # Find largest connected component of open cells
    visited = np.zeros_like(open_cells, dtype=bool)
    largest = None
    largest_size = 0
    for j in range(gy):
        for i in range(gx):
            if open_cells[j, i] and not visited[j, i]:
                # BFS
                stack = [(j, i)]
                comp = []
                while stack:
                    cj, ci = stack.pop()
                    if cj < 0 or cj >= gy or ci < 0 or ci >= gx:
                        continue
                    if visited[cj, ci] or not open_cells[cj, ci]:
                        continue
                    visited[cj, ci] = True
                    comp.append((cj, ci))
                    for nj, ni in [(cj - 1, ci), (cj + 1, ci), (cj, ci - 1), (cj, ci + 1)]:
                        stack.append((nj, ni))
                if len(comp) > largest_size:
                    largest_size = len(comp)
                    largest = comp

    if largest is None or largest_size == 0:
        print("ERROR: no open connected region found")
        return

    print(f"  largest open region: {largest_size} cells ({largest_size * cell_area:.2f} m^2)")

    # centroid of largest region
    js = np.array([c[0] for c in largest])
    is_ = np.array([c[1] for c in largest])
    cx_mean = float(np.mean(xmin + (is_ + 0.5) * args.grid_res))
    cy_mean = float(np.mean(ymin + (js + 0.5) * args.grid_res))

    # estimate floor z at this location from nearby floor faces
    dist_sq = (cx - cx_mean) ** 2 + (cy - cy_mean) ** 2
    near = dist_sq < (1.0 ** 2)
    if near.sum() > 0:
        cz_mean = float(np.median(centroids[floor_mask][near, 2]))
    else:
        cz_mean = floor_z

    result = {
        "start_xy": [round(cx_mean, 3), round(cy_mean, 3)],
        "start_z_floor": round(cz_mean, 3),
        "floor_z_global": round(floor_z, 3),
        "largest_open_area_m2": round(largest_size * cell_area, 2),
        "grid_res": args.grid_res,
        "bbox": {
            "x": [round(float(xmin), 3), round(float(xmax), 3)],
            "y": [round(float(ymin), 3), round(float(ymax), 3)],
            "z": [round(float(zmin), 3), round(float(zmax), 3)],
        },
        "mesh": {
            "verts": int(len(verts)),
            "faces": int(len(faces)),
        },
    }
    print(f"\n=== START POINT ===")
    print(f"  xy: ({cx_mean:.3f}, {cy_mean:.3f})")
    print(f"  floor z: {cz_mean:.3f}")
    print(f"  open area: {largest_size * cell_area:.2f} m^2")

    # Save heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax0 = axes[0]
    im0 = ax0.imshow(density, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="hot", aspect="equal")
    ax0.set_title("Floor face area density")
    ax0.plot(cx_mean, cy_mean, "gx", markersize=15, markeredgewidth=3)
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    ax1 = axes[1]
    im1 = ax1.imshow(open_frac * walkable, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="YlGn", aspect="equal")
    ax1.set_title("Open walkable region (green=open)")
    ax1.plot(cx_mean, cy_mean, "rx", markersize=15, markeredgewidth=3)
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    fig.tight_layout()
    png_path = os.path.join(args.output_dir, "start_point_analysis.png")
    fig.savefig(png_path, dpi=120)
    print(f"\nSaved: {png_path}")

    json_path = os.path.join(args.output_dir, "start_point.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
