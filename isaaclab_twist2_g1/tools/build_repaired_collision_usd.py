#!/usr/bin/env python3
import argparse
import json
import math
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


def compact_submesh(vertices, faces, keep_faces):
    kept = faces[keep_faces]
    used, inverse = np.unique(kept.reshape(-1), return_inverse=True)
    compact_vertices = vertices[used].astype(np.float32)
    compact_faces = inverse.reshape((-1, 3)).astype(np.int32)
    return compact_vertices, compact_faces


def write_collision_mesh(output_path: Path, prim_path: str, vertices, faces, invisible: bool):
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(output_path))
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    if invisible:
        mesh.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
    stage.GetRootLayer().defaultPrim = prim_path.strip("/").split("/")[0]
    stage.Save()


def write_wrapper(output_path: Path, scene_ref: str, obstacle_ref: str, floor_ref: str):
    text = f'''#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World" (
    prepend references = @{scene_ref}@</World>
)
{{
    over "mesh"
    {{
        bool physics:collisionEnabled = false
    }}

    def Mesh "collision_obstacles" (
        prepend references = @{obstacle_ref}@</collision_obstacles>
    )
    {{
    }}

    def Mesh "floor_repair" (
        prepend references = @{floor_ref}@</floor_repair>
    )
    {{
    }}
}}
'''
    output_path.write_text(text, encoding="utf-8")


def save_debug_plot(centroid, remove_floor, keep_obstacle, mask, x_grid, y_grid, start_xy, output_path: Path):
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=170)
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    ax.imshow(mask.astype(float), origin="lower", extent=extent, cmap="Greens", alpha=0.35, interpolation="nearest")

    keep_idx = np.where(keep_obstacle)[0]
    rem_idx = np.where(remove_floor)[0]
    if len(keep_idx) > 50000:
        keep_idx = rng.choice(keep_idx, size=50000, replace=False)
    if len(rem_idx) > 50000:
        rem_idx = rng.choice(rem_idx, size=50000, replace=False)

    ax.scatter(centroid[keep_idx, 0], centroid[keep_idx, 1], s=0.15, c="#4b5563", alpha=0.18, label="kept collision faces")
    ax.scatter(centroid[rem_idx, 0], centroid[rem_idx, 1], s=0.6, c="#ef4444", alpha=0.75, label="removed original floor faces")
    ax.scatter([start_xy[0]], [start_xy[1]], c="blue", s=45, marker="x", label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Original mesh collision split: red removed, gray kept, green repair mask")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--mesh_path", default="/World/mesh")
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--mask_key", choices=("reachable", "connected"), default="reachable")
    parser.add_argument("--floor_repair_ref", default="floor_repair_start_1p5_4p0_plane15_close35.usd")
    parser.add_argument("--obstacles_usd", type=Path, required=True)
    parser.add_argument("--wrapper_usda", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--scene_ref", default="small_warehouse_digital_twin_office_fixed_gauss_xneg90_ccm1.usda")
    parser.add_argument("--obstacles_ref", default=None)
    parser.add_argument("--delete_margin", type=float, default=0.10)
    parser.add_argument("--normal_deg", type=float, default=25.0)
    parser.add_argument("--plane_dist", type=float, default=0.16)
    parser.add_argument("--invisible_obstacles", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = load_world_mesh(args.scene, args.mesh_path)
    centroid, normal, area, valid = face_geometry(vertices, faces)

    data = np.load(args.masks)
    mask = data[args.mask_key].astype(bool)
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    coef = data["plane_coef"]
    start_xy = data["start_xy"]
    grid = float(data["grid"])

    margin_cells = max(0, int(round(args.delete_margin / grid)))
    if margin_cells:
        yy, xx = np.ogrid[-margin_cells : margin_cells + 1, -margin_cells : margin_cells + 1]
        disk = (xx * xx + yy * yy) <= margin_cells * margin_cells
        delete_mask = ndimage.binary_dilation(mask, structure=disk)
    else:
        delete_mask = mask

    x0 = float(x_grid[0, 0])
    y0 = float(y_grid[0, 0])
    ix = np.floor((centroid[:, 0] - x0) / grid).astype(np.int64)
    iy = np.floor((centroid[:, 1] - y0) / grid).astype(np.int64)
    in_grid = (ix >= 0) & (ix < delete_mask.shape[1]) & (iy >= 0) & (iy < delete_mask.shape[0])
    in_delete_xy = np.zeros(len(faces), dtype=bool)
    in_delete_xy[in_grid] = delete_mask[iy[in_grid], ix[in_grid]]

    normal_cos = math.cos(math.radians(args.normal_deg))
    horizontal = valid & (np.abs(normal[:, 2]) >= normal_cos)
    residual = np.abs(centroid[:, 2] - plane_z(coef, centroid[:, :2]))
    near_floor_plane = residual <= args.plane_dist
    remove_floor = in_delete_xy & horizontal & near_floor_plane
    keep_obstacle = ~remove_floor

    obstacle_vertices, obstacle_faces = compact_submesh(vertices, faces, keep_obstacle)
    args.obstacles_usd.parent.mkdir(parents=True, exist_ok=True)
    write_collision_mesh(args.obstacles_usd, "/collision_obstacles", obstacle_vertices, obstacle_faces, args.invisible_obstacles)

    obstacles_ref = args.obstacles_ref or args.obstacles_usd.name
    args.wrapper_usda.parent.mkdir(parents=True, exist_ok=True)
    write_wrapper(args.wrapper_usda, args.scene_ref, obstacles_ref, args.floor_repair_ref)

    save_debug_plot(
        centroid,
        remove_floor,
        keep_obstacle,
        delete_mask,
        x_grid,
        y_grid,
        start_xy,
        args.out_dir / "collision_split_removed_faces_overlay.png",
    )

    removed_area = float(area[remove_floor].sum())
    kept_area = float(area[keep_obstacle].sum())
    meta = {
        "scene": str(args.scene),
        "mesh_path": args.mesh_path,
        "masks": str(args.masks),
        "mask_key": args.mask_key,
        "delete_margin_m": args.delete_margin,
        "normal_deg": args.normal_deg,
        "plane_dist_m": args.plane_dist,
        "input_vertices": int(len(vertices)),
        "input_triangles": int(len(faces)),
        "removed_floor_triangles": int(remove_floor.sum()),
        "kept_obstacle_triangles": int(keep_obstacle.sum()),
        "removed_floor_area_m2": removed_area,
        "kept_obstacle_area_m2": kept_area,
        "obstacle_vertices": int(len(obstacle_vertices)),
        "obstacle_triangles": int(len(obstacle_faces)),
        "obstacles_usd": str(args.obstacles_usd),
        "wrapper_usda": str(args.wrapper_usda),
        "outputs": [
            "collision_split_removed_faces_overlay.png",
            "collision_split_metadata.json",
        ],
    }
    (args.out_dir / "collision_split_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
