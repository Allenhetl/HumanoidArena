#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def plane_z(coef, x, y):
    return coef[0] * x + coef[1] * y + coef[2]


def build_grid_mesh(mask, x_grid, y_grid, coef):
    h, w = mask.shape
    vertex_index = {}
    vertices = []

    def add_vertex(iy, ix):
        key = (iy, ix)
        if key in vertex_index:
            return vertex_index[key]
        x = float(x_grid[iy, ix])
        y = float(y_grid[iy, ix])
        z = float(plane_z(coef, x, y))
        vertex_index[key] = len(vertices)
        vertices.append((x, y, z))
        return vertex_index[key]

    faces = []
    # Use grid nodes around reachable cells. A cell at [iy, ix] spans
    # node corners (iy, ix), (iy, ix+1), (iy+1, ix+1), (iy+1, ix).
    for iy in range(h - 1):
        for ix in range(w - 1):
            if not mask[iy, ix]:
                continue
            v00 = add_vertex(iy, ix)
            v10 = add_vertex(iy, ix + 1)
            v11 = add_vertex(iy + 1, ix + 1)
            v01 = add_vertex(iy + 1, ix)
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    if not faces:
        raise RuntimeError("Reachable mask produced no triangles")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def write_usd(vertices, faces, output_path: Path, prim_path: str):
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(output_path))
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
    stage.GetRootLayer().defaultPrim = prim_path.strip("/").split("/")[0]
    stage.Save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output_usd", type=Path, required=True)
    parser.add_argument("--mask_key", choices=("reachable", "connected"), default="reachable")
    parser.add_argument("--prim_path", default="/floor_repair")
    parser.add_argument("--metadata_json", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.masks)
    mask = data[args.mask_key].astype(bool)
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    coef = data["plane_coef"]
    vertices, faces = build_grid_mesh(mask, x_grid, y_grid, coef)
    args.output_usd.parent.mkdir(parents=True, exist_ok=True)
    write_usd(vertices, faces, args.output_usd, args.prim_path)

    meta = {
        "masks": str(args.masks),
        "output_usd": str(args.output_usd),
        "mask_key": args.mask_key,
        "prim_path": args.prim_path,
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "bbox_min": vertices.min(axis=0).tolist(),
        "bbox_max": vertices.max(axis=0).tolist(),
        "plane_coef": coef.tolist(),
    }
    if args.metadata_json:
        args.metadata_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
