#!/usr/bin/env python3
"""Convert the SAM3D-reconstructed desk GLB into a single USD asset for IsaacLab.

Input:  desk0.glb (glTF 2.0, y-up, one mesh with POSITION+TEXCOORD_0 + PNG base
        color texture, physical table ~0.85x0.91x1.0 m).
Output: desk0.usd with:
  - /root/Desk  : Mesh (vertices/faces/UVs/normals), baked y-up -> z-up
                  (glTF is y-up; Isaac/PhysX scenes are z-up).
  - /root/materials/desk_mat : PreviewSurface + UsdUVTexture binding the PNG.
  - RigidBodyAPI on /root/Desk with kinematic=True + disable_gravity => static.
  - PhysxSDFMeshCollisionAPI on the mesh (approximation=sdf) so a bottle can
    rest on the tabletop and hands can reach under/around legs.

The desk is authored as a standalone asset; the task cfg positions it via
AssetBaseCfg.init_state (so its pose stays adjustable in the env YAML instead
of being welded into the room USD).

Run with the GMR conda env (has trimesh + pxr):
  python tools/convert_desk_glb_to_usd.py --glb assets/objects/desk_rec/desk0.glb \
      --out assets/objects/desk_rec/desk0.usd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUI

SDF_MATERIALS_SCOPE = "/root/materials"


def _extract_png_from_glb(glb_path: Path, out_png: Path) -> bool:
    """Extract the embedded PNG base-color texture from the GLB binary chunk."""
    import json
    import struct

    data = glb_path.read_bytes()
    if data[:4] != b"glTF":
        return False
    view = memoryview(data)
    # JSON chunk
    json_len = struct.unpack_from("<I", view, 12)[0]
    json_bytes = view[20 : 20 + json_len].tobytes()
    header = json.loads(json_bytes)
    # BIN chunk: length header at 20+json_len, payload follows (+8)
    bin_header = 20 + json_len
    bin_len = struct.unpack_from("<I", view, bin_header)[0]
    bin_data = view[bin_header + 8 : bin_header + 8 + bin_len]

    images = header.get("images", [])
    if not images:
        return False
    image = images[0]
    bv = header["bufferViews"][image["bufferView"]]
    offset = bv.get("byteOffset", 0)
    length = bv.get("byteLength", 0)
    png_bytes = bin_data[offset : offset + length]
    out_png.write_bytes(bytes(png_bytes))
    print(f"[desk] extracted texture -> {out_png}")
    return True


def _bake_yup_to_zup(points: np.ndarray) -> np.ndarray:
    """glTF y-up -> Isaac z-up: (x, y, z) -> (x, -z, y)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.stack([x, -z, y], axis=-1)


def convert(glb_path: Path, out_usd: Path, texture_dir: Path | None = None) -> Path:
    glb_path = glb_path.resolve()
    out_usd = out_usd.resolve()
    out_usd.parent.mkdir(parents=True, exist_ok=True)

    # ---- geometry ----
    scene = trimesh.load(str(glb_path), process=False, force="scene")
    mesh = None
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                mesh = geom
                break
    if mesh is None:
        raise RuntimeError("no Trimesh found in GLB")

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    uvs = None
    if mesh.visual and mesh.visual.uv is not None and len(mesh.visual.uv) == len(verts):
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32) if mesh.vertex_normals is not None else None

    verts = _bake_yup_to_zup(verts)
    if normals is not None:
        normals = _bake_yup_to_zup(normals)

    print(f"[desk] verts={len(verts)} faces={len(faces)} uvs={None if uvs is None else len(uvs)}")
    print(f"[desk] z-up bbox min={verts.min(axis=0).round(3)} max={verts.max(axis=0).round(3)}")

    # ---- USD ----
    stage = Usd.Stage.CreateNew(str(out_usd))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    root = stage.DefinePrim("/root", "Xform")
    stage.GetRootLayer().defaultPrim = "root"

    mesh_prim = stage.DefinePrim("/root/Desk", "Mesh")
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    usd_mesh.CreatePointsAttr(verts)
    usd_mesh.CreateFaceVertexCountsAttr(np.full(len(faces), 3, dtype=np.int32))
    usd_mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).astype(np.int32))
    usd_mesh.CreateSubdivisionSchemeAttr("none")
    if uvs is not None:
        tex_coords = np.zeros((len(verts), 2), dtype=np.float32)
        tex_coords[: len(uvs)] = uvs
        st = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
        st.Set(tex_coords)
    if normals is not None:
        usd_mesh.CreateNormalsAttr(normals)

    # static rigid body
    UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
    rb = UsdPhysics.RigidBodyAPI(mesh_prim)
    rb.CreateKinematicEnabledAttr().Set(True)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    coll = UsdPhysics.CollisionAPI(mesh_prim)
    coll.CreateCollisionEnabledAttr().Set(True)
    # SDF approximation (matches drink101 bottle physics: physics:approximation=sdf)
    mesh_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("sdf")

    # ---- material + texture ----
    texture_path = None
    if texture_dir is not None:
        texture_dir = texture_dir.resolve()
        texture_dir.mkdir(parents=True, exist_ok=True)
        texture_path = texture_dir / "desk0_albedo.png"
        _extract_png_from_glb(glb_path, texture_path)

    mat_path = "/root/materials/desk_mat"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    shader_prim = stage.DefinePrim(f"{mat_path}/PreviewSurface", "Shader")
    shader_prim.CreateAttribute("info:id", Sdf.ValueTypeNames.Token).Set("UsdPreviewSurface")
    shader_prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("id")

    shader = UsdShade.Shader(shader_prim)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.6, 0.6))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

    if texture_path is not None and texture_path.exists():
        tex_prim = stage.DefinePrim(f"{mat_path}/diffuseTexture", "Shader")
        tex_prim.CreateAttribute("info:id", Sdf.ValueTypeNames.Token).Set("UsdUVTexture")
        tex = UsdShade.Shader(tex_prim)
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path.name)
        tex.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).Set(Gf.Vec2f(0.0, 0.0))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        tex.CreateOutput("a", Sdf.ValueTypeNames.Float)
        shader.GetInput("diffuseColor").ConnectToSource(tex.GetOutput("rgb"))

    # wire material surface outputs to the shader surface
    usd_mat = UsdShade.Material(mat_prim)
    usd_mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.GetOutput("surface"))
    usd_mat.CreateSurfaceOutput("glslfx").ConnectToSource(shader.GetOutput("surface"))
    usd_mat.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(
        usd_mat, UsdShade.Tokens.weakerThanDescendants, "material"
    )

    stage.Save()
    print(f"[desk] wrote {out_usd}")
    return out_usd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, default=None,
                        help="directory to extract the PNG into (default: alongside --out)")
    args = parser.parse_args()
    tex_dir = args.texture_dir or args.out.parent
    convert(args.glb, args.out, texture_dir=tex_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
