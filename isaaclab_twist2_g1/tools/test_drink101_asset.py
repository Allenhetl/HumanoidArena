#!/usr/bin/env python3
"""Offline structural validation of the prepared drink101 USD assets.

Checks (no Isaac Sim / no GPU / no GUI needed, uses GMR env pxr):
  1. drink101_body.usd : exactly one rigid body (E_body_52), dynamic,
                         bottle meshes present.
  2. drink101_cap.usd  : exactly one rigid body (E_knob_58), dynamic,
                         cap collision mesh lifted clear of the bottle mouth.
  3. drink101_artic.usd: body + cap + breakable revolute joint with
                         axis=Z, limits 0..2*pi, breakForce>0.

Exit code 0 on success, 1 on failure. Run with GMR env:
  python tools/test_drink101_asset.py --assets-dir assets/objects/drink101
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdPhysics

BODY_PRIM = "E_body_52"
CAP_PRIM = "E_knob_58"
CAP_MESH = "P_6b115c4e8395a1c5"
MOUTH_TOP_Z = 0.258
CAP_CENTER_Z = 0.2649


def _rigid_bodies(stage: Usd.Stage) -> list:
    out = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            out.append(
                {
                    "path": str(prim.GetPath()),
                    "kinematic": bool(rb.GetKinematicEnabledAttr().Get()),
                    "rigid_enabled": bool(rb.GetRigidBodyEnabledAttr().Get()),
                }
            )
    return out


def check_body(assets_dir: Path) -> list[str]:
    errs = []
    path = assets_dir / "drink101_body.usd"
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return [f"cannot open {path}"]
    rbs = _rigid_bodies(stage)
    if len(rbs) != 1:
        errs.append(f"body usd: expected 1 rigid body, got {len(rbs)}")
    elif rbs[0]["path"] != f"/root/{BODY_PRIM}":
        errs.append(f"body usd: wrong rigid body path {rbs[0]['path']}")
    elif rbs[0]["kinematic"]:
        errs.append("body usd: bottle body must be dynamic (kinematic=False)")
    return errs


def check_cap(assets_dir: Path) -> list[str]:
    errs = []
    path = assets_dir / "drink101_cap.usd"
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return [f"cannot open {path}"]
    rbs = _rigid_bodies(stage)
    if len(rbs) != 1:
        errs.append(f"cap usd: expected 1 rigid body, got {len(rbs)}")
    elif rbs[0]["path"] != f"/root/{CAP_PRIM}":
        errs.append(f"cap usd: wrong rigid body path {rbs[0]['path']}")

    # collision mesh lifted clear of bottle mouth
    mesh = stage.GetPrimAtPath(f"/root/{CAP_PRIM}/{CAP_MESH}")
    if mesh is None or not mesh.IsValid():
        errs.append(f"cap usd: cap collision mesh {CAP_MESH} missing")
    else:
        from pxr import UsdGeom

        xf = UsdGeom.Xformable(mesh)
        translate_z = 0.0
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                translate_z = op.Get()[2]
        extent = mesh.GetAttribute("extent").Get()
        lower_edge = CAP_CENTER_Z + translate_z + extent[0][2]
        if lower_edge < MOUTH_TOP_Z - 1e-3:
            errs.append(
                f"cap usd: collision lower edge {lower_edge:.3f} still below "
                f"bottle mouth top {MOUTH_TOP_Z:.3f} (rotation would bite)"
            )
    return errs


def check_artic(assets_dir: Path) -> list[str]:
    errs = []
    path = assets_dir / "drink101_artic.usd"
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return [f"cannot open {path}"]
    rbs = _rigid_bodies(stage)
    if len(rbs) != 2:
        errs.append(f"artic usd: expected 2 rigid bodies, got {len(rbs)}")
    body_has_root = stage.GetPrimAtPath(f"/root/{BODY_PRIM}").HasAPI(
        UsdPhysics.ArticulationRootAPI
    )
    if not body_has_root:
        errs.append("artic usd: ArticulationRootAPI missing on body")

    joint = stage.GetPrimAtPath(f"/root/{CAP_PRIM}/RevoluteJoint_cap")
    if joint is None or not joint.IsValid():
        errs.append("artic usd: RevoluteJoint_cap missing")
    else:
        j = UsdPhysics.RevoluteJoint(joint)
        axis = j.GetAxisAttr().Get() if j.GetAxisAttr() else None
        low = j.GetLowerLimitAttr().Get() if j.GetLowerLimitAttr() else None
        high = j.GetUpperLimitAttr().Get() if j.GetUpperLimitAttr() else None
        bf = j.GetBreakForceAttr().Get() if j.GetBreakForceAttr() else None
        bt = j.GetBreakTorqueAttr().Get() if j.GetBreakTorqueAttr() else None
        if axis != "Z":
            errs.append(f"artic usd: joint axis={axis!r} (expected Z)")
        if not (low is not None and abs(low) < 1e-3):
            errs.append(f"artic usd: lower limit {low} (expected ~0)")
        if high is None or high < 6.28:
            errs.append(f"artic usd: upper limit {high} (expected >= 2*pi)")
        if not bf or bf <= 0:
            errs.append(f"artic usd: breakForce {bf} (expected > 0)")
    return errs


def check_desk(assets_dir: Path) -> list[str]:
    """Validate the SAM3D desk USD: single static rigid body, SDF collision,
    textured material, z-up, correct tabletop height/dimensions."""
    errs = []
    desk_dir = assets_dir.parent / "desk_rec"
    path = desk_dir / "desk0.usd"
    if not path.exists():
        return ["desk0.usd missing"]
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return [f"cannot open {path}"]
    if stage.GetMetadata("upAxis") != "Z":
        errs.append(f"desk: upAxis={stage.GetMetadata('upAxis')} (expected Z)")

    desk = stage.GetPrimAtPath("/root/Desk")
    if desk is None or not desk.IsValid():
        return ["desk: /root/Desk prim missing"]
    if not desk.HasAPI(UsdPhysics.RigidBodyAPI):
        errs.append("desk: RigidBodyAPI missing")
    else:
        rb = UsdPhysics.RigidBodyAPI(desk)
        if not rb.GetKinematicEnabledAttr().Get():
            errs.append("desk: kinematic_enabled must be True (static)")
    if not desk.HasAPI(UsdPhysics.CollisionAPI):
        errs.append("desk: CollisionAPI missing")
    approx = desk.GetAttribute("physics:approximation").Get()
    if approx != "sdf":
        errs.append(f"desk: physics:approximation={approx!r} (expected 'sdf')")

    from pxr import UsdGeom
    mesh = UsdGeom.Mesh(desk)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    if len(pts) == 0:
        errs.append("desk: empty mesh points")
    else:
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        height = bmax[2] - bmin[2]
        top = bmax[2]
        if not (0.9 < height < 1.1):
            errs.append(f"desk: height {height:.2f} (expected ~1.0)")
        if abs(top - 0.5) > 0.02:
            errs.append(f"desk: tabletop local z {top:.3f} (expected ~+0.5)")
        if not (0.7 < (bmax[0] - bmin[0]) < 1.0):
            errs.append(f"desk: x span {bmax[0]-bmin[0]:.2f} (expected ~0.85)")
        if not (0.8 < (bmax[1] - bmin[1]) < 1.05):
            errs.append(f"desk: y span {bmax[1]-bmin[1]:.2f} (expected ~0.91)")

    # texture + material binding
    mat_bind = desk.GetRelationship("material:binding:material")
    if not mat_bind or not mat_bind.GetTargets():
        errs.append("desk: material binding missing")
    mat = stage.GetPrimAtPath("/root/materials/desk_mat")
    if not mat or not mat.IsValid():
        errs.append("desk: material prim missing")
    tex = stage.GetPrimAtPath("/root/materials/desk_mat/diffuseTexture")
    if tex and tex.IsValid():
        f = tex.GetAttribute("inputs:file").Get()
        if not f:
            errs.append("desk: texture file attr empty")
        else:
            raw = str(f).strip().strip("@")
            tex_rel = Path(raw)
            if not tex_rel.is_absolute() and not (desk_dir / tex_rel).exists():
                errs.append(f"desk: texture file missing: {f}")
    else:
        errs.append("desk: diffuseTexture shader missing")
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    args = parser.parse_args()
    assets_dir = args.assets_dir.resolve()

    all_errs = []
    checks = (
        ("body", check_body),
        ("cap", check_cap),
        ("artic", check_artic),
        ("desk", check_desk),
    )
    for name, fn in checks:
        errs = fn(assets_dir)
        status = "PASS" if not errs else "FAIL"
        print(f"[drink_asset] {name}: {status}")
        for e in errs:
            print(f"    - {e}")
            all_errs.append(e)

    if all_errs:
        print(f"[drink_asset] FAILED ({len(all_errs)} errors)")
        return 1
    print("[drink_asset] SUCCESS: all assets structurally valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
