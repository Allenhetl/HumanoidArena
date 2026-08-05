#!/usr/bin/env python3
"""Prepare drink101 bottle USD for IsaacLab.

Original: model_beverage13.usd (Extwin) has two rigid bodies under /root:
  - E_body_52 (bottle body, kinematic=1, SDF collision, glass)
  - E_knob_58 (bottle cap, dynamic=0, SDF collision, plastic)
  They are NOT connected by any joint.

Outputs (kept in the same directory so relative sublayer/asset paths resolve):
  - drink101_body.usd : single rigid body (E_body_52, kinematic disabled)
  - drink101_cap.usd  : single rigid body (E_knob_58, dynamic)
  - drink101_artic.usd: body + cap + pre-baked breakable revolute joint
                        (axis=Z, limits 0..2*pi, breakForce=30, breakTorque=0)
                        with ArticulationRootAPI on the body. Load with
                        ArticulationCfg (open_door pattern) so PhysX parses the
                        joint from USD - runtime-created USD joints are NOT
                        picked up by the physics scene.

Run with the GMR conda env (has pxr):
  python tools/prepare_drink101_usd.py --source-dir assets/objects/drink101 --out-dir assets/objects/drink101
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics, UsdUtils

BODY_PRIM = "E_body_52"
CAP_PRIM = "E_knob_58"
SHARED_ROOT_CHILDREN = ("materials", "Plastic", "water")

CAP_JOINT_AXIS = "Z"
CAP_OFFSET_Z = 0.2649
CAP_JOINT_LIMIT_LOW = 0.0
CAP_JOINT_LIMIT_HIGH = 6.283185307179586  # 2*pi, one full turn
CAP_JOINT_BREAK_FORCE = 30.0
CAP_JOINT_BREAK_TORQUE = 0.0  # never break from twisting alone


def _flatten(src_usd: Path):
    src_stage = Usd.Stage.Open(str(src_usd))
    if src_stage is None:
        raise RuntimeError(f"failed to open {src_usd}")
    return UsdUtils.FlattenLayerStack(src_stage)


def _new_root_layer(out_usd: Path) -> Sdf.Layer:
    out_layer = Sdf.Layer.CreateNew(str(out_usd))
    root_spec = Sdf.CreatePrimInLayer(out_layer, "/root")
    root_spec.specifier = Sdf.SpecifierDef
    root_spec.typeName = "Xform"
    out_layer.defaultPrim = "root"
    return out_layer


def _copy_shared(flat, out_layer) -> None:
    for name in SHARED_ROOT_CHILDREN:
        if flat.GetPrimAtPath(f"/root/{name}"):
            Sdf.CopySpec(flat, f"/root/{name}", out_layer, f"/root/{name}")


def _lift_cap_collision(out_usd: Path) -> None:
    """Lift the cap collision mesh so its lower edge clears the bottle mouth.

    Cap mesh local extent z is [-0.0277, +0.0026] at cap center z=0.2649
    (body frame) => lower edge at 0.237, but the bottle mouth tops out at
    0.258.  The overlap (2.1 cm) makes the cap physically bite into the mouth
    and blocks rotation / destabilizes resets.  Raise the collision mesh by
    0.021 so it sits just on top of the mouth (visual mesh untouched).
    """
    stage = Usd.Stage.Open(str(out_usd))
    mesh = stage.GetPrimAtPath(f"/root/{CAP_PRIM}/P_6b115c4e8395a1c5")
    if mesh is None or not mesh.IsValid():
        print(f"[drink101] cap mesh not found in {out_usd.name}; skip lift")
        return
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(mesh)
    ops = xf.GetOrderedXformOps()
    existing = [o.GetOpName() for o in ops if o.GetOpName() == "xformOp:translate"]
    if existing:
        xf.GetOp("xformOp:translate").Set((0.0, 0.0, 0.021))
    else:
        xf.AddTranslateOp().Set((0.0, 0.0, 0.021))
    stage.GetRootLayer().Save()
    print(f"[drink101] lifted cap collision by 0.021 in {out_usd.name}")


def _emit(src_usd: Path, out_usd: Path, *, keep: str, drop: str, disable_kinematic: bool) -> None:
    flat = _flatten(src_usd)
    out_layer = _new_root_layer(out_usd)
    _copy_shared(flat, out_layer)
    Sdf.CopySpec(flat, f"/root/{keep}", out_layer, f"/root/{keep}")

    if disable_kinematic:
        prim_spec = out_layer.GetPrimAtPath(Sdf.Path(f"/root/{keep}"))
        if prim_spec is None:
            raise RuntimeError(f"prim spec missing: /root/{keep}")
        attr = prim_spec.attributes.get("physics:kinematicEnabled")
        if attr is None:
            raise RuntimeError(f"physics:kinematicEnabled missing on /root/{keep}")
        attr.default = False

    out_layer.Save()
    print(f"[drink101] wrote {out_usd}")


def _emit_artic(src_usd: Path, out_usd: Path) -> None:
    flat = _flatten(src_usd)
    out_layer = _new_root_layer(out_usd)
    _copy_shared(flat, out_layer)
    Sdf.CopySpec(flat, f"/root/{BODY_PRIM}", out_layer, f"/root/{BODY_PRIM}")
    Sdf.CopySpec(flat, f"/root/{CAP_PRIM}", out_layer, f"/root/{CAP_PRIM}")

    # body -> dynamic
    body_spec = out_layer.GetPrimAtPath(Sdf.Path(f"/root/{BODY_PRIM}"))
    if body_spec is None:
        raise RuntimeError("body spec missing")
    kin = body_spec.attributes.get("physics:kinematicEnabled")
    if kin is None:
        raise RuntimeError("physics:kinematicEnabled missing on body")
    kin.default = False
    out_layer.Save()

    # add ArticulationRootAPI on body (open_door pattern)
    tmp_stage = Usd.Stage.Open(str(out_usd))
    body_prim = tmp_stage.GetPrimAtPath(f"/root/{BODY_PRIM}")
    UsdPhysics.ArticulationRootAPI.Apply(body_prim)
    tmp_stage.Save()

    # append joint prim under cap (implicit body0=cap, body1=parent body)
    joint_layer = Sdf.Layer.FindOrOpen(str(out_usd))
    cap_prim = joint_layer.GetPrimAtPath(Sdf.Path(f"/root/{CAP_PRIM}"))
    if cap_prim is None:
        raise RuntimeError("cap prim spec missing")
    joint_spec = Sdf.CreatePrimInLayer(
        joint_layer, f"/root/{CAP_PRIM}/RevoluteJoint_cap"
    )
    joint_spec.specifier = Sdf.SpecifierDef
    joint_spec.typeName = "PhysicsRevoluteJoint"
    # open_door convention: body0 = connecting rigid body (body), body1 = the
    # rigid body that owns the joint prim (cap)
    rel0 = Sdf.RelationshipSpec(joint_spec, "physics:body0", True)
    rel0.targetPathList.explicitItems.append(Sdf.Path(f"/root/{BODY_PRIM}"))
    rel1 = Sdf.RelationshipSpec(joint_spec, "physics:body1", True)
    rel1.targetPathList.explicitItems.append(Sdf.Path(f"/root/{CAP_PRIM}"))
    attrs = (
        ("physics:axis", Sdf.ValueTypeNames.Token, CAP_JOINT_AXIS),
        ("physics:lowerLimit", Sdf.ValueTypeNames.Float, CAP_JOINT_LIMIT_LOW),
        ("physics:upperLimit", Sdf.ValueTypeNames.Float, CAP_JOINT_LIMIT_HIGH),
        ("physics:breakForce", Sdf.ValueTypeNames.Float, CAP_JOINT_BREAK_FORCE),
        ("physics:breakTorque", Sdf.ValueTypeNames.Float, CAP_JOINT_BREAK_TORQUE),
        # joint frame at cap center: in cap frame (0,0,0); in body frame the cap
        # sits at z=0.2649 (E_knob_58 translate)
        ("physics:localPos0", Sdf.ValueTypeNames.Point3f, (0.0, 0.0, 0.0)),
        ("physics:localPos1", Sdf.ValueTypeNames.Point3f, (0.0, 0.0, CAP_OFFSET_Z)),
    )
    for name, typ, val in attrs:
        a = Sdf.AttributeSpec(joint_spec, name, typ)
        a.default = val
    joint_layer.Save()
    print(f"[drink101] artic -> {out_usd}")


def prepare(source_dir: Path, out_dir: Path) -> dict[str, Path]:
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = source_dir / "model_beverage13.usd"
    if not src.exists():
        raise FileNotFoundError(f"missing {src}")

    body_path = out_dir / "drink101_body.usd"
    _emit(src, body_path, keep=BODY_PRIM, drop=CAP_PRIM, disable_kinematic=True)

    cap_path = out_dir / "drink101_cap.usd"
    _emit(src, cap_path, keep=CAP_PRIM, drop=BODY_PRIM, disable_kinematic=False)
    _lift_cap_collision(cap_path)

    artic_path = out_dir / "drink101_artic.usd"
    _emit_artic(src, artic_path)
    _lift_cap_collision(artic_path)

    return {"body": body_path, "cap": cap_path, "artic": artic_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
