#!/usr/bin/env python3
"""Build a MuJoCo G1 + Inspire (6-drive/12-joint) model from the GMR base model.

Takes GMR's g1_mocap_29dof.xml (29-DoF body, rubber_hand removed) and attaches the
xr_teleoperate inspire hand URDFs (6 drive + 6 mimic joints per hand) at the
left/right wrist_yaw_link bodies using the base-joint transform from the
dex-retarget URDF (rpy = -90deg about X, then 180deg about Z).

All 12 joints per hand are independent revolute joints (no MuJoCo equality mimic)
so the offline validation drives exactly the same 12 values as inspire_mapping.py
and the Isaac/USD control path.

Usage:
    python build_g1_inspire_mjcf.py \
        --base <g1_mocap_29dof.xml> \
        --urdf_right <inspire_hand_right.urdf> \
        --urdf_left  <inspire_hand_left.urdf> \
        --meshes <inspire_hand/meshes> \
        --out <g1_29dof_with_inspire_hand.xml>
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# URDF fixed joint origin for hand base -> wrist_yaw_link (from dex-retarget URDF)
# Hand base -> wrist_yaw_link transform.
# GMR's own g1_mocap_29dof_with_hands.xml places the palm directly along wrist +x
# (no rotation, pos along +x). The inspire hand extends fingers along hand_base -y,
# so a +90deg rotation about z maps -y -> +x to match the GMR convention.
_BASE_RPY = (0.0, 0.0, np.pi / 2.0)  # z +90deg
_URDF_NS = {"u": "http://www.robot.http://www.robot.com"}


def rpy_to_quat_wxyz(rpy):
    q_xyzw = R.from_euler("xyz", rpy).as_quat()
    return q_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz


def parse_urdf(path):
    """Return (links, joints) from a URDF file.

    links: dict name -> {mass, com_xyz, inertia, mesh, mesh_is_relative}
    joints: dict child -> {parent, type, origin_xyz, origin_rpy, axis, lower, upper, mimic}
    """
    tree = ET.parse(path)
    root = tree.getroot()
    links = {}
    joints = {}

    for link in root.findall("link"):
        name = link.attrib["name"]
        mass = 0.0
        com = np.zeros(3)
        inertia = None
        mesh = None
        inert = link.find("inertial")
        if inert is not None:
            m = inert.find("mass")
            if m is not None:
                mass = float(m.attrib.get("value", 0.0))
            o = inert.find("origin")
            if o is not None:
                com = np.array([float(v) for v in o.attrib.get("xyz", "0 0 0").split()])
            i = inert.find("inertia")
            if i is not None:
                inertia = np.array([
                    float(i.attrib["ixx"]), float(i.attrib["iyy"]), float(i.attrib["izz"]),
                    float(i.attrib["ixy"]), float(i.attrib["ixz"]), float(i.attrib["iyz"]),
                ])
        visual = link.find("visual")
        if visual is not None:
            mesh_el = visual.find(".//mesh")
            if mesh_el is not None:
                mesh = mesh_el.attrib.get("filename")
        links[name] = {
            "mass": mass,
            "com": com,
            "inertia": inertia,
            "mesh": mesh,
        }

    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        jtype = joint.attrib["type"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        oxyz = np.zeros(3)
        orpy = np.zeros(3)
        if origin is not None:
            if origin.attrib.get("xyz"):
                oxyz = np.array([float(v) for v in origin.attrib["xyz"].split()])
            if origin.attrib.get("rpy"):
                orpy = np.array([float(v) for v in origin.attrib["rpy"].split()])
        axis = np.array([0.0, 0.0, 1.0])
        ax = joint.find("axis")
        if ax is not None and ax.attrib.get("xyz"):
            axis = np.array([float(v) for v in ax.attrib["xyz"].split()])
        lower = upper = None
        lim = joint.find("limit")
        if lim is not None:
            lower = float(lim.attrib.get("lower", 0.0))
            upper = float(lim.attrib.get("upper", 0.0))
        mimic = None
        m = joint.find("mimic")
        if m is not None:
            mimic = (m.attrib["joint"], float(m.attrib.get("multiplier", 1.0)))
        joints[child] = {
            "parent": parent,
            "type": jtype,
            "origin_xyz": oxyz,
            "origin_rpy": orpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
            "mimic": mimic,
        }

    return links, joints


def link_to_mjcf(link_name, link, meshdir):
    """Build an <inertial> and mesh geom snippet for a link."""
    parts = []
    if link["mass"] > 0 and link["inertia"] is not None:
        ixx, iyy, izz, ixy, ixz, iyz = link["inertia"]
        parts.append(
            f'      <inertial pos="{" ".join(f"{v:.9g}" for v in link["com"])}" '
            f'mass="{link["mass"]:.9g}" '
            f'fullinertia="{ixx:.9g} {iyy:.9g} {izz:.9g} {ixy:.9g} {ixz:.9g} {iyz:.9g}"/>'
        )
    if link["mesh"]:
        meshname = Path(link["mesh"]).name
        parts.append(
            f'      <geom type="mesh" mesh="{meshname}" rgba="0.75 0.75 0.75 1"/>'
        )
    return "\n".join(parts)


def build_hand_mjcf(side: str, urdf_path: Path, meshdir: Path) -> str:
    """Convert one inspire hand URDF into an MJCF <body> subtree rooted at the hand base."""
    links, joints = parse_urdf(urdf_path)
    prefix = "R" if side == "right" else "L"

    # mesh names must match <mesh> asset declarations (use filename base)
    used_meshes = set()
    for lk in links.values():
        if lk["mesh"]:
            used_meshes.add(Path(lk["mesh"]).name)
    mesh_assets = "".join(
        f'    <mesh name="{m}" file="{m}"/>\n' for m in sorted(used_meshes)
    )

    base_name = f"{prefix}_hand_base_link"
    lines = []
    # root body: hand base, attach point transform applied by caller (wrist body)
    lines.append(f'  <body name="{base_name}">')
    lines.append(link_to_mjcf(base_name, links[base_name], meshdir))

    # children of base
    children = sorted(
        [c for c, j in joints.items() if j["parent"] == base_name and j["type"] == "revolute"],
        key=lambda c: joints[c]["origin_xyz"][1],  # stable order
    )
    for child in children:
        lines.append(_build_chain(child, links, joints, meshdir, depth=1))

    lines.append("  </body>")
    return mesh_assets, "\n".join(lines)


def _build_chain(link_name, links, joints, meshdir, depth):
    """Recursively build MJCF body for a URDF joint chain."""
    j = joints[link_name]
    q = rpy_to_quat_wxyz(j["origin_rpy"])
    pad = "  " * (depth + 1)

    # The URDF joint name already is the child link name + "_joint" in most cases
    # Normalize: derive a clean joint name
    clean_joint = link_name
    if not clean_joint.endswith("_joint"):
        clean_joint = clean_joint + "_joint"

    out = [
        f'{pad}<body name="{link_name}" '
        f'pos="{" ".join(f"{v:.9g}" for v in j["origin_xyz"])}" '
        f'quat="{" ".join(f"{v:.9g}" for v in q)}">',
    ]
    # joint (relative to parent, inside this body)
    if j["type"] == "revolute":
        out.append(
            f'{pad}  <joint name="{clean_joint}" '
            f'axis="{" ".join(f"{v:.9g}" for v in j["axis"])}" '
            f'range="{j["lower"]:.9g} {j["upper"]:.9g}" limited="true" '
            f'actuatorfrcrange="-1 1"/>'
        )
    # fixed joints: no joint element (child body becomes rigidly attached)
    # link visuals/inertial (link body content)
    lk = links[link_name]
    out.append(link_to_mjcf(link_name, lk, meshdir))

    # children
    children = [c for c, jj in joints.items() if jj["parent"] == link_name and c != link_name]
    for c in sorted(children):
        out.append(_build_chain(c, links, joints, meshdir, depth + 1))
    out.append(f"{pad}</body>")
    return "\n".join(out)


def remove_rubber_hands(base_xml: str) -> str:
    """Remove left/right rubber_hand body subtrees from the GMR MJCF."""
    # Simple, robust string surgery: find the exact body blocks.
    for side in ("left", "right"):
        start_marker = f'<body name="{side}_rubber_hand"'
        # find matching closing </body> after start
        idx = base_xml.find(start_marker)
        if idx == -1:
            continue
        depth = 0
        i = idx
        while i < len(base_xml):
            open_b = base_xml.find("<body", i)
            close_b = base_xml.find("</body>", i)
            if open_b != -1 and (close_b == -1 or open_b < close_b):
                depth += 1
                i = open_b + 5
            elif close_b != -1:
                depth -= 1
                if depth == 0:
                    end = close_b + len("</body>")
                    break
                i = close_b + 7
            else:
                break
        base_xml = base_xml[:idx] + base_xml[end:]
    return base_xml


def insert_hands_into_wrist(base_xml: str, left_body: str, right_body: str, q_base: np.ndarray) -> str:
    """Attach hand bodies under the wrist_yaw_link bodies, applying the base transform.

    The inspire hand URDF root (R/L_hand_base_link) is rotated by q_base relative to
    the G1 wrist_yaw_link body so fingers point along wrist +x (GMR convention).
    A small +x offset places the palm at the wrist end (matches GMR with_hands palm).
    """
    for side, body in (("left", left_body), ("right", right_body)):
        marker = f'<body name="{side}_wrist_yaw_link"'
        idx = base_xml.find(marker)
        if idx == -1:
            raise RuntimeError(f"wrist_yaw_link body not found for {side}")
        # insert after the opening tag's first '>'
        gt = base_xml.find(">", idx)
        wrapped = (
            f'\n  <body name="inspire_{side}_hand_attach" '
            f'pos="0.0415 0 0" quat="{" ".join(f"{v:.9g}" for v in q_base)}">\n'
            f"{body}\n  </body>"
        )
        base_xml = base_xml[: gt + 1] + wrapped + base_xml[gt + 1:]
    return base_xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="GMR g1_mocap_29dof.xml")
    ap.add_argument("--urdf_right", required=True)
    ap.add_argument("--urdf_left", required=True)
    ap.add_argument("--meshes", required=True, help="inspire_hand meshes dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None, help="scratch dir for mesh copy (default = out parent)")
    args = ap.parse_args()

    base_path = Path(args.base)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir) if args.workdir else out_path.parent
    # GMR base model uses meshdir="meshes" relative to the XML; place inspire STLs there too.
    meshdir = workdir / "meshes"
    meshdir.mkdir(parents=True, exist_ok=True)

    # copy hand meshes into the same meshes dir (coexist with GMR body meshes)
    src_meshes = Path(args.meshes)
    for f in src_meshes.glob("*.STL"):
        shutil.copy(f, meshdir / f.name)

    # copy GMR body meshes (base model references them via meshdir="meshes")
    base_meshdir = base_path.parent / "meshes"
    if base_meshdir.is_dir():
        for f in base_meshdir.glob("*.STL"):
            shutil.copy(f, meshdir / f.name)
    else:
        print("[build] WARNING: base model meshes dir not found:", base_meshdir)

    base_xml = base_path.read_text()

    # remove rubber hands
    base_xml = remove_rubber_hands(base_xml)

    # build hand subtrees
    right_meshes, right_body = build_hand_mjcf("right", Path(args.urdf_right), meshdir)
    left_meshes, left_body = build_hand_mjcf("left", Path(args.urdf_left), meshdir)

    # insert both hands: right under right_wrist, left under left_wrist, applying base transform
    q_base = rpy_to_quat_wxyz(_BASE_RPY)
    base_xml = insert_hands_into_wrist(base_xml, left_body, right_body, q_base)

    # add mesh assets for hands into the <asset><mesh> section
    # find closing </asset>
    asset_end = base_xml.find("</asset>")
    if asset_end == -1:
        raise RuntimeError("no </asset> section found")
    base_xml = base_xml[:asset_end] + right_meshes + left_meshes + base_xml[asset_end:]

    out_path.write_text(base_xml)
    print(f"[build] wrote {out_path}")
    print(f"[build] meshes -> {meshdir}")


if __name__ == "__main__":
    main()
