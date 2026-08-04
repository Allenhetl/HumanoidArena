#!/usr/bin/env python3
"""Offline G1 + Inspire retarget validation from recorded Pico body+hand data.

Pipeline:
  1. Body: 24-joint XRobot tracking -> GMR(xrobot->unitree_g1) -> 29-DoF G1 qpos.
  2. Hand : 26-joint OpenXR tracking -> geometric a_hw_6 -> dex-retarget
           expand_a_hw_to_q_sim -> 12-DoF inspire per hand.
  3. Visualize combined G1 skeleton + inspire joint bars -> mp4 for human review.

Run on remote (has GMR + gmr env):
  conda activate gmr
  export GMR_ROOT=/home/dreams/Users/taowen/HumanoidArena-inspire-ws/GMR
  python offline_g1_inspire_retarget.py --npz <recording.npz> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

GMR_ROOT = Path("/home/dreams/Users/taowen/HumanoidArena-inspire-ws/GMR")
sys.path.insert(0, str(GMR_ROOT))
sys.path.insert(0, "/home/dreams/Users/taowen/inspire_validation_lib")

BODY_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]

HAND_JOINT_NAMES = [
    "Palm", "Wrist",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
]

FINGERS = {
    "index": (6, 7, 8, 9, 10),
    "middle": (11, 12, 13, 14, 15),
    "ring": (16, 17, 18, 19, 20),
    "pinky": (21, 22, 23, 24, 25),
}
THUMB = (2, 3, 4, 5)
A_HW_NAMES = ("pinky_flex", "ring_flex", "middle_flex", "index_flex", "thumb_flex", "thumb_rotation")

Q_SIM_12_R = [
    "R_pinky_proximal_joint", "R_ring_proximal_joint", "R_middle_proximal_joint",
    "R_index_proximal_joint", "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint",
    "R_pinky_intermediate_joint", "R_ring_intermediate_joint", "R_middle_intermediate_joint",
    "R_index_intermediate_joint", "R_thumb_intermediate_joint", "R_thumb_distal_joint",
]
Q_SIM_12_L = [
    "L_pinky_proximal_joint", "L_ring_proximal_joint", "L_middle_proximal_joint",
    "L_index_proximal_joint", "L_thumb_proximal_pitch_joint", "L_thumb_proximal_yaw_joint",
    "L_pinky_intermediate_joint", "L_ring_intermediate_joint", "L_middle_intermediate_joint",
    "L_index_intermediate_joint", "L_thumb_intermediate_joint", "L_thumb_distal_joint",
]


def angle(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def hand_to_a_hw(kp: np.ndarray) -> np.ndarray:
    kp = np.asarray(kp, dtype=np.float64).reshape(26, 3)
    palm = kp[0]
    out = np.zeros(6, dtype=np.float64)
    for fname, a_name in zip(FINGERS, ("index_flex", "middle_flex", "ring_flex", "pinky_flex")):
        mc, pr, it, _d, _t = FINGERS[fname]
        out[A_HW_NAMES.index(a_name)] = np.pi - angle(kp[mc], kp[pr], kp[it])
    out[A_HW_NAMES.index("thumb_flex")] = np.pi - angle(kp[THUMB[0]], kp[THUMB[1]], kp[THUMB[3]])
    vt = kp[THUMB[0]] - palm
    vm = kp[FINGERS["middle"][0]] - palm
    vtx, vmx = np.array([vt[0], vt[2]]), np.array([vm[0], vm[2]])
    n1, n2 = np.linalg.norm(vtx), np.linalg.norm(vmx)
    out[A_HW_NAMES.index("thumb_rotation")] = (
        float(np.arccos(np.clip(np.dot(vtx, vmx) / (n1 * n2), -1.0, 1.0)))
        if n1 > 1e-8 and n2 > 1e-8 else 0.0
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="/home/dreams/Users/taowen/retarget_validation_out")
    ap.add_argument("--height", type=float, default=1.76)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    # --- load recording ---
    d = np.load(args.npz)
    body = d["body"]                 # (T,24,7) raw: [x,y,z,qx,qy,qz,qw] left-handed Unity
    left = d["left_hand"]            # (T,26,7)
    right = d["right_hand"]
    left_active = d["left_active"].astype(bool)
    right_active = d["right_active"].astype(bool)
    T = len(body)
    print(f"[offline] frames={T} body={body.shape} hand={left.shape}")

    # --- replicate XRobotStreamer Unity->robot coordinate transform ---
    # Same as general_motion_retargeting/xrobot_utils.py: coordinate_transform_unity_data
    # R maps Unity(left-hand, y-up) -> robot(right-hand, z-up); pos = xyz @ R.T ; quat = Rq * q
    from scipy.spatial.transform import Rotation as R

    RMAT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    RQ_WXYZ = R.from_matrix(RMAT).as_quat(scalar_first=True)  # wxyz

    def unity_to_robot_frame(b7):
        """b7 = [x,y,z, qx,qy,qz,qw] -> (pos_robot, quat_wxyz_robot)"""
        pos = np.asarray(b7[:3], dtype=np.float64) @ RMAT.T
        q_xyzw = np.asarray(b7[3:7], dtype=np.float64)
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        # quaternion multiply scalar-first: RQ * q
        rw, rx, ry, rz = RQ_WXYZ
        qw, qx, qy, qz = q_wxyz
        out = np.array([
            rw * qw - rx * qx - ry * qy - rz * qz,
            rw * qx + rx * qw + ry * qz - rz * qy,
            rw * qy - rx * qz + ry * qw + rz * qx,
            rw * qz + rx * qy - ry * qx + rz * qw,
        ])
        return pos, out

    # Pre-transform body into robot frame (24,7) with quat wxyz for GMR + overlay
    body_robot = np.zeros_like(body)
    for j in range(24):
        p, q = unity_to_robot_frame(body[0, j])
        body_robot[0, j] = np.concatenate([p, q])
    # vectorized transform for all frames
    for i in range(1, T):
        for j in range(24):
            p, q = unity_to_robot_frame(body[i, j])
            body_robot[i, j] = np.concatenate([p, q])

    # --- GMR setup ---
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    gmr = GMR(
        src_human="xrobot",
        tgt_robot="unitree_g1",
        actual_human_height=args.height,
        verbose=False,
    )
    print("[offline] GMR ready (xrobot -> unitree_g1)")

    qpos_all = np.zeros((T, gmr.model.nq), dtype=np.float64)
    a_hw_r_all = np.zeros((T, 6), dtype=np.float64)
    a_hw_l_all = np.zeros((T, 6), dtype=np.float64)

    for i in range(T):
        human = {}
        for j, name in enumerate(BODY_JOINT_NAMES):
            b = body_robot[i, j]
            if np.abs(b[:3]).sum() < 1e-6:
                continue
            human[name] = [b[:3].copy(), b[3:].copy()]
        if len(human) < 8:
            if i > 0:
                qpos_all[i] = qpos_all[i - 1]
            else:
                qpos_all[i] = gmr.configuration.data.qpos.copy()
        else:
            qpos_all[i] = gmr.retarget(human, offset_to_ground=False)

        a_hw_r_all[i] = hand_to_a_hw(right[i, :, :3]) if right_active[i] else a_hw_r_all[max(i - 1, 0)]
        a_hw_l_all[i] = hand_to_a_hw(left[i, :, :3]) if left_active[i] else a_hw_l_all[max(i - 1, 0)]

    # --- dex-retarget mapping -> inspire q_sim_12 ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "isaaclab_twist2_g1"))
    from action_provider.inspire_mapping import RIGHT, LEFT, expand_a_hw_to_q_sim, normalize_hw_command

    q12_r = np.zeros((T, 12), dtype=np.float64)
    q12_l = np.zeros((T, 12), dtype=np.float64)
    for i in range(T):
        norm_r = normalize_hw_command(a_hw_r_all[i], source="rad", clip=True)
        norm_l = normalize_hw_command(a_hw_l_all[i], source="rad", clip=True)
        qr = expand_a_hw_to_q_sim(norm_r, side=RIGHT, unit="normalized")
        ql = expand_a_hw_to_q_sim(norm_l, side=LEFT, unit="normalized")
        q12_r[i] = [qr[n] for n in Q_SIM_12_R]
        q12_l[i] = [ql[n] for n in Q_SIM_12_L]

    print("[offline] retarget done; qpos range=", round(float(qpos_all.min()), 2), round(float(qpos_all.max()), 2))
    print("[offline] a_hw_r fist frame idx =", int(np.argmax(a_hw_r_all[:, 3])))

    # --- render video ---
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render(
        gmr, qpos_all, q12_r, q12_l, a_hw_r_all, a_hw_l_all,
        left, right, left_active, right_active,
        body_robot, out, args.fps, args.step,
    )

    report = {
        "frames": int(T),
        "rendered": f"{out / 'g1_inspire_retarget.mp4'}",
        "qpos_shape": list(qpos_all.shape),
        "q12_r_shape": list(q12_r.shape),
        "q12_l_shape": list(q12_l.shape),
        "a_hw_r_names": list(A_HW_NAMES),
        "q12_r_names": Q_SIM_12_R,
        "q12_l_names": Q_SIM_12_L,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[offline] report -> {out / 'report.json'}")


def render(gmr, qpos, q12_r, q12_l, a_hw_r, a_hw_l, left, right, la, ra, body, out, fps, step):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco as mj

    model = gmr.model
    T = len(qpos)
    idxs = list(range(0, T, step))
    tmp = out / "_frames"
    tmp.mkdir(exist_ok=True)

    # MuJoCo offscreen renderer for the G1 3D body (real mesh, materials)
    ren = mj.Renderer(model, 640, 480)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    opt = mj.MjvOption()
    data = gmr.configuration.data

    # human body overlay skeleton (XRobot names)
    BODY = [
        "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
        "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
        "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
        "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
    ]
    BODY_CHAIN = [
        ("Pelvis", "Left_Hip"), ("Pelvis", "Right_Hip"), ("Pelvis", "Spine1"),
        ("Spine1", "Spine2"), ("Spine2", "Spine3"), ("Spine3", "Neck"), ("Neck", "Head"),
        ("Neck", "Left_Collar"), ("Left_Collar", "Left_Shoulder"),
        ("Left_Shoulder", "Left_Elbow"), ("Left_Elbow", "Left_Wrist"),
        ("Left_Wrist", "Left_Hand"), ("Neck", "Right_Collar"), ("Right_Collar", "Right_Shoulder"),
        ("Right_Shoulder", "Right_Elbow"), ("Right_Elbow", "Right_Wrist"),
        ("Right_Wrist", "Right_Hand"),
        ("Left_Hip", "Left_Knee"), ("Left_Knee", "Left_Ankle"), ("Left_Ankle", "Left_Foot"),
        ("Right_Hip", "Right_Knee"), ("Right_Knee", "Right_Ankle"), ("Right_Ankle", "Right_Foot"),
    ]

    def render_g1(q, body_frame, dist, elev, azim):
        data.qpos[:] = q
        mj.mj_forward(model, data)
        cam.lookat = data.xpos[gmr.robot_body_names["pelvis"]].copy()
        cam.distance = dist
        cam.elevation = elev
        cam.azimuth = azim
        ren.update_scene(data, camera=cam)
        # human reference dots placed to the RIGHT of G1 (offset) so they don't
        # visually merge with the robot mesh; cyan = human target pose
        scn = ren.scene
        bf = np.asarray(body_frame, dtype=np.float64).reshape(24, 7)
        ref_offset = np.array([0.8, 0.0, 0.0])
        for j, name in enumerate(BODY):
            p = bf[j, :3]
            if np.abs(p).sum() < 1e-6 or scn.ngeom >= scn.maxgeom:
                continue
            geom = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(
                geom,
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.028, 0, 0],
                pos=p + ref_offset,
                mat=np.eye(3).flatten(),
                rgba=[0.0, 1.0, 1.0, 1.0],
            )
            scn.ngeom += 1
        return ren.render()

    bar_names = Q_SIM_12_R
    bar_colors = ["#9467bd", "#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "#ff7f0e",
                  "#9467bd", "#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "#ff7f0e"]

    print(f"[render] {len(idxs)} frames -> {tmp}")
    for n, i in enumerate(idxs):
        img3d = render_g1(qpos[i], body[i], dist=3.0, elev=-8, azim=0)

        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 0.85, 1.15])
        ax_b = fig.add_subplot(gs[:, 0])
        ax_lh = fig.add_subplot(gs[0, 1])
        ax_rh = fig.add_subplot(gs[1, 1])
        ax_bar = fig.add_subplot(gs[:, 2])

        # G1 3D MuJoCo render with human overlay
        ax_b.clear()
        ax_b.set_title(f"G1 3D (GMR xrobot->g1)  frame {i}/{T}  t={i/50:.1f}s  cyan=human ref (right)", fontsize=8)
        ax_b.imshow(img3d)
        ax_b.axis("off")

        # hands skeleton
        draw_hand(ax_lh, left[i, :, :3], bool(la[i]), "LEFT hand (raw 26 joints)", "L")
        draw_hand(ax_rh, right[i, :, :3], bool(ra[i]), "RIGHT hand (raw 26 joints)", "R")

        # inspire bars
        ax_bar.clear()
        vals = q12_r[i]
        vis_vals = np.where(vals < 0.01, 0.01, vals)  # keep zero joints visible
        ypos = np.arange(12)[::-1]
        ax_bar.barh(ypos, vis_vals, color=bar_colors, height=0.7)
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(bar_names, fontsize=6.5)
        ax_bar.set_xlim(0, 1.8)
        ax_bar.set_xlabel("rad", fontsize=8)
        ax_bar.set_title("inspire q_sim_12 RIGHT (mimic from a_hw_6)", fontsize=9)
        ax_bar.grid(axis="x", alpha=0.3)
        ax_bar.axvline(1.7, color="k", lw=0.5, ls="--")
        ax_bar.axvline(0.5, color="k", lw=0.5, ls="--")

        fig.suptitle(f"G1+Inspire offline retarget  t={i/50:.1f}s", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(tmp / f"f{n:05d}.png", dpi=80)
        plt.close(fig)
        if (n + 1) % 60 == 0:
            print(f"  rendered {n+1}/{len(idxs)}")

    ren.close()
    mp4 = out / "g1_inspire_retarget.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4),
    ], check=True, capture_output=True)
    import shutil

    shutil.rmtree(tmp)
    print(f"[render] wrote {mp4}")


def draw_hand(ax, kp, active, title, side):
    import numpy as np

    ax.clear()
    kp = np.asarray(kp, dtype=np.float64).reshape(26, 3)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if np.abs(kp[..., :3]).sum() < 1e-6:
        ax.text(0, 0, "NO DATA", ha="center", va="center", fontsize=8)
        return
    c = kp[0]
    pts = kp[:, :3] - c
    chains = {
        "thumb": [2, 3, 4, 5], "index": [6, 7, 8, 9, 10], "middle": [11, 12, 13, 14, 15],
        "ring": [16, 17, 18, 19, 20], "pinky": [21, 22, 23, 24, 25],
    }
    colors = {"thumb": "#ff7f0e", "index": "#d62728", "middle": "#1f77b4", "ring": "#2ca02c", "pinky": "#9467bd"}
    for name, chain in chains.items():
        ax.plot([pts[j, 0] for j in chain], [pts[j, 1] for j in chain],
                "-o", color=colors[name], lw=1.4, ms=3)
    ax.set_aspect("equal")
    pad = 0.05
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.text(0.02, 0.98, f"active={int(active)} side={side}", transform=ax.transAxes, fontsize=7, va="top")


import mujoco as mj  # noqa: E402


if __name__ == "__main__":
    main()
