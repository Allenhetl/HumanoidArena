#!/usr/bin/env python3
"""Offline G1 + Inspire MuJoCo retarget validation (merged model, real hands).

Pipeline:
  1. Body: 24-joint XRobot tracking -> GMR(xrobot->unitree_g1) -> 29-DoF qpos.
  2. Hand : 26-joint OpenXR tracking -> geometric a_hw_6 -> inspire_mapping
           expand_a_hw_to_q_sim -> 12-DoF per hand (same code path as Isaac).
  3. Drive the MERGED MuJoCo model (GMR g1_mocap_29dof body + inspire hand 24 joints).
  4. Render two videos:
       - *_overlay.mp4  : cyan human reference skeleton beside the robot
       - *_clean.mp4    : no overlay

Run on remote (has GMR + gmr env + merged model):
  conda activate gmr
  export GMR_ROOT=/home/dreams/Users/taowen/HumanoidArena-inspire-ws/GMR
  python offline_g1_inspire_mujoco.py --npz <recording.npz> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

GMR_ROOT = Path("/home/dreams/Users/taowen/HumanoidArena-inspire-ws/GMR")
sys.path.insert(0, str(GMR_ROOT))
sys.path.insert(0, "/home/dreams/Users/taowen/inspire_validation_lib")
sys.path.insert(0, "/home/dreams/Users/taowen/HumanoidArena-inspire-ws/isaaclab_twist2_g1")

BODY_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
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

# Merged model path
MERGED_MODEL = "/home/dreams/Users/taowen/inspire_validation_lib/build/g1_29dof_with_inspire_hand.xml"

# Human overlay chain (XRobot body names)
BODY_CHAIN = [
    ("Pelvis", "Left_Hip"), ("Pelvis", "Right_Hip"), ("Pelvis", "Spine1"),
    ("Spine1", "Spine2"), ("Spine2", "Spine3"), ("Spine3", "Neck"), ("Neck", "Head"),
    ("Neck", "Left_Collar"), ("Left_Collar", "Left_Shoulder"),
    ("Left_Shoulder", "Left_Elbow"), ("Left_Elbow", "Left_Wrist"), ("Left_Wrist", "Left_Hand"),
    ("Neck", "Right_Collar"), ("Right_Collar", "Right_Shoulder"),
    ("Right_Shoulder", "Right_Elbow"), ("Right_Elbow", "Right_Wrist"), ("Right_Wrist", "Right_Hand"),
    ("Left_Hip", "Left_Knee"), ("Left_Knee", "Left_Ankle"), ("Left_Ankle", "Left_Foot"),
    ("Right_Hip", "Right_Knee"), ("Right_Knee", "Right_Ankle"), ("Right_Ankle", "Right_Foot"),
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
    ap.add_argument("--out", default="/home/dreams/Users/taowen/retarget_validation_mujoco")
    ap.add_argument("--height", type=float, default=1.76)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    import mujoco as mj
    from scipy.spatial.transform import Rotation as R

    # ---- load recording ----
    d = np.load(args.npz)
    body = d["body"]
    left = d["left_hand"]
    right = d["right_hand"]
    left_active = d["left_active"].astype(bool)
    right_active = d["right_active"].astype(bool)
    T = len(body)
    print(f"[offline] frames={T} body={body.shape} hand={left.shape}")

    # ---- Unity->robot transform (same as XRobotStreamer) ----
    RMAT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    RQ_WXYZ = R.from_matrix(RMAT).as_quat(scalar_first=True)

    def unity_to_robot_frame(b7):
        pos = np.asarray(b7[:3], dtype=np.float64) @ RMAT.T
        q_xyzw = np.asarray(b7[3:7], dtype=np.float64)
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        rw, rx, ry, rz = RQ_WXYZ
        qw, qx, qy, qz = q_wxyz
        out = np.array([
            rw * qw - rx * qx - ry * qy - rz * qz,
            rw * qx + rx * qw + ry * qz - rz * qy,
            rw * qy - rx * qz + ry * qw + rz * qx,
            rw * qz + rx * qy - ry * qx + rz * qw,
        ])
        return pos, out

    body_robot = np.zeros_like(body)
    for i in range(T):
        for j in range(24):
            p, q = unity_to_robot_frame(body[i, j])
            body_robot[i, j] = np.concatenate([p, q])

    # ---- GMR setup ----
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    gmr = GMR(src_human="xrobot", tgt_robot="unitree_g1", actual_human_height=args.height, verbose=False)
    gmr_body_dof = [nm for nm in gmr.robot_dof_names if nm != "pelvis"]

    # ---- merged model ----
    model = mj.MjModel.from_xml_path(MERGED_MODEL)
    data = mj.MjData(model)
    merged_joints = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    merged_body = [j for j in merged_joints if j != "pelvis" and not any(
        x in j for x in ["_index_", "_middle_", "_ring_", "_pinky_", "_thumb_"])]
    assert merged_body == gmr_body_dof, f"body joint mismatch:\n{merged_body}\n{gmr_body_dof}"
    # IMPORTANT: qpos indexing must use jnt_qposadr (freejoint occupies qpos[0:7])
    body_qpos_idx = [model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, nm)] for nm in gmr_body_dof]
    hand_qpos_idx = {nm: model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, nm)] for nm in Q_SIM_12_R + Q_SIM_12_L}
    print(f"[offline] merged model OK: nq={model.nq} body joints={len(body_qpos_idx)} hand joints={len(hand_qpos_idx)}")

    # ---- per-frame full qpos for merged model ----
    full_qpos = np.zeros((T, model.nq), dtype=np.float64)
    q12_r = np.zeros((T, 12), dtype=np.float64)
    q12_l = np.zeros((T, 12), dtype=np.float64)
    a_hw_r_all = np.zeros((T, 6), dtype=np.float64)
    a_hw_l_all = np.zeros((T, 6), dtype=np.float64)

    from action_provider.inspire_mapping import LEFT, RIGHT, expand_a_hw_to_q_sim, normalize_hw_command

    for i in range(T):
        human = {}
        for j, name in enumerate(BODY_JOINT_NAMES):
            b = body_robot[i, j]
            if np.abs(b[:3]).sum() < 1e-6:
                continue
            human[name] = [b[:3].copy(), b[3:].copy()]
        if len(human) < 8:
            if i > 0:
                full_qpos[i] = full_qpos[i - 1]
                q12_r[i] = q12_r[i - 1]
                q12_l[i] = q12_l[i - 1]
                a_hw_r_all[i] = a_hw_r_all[i - 1]
                a_hw_l_all[i] = a_hw_l_all[i - 1]
            continue
        q_gmr = gmr.retarget(human, offset_to_ground=True)
        # root position: keep GMR's, but ground it (feet at z=0) handled by offset_to_ground
        full_qpos[i, 0] = q_gmr[0]
        full_qpos[i, 1:4] = q_gmr[1:4]
        full_qpos[i, 4:7] = q_gmr[4:7]
        full_qpos[i, body_qpos_idx] = q_gmr[7:36]

        a_r = hand_to_a_hw(right[i, :, :3]) if right_active[i] else (a_hw_r_all[i - 1] if i else np.zeros(6))
        a_l = hand_to_a_hw(left[i, :, :3]) if left_active[i] else (a_hw_l_all[i - 1] if i else np.zeros(6))
        a_hw_r_all[i] = a_r
        a_hw_l_all[i] = a_l
        qr = expand_a_hw_to_q_sim(normalize_hw_command(a_r, source="rad", clip=True), side=RIGHT, unit="normalized")
        ql = expand_a_hw_to_q_sim(normalize_hw_command(a_l, source="rad", clip=True), side=LEFT, unit="normalized")
        q12_r[i] = [qr[n] for n in Q_SIM_12_R]
        q12_l[i] = [ql[n] for n in Q_SIM_12_L]
        full_qpos[i, [hand_qpos_idx[n] for n in Q_SIM_12_R]] = q12_r[i]
        full_qpos[i, [hand_qpos_idx[n] for n in Q_SIM_12_L]] = q12_l[i]

    print("[offline] retarget done; full_qpos range=", round(float(full_qpos.min()), 2), round(float(full_qpos.max()), 2))

    # ---- render two videos ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for overlay in (True, False):
        render_video(
            model, full_qpos, q12_r, q12_l, a_hw_r_all, a_hw_l_all,
            left, right, left_active, right_active, body_robot,
            out, args.fps, args.step, overlay=overlay,
        )

    report = {
        "frames": int(T),
        "merged_model": MERGED_MODEL,
        "full_qpos_shape": list(full_qpos.shape),
        "q12_r_shape": list(q12_r.shape),
        "q12_l_shape": list(q12_l.shape),
        "body_joints_verified": merged_body == gmr_body_dof,
        "videos": {
            "overlay": str(out / "g1_inspire_overlay.mp4"),
            "clean": str(out / "g1_inspire_clean.mp4"),
        },
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[offline] report -> {out / 'report.json'}")


def render_video(model, full_qpos, q12_r, q12_l, a_hw_r, a_hw_l, left, right, la, ra, body,
                 out, fps, step, overlay):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco as mj

    T = len(full_qpos)
    idxs = list(range(0, T, step))
    tmp = out / "_frames"
    tmp.mkdir(exist_ok=True)

    ren = mj.Renderer(model, 640, 480)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    data = mj.MjData(model)
    pel_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")

    bar_names = Q_SIM_12_R
    bar_colors = ["#9467bd", "#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "#ff7f0e",
                  "#9467bd", "#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "#ff7f0e"]

    suffix = "overlay" if overlay else "clean"
    print(f"[render:{suffix}] {len(idxs)} frames -> {tmp}")

    for n, i in enumerate(idxs):
        data.qpos[:] = full_qpos[i]
        mj.mj_forward(model, data)
        cam.lookat = data.xpos[pel_body].copy()
        cam.distance = 3.0
        cam.elevation = -8
        cam.azimuth = 0
        ren.update_scene(data, camera=cam)

        if overlay:
            scn = ren.scene
            bf = np.asarray(body[i], dtype=np.float64).reshape(24, 7)
            pos_map = {}
            for j, name in enumerate(BODY_JOINT_NAMES):
                p = bf[j, :3]
                if np.abs(p).sum() < 1e-6:
                    continue
                pos_map[name] = p.copy()
            # Ground the human reference the same way GMR offset_human_data_to_ground does:
            # lowest Foot z -> 0.1, so it aligns with the grounded G1.
            foot_z = [pos_map[n][2] for n in pos_map if "Foot" in n or "foot" in n]
            ground_dz = (min(foot_z) - 0.1) if foot_z else 0.0
            off = np.array([0.9, 0.0, 0.0])
            for a, bname in BODY_CHAIN:
                if a in pos_map and bname in pos_map and scn.ngeom + 2 < scn.maxgeom:
                    for p in (pos_map[a], pos_map[bname]):
                        pz = np.array([p[0], p[1], p[2] - ground_dz])
                        geom = scn.geoms[scn.ngeom]
                        mj.mjv_initGeom(
                            geom, type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0],
                            pos=pz + off, mat=np.eye(3).flatten(), rgba=[0.0, 1.0, 1.0, 1.0],
                        )
                        scn.ngeom += 1
        img3d = ren.render()

        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 0.85, 1.15])
        ax_b = fig.add_subplot(gs[:, 0])
        ax_lh = fig.add_subplot(gs[0, 1])
        ax_rh = fig.add_subplot(gs[1, 1])
        ax_bar = fig.add_subplot(gs[:, 2])

        ax_b.clear()
        ax_b.set_title(
            f"G1+Inspire (merged MJCF)  frame {i}/{T}  t={i/50:.1f}s" + ("  cyan=human ref" if overlay else ""),
            fontsize=8,
        )
        ax_b.imshow(img3d)
        ax_b.axis("off")

        draw_hand(ax_lh, left[i, :, :3], bool(la[i]), "LEFT inspire (raw 26)", "L")
        draw_hand(ax_rh, right[i, :, :3], bool(ra[i]), "RIGHT inspire (raw 26)", "R")

        ax_bar.clear()
        vals = q12_r[i]
        vis = np.where(vals < 0.01, 0.01, vals)
        ypos = np.arange(12)[::-1]
        ax_bar.barh(ypos, vis, color=bar_colors, height=0.7)
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(bar_names, fontsize=6.5)
        ax_bar.set_xlim(0, 1.8)
        ax_bar.set_xlabel("rad", fontsize=8)
        ax_bar.set_title("inspire q_sim_12 RIGHT", fontsize=9)
        ax_bar.grid(axis="x", alpha=0.3)
        ax_bar.axvline(1.7, color="k", lw=0.5, ls="--")
        ax_bar.axvline(0.5, color="k", lw=0.5, ls="--")

        fig.suptitle(f"G1+Inspire offline retarget ({suffix})  t={i/50:.1f}s", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(tmp / f"f{n:05d}.png", dpi=80)
        plt.close(fig)
        if (n + 1) % 60 == 0:
            print(f"  [{suffix}] rendered {n+1}/{len(idxs)}")

    ren.close()
    mp4 = out / f"g1_inspire_{suffix}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4),
    ], check=True, capture_output=True)
    shutil.rmtree(tmp)
    print(f"[render:{suffix}] wrote {mp4}")


def draw_hand(ax, kp, active, title, side):
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
        ax.plot([pts[j, 0] for j in chain], [pts[j, 1] for j in chain], "-o", color=colors[name], lw=1.4, ms=3)
    ax.set_aspect("equal")
    pad = 0.05
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.text(0.02, 0.98, f"active={int(active)} side={side}", transform=ax.transAxes, fontsize=7, va="top")


if __name__ == "__main__":
    main()
