# Copyright (c) 2025. All Rights Reserved.
# License: Apache License, Version 2.0
"""SonicActionProvider: POSE 模式全身遥操作，驱动 Isaac Lab 仿真。

POSE 模式数据流：
  Pico 头显 + 手腕控制器 + 脚踝 tracker
    → pico_manager_thread_server.py  (gear_sonic 侧，独立运行)
    → ZMQ "pose" topic（含完整 SMPL 数据）
    → SonicActionProvider._fetch_zmq_pose()
    → GEAR-SONIC ONNX encoder + decoder（全身 retargeting）
    → Isaac Lab 全身关节目标（29 DOF，含腿部）

关节顺序说明：
  - SONIC encoder/decoder 输入输出都是 SONIC IsaacLab order
  - 这个顺序对应 TWIST2 的 old_action_joints_names
  - 在 Isaac Lab 仿真中不需要顺序转换
  - 只有在真实硬件执行时才会转换成 MuJoCo/hardware order

ZMQ "pose" 消息关键字段（POSE 模式，Protocol v3）：
  smpl_joints   : (N, 24, 3)  float32  SMPL 24 关节局部坐标
  smpl_pose     : (N, 21, 3)  float32  SMPL 21 关节轴角旋转
  body_quat_w   : (N, 4)      float32  全局朝向四元数 [qw,qx,qy,qz]
  joint_pos     : (N, 29)     float32  G1 机器人关节位置（SONIC IsaacLab order）
  joint_vel     : (N, 29)     float32  G1 机器人关节速度（SONIC IsaacLab order）
  left_hand_joints  : (M,)    float32  左手关节
  right_hand_joints : (M,)    float32  右手关节

Usage:
    python sim_main.py --action_source sonic_wholebody \\
        --sonic_zmq_host localhost --sonic_zmq_port 5556 \\
        --sonic_encoder_path /path/to/model_encoder.onnx \\
        --sonic_decoder_path /path/to/model_decoder.onnx \\
        --task Isaac-Move-Cylinder-G129-Dex3-Wholebody \\
        --robot_type g129 --enable_dex3_dds --device cuda
"""

import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from action_provider.action_base import ActionProvider

# ---------------------------------------------------------------------------
# Resolve gear_sonic package path
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TWIST2_ROOT = os.path.dirname(_THIS_DIR)
# Fixed: GR00T-WholeBodyControl is at /home/dreams/Users/taowen/GR00T-WholeBodyControl
# not under HumanoidArena/
_GROOT_ROOT = "/home/dreams/Users/taowen/GR00T-WholeBodyControl"
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

try:
    from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
    _HAS_ZMQ_POLLER = True
except ImportError:
    _HAS_ZMQ_POLLER = False
    print("[SonicActionProvider] WARNING: gear_sonic ZMQPoller not found.")

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False
    print("[SonicActionProvider] WARNING: onnxruntime not found.")

# ---------------------------------------------------------------------------
# ZMQ pose 消息解析（Protocol v3，1280-byte JSON header + binary payload）
# ---------------------------------------------------------------------------
_HEADER_SIZE = 1280


def _parse_zmq_pose(raw: bytes) -> Optional[dict]:
    """解析 ZMQ 'pose' 消息，返回 {field_name: np.ndarray} 或 None。"""
    try:
        topic_end = raw.index(b"{")
        header_bytes = raw[topic_end: topic_end + _HEADER_SIZE]
        payload = raw[topic_end + _HEADER_SIZE:]
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))

        _DTYPE_MAP = {
            "f32": (np.float32, 4), "f64": (np.float64, 8),
            "i32": (np.int32, 4),   "i64": (np.int64, 8),
            "u8":  (np.uint8,  1),  "bool": (np.bool_,  1),
        }
        result, offset = {}, 0
        for f in header.get("fields", []):
            np_dtype, itemsize = _DTYPE_MAP.get(f["dtype"], (np.float32, 4))
            n = int(np.prod(f["shape"]))
            arr = np.frombuffer(payload[offset: offset + n * itemsize],
                                dtype=np_dtype).reshape(f["shape"])
            result[f["name"]] = arr
            offset += n * itemsize
        return result
    except Exception as e:
        print(f"[SonicActionProvider] pose parse error: {e}")
        return None


def quat_to_rotation_6d(quat: np.ndarray) -> np.ndarray:
    """
    将四元数转换为6D旋转表示（旋转矩阵的前2列，按行展开）

    这是SONIC encoder期望的输入格式，与C++实现保持一致。
    参考: gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:514-580

    Args:
        quat: (..., 4) 四元数 [w, x, y, z]

    Returns:
        rot6d: (..., 6) 6D旋转表示 [R00, R01, R10, R11, R20, R21]
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # 四元数转旋转矩阵（前2列）
    # 第0列
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    # 第1列
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    # 按行展开：[第0行的前2列, 第1行的前2列, 第2行的前2列]
    rot6d = np.stack([r00, r01, r10, r11, r20, r21], axis=-1)

    return rot6d.astype(np.float32)


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion from xyzw to wxyz."""
    quat = np.asarray(quat, dtype=np.float32)
    return np.concatenate([quat[..., 3:4], quat[..., 0:3]], axis=-1).astype(np.float32)


def quat_normalize_wxyz(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion in wxyz format."""
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / np.clip(norm, 1e-12, None)).astype(np.float32)


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    """Quaternion conjugate in wxyz format."""
    quat = np.asarray(quat, dtype=np.float32)
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out.astype(np.float32)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply in wxyz format."""
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    out = np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )
    return quat_normalize_wxyz(out)


def quat_angle_deg_wxyz(quat: np.ndarray) -> float:
    """Return rotation angle in degrees for a wxyz quaternion."""
    quat = quat_normalize_wxyz(quat)
    w = float(np.clip(np.abs(quat[..., 0]), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def quat_heading_wxyz(quat: np.ndarray) -> np.ndarray:
    """Extract z-up yaw-only quaternion in wxyz format."""
    quat = quat_normalize_wxyz(quat)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = 0.5 * yaw
    out = np.stack(
        [np.cos(half_yaw), np.zeros_like(half_yaw), np.zeros_like(half_yaw), np.sin(half_yaw)],
        axis=-1,
    )
    return quat_normalize_wxyz(out)


def quat_heading_inv_wxyz(quat: np.ndarray) -> np.ndarray:
    """Inverse of yaw-only quaternion in wxyz format."""
    return quat_conjugate_wxyz(quat_heading_wxyz(quat))


def array_range_str(arr: np.ndarray) -> str:
    """Compact range formatter for debug logs."""
    arr = np.asarray(arr)
    return f"[{arr.min():.4f}, {arr.max():.4f}]"


# ---------------------------------------------------------------------------
# Redis mode: locally port pico_manager_thread_server PoseStreamer.run_once
# (single-frame version, output keys compatible with _apply_pose_data()).
# ---------------------------------------------------------------------------

try:
    from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_quaternion,
        compute_human_joints,
        quat_apply,
        quat_inv,
        quaternion_to_angle_axis,
        quaternion_to_rotation_matrix,
    )
except ImportError as e:
    decompose_rotation_aa = None
    angle_axis_to_quaternion = None
    compute_human_joints = None
    quat_apply = None
    quat_inv = None
    quaternion_to_angle_axis = None
    quaternion_to_rotation_matrix = None
    print(f"[SonicActionProvider] WARNING: gear_sonic trl utils import failed: {e}")

try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None


_PICO_PARENT_INDICES = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
][:24]


def _process_smpl_joints(
    body_pose: torch.Tensor,
    global_orient: torch.Tensor,
    transl: torch.Tensor,
) -> dict:
    """
    Ported (minimal) from pico_manager_thread_server.py::process_smpl_joints.

    Args:
        body_pose: (1, 63) axis-angle for 21 body joints (excludes global root)
        global_orient: (1, 3) axis-angle for the root
        transl: (1, 3)
    """
    if smpl_root_ytoz_up is not None:
        global_orient_quat = angle_axis_to_quaternion(global_orient)
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
        global_orient_new = quaternion_to_angle_axis(global_orient_quat)
    else:
        # Keep the same semantics when optional transform isn't available.
        global_orient_quat = angle_axis_to_quaternion(global_orient)
        global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )  # (1, 24, 3)

    # Remove SMPL base rotation if configured.
    if remove_smpl_base_rot is not None:
        # pico uses w_last=False (i.e. quaternion is [w, x, y, z]).
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)  # (1,24,3)

    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)  # (1,3,3)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        # Raw SMPL body pose parameters (axis-angle), used later to compute wrists.
        "smpl_pose": body_pose,
        # Root-local joint positions (encoder uses smpl_joints_flat).
        "smpl_joints_local": smpl_joints_local,
        # Root orientation quaternion ([w, x, y, z] expected by action_provider_sonic).
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
        "adjusted_transl": transl,
    }


def _compute_from_body_poses(
    parent_indices: list[int],
    device: torch.device,
    body_poses_np: np.ndarray,
) -> dict:
    """
    Ported (minimal) from pico_manager_thread_server.py::compute_from_body_poses.

    Args:
        body_poses_np: (24, 7), each row [x,y,z,qx,qy,qz,qw] (scalar-last).
    """
    positions = body_poses_np[:, :3]  # (24,3)
    # Convert [qx,qy,qz,qw] -> [qw,qx,qy,qz] for scalar_first quaternion input.
    global_quats = body_poses_np[:, [6, 3, 4, 5]]  # (24,4)

    global_rots = R.from_quat(global_quats, scalar_first=True)
    global_rots = global_rots * R.from_euler("y", 180, degrees=True)

    local_rots = []
    for i in range(24):
        if parent_indices[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rot = global_rots[parent_indices[i]].inv() * global_rots[i]
            local_rots.append(local_rot)

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots], dtype=np.float32)  # (24,3)

    # Root orientation (axis-angle) + body joint axis-angles.
    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)  # (1,63)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)  # (1,3)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)  # (1,3)

    return _process_smpl_joints(body_pose, global_orient, transl)


def _process_3pt_pose(smpl_pose_np: np.ndarray) -> np.ndarray:
    """
    Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full SMPL body joint poses.
    Ported from pico_manager_thread_server.py::_process_3pt_pose.

    Args:
        smpl_pose_np: np.ndarray shape (24, 7) - 24 SMPL joints, each [x, y, z, qx, qy, qz, qw]
                      in Unity frame (scalar-last quaternion format)

    Returns:
        vr_3pt_pose: np.ndarray shape (3, 7) - 3 keypoints in robot frame
                     Each row is [x, y, z, qw, qx, qy, qz] (scalar-FIRST quaternion format)
                     Row 0: Left Wrist (SMPL joint 22)
                     Row 1: Right Wrist (SMPL joint 23)
                     Row 2: Neck (SMPL joint 12)
    """
    from scipy.spatial.transform import Rotation as sRot

    # Rotation offsets for each keypoint
    OFFSETS = [
        sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root
        sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist
        sRot.from_euler("xyz", [-90, 0, 180], degrees=True),  # R-Wrist
        sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck
    ]

    def _compute_rel_transform(pose, world_frame, scalar_first=True):
        """Transform pose from Unity to robot frame."""
        world_frame = world_frame.copy()
        Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
        pose[:3] = Q @ pose[:3]
        world_frame[:3] = Q @ world_frame[:3]
        rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
        rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
        rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
        rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
        return rel_pos, rel_rot.as_quat(scalar_first=True)

    # Defensive copy
    smpl_pose_np = smpl_pose_np.copy()

    # Transform all joints from Unity to robot frame
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos
        body_poses[i, 3:] = orn

    # Extract keypoints and apply offsets
    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue
        pos = positions[i]
        rel_i = [0, 22, 23, 12].index(i)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )
        kp_poses[rel_i, 3:] = rot_quat
        kp_poses[rel_i, :3] = pos

    # Make relative to root
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()

    for i in range(1, 4):
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # Return L-Wrist, R-Wrist, Neck (skip Root)
    return kp_poses[1:]


def _compute_wrist_joint_pos_from_smpl_pose(smpl_pose_np: np.ndarray) -> np.ndarray:
    """
    Ported (minimal) from pico_manager_thread_server.py::run_once.
    Writes wrist (roll/pitch/yaw) joints into the 29D SONIC IsaacLab order.

    Note:
        This matches pico's "directly setting the joint position" behavior:
        other joints remain 0.
    """
    if decompose_rotation_aa is None:
        raise RuntimeError("decompose_rotation_aa is not available (gear_sonic not installed?)")

    # Expected: smpl_pose_np shape (21,3), axis-angle (rotvec) for 21 joints.
    body_pose = smpl_pose_np.reshape(-1, 21, 3)  # (1,21,3)

    SMPL_L_ELBOW_IDX = 17
    SMPL_L_WRIST_IDX = 19
    SMPL_R_ELBOW_IDX = 18
    SMPL_R_WRIST_IDX = 20

    G1_L_WRIST_ROLL_IDX = 23
    G1_L_WRIST_PITCH_IDX = 25
    G1_L_WRIST_YAW_IDX = 27
    G1_R_WRIST_ROLL_IDX = 24
    G1_R_WRIST_PITCH_IDX = 26
    G1_R_WRIST_YAW_IDX = 28

    joint_pos = np.zeros(29, dtype=np.float32)

    smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]  # (1,3)
    smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]  # (1,3)
    smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]  # (1,3)
    smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]  # (1,3)

    g1_l_elbow_axis = np.array([0, 1, 0], dtype=np.float32)
    g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, g1_l_elbow_axis)

    g1_r_elbow_axis = np.array([0, 1, 0], dtype=np.float32)
    g1_r_elbow_q_twist, g1_r_elbow_q_swing = decompose_rotation_aa(smpl_r_elbow_aa, g1_r_elbow_axis)

    # Move elbow roll/yaw into wrist while preserving wrist pitch from SMPL.
    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)

    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    g1_l_wrist_pitch = -l_wrist_euler[:, 1]
    g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    g1_r_wrist_pitch = -r_wrist_euler[:, 1]
    g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0].astype(np.float32)
    joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0].astype(np.float32)
    joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0].astype(np.float32)

    joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0].astype(np.float32)
    joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0].astype(np.float32)
    joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0].astype(np.float32)

    return joint_pos


def _pico_single_frame_from_body_poses(
    *,
    device: torch.device,
    body_poses_24x7: np.ndarray,
    frame_index: int,
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
    joint_pos: Optional[np.ndarray],
    joint_vel: Optional[np.ndarray],
) -> dict:
    """
    Output dict compatible with SonicActionProvider._apply_pose_data().

    This function processes raw SMPL body poses and outputs a data dict that matches
    the format expected by pico_manager_thread_server.py::PoseStreamer.run_once.
    """
    if decompose_rotation_aa is None or compute_human_joints is None:
        raise RuntimeError("gear_sonic trl utils not available; cannot compute smpl fields.")

    latest_data = _compute_from_body_poses(_PICO_PARENT_INDICES, device, body_poses_24x7.astype(np.float32))

    smpl_pose_np = latest_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0].astype(np.float32)
    smpl_joints_np = latest_data["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
    body_quat_np = latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
    adjusted_transl_np = latest_data["adjusted_transl"].detach().cpu().numpy()[0].astype(np.float32)

    # Wrist joints from SMPL pose (roll/pitch/yaw only).
    joint_pos_smpl = _compute_wrist_joint_pos_from_smpl_pose(smpl_pose_np)
    joint_vel_out = np.zeros((1, 29), dtype=np.float32)

    # Teleop/Redis usually doesn't provide full robot joint_pos/joint_vel.
    # Keep the same strategy as existing redis code: use simulator proprioception,
    # then overwrite wrist joints with SMPL-derived wrists.
    if joint_pos is not None and joint_vel is not None:
        joint_pos_out = joint_pos.astype(np.float32).copy()
        wrist_indices = [23, 24, 25, 26, 27, 28]
        joint_pos_out[wrist_indices] = joint_pos_smpl[wrist_indices]
        joint_vel_out = joint_vel.astype(np.float32).copy()[None, :]
    else:
        joint_pos_out = joint_pos_smpl
        joint_vel_out = np.zeros((1, 29), dtype=np.float32)

    # ✨ CRITICAL: Compute VR 3-point pose (L-Wrist, R-Wrist, Neck)
    # This is required for SONIC encoder input and matches pico_manager_thread_server.py::run_once
    vr_3pt_pose = _process_3pt_pose(body_poses_24x7)  # (3, 7) - [L-Wrist, R-Wrist, Neck]

    data: dict = {
        "smpl_pose": smpl_pose_np[None, :, :],  # (1,21,3)
        "smpl_joints": smpl_joints_np[None, :, :],  # (1,24,3)
        "body_quat_w": body_quat_np[None, :],  # (1,4) wxyz
        "adjusted_transl": adjusted_transl_np[None, :],  # (1,3)
        "joint_pos": joint_pos_out[None, :],  # (1,29)
        "joint_vel": joint_vel_out,  # (1,29)
        "frame_index": np.array([frame_index], dtype=np.int64),
        # VR 3-point pose data (aligned with pico_manager_thread_server.py output)
        "vr_position": vr_3pt_pose[:, :3].flatten().astype(np.float32),  # (9,) - 3 points × 3D position
        "vr_orientation": vr_3pt_pose[:, 3:].flatten().astype(np.float32),  # (12,) - 3 points × 4D quat (wxyz)
    }

    if left_hand is not None:
        data["left_hand_joints"] = np.asarray(left_hand, dtype=np.float32).reshape(-1)
    if right_hand is not None:
        data["right_hand_joints"] = np.asarray(right_hand, dtype=np.float32).reshape(-1)

    return data


# ---------------------------------------------------------------------------
# SONIC 关节顺序（SONIC IsaacLab order，对应 GR00T 部署代码中的 IsaacLab order）
# 注意：SONIC encoder/decoder 输入输出都是这个顺序，不需要转换
# 这个顺序对应 TWIST2 的 old_action_joints_names
# ---------------------------------------------------------------------------
SONIC_ISAACLAB_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# SONIC 默认站立姿态（按 SONIC IsaacLab order）
# 参考 GR00T policy_parameters.hpp 中的 default_angles（MuJoCo order）
# 已转换为 SONIC IsaacLab order
SONIC_DEFAULT_POS = np.array([
    -0.2,    # left_hip_pitch_joint (TWIST2 腿部姿态)
    -0.2,    # right_hip_pitch_joint (TWIST2 腿部姿态)
    0.0,     # waist_yaw_joint
    0.0,     # left_hip_roll_joint
    0.0,     # right_hip_roll_joint
    0.0,     # waist_roll_joint
    0.0,     # left_hip_yaw_joint
    0.0,     # right_hip_yaw_joint
    0.0,     # waist_pitch_joint
    0.4,     # left_knee_joint (TWIST2 腿部姿态)
    0.4,     # right_knee_joint (TWIST2 腿部姿态)
    0.0,     # left_shoulder_pitch_joint (URDF 零位)
    0.0,     # right_shoulder_pitch_joint (URDF 零位)
    -0.2,    # left_ankle_pitch_joint (TWIST2 腿部姿态)
    -0.2,    # right_ankle_pitch_joint (TWIST2 腿部姿态)
    0.0,     # left_shoulder_roll_joint (URDF 零位，手臂在躯干两侧)
    0.0,     # right_shoulder_roll_joint (URDF 零位，手臂在躯干两侧)
    0.0,     # left_ankle_roll_joint
    0.0,     # right_ankle_roll_joint
    0.0,     # left_shoulder_yaw_joint
    0.0,     # right_shoulder_yaw_joint
    0.0,     # left_elbow_joint (URDF 零位，屈肘状态)
    0.0,     # right_elbow_joint (URDF 零位，屈肘状态)
    0.0,     # left_wrist_roll_joint
    0.0,     # right_wrist_roll_joint
    0.0,     # left_wrist_pitch_joint
    0.0,     # right_wrist_pitch_joint
    0.0,     # left_wrist_yaw_joint
    0.0,     # right_wrist_yaw_joint
], dtype=np.float32)

# SONIC 动作缩放系数（按 SONIC IsaacLab order）
# 参考 GR00T policy_parameters.hpp 中的 g1_action_scale（MuJoCo order）
# 已转换为 SONIC IsaacLab order
# 公式: action_scale = 0.25 * effort_limit / stiffness
G1_ACTION_SCALE_ISAACLAB = np.array([
    0.3506614566,  # 0: left_hip_pitch_joint
    0.3506614566,  # 1: right_hip_pitch_joint
    0.5475464463,  # 2: waist_yaw_joint
    0.3506614566,  # 3: left_hip_roll_joint
    0.3506614566,  # 4: right_hip_roll_joint
    0.4385773242,  # 5: waist_roll_joint
    0.5475464463,  # 6: left_hip_yaw_joint
    0.5475464463,  # 7: right_hip_yaw_joint                                                                        d
    0.4385773242,  # 8: waist_pitch_joint
    0.3506614566,  # 9: left_knee_joint
    0.3506614566,  # 10: right_knee_joint
    0.4385773242,  # 11: left_shoulder_pitch_joint
    0.4385773242,  # 12: right_shoulder_pitch_joint
    0.4385773242,  # 13: left_ankle_pitch_joint
    0.4385773242,  # 14: right_ankle_pitch_joint
    0.4385773242,  # 15: left_shoulder_roll_joint
    0.4385773242,  # 16: right_shoulder_roll_joint
    0.4385773242,  # 17: left_ankle_roll_joint
    0.4385773242,  # 18: right_ankle_roll_joint
    0.4385773242,  # 19: left_shoulder_yaw_joint
    0.4385773242,  # 20: right_shoulder_yaw_joint
    0.4385773242,  # 21: left_elbow_joint
    0.4385773242,  # 22: right_elbow_joint
    0.4385773242,  # 23: left_wrist_roll_joint
    0.4385773242,  # 24: right_wrist_roll_joint
    0.0745008737,  # 25: left_wrist_pitch_joint
    0.0745008737,  # 26: right_wrist_pitch_joint
    0.0745008737,  # 27: left_wrist_yaw_joint
    0.0745008737,  # 28: right_wrist_yaw_joint
], dtype=np.float32)

# SMPL 参数维度
_N_SMPL_JOINTS = 24   # smpl_joints: (N, 24, 3)
_N_SMPL_POSES  = 21   # smpl_pose:   (N, 21, 3)
_STEP1_FRAMES = 10
_STEP5_FRAMES = 10
_STEP5_STRIDE = 5
_STEP5_HISTORY_LEN = (_STEP5_FRAMES - 1) * _STEP5_STRIDE + 1

OFFICIAL_WRIST_INDICES = [23, 24, 25, 26, 27, 28]
OFFICIAL_LOWERBODY_INDICES = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]
SMPL_MODE_ACTIVE_BLOCKS = [
    "encoder_mode_4",
    "smpl_joints_10frame_step1",
    "smpl_anchor_orientation_10frame_step1",
    "motion_joint_positions_wrists_10frame_step1",
]
SMPL_MODE_ZEROED_BLOCKS = [
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    "motion_root_z_position_10frame_step5",
    "motion_root_z_position",
    "motion_anchor_orientation",
    "motion_anchor_orientation_10frame_step5",
    "motion_joint_positions_lowerbody_10frame_step5",
    "motion_joint_velocities_lowerbody_10frame_step5",
    "vr_3point_local_target",
    "vr_3point_local_orn_target",
]


def gather_temporal_window(hist: np.ndarray, num_frames: int, stride: int) -> np.ndarray:
    """Take the latest temporal window using `stride` over a history buffer."""
    hist = np.asarray(hist, dtype=np.float32)
    required = (num_frames - 1) * stride + 1
    if hist.shape[0] < required:
        raise ValueError(
            f"History too short for num_frames={num_frames}, stride={stride}: "
            f"len={hist.shape[0]}, required={required}"
        )
    start = hist.shape[0] - required
    window = hist[start::stride]
    if window.shape[0] != num_frames:
        raise ValueError(
            f"Temporal window shape mismatch: got {window.shape[0]}, expected {num_frames}"
        )
    return window.astype(np.float32)


class SonicActionProvider(ActionProvider):
    """POSE 模式全身遥操作 ActionProvider。

    使用 Pico 脚踝 tracker 提供的完整 SMPL 全身姿态，
    通过 GEAR-SONIC encoder+decoder ONNX 模型做全身 retargeting，
    输出 G1 机器人 29 DOF 关节目标（上半身 + 下半身均来自 SMPL 跟踪）。
    """

    def __init__(self, env, args_cli):
        super().__init__("SonicActionProvider")
        self.env = env
        self.device = env.device

        # Debug/perf knobs (默认关闭高频打印，否则会把控制环拖到个位数 Hz)
        # - SONIC_DEBUG=1: 打开详细日志
        # - SONIC_LOG_EVERY=50: 每 N 帧打印一次（仍会在前几帧打印）
        # self._sonic_debug = bool(int(os.environ.get("SONIC_DEBUG", "0") or "0"))
        self._sonic_log_every = int(os.environ.get("SONIC_LOG_EVERY", "50") or 50)
        # 也允许通过 CLI 参数覆盖（若 sim_main.py 透传了这些字段）
        # self._sonic_debug = bool(getattr(args_cli, "sonic_debug", self._sonic_debug))
        self._sonic_debug = True
        self._sonic_log_every = int(getattr(args_cli, "sonic_log_every", self._sonic_log_every))

        self.enable_dex3    = getattr(args_cli, "enable_dex3_dds",   False)
        self.enable_gripper = getattr(args_cli, "enable_dex1_dds",   False)
        self._pose_source   = getattr(args_cli, "sonic_pose_source", "zmq")  # "zmq" | "redis"
        self.zmq_host       = getattr(args_cli, "sonic_zmq_host",    "localhost")
        self.zmq_port       = getattr(args_cli, "sonic_zmq_port",    5556)
        self.redis_host     = getattr(args_cli, "sonic_redis_host",  "localhost")
        self.redis_port     = getattr(args_cli, "sonic_redis_port",  6379)
        self.encoder_path   = getattr(args_cli, "sonic_encoder_path", "")
        self.decoder_path   = getattr(args_cli, "sonic_decoder_path", "")
        # self._sonic_warmup_steps = int(getattr(args_cli, "sonic_warmup_steps", 50))  # warmup 已注释，仅用 history_ready
        self._sonic_warmup_steps = 0
        self._sonic_smooth_steps = int(getattr(args_cli, "sonic_smooth_steps", 20))
        cfg = getattr(env, "cfg", None)
        self._decimation    = int(getattr(cfg, "decimation", 4))

        self._setup_joint_mapping()
        if self._pose_source == "redis":
            self._setup_redis()
        else:
            self._setup_zmq()
        self._setup_policy()
        self._setup_buffers()
        self._setup_hand_dds(args_cli)

        print(f"[SonicActionProvider] POSE mode ready  "
              f"pose_source={self._pose_source}  "
              f"(zmq={self.zmq_host}:{self.zmq_port}  redis={self.redis_host}:{self.redis_port})  "
              f"encoder={self.encoder_path}  decoder={self.decoder_path}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_joint_mapping(self):
        all_names = list(self.env.scene["robot"].data.joint_names)
        idx_map = {n: i for i, n in enumerate(all_names)}

        # 使用 SONIC IsaacLab 关节顺序
        missing = [n for n in SONIC_ISAACLAB_JOINT_ORDER if n not in idx_map]
        if missing:
            raise ValueError(f"[SonicActionProvider] joints missing: {missing}")

        # SONIC IsaacLab 关节索引（用于读取/写入 Isaac Lab）
        self._sonic_idx = torch.tensor(
            [idx_map[n] for n in SONIC_ISAACLAB_JOINT_ORDER],
            dtype=torch.long, device=self.device)

        # 默认姿态（Isaac Lab 完整关节）
        self._default_pos = self.env.scene["robot"].data.default_joint_pos.clone()

        # 调试：打印默认姿态
        print(f"\n[DEBUG] Isaac Lab default_joint_pos (full):")
        default_sonic = self._default_pos[0, self._sonic_idx].cpu().numpy()
        print(f"  left_shoulder_pitch (idx 11): {default_sonic[11]:.3f}")
        print(f"  left_shoulder_roll (idx 15): {default_sonic[15]:.3f}")
        print(f"  left_elbow (idx 21): {default_sonic[21]:.3f}")
        print(f"  right_shoulder_pitch (idx 12): {default_sonic[12]:.3f}")
        print(f"  right_shoulder_roll (idx 16): {default_sonic[16]:.3f}")
        print(f"  right_elbow (idx 22): {default_sonic[22]:.3f}")
        self._init_sonic_target_np = default_sonic.astype(np.float32).copy()

        # SONIC 默认姿态（SONIC IsaacLab order）
        self._sonic_default_np = SONIC_DEFAULT_POS.copy()

        if self.enable_dex3:
            left_names  = ["left_hand_thumb_0_joint","left_hand_thumb_1_joint",
                           "left_hand_thumb_2_joint","left_hand_middle_0_joint",
                           "left_hand_middle_1_joint","left_hand_index_0_joint",
                           "left_hand_index_1_joint"]
            right_names = ["right_hand_thumb_0_joint","right_hand_thumb_1_joint",
                           "right_hand_thumb_2_joint","right_hand_middle_0_joint",
                           "right_hand_middle_1_joint","right_hand_index_0_joint",
                           "right_hand_index_1_joint"]
            self._left_hand_idx  = torch.tensor(
                [idx_map[n] for n in left_names  if n in idx_map],
                dtype=torch.long, device=self.device)
            self._right_hand_idx = torch.tensor(
                [idx_map[n] for n in right_names if n in idx_map],
                dtype=torch.long, device=self.device)

    def _setup_zmq(self):
        self._zmq_poller = None
        if not _HAS_ZMQ_POLLER:
            print("[SonicActionProvider] WARNING: ZMQPoller not available (_HAS_ZMQ_POLLER=False)")
            return
        try:
            print(f"[SonicActionProvider] Attempting to connect ZMQ: tcp://{self.zmq_host}:{self.zmq_port} topic=pose")
            self._zmq_poller = ZMQPoller(
                host=self.zmq_host, port=self.zmq_port, topic="pose")
            print(f"[SonicActionProvider] ✓ ZMQ connected successfully "
                  f"tcp://{self.zmq_host}:{self.zmq_port} topic=pose")
            print(f"[SonicActionProvider] ZMQ Poller object: {self._zmq_poller}")

            # Wait for ZMQ subscription to establish (fixes "slow joiner" problem)
            print("[SonicActionProvider] Waiting for ZMQ subscription to establish...")
            time.sleep(1.0)
            print("[SonicActionProvider] ZMQ subscription ready")
        except Exception as e:
            print(f"[SonicActionProvider] ✗ ZMQ init failed: {e}")
            import traceback
            traceback.print_exc()

    def _setup_redis(self):
        """Redis 直连：从 human_smplx_data_unitree_g1_with_hands 读 pose，不再经 ZMQ。"""
        self._redis_client = None
        self._redis_frame_index = 0
        if not _HAS_REDIS:
            print("[SonicActionProvider] WARNING: redis not installed. pip install redis")
            return
        try:
            self._redis_client = redis.Redis(
                host=self.redis_host, port=self.redis_port, db=0, decode_responses=False
            )
            self._redis_client.ping()
            print(f"[SonicActionProvider] Redis connected "
                  f"{self.redis_host}:{self.redis_port} key=human_smplx_data_unitree_g1_with_hands")
        except Exception as e:
            print(f"[SonicActionProvider] Redis init failed: {e}")

    def _make_session(self, path: str):
        """创建 ONNX InferenceSession，优先使用 CUDA。"""
        if not path:
            print(f"[SonicActionProvider] model path is empty, skipping load")
            return None
        if not os.path.isfile(path):
            print(f"[SonicActionProvider] model file not found: {path}")
            return None
        if not _HAS_ORT:
            return None
        # Provider 选择逻辑：
        # - 如果当前 onnxruntime 没编译 CUDA，则 get_available_providers() 不会包含 CUDAExecutionProvider
        # - 这里明确打印可用 provider，便于定位环境问题
        avail = ort.get_available_providers()
        if not hasattr(self, "_ort_avail_logged"):
            self._ort_avail_logged = True
            print(f"[SonicActionProvider] onnxruntime available_providers={avail}")

        providers: list[str] = []
        if str(self.device).startswith("cuda") and "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        try:
            sess = ort.InferenceSession(path, providers=providers)
            print(f"[SonicActionProvider] loaded {os.path.basename(path)} "
                  f"providers={sess.get_providers()}")
            return sess
        except Exception as e:
            print(f"[SonicActionProvider] failed to load {path}: {e}")
            return None

    def _setup_policy(self):
        """加载 GEAR-SONIC encoder 和 decoder ONNX 模型。"""
        self._encoder = self._make_session(self.encoder_path)
        self._decoder = self._make_session(self.decoder_path)
        print('Successful load sonic model')
        if self._encoder is None or self._decoder is None:
            print("[SonicActionProvider] encoder/decoder not loaded, "
                  "will hold default standing pose.")

    def _setup_buffers(self):
        # SMPL 历史帧缓冲（encoder 需要 10 帧）
        self._smpl_joints_buf = np.zeros(
            (_STEP1_FRAMES, _N_SMPL_JOINTS, 3), dtype=np.float32)   # (10, 24, 3)
        self._smpl_pose_buf   = np.zeros(
            (_STEP1_FRAMES, _N_SMPL_POSES,  3), dtype=np.float32)   # (10, 21, 3)
        self._body_rot6d_buf  = np.tile(
            np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32), (_STEP1_FRAMES, 1))  # (10, 6)

        # 机器人状态历史缓冲（SONIC IsaacLab order）
        # 用于 encoder 输入（step5 采样）和 decoder 输入（step1 连续10帧）
        self._robot_joint_pos_hist = np.tile(
            self._sonic_default_np[np.newaxis], (_STEP1_FRAMES, 1))  # (10, 29)
        self._robot_joint_vel_hist = np.zeros((_STEP1_FRAMES, 29), dtype=np.float32)

        # 参考 motion 历史缓冲（来自 ZMQ replay/reference，而不是当前仿真机器人）
        self._motion_joint_pos_hist = np.tile(
            self._sonic_default_np[np.newaxis], (_STEP5_HISTORY_LEN, 1)
        )
        self._motion_joint_vel_hist = np.zeros((_STEP5_HISTORY_LEN, 29), dtype=np.float32)
        self._motion_root_z_hist = np.zeros((_STEP5_HISTORY_LEN,), dtype=np.float32)
        self._motion_anchor_rot6d_hist = np.tile(
            np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32), (_STEP5_HISTORY_LEN, 1)
        )

        # Decoder 需要的额外历史缓冲
        # his_base_angular_velocity_10frame_step1: (10, 3)
        self._ang_vel_hist    = np.zeros((_STEP1_FRAMES, 3),  dtype=np.float32)
        # his_gravity_dir_10frame_step1: (10, 3)
        self._grav_dir_hist   = np.zeros((_STEP1_FRAMES, 3),  dtype=np.float32)
        # his_last_actions_10frame_step1: (10, 29)
        self._last_action_hist = np.tile(
            self._sonic_default_np[np.newaxis], (_STEP1_FRAMES, 1))  # (10, 29)

        # 手部关节目标
        self._left_hand_target  = np.zeros(7, dtype=np.float32)
        self._right_hand_target = np.zeros(7, dtype=np.float32)

        # VR 3点姿态缓冲（来自 SMPL，用于 encoder 输入）
        # vr_position: (9,) - 3 points × 3D position [L-Wrist, R-Wrist, Neck]
        # vr_orientation: (12,) - 3 points × 4D quat wxyz
        self._vr_3pt_position = np.zeros(9, dtype=np.float32)
        self._vr_3pt_orientation = np.zeros(12, dtype=np.float32)

        # encoder 输出的 latent（首次推理前为零）
        self._latent = None

        # SMPL数据有效性标志（用于检测是否接收到有效的遥操数据）
        self._smpl_data_valid = False
        self._frame_count = 0
        self._smpl_history_fill = 0
        self._anchor_heading_initialized = False
        self._anchor_use_heading_align = False
        self._anchor_init_base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._anchor_init_ref_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._anchor_heading_align_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Joint tracking monitoring (for PD coefficient tuning)
        self._tracking_target_buffer = np.zeros(29, dtype=np.float32)
        self._tracking_log_interval = 1  # Print every N frames (10 = more frequent for tuning)

    def on_env_reset(self):
        try:
            robot = self.env.scene["robot"].data
            joint_pos_sonic = robot.joint_pos[0, self._sonic_idx].cpu().numpy().astype(np.float32)
            joint_vel_sonic = robot.joint_vel[0, self._sonic_idx].cpu().numpy().astype(np.float32)
        except Exception:
            joint_pos_sonic = self._init_sonic_target_np.copy()
            joint_vel_sonic = np.zeros(29, dtype=np.float32)
            root_z = 0.0
        else:
            root_z = float(robot.root_state_w[0, 2].cpu().numpy())

        self._frame_count = 0
        self._smpl_data_valid = False
        self._smpl_history_fill = 0
        self._latent = None
        self._anchor_heading_initialized = False
        self._anchor_use_heading_align = False
        self._anchor_init_base_quat_wxyz[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._anchor_init_ref_quat_wxyz[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._anchor_heading_align_quat_wxyz[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._smpl_joints_buf.fill(0.0)
        self._smpl_pose_buf.fill(0.0)
        self._body_rot6d_buf[:] = np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32)
        self._robot_joint_pos_hist[:] = joint_pos_sonic
        self._robot_joint_vel_hist[:] = joint_vel_sonic
        self._motion_joint_pos_hist[:] = joint_pos_sonic
        self._motion_joint_vel_hist[:] = joint_vel_sonic
        self._motion_root_z_hist[:] = root_z
        self._motion_anchor_rot6d_hist[:] = self._body_rot6d_buf[-1]
        self._ang_vel_hist.fill(0.0)
        self._grav_dir_hist.fill(0.0)
        self._last_action_hist.fill(0.0)
        # print(f"[SONIC] on_env_reset: frame_count reset, warmup={self._sonic_warmup_steps}")  # warmup 已注释
        print(f"[SONIC] on_env_reset: frame_count and history reset")

    def _setup_hand_dds(self, args_cli):
        self._dex3_dds = None
        try:
            from dds.dds_master import dds_manager
            if self.enable_dex3:
                self._dex3_dds = dds_manager.get_object("dex3")
        except Exception as e:
            print(f"[SonicActionProvider] hand DDS init skipped: {e}")

    def _update_robot_hist_from_env(self):
        """仅用当前仿真中的机器人状态更新 _robot_joint_pos_hist / _robot_joint_vel_hist（原 warmup/现 history 未满时填历史用）。"""
        try:
            robot = self.env.scene["robot"].data
            joint_pos_sonic = robot.joint_pos[0, self._sonic_idx].cpu().numpy()
            joint_vel_sonic = robot.joint_vel[0, self._sonic_idx].cpu().numpy()
            root_z = float(robot.root_state_w[0, 2].cpu().numpy())

            self._robot_joint_pos_hist = np.roll(self._robot_joint_pos_hist, -1, axis=0)
            self._robot_joint_pos_hist[-1] = joint_pos_sonic
            self._robot_joint_vel_hist = np.roll(self._robot_joint_vel_hist, -1, axis=0)
            self._robot_joint_vel_hist[-1] = joint_vel_sonic

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-step: 读取 POSE（ZMQ 或 Redis）
    # ------------------------------------------------------------------

    def _fetch_zmq_pose(self):
        """从 ZMQ 读取最新 POSE 消息，更新 SMPL 历史缓冲。"""
        if self._zmq_poller is None:
            if self._frame_count <= 3:
                print("[ZMQ] ERROR: _zmq_poller is None!")
            return
        raw = self._zmq_poller.get_data()
        if raw is None:
            if self._frame_count <= 3 or self._frame_count % 50 == 0:
                print(f"[ZMQ] No data available (frame={self._frame_count})")
            return
        if self._frame_count <= 3 or self._frame_count % 50 == 0:
            print(f"[ZMQ] Received raw data, size={len(raw)} bytes (frame={self._frame_count})")
        data = _parse_zmq_pose(raw)
        if data is None:
            print(f"[ZMQ] ERROR: Failed to parse ZMQ data (frame={self._frame_count})")
            return
        if self._frame_count <= 3 or self._frame_count % 50 == 0:
            print(f"[ZMQ] Successfully parsed data, calling _apply_pose_data (frame={self._frame_count})")
        self._apply_pose_data(data, "zmq")

    def _fetch_redis_pose(self):
        """从 Redis 读取遥操 pose（human_smplx_data_unitree_g1_with_hands），转成与 ZMQ 同格式后更新缓冲。"""
        if self._redis_client is None:
            print("[REDIS] Redis client is None, skipping fetch")
            return
        # Read required keys (removed debug packet check since sender doesn't provide it)
        try:
            raw_smplx = self._redis_client.get("human_smplx_data_unitree_g1_with_hands")
            raw_left = self._redis_client.get("action_hand_left_unitree_g1_with_hands")
            raw_right = self._redis_client.get("action_hand_right_unitree_g1_with_hands")
        except Exception as e:
            print(f"[REDIS] Failed to read from Redis: {e}")
            return
        raw = raw_smplx
        if raw is None:
            print("[REDIS] raw_smplx is None, no data in Redis")
            return
        try:
            s = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            frame = json.loads(s)
            print(f"[REDIS] Successfully parsed JSON, frame keys: {list(frame.keys())[:5]}...")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[REDIS] JSON decode failed: {e}")
            return
        if not isinstance(frame, dict):
            print(f"[REDIS] Frame is not a dict, got {type(frame)}")
            return
        # Validate that the received SMPL frame matches the exact joint names
        # expected by tools/sonic_pose_npz_replay_server.py::_frame_to_pose_fields.
        try:
            from tools.sonic_pose_npz_replay_server import SMPL_JOINT_ORDER_24
        except ImportError as e:
            print(f"[REDIS] Failed to import SMPL_JOINT_ORDER_24: {e}")
            return

        missing = [name for name in SMPL_JOINT_ORDER_24 if name not in frame]
        if missing:
            print(f"[REDIS][SMPL_FRAME_MISSING_KEYS] missing={missing}", flush=True)
            return

        # Basic structure check: each joint entry should be [pos3, quat4]
        # where pos3 has 3 numbers and quat4 has 4 numbers.
        try:
            for name in SMPL_JOINT_ORDER_24:
                v = frame.get(name)
                if not isinstance(v, (list, tuple)) or len(v) < 2:
                    raise TypeError(f"{name} must be [pos3, quat4], got {type(v)} len={len(v) if hasattr(v,'__len__') else 'n/a'}")
                pos3, quat4 = v[0], v[1]
                if not isinstance(pos3, (list, tuple)) or len(pos3) != 3:
                    raise TypeError(f"{name}[0] pos3 must have 3 numbers, got {pos3}")
                if not isinstance(quat4, (list, tuple)) or len(quat4) != 4:
                    raise TypeError(f"{name}[1] quat4 must have 4 numbers, got {quat4}")
            print(f"[REDIS] SMPL frame structure validation passed")
        except Exception as e:
            print(f"[REDIS][SMPL_FRAME_BAD_FORMAT] {e}", flush=True)
            return
        left_hand = right_hand = None
        for key, which in [
            ("action_hand_left_unitree_g1_with_hands", "left"),
            ("action_hand_right_unitree_g1_with_hands", "right"),
        ]:
            try:
                raw_h = raw_left if which == "left" else raw_right
                if raw_h is not None:
                    sh = raw_h.decode("utf-8") if isinstance(raw_h, bytes) else raw_h
                    arr = np.asarray(json.loads(sh), dtype=np.float32)
                    if arr.size == 7:
                        if which == "left":
                            left_hand = arr
                        else:
                            right_hand = arr
            except Exception:
                pass
        # Convert frame dict to (24, 7) numpy array format expected by _frame_to_pose_fields
        # Format: each row is [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        # Redis format: frame[joint_name] = [[pos_x, pos_y, pos_z], [quat_x, quat_y, quat_z, quat_w]]
        try:
            body_poses_24x7 = np.zeros((24, 7), dtype=np.float32)
            for i, joint_name in enumerate(SMPL_JOINT_ORDER_24):
                joint_data = frame[joint_name]  # [[pos3], [quat4]]
                pos3 = joint_data[0]  # [x, y, z]
                quat4 = joint_data[1]  # [x, y, z, w] from scipy.spatial.transform.Rotation

                # Store as [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
                body_poses_24x7[i, 0:3] = pos3
                body_poses_24x7[i, 3:7] = quat4  # Already in [x, y, z, w] format

            print(f"[REDIS] Converted frame dict to body_poses_24x7 shape: {body_poses_24x7.shape}")
        except Exception as e:
            print(f"[REDIS] Failed to convert frame dict to body_poses_24x7: {e}")
            return

        self._redis_frame_index += 1
        # Provide proprioception (joint_pos/joint_vel) from the current Isaac Lab robot state.
        # Replay/ZMQ often carries this; teleop/Redis typically doesn't, so we must source it here.
        joint_pos = joint_vel = None
        try:
            robot = self.env.scene["robot"].data
            joint_pos = robot.joint_pos[0, self._sonic_idx].detach().cpu().numpy().astype(np.float32)
            joint_vel = robot.joint_vel[0, self._sonic_idx].detach().cpu().numpy().astype(np.float32)
            if joint_pos.shape[0] != 29 or joint_vel.shape[0] != 29:
                joint_pos = None
                joint_vel = None
        except Exception:
            joint_pos = None
            joint_vel = None
        try:
            # Ported from pico_manager_thread_server.py PoseStreamer.run_once:
            # dict(frame) -> numpy (24,7) -> compute smpl_pose/smpl_joints/body_quat_w
            # + wrist joint_pos, then overwrite with simulator proprioception.
            data = _pico_single_frame_from_body_poses(
                device=self.device,
                body_poses_24x7=body_poses_24x7,
                frame_index=self._redis_frame_index,
                left_hand=left_hand,
                right_hand=right_hand,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
            )
            print(f"[REDIS] Successfully converted frame to pose fields, data keys: {list(data.keys())}")
        except Exception as e:
            print(f"[REDIS] Failed to convert frame to pose fields: {e}")
            return
        print(f"[REDIS] About to call _apply_pose_data with {len(data)} fields")
        self._apply_pose_data(data, "redis")

    def _apply_pose_data(self, data: dict, source: str = "zmq"):
        """用解析后的 pose 字典更新 SMPL/机器人/手部缓冲。data 格式与 ZMQ v3 一致。"""
        tag = source.upper()
        print(f"[{tag}] Received data keys: {list(data.keys())}")
        got_pose_frame = False
        root_z_value = None
        root_z_source = None

        # smpl_joints: (N, 24, 3) — 取最新一帧
        if "smpl_joints" in data:
            sj = data["smpl_joints"].astype(np.float32)  # (N, 24, 3)
            frame = sj[-1]  # (24, 3)
            print(f"[{tag}] smpl_joints shape: {sj.shape}, latest frame sum: {np.abs(frame).sum():.4f}")
            print(f"[{tag}] smpl_joints latest frame:\n{frame}")
            # 检查是否为有效数据（非全0）
            if np.abs(frame).sum() > 0.01:
                self._smpl_data_valid = True
                print(f"[{tag}] SMPL data marked as VALID")
            self._smpl_joints_buf = np.roll(self._smpl_joints_buf, -1, axis=0)
            self._smpl_joints_buf[-1] = frame
            got_pose_frame = True

        # smpl_pose: (N, 21, 3)
        if "smpl_pose" in data:
            sp = data["smpl_pose"].astype(np.float32)    # (N, 21, 3)
            print(f"[{tag}] smpl_pose shape: {sp.shape}, latest frame:\n{sp[-1]}")
            self._smpl_pose_buf = np.roll(self._smpl_pose_buf, -1, axis=0)
            self._smpl_pose_buf[-1] = sp[-1]

        # body_quat_w: (N, 4) → 转换为6D旋转表示
        # ✨ CRITICAL FIX: 原版 C++ 代码计算相对旋转（ref相对于robot base）
        # motion_anchor_orientation 是机器人局部坐标系下的相对旋转
        # 公式: base_to_ref = base^(-1) * ref
        if "body_quat_w" in data:
            bq = data["body_quat_w"].astype(np.float32)  # (N, 4) wxyz
            print(f"[{tag}] body_quat_w shape: {bq.shape}, latest: {bq[-1]}")
            got_pose_frame = True

            # 获取机器人当前朝向（从Isaac Lab）
            robot = self.env.scene["robot"].data
            base_quat_wxyz = robot.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)  # [w,x,y,z]

            # 获取参考数据（SMPL）的朝向
            ref_quat_wxyz = quat_normalize_wxyz(bq[-1])  # [w,x,y,z]

            if not self._anchor_heading_initialized:
                self._anchor_init_base_quat_wxyz[:] = quat_normalize_wxyz(base_quat_wxyz)
                self._anchor_init_ref_quat_wxyz[:] = ref_quat_wxyz
                self._anchor_heading_align_quat_wxyz[:] = quat_mul_wxyz(
                    quat_heading_wxyz(self._anchor_init_base_quat_wxyz),
                    quat_conjugate_wxyz(quat_heading_wxyz(self._anchor_init_ref_quat_wxyz)),
                )
                self._anchor_heading_initialized = True
                self._anchor_use_heading_align = True
                raw_init_angle_deg = quat_angle_deg_wxyz(
                    quat_mul_wxyz(
                        quat_conjugate_wxyz(self._anchor_init_base_quat_wxyz),
                        self._anchor_init_ref_quat_wxyz,
                    )
                )
                aligned_init_angle_deg = quat_angle_deg_wxyz(
                    quat_mul_wxyz(
                        quat_conjugate_wxyz(self._anchor_init_base_quat_wxyz),
                        quat_mul_wxyz(
                            self._anchor_heading_align_quat_wxyz,
                            self._anchor_init_ref_quat_wxyz,
                        ),
                    )
                )
                print(
                    f"[{tag}][ANCHOR_INIT] "
                    f"raw_init_angle_deg={raw_init_angle_deg:.2f} "
                    f"aligned_init_angle_deg={aligned_init_angle_deg:.2f} "
                    f"use_heading_align={self._anchor_use_heading_align}"
                )

            aligned_ref_quat_wxyz = ref_quat_wxyz.copy()
            if self._anchor_use_heading_align:
                aligned_ref_quat_wxyz = quat_mul_wxyz(
                    self._anchor_heading_align_quat_wxyz,
                    ref_quat_wxyz,
                )

            # 计算相对旋转：ref 相对于 base（与C++一致）
            # base_to_ref = base^(-1) * ref
            rel_quat_wxyz = quat_mul_wxyz(
                quat_conjugate_wxyz(base_quat_wxyz),  # base^(-1)
                aligned_ref_quat_wxyz                  # heading-aligned ref
            )

            # 转换为rot6d
            rot6d_latest = quat_to_rotation_6d(rel_quat_wxyz.reshape(1, 4))[0]

            if self._frame_count <= 3 or self._frame_count % 25 == 0:
                print(
                    f"[{tag}][ANCHOR] "
                    f"base_quat={base_quat_wxyz} "
                    f"ref_quat={ref_quat_wxyz} "
                    f"aligned_ref_quat={aligned_ref_quat_wxyz} "
                    f"rel_quat={rel_quat_wxyz} "
                    f"rot6d={rot6d_latest} "
                    f"use_heading_align={self._anchor_use_heading_align} "
                    f"(relative to robot base, C++ convention)"
                )

            print(f"[{tag}] converted to rot6d latest: {rot6d_latest}")
            self._body_rot6d_buf = np.roll(self._body_rot6d_buf, -1, axis=0)
            self._body_rot6d_buf[-1] = rot6d_latest  # (6,)
            self._motion_anchor_rot6d_hist = np.roll(self._motion_anchor_rot6d_hist, -1, axis=0)
            self._motion_anchor_rot6d_hist[-1] = rot6d_latest

        if "adjusted_transl" in data:
            adjusted_transl = data["adjusted_transl"].astype(np.float32)
            latest_transl = adjusted_transl[-1] if adjusted_transl.ndim > 1 else adjusted_transl
            if latest_transl.shape[0] >= 3:
                root_z_value = float(latest_transl[2])
                root_z_source = "adjusted_transl"

        if root_z_value is None and got_pose_frame:
            try:
                robot = self.env.scene["robot"].data
                root_z_value = float(robot.root_state_w[0, 2].cpu().numpy())
                root_z_source = "sim_fallback"
            except Exception:
                root_z_value = None
                root_z_source = None

        if root_z_value is not None:
            self._motion_root_z_hist = np.roll(self._motion_root_z_hist, -1, axis=0)
            self._motion_root_z_hist[-1] = root_z_value
            if self._frame_count <= 3 or self._frame_count % 25 == 0:
                print(
                    f"[{tag}][ROOT_Z] "
                    f"source={root_z_source} "
                    f"value={root_z_value:.4f}"
                )

        if got_pose_frame:
            self._smpl_history_fill = min(_STEP1_FRAMES, self._smpl_history_fill + 1)
            if self._frame_count <= 3 or self._frame_count % 25 == 0:
                print(
                    f"[{tag}][HISTORY] "
                    f"smpl_history_fill={self._smpl_history_fill}/{_STEP1_FRAMES} "
                    f"smpl_valid={self._smpl_data_valid}"
                )

        # 机器人关节状态（来自 ZMQ，用于 obs 构建）
        if "joint_pos" in data:
            jp = data["joint_pos"].astype(np.float32)
            self._robot_joint_pos = jp[-1]
            self._motion_joint_pos_hist = np.roll(self._motion_joint_pos_hist, -1, axis=0)
            self._motion_joint_pos_hist[-1] = self._robot_joint_pos
            if self._frame_count <= 3 or self._frame_count % 25 == 0:
                wrist_ref = self._motion_joint_pos_hist[-1, OFFICIAL_WRIST_INDICES]
                print(
                    f"[{tag}][REF_JOINT_POS] "
                    f"range={array_range_str(self._robot_joint_pos)} "
                    f"wrist_range={array_range_str(wrist_ref)} "
                    f"heading_init={'YES' if self._anchor_heading_initialized else 'NO'}"
                )
        if "joint_vel" in data:
            jv = data["joint_vel"].astype(np.float32)
            self._robot_joint_vel = jv[-1]
            self._motion_joint_vel_hist = np.roll(self._motion_joint_vel_hist, -1, axis=0)
            self._motion_joint_vel_hist[-1] = self._robot_joint_vel
            if self._frame_count <= 3 or self._frame_count % 25 == 0:
                print(
                    f"[{tag}][REF_JOINT_VEL] "
                    f"range={array_range_str(self._robot_joint_vel)}"
                )

        # 手部关节
        if "left_hand_joints" in data:
            lh = data["left_hand_joints"].flatten().astype(np.float32)
            self._left_hand_target[:len(lh)] = lh[:7]
        if "right_hand_joints" in data:
            rh = data["right_hand_joints"].flatten().astype(np.float32)
            self._right_hand_target[:len(rh)] = rh[:7]

        # VR 3点姿态（来自 SMPL，用于 encoder 输入）
        if "vr_position" in data:
            vr_pos = data["vr_position"].astype(np.float32)  # (9,)
            self._vr_3pt_position = vr_pos.flatten()
            if self._frame_count <= 3 or self._frame_count % 25 == 0:
                print(
                    f"[{tag}][VR_3PT_POS] "
                    f"range={array_range_str(self._vr_3pt_position)}"
                )
        if "vr_orientation" in data:
            vr_orn = data["vr_orientation"].astype(np.float32)  # (12,)
            self._vr_3pt_orientation = vr_orn.flatten()
            if self._frame_count <= 3 or self._frame_count % 25 == 0:
                print(
                    f"[{tag}][VR_3PT_ORN] "
                    f"range={array_range_str(self._vr_3pt_orientation)}"
                )

    # ------------------------------------------------------------------
    # GEAR-SONIC encoder + decoder 推理
    # ------------------------------------------------------------------

    def _run_gear_sonic(self) -> np.ndarray:
        """运行 GEAR-SONIC encoder+decoder，返回 SONIC IsaacLab 顺序的 29 DOF 关节目标。

        Encoder输入: 1762维，包含所有启用的观察值
        - encoder_mode_4: 4   固定 SMPL 模式
        - motion_joint_positions_10frame_step5: 290  机器人关节位置历史，step5采样
        - motion_joint_velocities_10frame_step5: 290  机器人关节速度历史，step5采样
        - motion_root_z_position_10frame_step5: 10  机器人根位置历史，step5采样
        - motion_root_z_position: 1  机器人根位置
        - motion_anchor_orientation: 6  机器人锚点旋转（6D）
        - motion_anchor_orientation_10frame_step5: 60  机器人锚点旋转历史，step5采样
        - motion_joint_positions_lowerbody_10frame_step5: 120  机器人下半身关节位置历史，step5采样
        - motion_joint_velocities_lowerbody_10frame_step5: 120  机器人下半身关节速度历史，step5采样
        - vr_3point_local_target: 9  虚拟目标点位置（3个点，每个3维）
        - vr_3point_local_orn_target: 12  虚拟目标点旋转（3个点，每个6D）
        - smpl_joints_10frame_step1: 720  SMPL 关节位置历史，step1采样
        - smpl_anchor_orientation_10frame_step1: 60  SMPL 锚点旋转历史，step1采样
        - motion_joint_velocities_wrists_10frame_step1: 60  机器人手腕关节速度历史，step1采样
        """
        do_log = self._sonic_debug and (
            self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0
        )
        if do_log:
            print(f"[SONIC] _run_gear_sonic called")
            print(f"[SONIC] encoder={self._encoder is not None}, decoder={self._decoder is not None}")
            print(f"[SONIC] _smpl_data_valid={self._smpl_data_valid}")

        if self._encoder is None or self._decoder is None:
            print(f"[SONIC] Encoder/Decoder not loaded, returning default pose")
            return self._sonic_default_np.copy()

        # 检查历史缓冲区是否有有效数据（即使ZMQ暂时没有新数据）
        smpl_joints_sum = np.abs(self._smpl_joints_buf).sum()
        if do_log:
            print(f"[SONIC] SMPL joints buffer sum: {smpl_joints_sum:.4f}")

        if smpl_joints_sum > 1.0:
            # 历史缓冲区有有效数据，强制设置标志
            if not self._smpl_data_valid:
                print(f"[SONIC] Forcing _smpl_data_valid=True based on buffer data")
                self._smpl_data_valid = True

        # 如果SMPL数据无效（全0或未接收），直接返回默认站立姿态
        if not self._smpl_data_valid:
            print(f"[SONIC] SMPL data invalid, returning default pose")
            return self._sonic_default_np.copy()

        try:
            t0 = time.perf_counter()
            # 从Isaac Lab读取当前机器人状态
            robot = self.env.scene["robot"].data
            joint_pos_sonic = robot.joint_pos[0, self._sonic_idx].cpu().numpy()  # (29,)
            joint_vel_sonic = robot.joint_vel[0, self._sonic_idx].cpu().numpy()  # (29,)

            # 更新历史缓冲区
            self._robot_joint_pos_hist = np.roll(self._robot_joint_pos_hist, -1, axis=0)
            self._robot_joint_pos_hist[-1] = joint_pos_sonic
            self._robot_joint_vel_hist = np.roll(self._robot_joint_vel_hist, -1, axis=0)
            self._robot_joint_vel_hist[-1] = joint_vel_sonic

            # 构建完整的1762维encoder输入
            # 按照observation_config.yaml的顺序

            # 1. encoder_mode_4 (4) - 官方格式不是 one-hot，而是 [mode_id, 0, 0, 0]
            # 参考 g1_deploy_onnx_ref.cpp: GatherEncoderMode(..., fill_zeros_num=3)
            encoder_mode = np.array([2., 0., 0., 0.], dtype=np.float32)

            motion_joint_pos_step5_ref = gather_temporal_window(
                self._motion_joint_pos_hist, _STEP5_FRAMES, _STEP5_STRIDE
            )
            motion_joint_vel_step5_ref = gather_temporal_window(
                self._motion_joint_vel_hist, _STEP5_FRAMES, _STEP5_STRIDE
            )
            lowerbody_indices = OFFICIAL_LOWERBODY_INDICES
            motion_joint_pos_lowerbody_ref = motion_joint_pos_step5_ref[:, lowerbody_indices]
            motion_joint_vel_lowerbody_ref = motion_joint_vel_step5_ref[:, lowerbody_indices]

            # ✨ CRITICAL FIX: Use reference motion data from ZMQ instead of zeros
            # In SMPL mode, the encoder still needs robot proprioception for better tracking
            motion_joint_pos_step5_full = motion_joint_pos_step5_ref.reshape(-1).astype(np.float32)
            motion_joint_vel_step5_full = motion_joint_vel_step5_ref.reshape(-1).astype(np.float32)

            # Extract root z position from motion history
            motion_root_z_step5 = gather_temporal_window(
                self._motion_root_z_hist[:, None], _STEP5_FRAMES, _STEP5_STRIDE
            ).reshape(-1).astype(np.float32)
            motion_root_z = np.array([self._motion_root_z_hist[-1]], dtype=np.float32)
            print(f"[SONIC Root z step 5:{motion_root_z_step5}]")
            print(f"[SONIC Root z:{motion_root_z}]")


            # Use anchor orientation from SMPL data
            motion_anchor_orient = self._body_rot6d_buf[-1].copy()  # Latest anchor orientation
            motion_anchor_orient_step5_full = gather_temporal_window(
                self._motion_anchor_rot6d_hist, _STEP5_FRAMES, _STEP5_STRIDE
            ).reshape(-1).astype(np.float32)

            motion_joint_pos_lowerbody_full = motion_joint_pos_lowerbody_ref.reshape(-1).astype(np.float32)
            motion_joint_vel_lowerbody_full = motion_joint_vel_lowerbody_ref.reshape(-1).astype(np.float32)

            # ✨ CRITICAL: Use actual VR 3-point pose data from SMPL processing
            # This data comes from _apply_pose_data which processes vr_position and vr_orientation
            # from the ZMQ/Redis pose message
            vr_3pt_pos = self._vr_3pt_position.copy()  # (9,) - 3 points × 3D position
            vr_3pt_orn = self._vr_3pt_orientation.copy()  # (12,) - 3 points × 4D quat wxyz
            # Note: Despite the comment saying "每个6D", the encoder actually expects quaternions (4D)
            # Total encoder input: 1762 = ... + 9 (vr_pos) + 12 (vr_orn) + ...

            smpl_joints_flat = self._smpl_joints_buf.reshape(-1)  # (10, 24, 3) → (720,)
            smpl_anchor_orient_flat = self._body_rot6d_buf.reshape(-1)  # (10, 6) → (60,)
            wrist_indices = OFFICIAL_WRIST_INDICES
            motion_wrist_window = gather_temporal_window(self._motion_joint_pos_hist, _STEP1_FRAMES, 1)
            motion_wrist_pos = motion_wrist_window[:, wrist_indices].reshape(-1)

            # 拼接所有观察值
            encoder_input = np.concatenate([
                encoder_mode,                           # 4
                motion_joint_pos_step5_full,            # 290
                motion_joint_vel_step5_full,            # 290
                motion_root_z_step5,                    # 10
                motion_root_z,                          # 1
                motion_anchor_orient,                   # 6
                motion_anchor_orient_step5_full,        # 60
                motion_joint_pos_lowerbody_full,        # 120
                motion_joint_vel_lowerbody_full,        # 120
                vr_3pt_pos,                             # 9
                vr_3pt_orn,                             # 12
                smpl_joints_flat,                       # 720
                smpl_anchor_orient_flat,                # 60
                motion_wrist_pos,                       # 60
            ])[np.newaxis]  # (1, 1762)

            if do_log:
                print(f"[SONIC] Encoder input shape: {encoder_input.shape}, expected: (1, 1762)")
                print(f"[SONIC] Encoder input dtype: {encoder_input.dtype}")
                print(f"[SONIC] Encoder input range: [{encoder_input.min():.4f}, {encoder_input.max():.4f}]")
                print(f"[SONIC] SMPL joints sum: {np.abs(smpl_joints_flat).sum():.4f}")

            if do_log or (self._sonic_debug and np.max(np.abs(encoder_input)) > 8.0):
                sim_wrist_pos = self._robot_joint_pos_hist[:, wrist_indices].reshape(-1)
                wrist_l2 = float(np.linalg.norm(motion_wrist_pos - sim_wrist_pos))
                print(
                    "[SONIC][SMPL_MODE] "
                    f"encoder_mode_vec={encoder_mode.tolist()} "
                    f"active={SMPL_MODE_ACTIVE_BLOCKS} "
                    f"zeroed={SMPL_MODE_ZEROED_BLOCKS}"
                )
                print(
                    "[SONIC][ENCODER_BLOCKS] "
                    f"motion_pos_step5={array_range_str(motion_joint_pos_step5_full)} "
                    f"motion_vel_step5={array_range_str(motion_joint_vel_step5_full)} "
                    f"motion_root_z_step5={array_range_str(motion_root_z_step5)} "
                    f"anchor={array_range_str(motion_anchor_orient)} "
                    f"anchor_step5={array_range_str(motion_anchor_orient_step5_full)}"
                )
                print(
                    "[SONIC][REF_DIAG] "
                    f"ignored_ref_motion_pos_step5={array_range_str(motion_joint_pos_step5_ref.reshape(-1))} "
                    f"ignored_ref_motion_vel_step5={array_range_str(motion_joint_vel_step5_ref.reshape(-1))} "
                    f"ignored_ref_lowerbody_pos={array_range_str(motion_joint_pos_lowerbody_ref.reshape(-1))} "
                    f"ignored_ref_lowerbody_vel={array_range_str(motion_joint_vel_lowerbody_ref.reshape(-1))}"
                )
                print(
                    "[SONIC][ENCODER_BLOCKS] "
                    f"lowerbody_pos={array_range_str(motion_joint_pos_lowerbody_full)} "
                    f"lowerbody_vel={array_range_str(motion_joint_vel_lowerbody_full)} "
                    f"smpl_joints={array_range_str(smpl_joints_flat)} "
                    f"smpl_anchor={array_range_str(smpl_anchor_orient_flat)} "
                    f"wrist_pos={array_range_str(motion_wrist_pos)}"
                )
                print(
                    "[SONIC][WRIST_DIAG] "
                    f"ref_wrist={array_range_str(motion_wrist_pos)} "
                    f"sim_wrist={array_range_str(sim_wrist_pos)} "
                    f"ref_vs_sim_l2={wrist_l2:.4f} "
                    f"lowerbody_idx={lowerbody_indices} "
                    f"wrist_idx={wrist_indices}"
                )

            # Encoder推理
            enc_inputs = {
                self._encoder.get_inputs()[0].name: encoder_input
            }
            t_enc0 = time.perf_counter()
            latent = self._encoder.run(None, enc_inputs)[0]
            t_enc1 = time.perf_counter()
            self._latent = latent
            if do_log:
                print(f"[SONIC] ✓ Encoder output latent shape: {latent.shape}")
                print(f"[SONIC] Latent range: [{latent.min():.4f}, {latent.max():.4f}]")

            # Decoder 输入：994维 policy observation 向量
            # = token_state(64) + ang_vel_hist(30) + joint_pos_hist(290)
            #   + joint_vel_hist(290) + last_action_hist(290) + grav_dir_hist(30)
            ang_vel   = robot.root_ang_vel_b[0].cpu().numpy()    # (3,)
            proj_grav = robot.projected_gravity_b[0].cpu().numpy()  # (3,)

            # 更新 decoder 历史缓冲区
            self._ang_vel_hist   = np.roll(self._ang_vel_hist,   -1, axis=0)
            self._ang_vel_hist[-1] = ang_vel
            self._grav_dir_hist  = np.roll(self._grav_dir_hist,  -1, axis=0)
            self._grav_dir_hist[-1] = proj_grav
            # NOTE: last_action_hist will be updated AFTER decoder inference with raw action

            # 构建 994 维 decoder 输入
            dec_obs = np.concatenate([
                latent.flatten(),                          # token_state: 64
                self._ang_vel_hist.flatten(),              # his_base_angular_velocity: 30
                self._robot_joint_pos_hist.flatten(),      # his_body_joint_positions: 290
                self._robot_joint_vel_hist.flatten(),      # his_body_joint_velocities: 290
                self._last_action_hist.flatten(),          # his_last_actions: 290
                self._grav_dir_hist.flatten(),             # his_gravity_dir: 30
            ])[np.newaxis].astype(np.float32)  # (1, 994)

            if do_log:
                print(f"[SONIC] Decoder input shape: {dec_obs.shape}, expected: (1, 994)")
            dec_inputs = {self._decoder.get_inputs()[0].name: dec_obs}
            t_dec0 = time.perf_counter()
            action_sonic = self._decoder.run(None, dec_inputs)[0]
            t_dec1 = time.perf_counter()
            raw_sonic = action_sonic.flatten()[:29]
            if do_log:
                print(f"[SONIC] ✓ Decoder output shape: {action_sonic.shape}")
                print(f"[SONIC] Raw sonic range (before clip): [{raw_sonic.min():.4f}, {raw_sonic.max():.4f}]")

            # ✨ CRITICAL FIX: Clip raw action to reasonable range
            # Normal decoder output should be in [-2, 2] range
            raw_sonic = np.clip(raw_sonic, -2.0, 2.0)
            if do_log:
                print(f"[SONIC] Raw sonic range (after clip): [{raw_sonic.min():.4f}, {raw_sonic.max():.4f}]")

            # 更新 last_action_hist with raw action (before scaling)
            # 参考 GR00T g1_deploy_onnx_ref.cpp:269, 2825
            self._last_action_hist = np.roll(self._last_action_hist, -1, axis=0)
            self._last_action_hist[-1] = raw_sonic

            # 后处理：per-joint action_scale + default（SONIC IsaacLab order）
            # 参考 GR00T g1_deploy_onnx_ref.cpp:2824
            # target = action * action_scale + default_angle
            target_sonic = raw_sonic * G1_ACTION_SCALE_ISAACLAB + self._sonic_default_np
            if do_log:
                print(f"[SONIC] ✓ Final target range (before safety clip): [{target_sonic.min():.4f}, {target_sonic.max():.4f}]")

            # ✨ Safety clip: Ensure final targets are within reasonable joint limits
            # Typical G1 joint limits are around [-3.14, 3.14] rad
            target_sonic = np.clip(target_sonic, -3.0, 3.0)
            if do_log:
                print(f"[SONIC] ✓ Final target range (after safety clip): [{target_sonic.min():.4f}, {target_sonic.max():.4f}]")

            if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0):
                t1 = time.perf_counter()
                enc_ms = (t_enc1 - t_enc0) * 1000.0
                dec_ms = (t_dec1 - t_dec0) * 1000.0
                total_ms = (t1 - t0) * 1000.0
                print(
                    "[SONIC][PERF] "
                    f"encoder_ms={enc_ms:.2f} decoder_ms={dec_ms:.2f} total_ms={total_ms:.2f}"
                )

            return target_sonic.astype(np.float32)

        except Exception as e:
            print(f"[SonicActionProvider] GEAR-SONIC inference error: {e}")
            import traceback
            traceback.print_exc()
            return self._sonic_default_np.copy()


    def get_action(self, env) -> Optional[torch.Tensor]:
        try:
            if not hasattr(self, "_runtime_logged"):
                self._runtime_logged = True
                self._frame_count = 0
                print("\n[SONIC] Real get_action path enabled")
            self._frame_count += 1

            # 1. 读取 POSE（ZMQ 或 Redis）
            t_step0 = time.perf_counter()
            if self._pose_source == "redis":
                self._fetch_redis_pose()
                if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0):
                    print("redis pose")
            else:
                self._fetch_zmq_pose()
                if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0):
                    print("zmq pose")
            # warmup_active = self._frame_count <= self._sonic_warmup_steps  # warmup 已注释
            history_ready = self._smpl_history_fill >= _STEP1_FRAMES and self._smpl_data_valid

            # if warmup_active or not history_ready:  # 原 warmup 逻辑：前 N 步或历史未满则 hold
            if not history_ready:
                self._update_robot_hist_from_env()
                robot = self.env.scene["robot"].data
                sonic_targets = (
                    robot.joint_pos[0, self._sonic_idx].detach().cpu().numpy().astype(np.float32)
                )
                # Warmup/hold 时，decoder 仍会用 his_last_actions_10frame_step1。
                # 为了让 frame=10 第一次推理输入与“我们实际上在 hold 的关节目标”一致，
                # 这里把 hold 的目标 qpos 反算成 raw action（逆 target = raw*scale + default），并写入 last_action_hist。
                # 这样可以避免 last_action_hist 在 warmup 期间一直为 0，导致第一次推理出现输入突变。
                raw_action_equiv = (sonic_targets - self._sonic_default_np) / G1_ACTION_SCALE_ISAACLAB
                raw_action_equiv = np.clip(raw_action_equiv, -2.0, 2.0).astype(np.float32)
                self._last_action_hist = np.roll(self._last_action_hist, -1, axis=0)
                self._last_action_hist[-1] = raw_action_equiv
                if self._frame_count <= 3 or self._frame_count % 10 == 0:
                    # print("[SONIC][WARMUP] " f"frame=..." f"warmup_active={warmup_active} " ...)  # warmup 已注释
                    print(
                        "[SONIC][HISTORY] "
                        f"frame={self._frame_count} "
                        f"smpl_history_fill={self._smpl_history_fill}/{_STEP1_FRAMES} "
                        f"smpl_valid={self._smpl_data_valid} "
                        "action=hold_current_pose"
                    )
            else:
                sonic_targets = self._run_gear_sonic()

            if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0):
                sonic_targets_str = np.array2string(
                    sonic_targets,
                    precision=6,
                    separator=", ",
                    suppress_small=False,
                    max_line_width=2000
                )
                print("\n" + "=" * 120)
                print(
                    f"[SONIC_29_QPOS] frame={self._frame_count}  "
                    f"history_ready={history_ready}"
                )
                print(f"[SONIC_29_QPOS_VALUES] {sonic_targets_str}")
                print(f"[SONIC_29_QPOS_RANGE] min={sonic_targets.min():.6f}, max={sonic_targets.max():.6f}")
                print("=" * 120)

            # 3. 构建完整 Isaac 动作
            full_action = self._default_pos.clone().squeeze(0)
            sonic_t = torch.tensor(sonic_targets, dtype=torch.float32, device=self.device)
            full_action.index_copy_(0, self._sonic_idx, sonic_t)

            # 4. 手部关节
            self._apply_hand_targets(full_action)

            # 5. 步进仿真（decimation）
            t_sim0 = time.perf_counter()
            for _ in range(self._decimation):
                env.scene["robot"].set_joint_position_target(full_action)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
            t_sim1 = time.perf_counter()

            # Joint tracking comparison (for PD coefficient tuning)
            # Store targets for next frame comparison
            self._tracking_target_buffer = sonic_targets.copy()

            if self._frame_count % self._tracking_log_interval == 0:
                current_pos = env.scene["robot"].data.joint_pos[0, self._sonic_idx].cpu().numpy()
                current_vel = env.scene["robot"].data.joint_vel[0, self._sonic_idx].cpu().numpy()

                # Compute tracking errors
                pos_error = current_pos - sonic_targets
                pos_error_abs = np.abs(pos_error)

                print(f"\n[SONIC_TRACKING] Frame {self._frame_count} - Joint Position Tracking")
                print("=" * 80)
                print(f"{'Joint':<25} {'Target':>10} {'Current':>10} {'Error':>10} {'|Error|':>10}")
                print("-" * 80)

                # Key joints for monitoring (same as action_provider_wh_twist2.py reference)
                key_joints = {
                    "left_elbow": 21,
                    "right_elbow": 22,
                    "left_shoulder_pitch": 11,
                    "right_shoulder_pitch": 12,
                    "left_shoulder_roll": 15,
                    "right_shoulder_roll": 16,
                    "left_knee": 9,
                    "right_knee": 10,
                }

                for joint_name, idx in key_joints.items():
                    print(f"{joint_name:<25} {sonic_targets[idx]:>10.4f} {current_pos[idx]:>10.4f} "
                          f"{pos_error[idx]:>10.4f} {pos_error_abs[idx]:>10.4f}")

                print("-" * 80)
                print(f"Max absolute error: {pos_error_abs.max():.4f} rad ({np.degrees(pos_error_abs.max()):.2f}°)")
                print(f"Mean absolute error: {pos_error_abs.mean():.4f} rad ({np.degrees(pos_error_abs.mean()):.2f}°)")
                print(f"RMS error: {np.sqrt(np.mean(pos_error**2)):.4f} rad ({np.degrees(np.sqrt(np.mean(pos_error**2))):.2f}°)")
                print("=" * 80)

            env.sim.render()
            env.observation_manager.compute()

            if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0):
                t_step1 = time.perf_counter()
                sim_ms = (t_sim1 - t_sim0) * 1000.0
                step_ms = (t_step1 - t_step0) * 1000.0
                print(f"[SONIC][PERF] step_total_ms={step_ms:.2f} sim_loop_ms={sim_ms:.2f}")
            return full_action

        except Exception as e:
            print(f"[SonicActionProvider] get_action error: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _apply_hand_targets(self, full_action: torch.Tensor):
        if self._dex3_dds is not None:
            try:
                cmds = self._dex3_dds.get_hand_commands()
                if cmds:
                    lp = cmds.get("left_hand_cmd",  {}).get("positions", [])
                    rp = cmds.get("right_hand_cmd", {}).get("positions", [])
                    if len(lp) >= 7 and hasattr(self, "_left_hand_idx"):
                        full_action.index_copy_(
                            0, self._left_hand_idx,
                            torch.tensor(lp[:7], dtype=torch.float32,
                                         device=self.device))
                    if len(rp) >= 7 and hasattr(self, "_right_hand_idx"):
                        full_action.index_copy_(
                            0, self._right_hand_idx,
                            torch.tensor(rp[:7], dtype=torch.float32,
                                         device=self.device))
                    return
            except Exception:
                pass
        # fallback: ZMQ 手部数据
        if hasattr(self, "_left_hand_idx") and self._left_hand_idx.numel() > 0:
            full_action.index_copy_(
                0, self._left_hand_idx,
                torch.tensor(self._left_hand_target, dtype=torch.float32,
                             device=self.device))
        if hasattr(self, "_right_hand_idx") and self._right_hand_idx.numel() > 0:
            full_action.index_copy_(
                0, self._right_hand_idx,
                torch.tensor(self._right_hand_target, dtype=torch.float32,
                             device=self.device))

    def cleanup(self):
        if self._zmq_poller is not None:
            try:
                self._zmq_poller.close()
            except Exception:
                pass
