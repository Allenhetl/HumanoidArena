# Copyright (c) 2025. All Rights Reserved.
# License: Apache License, Version 2.0
"""SonicActionProvider: POSE 模式全身遥操作，驱动 Isaac Lab 仿真。

POSE 模式数据流：
  Pico 头显 + 手腕控制器 + 脚踝 tracker
    → pico_server/pico_server_pose_only.py
    → Redis keys（主链路）/ legacy ZMQ "pose"（兼容链路）
    → SonicActionProvider._fetch_*_pose()
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
        --sonic_pose_source redis --sonic_redis_host localhost --sonic_redis_port 6379 \\
        --sonic_encoder_path /path/to/model_encoder.onnx \\
        --sonic_decoder_path /path/to/model_decoder.onnx \\
        --task Isaac-Move-Cylinder-G129-Dex3-Wholebody \\
        --robot_type g129 --enable_dex3_dds --device cuda
"""

import json
import os
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from action_provider.action_base import ActionProvider
from action_provider.recording_common import AsyncEpisodeRecorder
from action_provider.reset_control import consume_reset_complete, get_input_ready_key, publish_reset_command

# Isaac Sim imports for RTF monitoring
try:
    import omni.timeline
except ImportError:
    omni = None

try:
    from pico_server.sonic_tools.utils.teleop.zmq.zmq_poller import ZMQPoller
    _HAS_ZMQ_POLLER = True
except ImportError:
    _HAS_ZMQ_POLLER = False
    print("[SonicActionProvider] WARNING: local sonic_tools ZMQPoller not found.")

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


def gravity_dir_from_base_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Match SONIC deploy: gravity_dir = quat_conjugate(base_quat) rotate [0, 0, -1]."""
    quat = quat_normalize_wxyz(quat)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_quat = np.array([0.0, gravity_world[0], gravity_world[1], gravity_world[2]], dtype=np.float32)
    rotated = quat_mul_wxyz(
        quat_mul_wxyz(quat_conjugate_wxyz(quat), gravity_quat),
        quat,
    )
    return rotated[1:].astype(np.float32)


def array_range_str(arr: np.ndarray) -> str:
    """Compact range formatter for debug logs."""
    arr = np.asarray(arr)
    return f"[{arr.min():.4f}, {arr.max():.4f}]"


def topk_joint_abs_str(
    values: np.ndarray,
    joint_names: list[str],
    k: int = 6,
    *,
    signed: bool = True,
) -> str:
    """Format top-k joints by absolute magnitude for debug logs."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return "[]"

    k = max(1, min(int(k), values.size))
    top_indices = np.argsort(np.abs(values))[-k:][::-1]
    items = []
    for idx in top_indices:
        name = joint_names[idx] if idx < len(joint_names) else f"joint_{idx}"
        val = float(values[idx])
        if signed:
            items.append(f"{name}={val:+.4f}")
        else:
            items.append(f"{name}={abs(val):.4f}")
    return "[" + ", ".join(items) + "]"


def joint_slice_str(
    values: np.ndarray,
    joint_names: list[str],
    indices: list[int],
    *,
    signed: bool = True,
) -> str:
    """Format a selected joint subset for debug logs."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    items = []
    for idx in indices:
        if idx >= values.size:
            continue
        name = joint_names[idx] if idx < len(joint_names) else f"joint_{idx}"
        val = float(values[idx])
        if signed:
            items.append(f"{name}={val:+.4f}")
        else:
            items.append(f"{name}={abs(val):.4f}")
    return "[" + ", ".join(items) + "]"


# ---------------------------------------------------------------------------
# Redis mode: locally port pico_manager_thread_server PoseStreamer.run_once
# (single-frame version, output keys compatible with _apply_pose_data()).
# ---------------------------------------------------------------------------

try:
    from pico_server.sonic_tools.trl.utils.rotation_conversion import decompose_rotation_aa
    from pico_server.sonic_tools.trl.utils.torch_transform import (
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
    print(f"[SonicActionProvider] WARNING: local sonic_tools trl utils import failed: {e}")

try:
    from pico_server.sonic_tools.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
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
        raise RuntimeError("decompose_rotation_aa is not available (local sonic_tools not installed?)")

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
        raise RuntimeError("local sonic_tools trl utils not available; cannot compute smpl fields.")

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
    -0.312,  # left_hip_pitch_joint
    -0.312,  # right_hip_pitch_joint
    0.0,     # waist_yaw_joint
    0.0,     # left_hip_roll_joint
    0.0,     # right_hip_roll_joint
    0.0,     # waist_roll_joint
    0.0,     # left_hip_yaw_joint
    0.0,     # right_hip_yaw_joint
    0.0,     # waist_pitch_joint
    0.669,   # left_knee_joint
    0.669,   # right_knee_joint
    0.2,     # left_shoulder_pitch_joint
    0.2,     # right_shoulder_pitch_joint
    -0.363,  # left_ankle_pitch_joint
    -0.363,  # right_ankle_pitch_joint
    0.2,     # left_shoulder_roll_joint
    -0.2,    # right_shoulder_roll_joint
    0.0,     # left_ankle_roll_joint
    0.0,     # right_ankle_roll_joint
    0.0,     # left_shoulder_yaw_joint
    0.0,     # right_shoulder_yaw_joint
    0.6,     # left_elbow_joint
    0.6,     # right_elbow_joint
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

# Deploy-time SONIC PD/effort parameters from policy_parameters.hpp, remapped to
# SONIC IsaacLab order so they stay consistent with G1_ACTION_SCALE_ISAACLAB.
SONIC_PD_KP = np.array([
    99.02755,   # 0: left_hip_pitch_joint
    99.02755,   # 1: right_hip_pitch_joint
    40.168964,  # 2: waist_yaw_joint
    99.02755,   # 3: left_hip_roll_joint
    99.02755,   # 4: right_hip_roll_joint
    28.445627,  # 5: waist_roll_joint
    40.168964,  # 6: left_hip_yaw_joint
    40.168964,  # 7: right_hip_yaw_joint
    28.445627,  # 8: waist_pitch_joint
    99.02755,   # 9: left_knee_joint
    99.02755,   # 10: right_knee_joint
    14.222814,  # 11: left_shoulder_pitch_joint
    14.222814,  # 12: right_shoulder_pitch_joint
    # 28.445627,  # 13: left_ankle_pitch_joint
    # 28.445627,  # 14: right_ankle_pitch_joint
    48.445627,  # 13: left_ankle_pitch_joint
    48.445627,  # 14: right_ankle_pitch_joint
    # 14.222814,  # 15: left_shoulder_roll_joint
    # 14.222814,  # 16: right_shoulder_roll_joint
    34.222814,  # 15: left_shoulder_roll_joint
    34.222814,  # 16: right_shoulder_roll_joint
    28.445627,  # 17: left_ankle_roll_joint
    28.445627,  # 18: right_ankle_roll_joint
    # 14.222814,  # 19: left_shoulder_yaw_joint
    # 14.222814,  # 20: right_shoulder_yaw_joint
    24.222814,  # 19: left_shoulder_yaw_joint
    24.222814,  # 20: right_shoulder_yaw_joint
    # 14.222814,  # 21: left_elbow_joint
    # 14.222814,  # 22: right_elbow_joint
    19.222814,  # 21: left_elbow_joint
    19.222814,  # 22: right_elbow_joint
    14.222814,  # 23: left_wrist_roll_joint
    14.222814,  # 24: right_wrist_roll_joint
    16.77876,   # 25: left_wrist_pitch_joint
    16.77876,   # 26: right_wrist_pitch_joint
    16.77876,   # 27: left_wrist_yaw_joint
    16.77876,   # 28: right_wrist_yaw_joint
], dtype=np.float32)

SONIC_PD_KD = np.array([
    3.1559467,   # 0: left_hip_pitch_joint
    3.1559467,   # 1: right_hip_pitch_joint
    1.2792531,   # 2: waist_yaw_joint
    3.1559467,   # 3: left_hip_roll_joint
    3.1559467,   # 4: right_hip_roll_joint
    0.90733415,  # 5: waist_roll_joint
    1.2792531,   # 6: left_hip_yaw_joint
    1.2792531,   # 7: right_hip_yaw_joint
    0.90733415,  # 8: waist_pitch_joint
    3.1559467,   # 9: left_knee_joint
    3.1559467,   # 10: right_knee_joint
    0.45366707,  # 11: left_shoulder_pitch_joint
    0.45366707,  # 12: right_shoulder_pitch_joint
    0.90733415,  # 13: left_ankle_pitch_joint
    0.90733415,  # 14: right_ankle_pitch_joint
    0.45366707,  # 15: left_shoulder_roll_joint
    0.45366707,  # 16: right_shoulder_roll_joint
    0.90733415,  # 17: left_ankle_roll_joint
    0.90733415,  # 18: right_ankle_roll_joint
    0.45366707,  # 19: left_shoulder_yaw_joint
    0.45366707,  # 20: right_shoulder_yaw_joint
    0.45366707,  # 21: left_elbow_joint
    0.45366707,  # 22: right_elbow_joint
    0.45366707,  # 23: left_wrist_roll_joint
    0.45366707,  # 24: right_wrist_roll_joint
    0.53407073,  # 25: left_wrist_pitch_joint
    0.53407073,  # 26: right_wrist_pitch_joint
    0.53407073,  # 27: left_wrist_yaw_joint
    0.53407073,  # 28: right_wrist_yaw_joint
], dtype=np.float32)

SONIC_EFFORT_LIMIT = np.array([
    139.0, 139.0, 88.0, 139.0, 139.0, 25.0, 88.0, 88.0, 25.0, 139.0, 139.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0,
    25.0, 25.0, 5.0, 5.0, 5.0, 5.0,
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


def build_future_window(window: np.ndarray, num_frames: int) -> np.ndarray:
    """Build a current->future window by truncating or repeating the last frame."""
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("build_future_window expects at least 1D input")
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.shape[0] <= 0:
        raise ValueError("build_future_window received empty window")
    if arr.shape[0] >= num_frames:
        return arr[:num_frames].astype(np.float32)
    pad = np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)


def sorted_insert_unique(sorted_list: list[int], value: int) -> None:
    """Insert value into sorted_list if not already present."""
    pos = bisect_left(sorted_list, value)
    if pos >= len(sorted_list) or sorted_list[pos] != value:
        sorted_list.insert(pos, value)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_string(value: Any) -> str:
    return json.dumps(value, default=_json_default, separators=(",", ":"))


def _extract_controller_binary_signals(controller_data: dict | None) -> dict[str, bool]:
    def _get_side_data(side_name: str) -> dict:
        if not isinstance(controller_data, dict):
            return {}
        side_data = controller_data.get(side_name, {})
        if not isinstance(side_data, dict):
            return {}
        return side_data

    def _get_grip_binary(side_name: str) -> bool:
        side_data = _get_side_data(side_name)
        if "grip_binary" in side_data:
            return bool(side_data.get("grip_binary"))
        try:
            if float(side_data.get("index_trig", 0.0)) > 0.5:
                return True
            if float(side_data.get("grip", 0.0)) > 0.5:
                return False
        except Exception:
            pass
        return False

    def _get_close_trigger_binary(side_name: str) -> bool:
        side_data = _get_side_data(side_name)
        if "close_trigger_binary" in side_data:
            return bool(side_data.get("close_trigger_binary"))
        try:
            return bool(float(side_data.get("index_trig", 0.0)) > 0.5)
        except Exception:
            return False

    def _get_open_trigger_binary(side_name: str) -> bool:
        side_data = _get_side_data(side_name)
        if "open_trigger_binary" in side_data:
            return bool(side_data.get("open_trigger_binary"))
        try:
            return bool(float(side_data.get("grip", 0.0)) > 0.5)
        except Exception:
            return False

    return {
        "left_grip_binary": _get_grip_binary("LeftController"),
        "right_grip_binary": _get_grip_binary("RightController"),
        "left_close_trigger_binary": _get_close_trigger_binary("LeftController"),
        "right_close_trigger_binary": _get_close_trigger_binary("RightController"),
        "left_open_trigger_binary": _get_open_trigger_binary("LeftController"),
        "right_open_trigger_binary": _get_open_trigger_binary("RightController"),
    }


def _organize_sonic_episode(data_buffer: list[dict[str, Any]], timestamp_us: int) -> dict[str, Any]:
    if not data_buffer:
        raise ValueError("empty sonic recording buffer")

    first = data_buffer[0]
    meta = first["meta"]
    num_frames = len(data_buffer)

    def _stack(path: tuple[str, ...], dtype=None):
        values = []
        for frame in data_buffer:
            current = frame
            for key in path:
                current = current[key]
            values.append(np.asarray(current))
        return np.stack(values).astype(dtype) if dtype is not None else np.stack(values)

    organized: dict[str, Any] = {
        "schema_version": np.array("sonic_episode_v1"),
        "task": np.array(meta["task"]),
        "episode_id": np.array(meta["episode_id"], dtype=np.int64),
        "save_timestamp_us": np.array(timestamp_us, dtype=np.int64),
        "num_frames": np.array(num_frames, dtype=np.int32),
        "meta_control_dt": np.array(meta["control_dt"], dtype=np.float32),
        "meta_physics_dt": np.array(meta["physics_dt"], dtype=np.float32),
        "meta_decimation": np.array(meta["decimation"], dtype=np.int32),
        "meta_pose_source": np.array(meta["pose_source"]),
        "meta_encoder_path": np.array(meta["encoder_path"]),
        "meta_decoder_path": np.array(meta["decoder_path"]),
        "frame_index": _stack(("markers", "frame_index"), np.int64),
        "episode_step": _stack(("markers", "episode_step"), np.int64),
        "timestamp_wall": _stack(("markers", "timestamp_wall"), np.float64),
        "timestamp_monotonic": _stack(("markers", "timestamp_monotonic"), np.float64),
        "timestamp_realtime": _stack(("markers", "timestamp_realtime"), np.float64),
        "recording_command": np.array(
            [frame["markers"]["recording_command"] for frame in data_buffer]
        ),
        "reset_requested": _stack(("markers", "reset_requested"), np.bool_),
        "reset_completed": _stack(("markers", "reset_completed"), np.bool_),
        "save_triggered": _stack(("markers", "save_triggered"), np.bool_),
        "human_left_hand": _stack(("human_raw", "left_hand"), np.float32),
        "human_right_hand": _stack(("human_raw", "right_hand"), np.float32),
        "pico_left_grip_binary": _stack(
            ("human_raw", "controller_binary", "left_grip_binary"), np.bool_
        ),
        "pico_right_grip_binary": _stack(
            ("human_raw", "controller_binary", "right_grip_binary"), np.bool_
        ),
        "pico_left_close_trigger_binary": _stack(
            ("human_raw", "controller_binary", "left_close_trigger_binary"), np.bool_
        ),
        "pico_right_close_trigger_binary": _stack(
            ("human_raw", "controller_binary", "right_close_trigger_binary"), np.bool_
        ),
        "pico_left_open_trigger_binary": _stack(
            ("human_raw", "controller_binary", "left_open_trigger_binary"), np.bool_
        ),
        "pico_right_open_trigger_binary": _stack(
            ("human_raw", "controller_binary", "right_open_trigger_binary"), np.bool_
        ),
        "human_smpl_joints": _stack(("human_processed", "smpl_joints"), np.float32),
        "human_smpl_pose": _stack(("human_processed", "smpl_pose"), np.float32),
        "human_body_quat_w": _stack(("human_processed", "body_quat_w"), np.float32),
        "human_vr_position": _stack(("human_processed", "vr_position"), np.float32),
        "human_vr_orientation": _stack(("human_processed", "vr_orientation"), np.float32),
        "human_heading_increment": _stack(("human_processed", "heading_increment"), np.float32),
        "anchor_heading_initialized": _stack(
            ("human_processed", "anchor_heading_initialized"), np.bool_
        ),
        "anchor_use_heading_align": _stack(
            ("human_processed", "anchor_use_heading_align"), np.bool_
        ),
        "anchor_init_base_quat_wxyz": _stack(
            ("human_processed", "anchor_init_base_quat_wxyz"), np.float32
        ),
        "anchor_init_ref_quat_wxyz": _stack(
            ("human_processed", "anchor_init_ref_quat_wxyz"), np.float32
        ),
        "anchor_heading_align_quat_wxyz": _stack(
            ("human_processed", "anchor_heading_align_quat_wxyz"), np.float32
        ),
        "encoder_input": _stack(("sonic_model_io", "encoder_input"), np.float32),
        "encoder_smpl_joint_window": _stack(
            ("sonic_model_io", "smpl_joint_window"), np.float32
        ),
        "encoder_anchor_window": _stack(("sonic_model_io", "anchor_window"), np.float32),
        "encoder_wrist_window": _stack(("sonic_model_io", "wrist_window"), np.float32),
        "encoder_motion_joint_pos_hist": _stack(
            ("sonic_model_io", "motion_joint_pos_hist"), np.float32
        ),
        "encoder_motion_joint_vel_hist": _stack(
            ("sonic_model_io", "motion_joint_vel_hist"), np.float32
        ),
        "encoder_motion_root_z_hist": _stack(
            ("sonic_model_io", "motion_root_z_hist"), np.float32
        ),
        "encoder_motion_anchor_rot6d_hist": _stack(
            ("sonic_model_io", "motion_anchor_rot6d_hist"), np.float32
        ),
        "encoder_robot_joint_pos_hist": _stack(
            ("sonic_model_io", "robot_joint_pos_hist"), np.float32
        ),
        "encoder_robot_joint_vel_hist": _stack(
            ("sonic_model_io", "robot_joint_vel_hist"), np.float32
        ),
        "encoder_latent": _stack(("sonic_model_io", "encoder_latent"), np.float32),
        "decoder_obs": _stack(("sonic_model_io", "decoder_obs"), np.float32),
        "decoder_ang_vel_hist": _stack(("sonic_model_io", "ang_vel_hist"), np.float32),
        "decoder_gravity_dir_hist": _stack(("sonic_model_io", "gravity_dir_hist"), np.float32),
        "decoder_last_action_hist": _stack(("sonic_model_io", "last_action_hist"), np.float32),
        "decoder_raw_action": _stack(("sonic_model_io", "decoder_raw_action"), np.float32),
        "decoder_target_action": _stack(("sonic_model_io", "decoder_target_action"), np.float32),
        "robot_qpos_before_decimation": _stack(
            ("robot", "qpos_before_decimation"), np.float32
        ),
        "robot_qvel_before_decimation": _stack(
            ("robot", "qvel_before_decimation"), np.float32
        ),
        "robot_root_position": _stack(("robot", "root_position"), np.float32),
        "robot_root_orientation": _stack(("robot", "root_orientation"), np.float32),
        "robot_root_lin_vel_local": _stack(("robot", "root_lin_vel_local"), np.float32),
        "robot_root_ang_vel_local": _stack(("robot", "root_ang_vel_local"), np.float32),
        "robot_root_lin_vel_world": _stack(("robot", "root_lin_vel_world"), np.float32),
        "robot_root_ang_vel_world": _stack(("robot", "root_ang_vel_world"), np.float32),
        "final_body_action_29dof": _stack(("action", "body_action_29dof"), np.float32),
        "final_full_action": _stack(("action", "full_action"), np.float32),
        "body_effort_target": _stack(("action", "body_effort_target"), np.float32),
        "hand_action_left": _stack(("action", "hand_action_left"), np.float32),
        "hand_action_right": _stack(("action", "hand_action_right"), np.float32),
        "human_raw_smplx_json": np.array(
            _json_string([frame["human_raw"]["smplx_frame"] for frame in data_buffer])
        ),
        "human_controller_json": np.array(
            _json_string([frame["human_raw"]["controller_data"] for frame in data_buffer])
        ),
        "human_recording_control_json": np.array(
            _json_string([frame["human_raw"]["recording_control"] for frame in data_buffer])
        ),
    }

    football_states = [frame["env"]["football"] for frame in data_buffer]
    if any(state is not None for state in football_states):
        organized["env_obj_football_position"] = np.array(
            [
                state["position"] if state is not None else np.zeros(3, dtype=np.float32)
                for state in football_states
            ],
            dtype=np.float32,
        )
        organized["env_obj_football_linear_velocity"] = np.array(
            [
                state["linear_velocity"] if state is not None else np.zeros(3, dtype=np.float32)
                for state in football_states
            ],
            dtype=np.float32,
        )
        organized["env_obj_football_angular_velocity"] = np.array(
            [
                state["angular_velocity"] if state is not None else np.zeros(3, dtype=np.float32)
                for state in football_states
            ],
            dtype=np.float32,
        )

    table_states = [frame["env"]["table_drink"] for frame in data_buffer]
    if any(state is not None for state in table_states):
        organized["env_obj_table_drink_position"] = np.array(
            [
                state["position"] if state is not None else np.zeros(3, dtype=np.float32)
                for state in table_states
            ],
            dtype=np.float32,
        )
        organized["env_obj_table_drink_linear_velocity"] = np.array(
            [
                state["linear_velocity"] if state is not None else np.zeros(3, dtype=np.float32)
                for state in table_states
            ],
            dtype=np.float32,
        )
        organized["env_obj_table_drink_angular_velocity"] = np.array(
            [
                state["angular_velocity"] if state is not None else np.zeros(3, dtype=np.float32)
                for state in table_states
            ],
            dtype=np.float32,
        )

    rgb_frames = []
    depth_frames = []
    vision_indices = []
    for idx, frame in enumerate(data_buffer):
        rgb = frame["env"]["vision"].get("rgb")
        depth = frame["env"]["vision"].get("depth")
        if rgb is not None:
            rgb_frames.append(rgb)
            vision_indices.append(idx)
        if depth is not None:
            depth_frames.append(depth)
    if rgb_frames:
        organized["vision_rgb"] = np.array(rgb_frames)
        organized["vision_frame_indices"] = np.array(vision_indices, dtype=np.int32)
    if depth_frames:
        organized["vision_depth"] = np.array(depth_frames)

    return organized


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
        self.task_name = getattr(args_cli, "task", "sonic")

        # Debug/perf knobs (默认关闭高频打印，否则会把控制环拖到个位数 Hz)
        # - SONIC_DEBUG=1: 打开详细日志
        # - SONIC_LOG_EVERY=50: 每 N 帧打印一次（仍会在前几帧打印）
        self._sonic_debug = bool(int(os.environ.get("SONIC_DEBUG", "0") or "0"))
        self._sonic_log_every = int(os.environ.get("SONIC_LOG_EVERY", "50") or 50)
        # 也允许通过 CLI 参数覆盖（若 sim_main.py 透传了这些字段）
        self._sonic_debug = bool(getattr(args_cli, "sonic_debug", self._sonic_debug))
        self._sonic_log_every = int(getattr(args_cli, "sonic_log_every", self._sonic_log_every))
        self._enable_rtf_monitor = getattr(args_cli, "enable_rtf_monitor", False)
        self._use_effort_control = bool(getattr(args_cli, "sonic_effort_control", False))

        self.enable_dex3    = getattr(args_cli, "enable_dex3_dds",   False)
        self.enable_gripper = getattr(args_cli, "enable_dex1_dds",   False)
        self._pose_source   = getattr(args_cli, "sonic_pose_source", "redis")  # "zmq" | "redis"
        self.zmq_host       = getattr(args_cli, "sonic_zmq_host",    "localhost")
        self.zmq_port       = getattr(args_cli, "sonic_zmq_port",    5556)
        self.redis_host     = getattr(args_cli, "sonic_redis_host",  "localhost")
        self.redis_port     = getattr(args_cli, "sonic_redis_port",  6379)
        self.encoder_path   = getattr(args_cli, "sonic_encoder_path", "")
        self.decoder_path   = getattr(args_cli, "sonic_decoder_path", "")
        self._replay_file   = getattr(args_cli, "replay_file", "")
        self._replay_mode   = self._normalize_replay_mode(getattr(args_cli, "replay_mode", "inference_replay"))
        self._replay_loop   = bool(getattr(args_cli, "replay_loop", False))
        self._replay_enabled = bool(self._replay_file)
        if self._replay_enabled:
            self._pose_source = "replay"
        self._replay_anchor_heading_initialized = None
        self._replay_anchor_use_heading_align = None
        self._replay_anchor_init_base_quat_wxyz = None
        self._replay_anchor_init_ref_quat_wxyz = None
        self._replay_anchor_heading_align_quat_wxyz = None
        self._replay_encoder_input = None
        self._replay_encoder_smpl_joint_window = None
        self._replay_encoder_anchor_window = None
        self._replay_encoder_wrist_window = None
        self._replay_encoder_motion_joint_pos_hist = None
        self._replay_encoder_motion_joint_vel_hist = None
        self._replay_encoder_motion_root_z_hist = None
        self._replay_encoder_motion_anchor_rot6d_hist = None
        self._replay_encoder_robot_joint_pos_hist = None
        self._replay_encoder_robot_joint_vel_hist = None
        self._replay_decoder_obs = None
        self._replay_decoder_ang_vel_hist = None
        self._replay_decoder_gravity_dir_hist = None
        self._replay_decoder_last_action_hist = None
        self._replay_object_states = {}
        self._replay_joint_mae_sum = 0.0
        self._replay_joint_mae_count = 0
        self._replay_joint_err_log_interval = 10
        self._replay_object_err_sums = {}
        self._replay_object_err_counts = {}
        self.recording_manager = AsyncEpisodeRecorder(
            save_dir=getattr(args_cli, "recording_save_dir", "./recording_data"),
            task_name=f"{self.task_name}_sonic",
            organize_fn=_organize_sonic_episode,
            max_frames=10000,
        )
        self._should_start_recording_on_first_call = not self._replay_enabled
        self._recording_command = "none"
        self._recording_active = False
        self._recording_display_state = "idle"
        self._recording_display_counter = 0
        self._recording_display_duration = 10
        self._save_in_progress = False
        self._save_completion_state = None
        self._waiting_for_reset_complete = False
        self._reset_complete_received = False
        self._input_ready_key = get_input_ready_key("sonic")
        self._input_ready_epoch_id = -1
        self._input_ready_timestamp_realtime = 0.0
        self._input_ready_timestamp_monotonic = 0.0
        self._stale_input_drop_logged_epoch = -1
        self._episode_id = 0
        self._latest_human_smplx_frame = None
        self._latest_controller_data = None
        self._latest_recording_control = None
        self._latest_recording_control_sequence = -1
        self._latest_pose_payload = {}
        self._latest_encoder_input = np.zeros((1762,), dtype=np.float32)
        self._latest_smpl_joint_window = np.zeros((_STEP1_FRAMES, _N_SMPL_JOINTS, 3), dtype=np.float32)
        self._latest_anchor_window = np.zeros((_STEP1_FRAMES, 6), dtype=np.float32)
        self._latest_wrist_window = np.zeros((_STEP1_FRAMES, len(OFFICIAL_WRIST_INDICES)), dtype=np.float32)
        self._latest_decoder_obs = np.zeros((994,), dtype=np.float32)
        self._latest_decoder_raw_action = np.zeros((29,), dtype=np.float32)
        self._latest_decoder_target = self._sonic_default_np.copy() if hasattr(self, "_sonic_default_np") else np.zeros((29,), dtype=np.float32)
        self._latest_decoder_body_effort = np.zeros(29, dtype=np.float32)
        self._latest_frame_index = -1
        self._latest_timestamp_realtime = 0.0
        self._latest_timestamp_monotonic = 0.0
        self._latest_heading_increment = 0.0
        self._command_edge_this_frame = "none"
        # self._sonic_warmup_steps = int(getattr(args_cli, "sonic_warmup_steps", 50))  # warmup 已注释，仅用 history_ready
        self._sonic_warmup_steps = 0
        self._sonic_smooth_steps = int(getattr(args_cli, "sonic_smooth_steps", 20))
        cfg = getattr(env, "cfg", None)
        self._decimation    = int(getattr(cfg, "decimation", 4))

        self._setup_joint_mapping()
        self._setup_pd_controller()
        if self._replay_enabled:
            self._setup_local_replay()
        elif self._pose_source == "redis":
            self._setup_redis()
        else:
            self._setup_zmq()
        self._setup_policy()
        self._setup_buffers()
        self._latest_decoder_target = self._sonic_default_np.copy()
        self._setup_hand_dds(args_cli)

        print(f"[SonicActionProvider] POSE mode ready  "
              f"pose_source={self._pose_source}  "
              f"(zmq={self.zmq_host}:{self.zmq_port}  redis={self.redis_host}:{self.redis_port})  "
              f"encoder={self.encoder_path}  decoder={self.decoder_path}")
        if self._replay_enabled:
            print(
                f"[SonicActionProvider] replay enabled  "
                f"file={self._replay_file}  mode={self._replay_mode}  loop={self._replay_loop}"
            )

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

    def _setup_pd_controller(self):
        self._pd_kp_t = torch.tensor(SONIC_PD_KP, dtype=torch.float32, device=self.device)
        self._pd_kd_t = torch.tensor(SONIC_PD_KD, dtype=torch.float32, device=self.device)
        self._pd_effort_limit_t = torch.tensor(SONIC_EFFORT_LIMIT, dtype=torch.float32, device=self.device)
        self._zero_vel_t = torch.zeros_like(self._pd_kp_t)
        self._effort_mode_runtime_configured = False
        self._position_mode_runtime_configured = False

    def _ensure_effort_mode_runtime_config(self, env):
        if not self._use_effort_control or self._effort_mode_runtime_configured:
            return
        robot = env.scene["robot"]
        # Disable implicit drive on the 29 SONIC body joints so the explicit PD torque path
        # is not fighting Isaac Lab's built-in stiffness/damping controller.
        robot.write_joint_stiffness_to_sim(0.0, joint_ids=self._sonic_idx)
        robot.write_joint_damping_to_sim(0.0, joint_ids=self._sonic_idx)
        self._effort_mode_runtime_configured = True
        print("[SONIC] effort mode configured: body joint stiffness/damping set to 0")

    def _ensure_position_mode_runtime_config(self, env):
        if self._use_effort_control or self._position_mode_runtime_configured:
            return
        robot = env.scene["robot"]
        kp = self._pd_kp_t.unsqueeze(0)
        kd = self._pd_kd_t.unsqueeze(0)
        robot.write_joint_stiffness_to_sim(kp, joint_ids=self._sonic_idx)
        robot.write_joint_damping_to_sim(kd, joint_ids=self._sonic_idx)
        self._position_mode_runtime_configured = True
        print("[SONIC] position mode configured: body joint stiffness/damping written to SONIC gains")

    def _compute_sonic_effort(self, target_sonic: np.ndarray) -> torch.Tensor:
        robot = self.env.scene["robot"].data
        q = robot.joint_pos[0, self._sonic_idx]
        dq = robot.joint_vel[0, self._sonic_idx]
        q_des = torch.as_tensor(target_sonic, dtype=torch.float32, device=self.device)

        effort = self._pd_kp_t * (q_des - q) + self._pd_kd_t * (self._zero_vel_t - dq)
        effort = torch.clamp(effort, -self._pd_effort_limit_t, self._pd_effort_limit_t)
        return effort

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
        self._redis_control_client = None
        self._redis_frame_index = 0
        if not _HAS_REDIS:
            print("[SonicActionProvider] WARNING: redis not installed. pip install redis")
            return
        try:
            self._redis_client = redis.Redis(
                host=self.redis_host, port=self.redis_port, db=0, decode_responses=False
            )
            self._redis_control_client = redis.Redis(
                host=self.redis_host, port=self.redis_port, db=0, decode_responses=True
            )
            self._redis_client.ping()
            self._redis_control_client.ping()
            print(f"[SonicActionProvider] Redis connected "
                  f"{self.redis_host}:{self.redis_port} key=human_smplx_data_unitree_g1_with_hands")
        except Exception as e:
            print(f"[SonicActionProvider] Redis init failed: {e}")

    def _setup_local_replay(self):
        replay_path = Path(self._replay_file).expanduser().resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"[SonicActionProvider] replay file not found: {replay_path}")

        def _load_array(data, *keys, required=False):
            for key in keys:
                if key in data:
                    return np.asarray(data[key]).copy()
            if required:
                raise KeyError(f"[SonicActionProvider] replay npz missing keys: {keys}")
            return None

        with np.load(replay_path, allow_pickle=True) as replay_data:
            self._replay_body_targets = _load_array(
                replay_data,
                "final_body_action_29dof",
                "decoder_target_action",
                required=self._replay_mode == "direct_replay",
            )
            self._replay_hand_left = _load_array(replay_data, "hand_action_left", "human_left_hand")
            self._replay_hand_right = _load_array(replay_data, "hand_action_right", "human_right_hand")
            self._replay_smpl_joints = _load_array(replay_data, "human_smpl_joints")
            self._replay_smpl_pose = _load_array(replay_data, "human_smpl_pose")
            self._replay_body_quat_w = _load_array(replay_data, "human_body_quat_w")
            self._replay_joint_pos = _load_array(replay_data, "robot_qpos_before_decimation")
            self._replay_joint_vel = _load_array(replay_data, "robot_qvel_before_decimation")
            self._replay_frame_indices = _load_array(replay_data, "frame_index")
            self._replay_heading_increment = _load_array(replay_data, "human_heading_increment")
            self._replay_anchor_heading_initialized = _load_array(replay_data, "anchor_heading_initialized")
            self._replay_anchor_use_heading_align = _load_array(replay_data, "anchor_use_heading_align")
            self._replay_anchor_init_base_quat_wxyz = _load_array(
                replay_data,
                "anchor_init_base_quat_wxyz",
            )
            self._replay_anchor_init_ref_quat_wxyz = _load_array(
                replay_data,
                "anchor_init_ref_quat_wxyz",
            )
            self._replay_anchor_heading_align_quat_wxyz = _load_array(
                replay_data,
                "anchor_heading_align_quat_wxyz",
            )
            self._replay_encoder_input = _load_array(replay_data, "encoder_input")
            self._replay_encoder_smpl_joint_window = _load_array(replay_data, "encoder_smpl_joint_window")
            self._replay_encoder_anchor_window = _load_array(replay_data, "encoder_anchor_window")
            self._replay_encoder_wrist_window = _load_array(replay_data, "encoder_wrist_window")
            self._replay_encoder_motion_joint_pos_hist = _load_array(
                replay_data,
                "encoder_motion_joint_pos_hist",
            )
            self._replay_encoder_motion_joint_vel_hist = _load_array(
                replay_data,
                "encoder_motion_joint_vel_hist",
            )
            self._replay_encoder_motion_root_z_hist = _load_array(
                replay_data,
                "encoder_motion_root_z_hist",
            )
            self._replay_encoder_motion_anchor_rot6d_hist = _load_array(
                replay_data,
                "encoder_motion_anchor_rot6d_hist",
            )
            self._replay_encoder_robot_joint_pos_hist = _load_array(
                replay_data,
                "encoder_robot_joint_pos_hist",
            )
            self._replay_encoder_robot_joint_vel_hist = _load_array(
                replay_data,
                "encoder_robot_joint_vel_hist",
            )
            self._replay_decoder_obs = _load_array(replay_data, "decoder_obs")
            self._replay_decoder_ang_vel_hist = _load_array(replay_data, "decoder_ang_vel_hist")
            self._replay_decoder_gravity_dir_hist = _load_array(replay_data, "decoder_gravity_dir_hist")
            self._replay_decoder_last_action_hist = _load_array(replay_data, "decoder_last_action_hist")
            self._replay_object_states = {}
            replay_object_suffixes = {
                "_position": "position",
                "_linear_velocity": "linear_velocity",
                "_angular_velocity": "angular_velocity",
            }
            for key in replay_data.files:
                if not key.startswith("env_obj_"):
                    continue
                for suffix, field_name in replay_object_suffixes.items():
                    if key.endswith(suffix):
                        object_name = key[len("env_obj_") : -len(suffix)]
                        self._replay_object_states.setdefault(object_name, {})[field_name] = np.asarray(
                            replay_data[key]
                        ).copy()
                        break
            if "num_frames" in replay_data:
                self._replay_num_frames = int(np.asarray(replay_data["num_frames"]).item())
            else:
                candidate_arrays = [
                    self._replay_body_targets,
                    self._replay_smpl_joints,
                    self._replay_joint_pos,
                    self._replay_hand_left,
                    self._replay_hand_right,
                ]
                candidate_arrays = [arr for arr in candidate_arrays if arr is not None]
                if not candidate_arrays:
                    raise ValueError("[SonicActionProvider] replay npz contains no usable frame arrays")
                self._replay_num_frames = int(candidate_arrays[0].shape[0])

        if self._replay_mode == "inference_replay":
            required_arrays = {
                "human_smpl_joints": self._replay_smpl_joints,
                "human_smpl_pose": self._replay_smpl_pose,
                "human_body_quat_w": self._replay_body_quat_w,
                "robot_qpos_before_decimation": self._replay_joint_pos,
            }
            missing = [name for name, value in required_arrays.items() if value is None]
            if missing:
                raise KeyError(f"[SonicActionProvider] inference_replay missing arrays: {missing}")

        self._replay_file = str(replay_path)
        self._replay_cursor = 0
        print(
            f"[SonicActionProvider] loaded replay npz: {replay_path}  "
            f"frames={self._replay_num_frames}"
        )

    def _normalize_replay_mode(self, replay_mode: str) -> str:
        if replay_mode in ("direct", "direct_replay"):
            return "direct_replay"
        if replay_mode in ("inference", "inference_replay"):
            return "inference_replay"
        raise ValueError(f"[SonicActionProvider] Unsupported replay_mode: {replay_mode}")

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
        if "CUDAExecutionProvider" in avail:
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
        self._ref_smpl_joints_window = np.zeros((_STEP1_FRAMES, _N_SMPL_JOINTS, 3), dtype=np.float32)
        self._ref_body_quat_window = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (_STEP1_FRAMES, 1)
        )
        self._ref_joint_pos_window = np.tile(
            self._sonic_default_np[np.newaxis], (_STEP1_FRAMES, 1)
        )
        self._ref_window_valid = False

        # 机器人状态历史缓冲（SONIC IsaacLab order）
        # 用于 encoder 输入（step5 采样）和 decoder 输入（step1 连续10帧）
        self._robot_joint_pos_hist = np.zeros((_STEP1_FRAMES, 29), dtype=np.float32)
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

        # Streamed-motion style reference timeline for POSE mode.
        # This approximates SONIC deploy's ZMQEndpointInterface + StreamedMotionMerger:
        # incoming recent chunks are merged onto a global frame-index timeline, and
        # encoder active blocks are gathered from a playback cursor over that timeline.
        self._stream_ref_frames: dict[int, dict[str, np.ndarray]] = {}
        self._stream_ref_indices: list[int] = []
        self._stream_playback_frame_idx: int | None = None
        self._stream_window_start = 0
        self._stream_current_frame = 0
        self._stream_frame_step = 1
        self._stream_history_keep = 5
        self._stream_max_gap_frames = 200
        self._stream_max_frames = 80

        # RTF (Real Time Factor) monitoring
        self._rtf_wall_time_start = None
        self._rtf_sim_time_start = None
        self._rtf_log_interval = 50  # Print RTF every N frames

        # Performance profiling buffers
        self._perf_fetch_pose_ms = []
        self._perf_encoder_ms = []
        self._perf_decoder_ms = []
        self._perf_sim_step_ms = []
        self._perf_render_ms = []
        self._perf_total_ms = []
        self._perf_buffer_size = 50  # Keep last N frames for statistics
        self._perf_report_interval = 50  # Print performance report every N frames

    def _next_replay_frame_idx(self) -> int | None:
        if not self._replay_enabled or self._replay_num_frames <= 0:
            return None
        if self._replay_cursor >= self._replay_num_frames:
            if not self._replay_loop:
                return None
            self._replay_cursor = 0
        frame_idx = int(self._replay_cursor)
        self._replay_cursor += 1
        return frame_idx

    def _set_replay_hand_targets(self, frame_idx: int) -> None:
        self._left_hand_target.fill(0.0)
        self._right_hand_target.fill(0.0)
        if self._replay_hand_left is not None and frame_idx < len(self._replay_hand_left):
            left = np.asarray(self._replay_hand_left[frame_idx], dtype=np.float32).reshape(-1)
            self._left_hand_target[: min(7, left.shape[0])] = left[:7]
        if self._replay_hand_right is not None and frame_idx < len(self._replay_hand_right):
            right = np.asarray(self._replay_hand_right[frame_idx], dtype=np.float32).reshape(-1)
            self._right_hand_target[: min(7, right.shape[0])] = right[:7]

    def _apply_replay_anchor_state(self, frame_idx: int) -> None:
        if (
            self._replay_anchor_heading_initialized is None
            or frame_idx >= len(self._replay_anchor_heading_initialized)
        ):
            return
        self._anchor_heading_initialized = bool(
            np.asarray(self._replay_anchor_heading_initialized[frame_idx]).reshape(-1)[-1]
        )
        if (
            self._replay_anchor_use_heading_align is not None
            and frame_idx < len(self._replay_anchor_use_heading_align)
        ):
            self._anchor_use_heading_align = bool(
                np.asarray(self._replay_anchor_use_heading_align[frame_idx]).reshape(-1)[-1]
            )
        if (
            self._replay_anchor_init_base_quat_wxyz is not None
            and frame_idx < len(self._replay_anchor_init_base_quat_wxyz)
        ):
            self._anchor_init_base_quat_wxyz[:] = np.asarray(
                self._replay_anchor_init_base_quat_wxyz[frame_idx], dtype=np.float32
            ).reshape(-1)[:4]
        if (
            self._replay_anchor_init_ref_quat_wxyz is not None
            and frame_idx < len(self._replay_anchor_init_ref_quat_wxyz)
        ):
            self._anchor_init_ref_quat_wxyz[:] = np.asarray(
                self._replay_anchor_init_ref_quat_wxyz[frame_idx], dtype=np.float32
            ).reshape(-1)[:4]
        if (
            self._replay_anchor_heading_align_quat_wxyz is not None
            and frame_idx < len(self._replay_anchor_heading_align_quat_wxyz)
        ):
            self._anchor_heading_align_quat_wxyz[:] = np.asarray(
                self._replay_anchor_heading_align_quat_wxyz[frame_idx], dtype=np.float32
            ).reshape(-1)[:4]

    def _prepare_replay_frame(self, frame_idx: int) -> np.ndarray | None:
        self._set_replay_hand_targets(frame_idx)
        self._apply_replay_anchor_state(frame_idx)
        if self._replay_frame_indices is not None and frame_idx < len(self._replay_frame_indices):
            self._latest_frame_index = int(np.asarray(self._replay_frame_indices[frame_idx]).reshape(-1)[-1])
        else:
            self._latest_frame_index = frame_idx
        if self._replay_heading_increment is not None and frame_idx < len(self._replay_heading_increment):
            self._latest_heading_increment = float(np.asarray(self._replay_heading_increment[frame_idx]).reshape(-1)[-1])

        if self._replay_mode == "direct_replay":
            if self._replay_body_targets is None or frame_idx >= len(self._replay_body_targets):
                return None
            sonic_targets = np.asarray(self._replay_body_targets[frame_idx], dtype=np.float32).reshape(-1)
            if sonic_targets.shape[0] != 29:
                raise ValueError(
                    f"[SonicActionProvider] direct_replay target must have 29 dims, got {sonic_targets.shape}"
                )
            self._latest_decoder_target = sonic_targets.copy()
            return sonic_targets

        # For inference replay, restore the exact recorded model inputs and history
        # buffers. Do not call _apply_pose_data() here: that path rolls temporal
        # windows forward once more and turns replay into a reconstructed history
        # instead of an exact restore of the recorded model state.
        restored_current_frame = False
        if self._replay_encoder_smpl_joint_window is not None and frame_idx < len(self._replay_encoder_smpl_joint_window):
            self._latest_smpl_joint_window = np.asarray(
                self._replay_encoder_smpl_joint_window[frame_idx], dtype=np.float32
            ).copy()
            if self._latest_smpl_joint_window.shape == self._smpl_joints_buf.shape:
                self._smpl_joints_buf[:] = self._latest_smpl_joint_window
                self._ref_smpl_joints_window[:] = self._latest_smpl_joint_window
                self._smpl_data_valid = bool(np.abs(self._latest_smpl_joint_window[-1]).sum() > 0.01)
                self._smpl_history_fill = _STEP1_FRAMES if self._smpl_data_valid else 0
                restored_current_frame = True
        if self._replay_encoder_anchor_window is not None and frame_idx < len(self._replay_encoder_anchor_window):
            self._latest_anchor_window = np.asarray(
                self._replay_encoder_anchor_window[frame_idx], dtype=np.float32
            ).copy()
            if self._latest_anchor_window.shape == self._body_rot6d_buf.shape:
                self._body_rot6d_buf[:] = self._latest_anchor_window
                restored_current_frame = True
        if self._replay_encoder_wrist_window is not None and frame_idx < len(self._replay_encoder_wrist_window):
            self._latest_wrist_window = np.asarray(
                self._replay_encoder_wrist_window[frame_idx], dtype=np.float32
            ).copy()
        if (
            self._replay_encoder_motion_joint_pos_hist is not None
            and frame_idx < len(self._replay_encoder_motion_joint_pos_hist)
        ):
            hist = np.asarray(self._replay_encoder_motion_joint_pos_hist[frame_idx], dtype=np.float32)
            if hist.shape == self._motion_joint_pos_hist.shape:
                self._motion_joint_pos_hist[:] = hist
        if (
            self._replay_encoder_motion_joint_vel_hist is not None
            and frame_idx < len(self._replay_encoder_motion_joint_vel_hist)
        ):
            hist = np.asarray(self._replay_encoder_motion_joint_vel_hist[frame_idx], dtype=np.float32)
            if hist.shape == self._motion_joint_vel_hist.shape:
                self._motion_joint_vel_hist[:] = hist
        if (
            self._replay_encoder_motion_root_z_hist is not None
            and frame_idx < len(self._replay_encoder_motion_root_z_hist)
        ):
            hist = np.asarray(self._replay_encoder_motion_root_z_hist[frame_idx], dtype=np.float32).reshape(-1)
            if hist.shape == self._motion_root_z_hist.shape:
                self._motion_root_z_hist[:] = hist
        if (
            self._replay_encoder_motion_anchor_rot6d_hist is not None
            and frame_idx < len(self._replay_encoder_motion_anchor_rot6d_hist)
        ):
            hist = np.asarray(self._replay_encoder_motion_anchor_rot6d_hist[frame_idx], dtype=np.float32)
            if hist.shape == self._motion_anchor_rot6d_hist.shape:
                self._motion_anchor_rot6d_hist[:] = hist
        if (
            self._replay_encoder_robot_joint_pos_hist is not None
            and frame_idx < len(self._replay_encoder_robot_joint_pos_hist)
        ):
            hist = np.asarray(self._replay_encoder_robot_joint_pos_hist[frame_idx], dtype=np.float32)
            if hist.shape == self._robot_joint_pos_hist.shape:
                self._robot_joint_pos_hist[:] = hist
        if (
            self._replay_encoder_robot_joint_vel_hist is not None
            and frame_idx < len(self._replay_encoder_robot_joint_vel_hist)
        ):
            hist = np.asarray(self._replay_encoder_robot_joint_vel_hist[frame_idx], dtype=np.float32)
            if hist.shape == self._robot_joint_vel_hist.shape:
                self._robot_joint_vel_hist[:] = hist
        if self._replay_decoder_ang_vel_hist is not None and frame_idx < len(self._replay_decoder_ang_vel_hist):
            hist = np.asarray(self._replay_decoder_ang_vel_hist[frame_idx], dtype=np.float32)
            if hist.shape == self._ang_vel_hist.shape:
                self._ang_vel_hist[:] = hist
        if (
            self._replay_decoder_gravity_dir_hist is not None
            and frame_idx < len(self._replay_decoder_gravity_dir_hist)
        ):
            hist = np.asarray(self._replay_decoder_gravity_dir_hist[frame_idx], dtype=np.float32)
            if hist.shape == self._grav_dir_hist.shape:
                self._grav_dir_hist[:] = hist
        if (
            self._replay_decoder_last_action_hist is not None
            and frame_idx < len(self._replay_decoder_last_action_hist)
        ):
            hist = np.asarray(self._replay_decoder_last_action_hist[frame_idx], dtype=np.float32)
            if hist.shape == self._last_action_hist.shape:
                self._last_action_hist[:] = hist
        if self._replay_encoder_input is not None and frame_idx < len(self._replay_encoder_input):
            self._latest_encoder_input = np.asarray(self._replay_encoder_input[frame_idx], dtype=np.float32).reshape(-1).copy()
        if self._replay_decoder_obs is not None and frame_idx < len(self._replay_decoder_obs):
            self._latest_decoder_obs = np.asarray(self._replay_decoder_obs[frame_idx], dtype=np.float32).reshape(-1).copy()
        if self._replay_smpl_joints is not None and frame_idx < len(self._replay_smpl_joints):
            current = np.asarray(self._replay_smpl_joints[frame_idx], dtype=np.float32)
            if current.shape == self._smpl_joints_buf[-1].shape:
                self._smpl_joints_buf[-1] = current
                self._ref_smpl_joints_window[-1] = current
                self._smpl_data_valid = bool(np.abs(current).sum() > 0.01)
                if self._smpl_data_valid and self._smpl_history_fill == 0:
                    self._smpl_history_fill = 1
                restored_current_frame = True
        if self._replay_smpl_pose is not None and frame_idx < len(self._replay_smpl_pose):
            current = np.asarray(self._replay_smpl_pose[frame_idx], dtype=np.float32)
            if current.shape == self._smpl_pose_buf[-1].shape:
                self._smpl_pose_buf[-1] = current
        if self._replay_body_quat_w is not None and frame_idx < len(self._replay_body_quat_w):
            current = np.asarray(self._replay_body_quat_w[frame_idx], dtype=np.float32).reshape(-1)
            if current.shape[0] >= 4:
                self._ref_body_quat_window[-1] = current[:4]
                restored_current_frame = True
        if self._replay_joint_pos is not None and frame_idx < len(self._replay_joint_pos):
            current = np.asarray(self._replay_joint_pos[frame_idx], dtype=np.float32).reshape(-1)
            if current.shape == self._robot_joint_pos_hist[-1].shape:
                self._robot_joint_pos = current.copy()
                if self._replay_encoder_robot_joint_pos_hist is None:
                    self._robot_joint_pos_hist[-1] = current
                if self._replay_encoder_motion_joint_pos_hist is None:
                    self._motion_joint_pos_hist[-1] = current
        if self._replay_joint_vel is not None and frame_idx < len(self._replay_joint_vel):
            current = np.asarray(self._replay_joint_vel[frame_idx], dtype=np.float32).reshape(-1)
            if current.shape == self._robot_joint_vel_hist[-1].shape:
                self._robot_joint_vel = current.copy()
                if self._replay_encoder_robot_joint_vel_hist is None:
                    self._robot_joint_vel_hist[-1] = current
                if self._replay_encoder_motion_joint_vel_hist is None:
                    self._motion_joint_vel_hist[-1] = current
        if restored_current_frame:
            self._ref_window_valid = True
        return None

    def _run_gear_sonic_replay_inference(self, frame_idx: int) -> np.ndarray:
        if (
            self._replay_encoder_input is None
            or self._replay_decoder_obs is None
            or frame_idx >= len(self._replay_encoder_input)
            or frame_idx >= len(self._replay_decoder_obs)
        ):
            return self._run_gear_sonic()

        # If the recorded frame contains no valid SMPL input, mimic the live provider
        # behavior and return the default standing target instead of forcing ONNX.
        if (
            self._replay_smpl_joints is not None
            and frame_idx < len(self._replay_smpl_joints)
            and float(np.abs(np.asarray(self._replay_smpl_joints[frame_idx], dtype=np.float32)).sum()) <= 0.01
        ):
            self._latest_decoder_raw_action = np.zeros((29,), dtype=np.float32)
            self._latest_decoder_target = self._sonic_default_np.copy()
            return self._sonic_default_np.copy()

        try:
            enc_input = np.asarray(self._replay_encoder_input[frame_idx], dtype=np.float32).reshape(1, -1)
            self._latest_encoder_input = enc_input[0].copy()
            t_enc0 = time.perf_counter()
            latent = self._encoder.run(None, {self._encoder.get_inputs()[0].name: enc_input})[0]
            t_enc1 = time.perf_counter()
            self._latent = latent

            dec_obs = np.asarray(self._replay_decoder_obs[frame_idx], dtype=np.float32).reshape(1, -1).copy()
            latent_flat = latent.reshape(-1).astype(np.float32, copy=False)
            if dec_obs.shape[1] >= latent_flat.shape[0]:
                dec_obs[0, :latent_flat.shape[0]] = latent_flat
            self._latest_decoder_obs = dec_obs[0].copy()

            t_dec0 = time.perf_counter()
            action_sonic = self._decoder.run(None, {self._decoder.get_inputs()[0].name: dec_obs})[0]
            t_dec1 = time.perf_counter()

            raw_sonic_unclipped = action_sonic.flatten()[:29].astype(np.float32, copy=False)
            self._latest_decoder_raw_action = raw_sonic_unclipped.astype(np.float32, copy=True)
            target_sonic = raw_sonic_unclipped * G1_ACTION_SCALE_ISAACLAB + self._sonic_default_np
            self._latest_decoder_target = target_sonic.astype(np.float32, copy=True)

            enc_ms = (t_enc1 - t_enc0) * 1000.0
            dec_ms = (t_dec1 - t_dec0) * 1000.0
            self._perf_encoder_ms.append(enc_ms)
            self._perf_decoder_ms.append(dec_ms)
            if len(self._perf_encoder_ms) > self._perf_buffer_size:
                self._perf_encoder_ms.pop(0)
            if len(self._perf_decoder_ms) > self._perf_buffer_size:
                self._perf_decoder_ms.pop(0)

            return target_sonic.astype(np.float32)
        except Exception as e:
            print(f"[SonicActionProvider] replay inference error at frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return self._run_gear_sonic()

    def _log_replay_joint_position_error(self, current_pos: np.ndarray, frame_idx: int) -> None:
        if self._replay_joint_pos is None or len(self._replay_joint_pos) == 0:
            return
        compare_idx = min(frame_idx + 1, len(self._replay_joint_pos) - 1)
        recorded = np.asarray(self._replay_joint_pos[compare_idx], dtype=np.float32).reshape(-1)
        current = np.asarray(current_pos, dtype=np.float32).reshape(-1)
        dims = min(current.shape[0], recorded.shape[0])
        if dims <= 0:
            return
        err = np.abs(current[:dims] - recorded[:dims])
        mae = float(err.mean())
        max_err = float(err.max())
        self._replay_joint_mae_sum += mae
        self._replay_joint_mae_count += 1
        running_mae = self._replay_joint_mae_sum / max(1, self._replay_joint_mae_count)
        if frame_idx < 3 or frame_idx % self._replay_joint_err_log_interval == 0:
            print(
                f"[SonicActionProvider] Replay joint err: frame={frame_idx} "
                f"compare_to_recorded_frame={compare_idx} "
                f"mae={mae:.6f} rad max={max_err:.6f} rad "
                f"running_mae={running_mae:.6f} rad"
            )

    def _resolve_replay_object_scene_key(self, object_name: str) -> str | None:
        alias_candidates = {
            "football": ("object", "football"),
            "table_drink": ("table_drink",),
        }
        for scene_key in alias_candidates.get(object_name, (object_name,)):
            if scene_key in self.env.scene.keys():
                return scene_key
        return None

    def _get_current_replay_object_state(self, object_name: str) -> dict | None:
        scene_key = self._resolve_replay_object_scene_key(object_name)
        if scene_key is None:
            return None
        try:
            obj = self.env.scene[scene_key]
            root_state = obj.data.root_state_w
            return {
                "position": root_state[0, 0:3].detach().cpu().numpy().astype(np.float32),
                "linear_velocity": root_state[0, 7:10].detach().cpu().numpy().astype(np.float32),
                "angular_velocity": root_state[0, 10:13].detach().cpu().numpy().astype(np.float32),
            }
        except Exception:
            return None

    def _log_replay_object_state_error(self, frame_idx: int) -> None:
        if not self._replay_object_states:
            return
        if not (frame_idx < 3 or frame_idx % self._replay_joint_err_log_interval == 0):
            return

        field_units = {
            "position": "m",
            "linear_velocity": "m/s",
            "angular_velocity": "rad/s",
        }
        field_labels = {
            "position": "pos",
            "linear_velocity": "lin_vel",
            "angular_velocity": "ang_vel",
        }

        for object_name, recorded_fields in self._replay_object_states.items():
            current_state = self._get_current_replay_object_state(object_name)
            if current_state is None:
                continue

            parts = []
            object_compare_idx = None
            for field_name in ("position", "linear_velocity", "angular_velocity"):
                recorded_series = recorded_fields.get(field_name)
                if recorded_series is None or len(recorded_series) == 0:
                    continue
                compare_idx = min(frame_idx + 1, len(recorded_series) - 1)
                object_compare_idx = compare_idx
                recorded = np.asarray(recorded_series[compare_idx], dtype=np.float32).reshape(-1)
                current = np.asarray(current_state[field_name], dtype=np.float32).reshape(-1)
                dims = min(current.shape[0], recorded.shape[0])
                if dims <= 0:
                    continue
                err = np.abs(current[:dims] - recorded[:dims])
                mae = float(err.mean())
                max_err = float(err.max())
                stat_key = (object_name, field_name)
                self._replay_object_err_sums[stat_key] = self._replay_object_err_sums.get(stat_key, 0.0) + mae
                self._replay_object_err_counts[stat_key] = self._replay_object_err_counts.get(stat_key, 0) + 1
                running_mae = self._replay_object_err_sums[stat_key] / self._replay_object_err_counts[stat_key]
                unit = field_units[field_name]
                parts.append(
                    f"{field_labels[field_name]}_mae={mae:.6f} {unit} "
                    f"max={max_err:.6f} {unit} running={running_mae:.6f} {unit}"
                )

            if parts:
                print(
                    f"[SonicActionProvider] Replay env err: frame={frame_idx} "
                    f"compare_to_recorded_frame={object_compare_idx if object_compare_idx is not None else frame_idx} "
                    f"object={object_name} "
                    + " ".join(parts)
                )

    def on_env_reset(self):
        try:
            robot = self.env.scene["robot"].data
            joint_pos_sonic = robot.joint_pos[0, self._sonic_idx].cpu().numpy().astype(np.float32)
            joint_vel_sonic = robot.joint_vel[0, self._sonic_idx].cpu().numpy().astype(np.float32)
        except Exception:
            joint_pos_sonic = np.zeros(29, dtype=np.float32)
            joint_vel_sonic = np.zeros(29, dtype=np.float32)
            root_z = 0.0
        else:
            joint_pos_sonic = joint_pos_sonic - self._sonic_default_np
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
        self._ref_smpl_joints_window.fill(0.0)
        self._ref_body_quat_window[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._ref_joint_pos_window[:] = self._sonic_default_np
        self._ref_window_valid = False
        self._robot_joint_pos_hist[:] = joint_pos_sonic
        self._robot_joint_vel_hist[:] = joint_vel_sonic
        self._motion_joint_pos_hist[:] = joint_pos_sonic
        self._motion_joint_vel_hist[:] = joint_vel_sonic
        self._motion_root_z_hist[:] = root_z
        self._motion_anchor_rot6d_hist[:] = self._body_rot6d_buf[-1]
        self._ang_vel_hist.fill(0.0)
        self._grav_dir_hist.fill(0.0)
        self._last_action_hist.fill(0.0)
        self._effort_mode_runtime_configured = False
        self._position_mode_runtime_configured = False
        self._stream_ref_frames.clear()
        self._stream_ref_indices.clear()
        self._stream_playback_frame_idx = None
        self._stream_window_start = 0
        self._stream_current_frame = 0
        self._stream_frame_step = 1
        if self._replay_enabled:
            self._replay_cursor = 0
            self._replay_joint_mae_sum = 0.0
            self._replay_joint_mae_count = 0
            self._replay_object_err_sums = {}
            self._replay_object_err_counts = {}
        self._latest_controller_data = None
        self._latest_recording_control = None
        self._latest_timestamp_realtime = 0.0
        self._latest_timestamp_monotonic = 0.0
        # print(f"[SONIC] on_env_reset: frame_count reset, warmup={self._sonic_warmup_steps}")  # warmup 已注释
        print(f"[SONIC] on_env_reset: frame_count and history reset")

    def _begin_episode_recording(self) -> None:
        if not self.recording_manager.is_recording:
            self.recording_manager.start_recording()
        self._recording_active = True
        self._recording_display_state = "recording"
        self._recording_display_counter = 0
        self._save_in_progress = False

    def is_recording_active(self):
        return self._recording_active

    def get_recording_command(self):
        return self._recording_command

    def _update_input_ready_guard(self, raw_guard) -> None:
        guard = None
        if raw_guard is not None:
            try:
                if isinstance(raw_guard, (bytes, bytearray)):
                    raw_guard = raw_guard.decode("utf-8")
                parsed = json.loads(raw_guard)
                if isinstance(parsed, dict):
                    guard = parsed
            except Exception:
                guard = None
        new_epoch = int(guard.get("epoch_id", -1)) if guard is not None else -1
        new_realtime = float(guard.get("ready_timestamp_realtime", 0.0)) if guard is not None else 0.0
        new_monotonic = float(guard.get("ready_timestamp_monotonic", 0.0)) if guard is not None else 0.0
        if (
            new_epoch != self._input_ready_epoch_id
            or new_realtime != self._input_ready_timestamp_realtime
            or new_monotonic != self._input_ready_timestamp_monotonic
        ):
            self._stale_input_drop_logged_epoch = -1
        self._input_ready_epoch_id = new_epoch
        self._input_ready_timestamp_realtime = new_realtime
        self._input_ready_timestamp_monotonic = new_monotonic

    def _is_stale_controller_payload(self, controller_data: dict | None) -> bool:
        if self._input_ready_timestamp_monotonic <= 0.0 and self._input_ready_timestamp_realtime <= 0.0:
            return False
        if not isinstance(controller_data, dict):
            return True
        monotonic_ts = controller_data.get("timestamp_monotonic")
        realtime_ts = controller_data.get("timestamp_realtime")
        try:
            monotonic_ts = float(monotonic_ts) if monotonic_ts is not None else None
        except Exception:
            monotonic_ts = None
        try:
            realtime_ts = float(realtime_ts) if realtime_ts is not None else None
        except Exception:
            realtime_ts = None
        if self._input_ready_timestamp_monotonic > 0.0 and monotonic_ts is not None:
            return monotonic_ts <= self._input_ready_timestamp_monotonic
        if self._input_ready_timestamp_realtime > 0.0 and realtime_ts is not None:
            return realtime_ts <= self._input_ready_timestamp_realtime
        return True

    def _update_recording_display_state(self) -> None:
        if self._save_completion_state is not None:
            if self._save_completion_state == "success":
                self._recording_display_state = "saved"
            else:
                self._recording_display_state = "discard"
            self._recording_display_counter = 0
            self._save_in_progress = False
            self._save_completion_state = None
        elif self._save_in_progress:
            return
        elif self._recording_display_state in ["saved", "discard"]:
            self._recording_display_counter += 1
            if self._recording_display_counter >= self._recording_display_duration:
                self._recording_display_counter = 0
                if not self.recording_manager.is_recording and not self._waiting_for_reset_complete:
                    self._begin_episode_recording()
                else:
                    self._recording_display_state = "recording" if self.recording_manager.is_recording else "idle"
        elif self.recording_manager.is_recording:
            self._recording_display_state = "recording"
        elif self._recording_display_state == "recording":
            self._recording_display_state = "idle"

    def _trigger_complete_reset(self) -> None:
        try:
            publish_reset_command(
                reset_category="3",
                redis_client=self._redis_control_client,
                host=self.redis_host,
                port=self.redis_port,
            )
            print("[SONIC] complete reset requested via Redis")
        except Exception as exc:
            print(f"[SONIC] failed to send reset trigger: {exc}")

    def _check_reset_complete(self) -> bool:
        try:
            return consume_reset_complete(
                redis_client=self._redis_control_client,
                host=self.redis_host,
                port=self.redis_port,
            )
        except Exception as exc:
            print(f"[SONIC] failed to check reset complete: {exc}")
            return False

    def _handle_recording_command(self) -> None:
        command = self._recording_command
        if command == "none":
            return
        print(f"[SONIC] recording command received: {command}")

        if command == "start":
            self._begin_episode_recording()

        elif command == "save":
            if self.recording_manager.is_recording:
                print("[SONIC] saving recording...")
                self._recording_display_state = "saving"
                self._recording_display_counter = 0
                self._save_in_progress = True
                self.recording_manager.save_recording(completion_callback=None)
                self.recording_manager.save_queue.join()
                print("[SONIC] save completed")
                self._save_in_progress = False
                self._recording_active = False
                self._recording_display_state = "saved"
                self._recording_display_counter = 0
                self._save_completion_state = None
                self._episode_id += 1

        elif command == "cancel":
            self.recording_manager.cancel_recording()
            self._recording_active = False
            self._recording_display_state = "discard"
            self._recording_display_counter = 0
            self._save_in_progress = False

        elif command == "save_and_reset":
            if self.recording_manager.is_recording:
                print("[SONIC] save_and_reset command received")
                self._recording_display_state = "saving"
                self._recording_display_counter = 0
                self._save_in_progress = True
                print("[SONIC] saving recording...")
                self.recording_manager.save_recording(completion_callback=None)
                self.recording_manager.save_queue.join()
                print("[SONIC] save completed")
                self._save_in_progress = False
            self._recording_active = False
            print("[SONIC] triggering complete reset...")
            self._trigger_complete_reset()
            self._waiting_for_reset_complete = True
            self._reset_complete_received = False

        elif command == "discard_and_reset":
            print("[SONIC] discard_and_reset command received")
            self.recording_manager.cancel_recording()
            self._recording_active = False
            self._recording_display_state = "discard"
            self._recording_display_counter = 0
            self._save_in_progress = False
            print("[SONIC] triggering complete reset...")
            self._trigger_complete_reset()
            self._waiting_for_reset_complete = True
            self._reset_complete_received = False

        self._recording_command = "none"

    def _collect_env_state(self) -> dict[str, Any]:
        env_state = {"football": None, "table_drink": None}
        try:
            if "object" in self.env.scene.keys():
                obj = self.env.scene["object"]
                root_state = obj.data.root_state_w
                env_state["football"] = {
                    "position": root_state[0, 0:3].cpu().numpy(),
                    "linear_velocity": root_state[0, 7:10].cpu().numpy(),
                    "angular_velocity": root_state[0, 10:13].cpu().numpy(),
                }
        except Exception:
            env_state["football"] = None
        try:
            if "table_drink" in self.env.scene.keys():
                obj = self.env.scene["table_drink"]
                root_state = obj.data.root_state_w
                env_state["table_drink"] = {
                    "position": root_state[0, 0:3].cpu().numpy(),
                    "linear_velocity": root_state[0, 7:10].cpu().numpy(),
                    "angular_velocity": root_state[0, 10:13].cpu().numpy(),
                }
        except Exception:
            env_state["table_drink"] = None
        return env_state

    def _collect_vision_state(self) -> dict[str, Any]:
        vision = {"rgb": None, "depth": None}
        try:
            if "front_camera" not in self.env.scene.keys():
                return vision
            camera = self.env.scene["front_camera"]
            if "rgb" in camera.data.output:
                vision["rgb"] = camera.data.output["rgb"][0].cpu().numpy().copy()
            if "distance_to_image_plane" in camera.data.output:
                vision["depth"] = camera.data.output["distance_to_image_plane"][0].cpu().numpy().copy()
        except Exception:
            return vision
        return vision

    def _collect_recording_data(
        self,
        *,
        full_action: torch.Tensor,
        sonic_targets: np.ndarray,
        body_effort_target: np.ndarray,
    ) -> dict[str, Any]:
        robot = self.env.scene["robot"].data
        root_state = robot.root_state_w
        joint_pos = robot.joint_pos[0, self._sonic_idx].cpu().numpy().astype(np.float32)
        joint_vel = robot.joint_vel[0, self._sonic_idx].cpu().numpy().astype(np.float32)
        return {
            "meta": {
                "task": self.task_name,
                "episode_id": self._episode_id,
                "control_dt": float(self.env.physics_dt * self._decimation),
                "physics_dt": float(self.env.physics_dt),
                "decimation": int(self._decimation),
                "pose_source": self._pose_source,
                "encoder_path": self.encoder_path,
                "decoder_path": self.decoder_path,
            },
            "markers": {
                "frame_index": int(self._latest_frame_index),
                "episode_step": int(self.recording_manager.frame_count),
                "timestamp_wall": float(time.time()),
                "timestamp_monotonic": float(self._latest_timestamp_monotonic),
                "timestamp_realtime": float(self._latest_timestamp_realtime),
                "recording_command": self._command_edge_this_frame,
                "reset_requested": bool(
                    self._waiting_for_reset_complete
                    or self._command_edge_this_frame in {"save_and_reset", "discard_and_reset"}
                ),
                "reset_completed": bool(self._reset_complete_received),
                "save_triggered": bool(self._command_edge_this_frame == "save"),
            },
            "human_raw": {
                "smplx_frame": self._latest_human_smplx_frame,
                "left_hand": self._left_hand_target.copy(),
                "right_hand": self._right_hand_target.copy(),
                "controller_data": self._latest_controller_data,
                "controller_binary": _extract_controller_binary_signals(self._latest_controller_data),
                "recording_control": self._latest_recording_control,
            },
            "human_processed": {
                "smpl_joints": self._smpl_joints_buf[-1].copy(),
                "smpl_pose": self._smpl_pose_buf[-1].copy(),
                "body_quat_w": self._ref_body_quat_window[-1].copy(),
                "vr_position": self._vr_3pt_position.copy(),
                "vr_orientation": self._vr_3pt_orientation.copy(),
                "heading_increment": np.array([self._latest_heading_increment], dtype=np.float32),
                "anchor_heading_initialized": np.array([self._anchor_heading_initialized], dtype=np.bool_),
                "anchor_use_heading_align": np.array([self._anchor_use_heading_align], dtype=np.bool_),
                "anchor_init_base_quat_wxyz": self._anchor_init_base_quat_wxyz.copy(),
                "anchor_init_ref_quat_wxyz": self._anchor_init_ref_quat_wxyz.copy(),
                "anchor_heading_align_quat_wxyz": self._anchor_heading_align_quat_wxyz.copy(),
            },
            "sonic_model_io": {
                "encoder_input": self._latest_encoder_input.copy(),
                "smpl_joint_window": self._latest_smpl_joint_window.copy(),
                "anchor_window": self._latest_anchor_window.copy(),
                "wrist_window": self._latest_wrist_window.copy(),
                "motion_joint_pos_hist": self._motion_joint_pos_hist.copy(),
                "motion_joint_vel_hist": self._motion_joint_vel_hist.copy(),
                "motion_root_z_hist": self._motion_root_z_hist.copy(),
                "motion_anchor_rot6d_hist": self._motion_anchor_rot6d_hist.copy(),
                "robot_joint_pos_hist": self._robot_joint_pos_hist.copy(),
                "robot_joint_vel_hist": self._robot_joint_vel_hist.copy(),
                "encoder_latent": np.zeros((64,), dtype=np.float32)
                if self._latent is None
                else np.asarray(self._latent).reshape(-1).astype(np.float32),
                "decoder_obs": self._latest_decoder_obs.copy(),
                "ang_vel_hist": self._ang_vel_hist.copy(),
                "gravity_dir_hist": self._grav_dir_hist.copy(),
                "last_action_hist": self._last_action_hist.copy(),
                "decoder_raw_action": self._latest_decoder_raw_action.copy(),
                "decoder_target_action": self._latest_decoder_target.copy(),
            },
            "robot": {
                "qpos_before_decimation": joint_pos,
                "qvel_before_decimation": joint_vel,
                "root_position": root_state[0, 0:3].cpu().numpy(),
                "root_orientation": root_state[0, 3:7].cpu().numpy(),
                "root_lin_vel_local": robot.root_lin_vel_b[0].cpu().numpy(),
                "root_ang_vel_local": robot.root_ang_vel_b[0].cpu().numpy(),
                "root_lin_vel_world": root_state[0, 7:10].cpu().numpy(),
                "root_ang_vel_world": root_state[0, 10:13].cpu().numpy(),
            },
            "action": {
                "body_action_29dof": sonic_targets.astype(np.float32).copy(),
                "full_action": full_action.detach().cpu().numpy().astype(np.float32),
                "body_effort_target": body_effort_target.astype(np.float32).copy(),
                "hand_action_left": self._left_hand_target.copy(),
                "hand_action_right": self._right_hand_target.copy(),
            },
            "env": {
                **self._collect_env_state(),
                "vision": self._collect_vision_state(),
            },
        }

    def _prune_stream_reference_timeline(self):
        """Keep a bounded streamed reference timeline around the playback cursor."""
        if not self._stream_ref_indices:
            return

        newest = self._stream_ref_indices[-1]
        oldest_keep = newest - self._stream_max_frames + 1
        current_global_frame = self._get_stream_global_playback_frame()
        if current_global_frame is not None:
            oldest_keep = min(
                oldest_keep,
                current_global_frame - self._stream_history_keep * self._stream_frame_step,
            )

        drop_until = bisect_left(self._stream_ref_indices, oldest_keep)
        if drop_until <= 0:
            return

        for frame_idx in self._stream_ref_indices[:drop_until]:
            self._stream_ref_frames.pop(frame_idx, None)
        del self._stream_ref_indices[:drop_until]

    def _get_stream_global_playback_frame(self) -> int | None:
        if self._stream_playback_frame_idx is not None:
            return int(self._stream_playback_frame_idx)
        if not self._stream_ref_indices:
            return None
        return int(self._stream_window_start + self._stream_frame_step * self._stream_current_frame)

    def _merge_stream_reference_data(
        self,
        frame_indices: np.ndarray,
        smpl_joints: np.ndarray | None,
        body_quat_w: np.ndarray | None,
        joint_pos: np.ndarray | None,
    ) -> None:
        """Merge incoming streamed frames into a sorted reference timeline."""
        frame_indices = np.asarray(frame_indices).reshape(-1)
        if frame_indices.size == 0:
            return

        incoming_frame_start = int(frame_indices[0])
        incoming_frame_end = int(frame_indices[-1])
        had_existing_window = bool(self._stream_ref_indices)
        old_window_start = self._stream_window_start
        old_window_end = self._stream_ref_indices[-1] if self._stream_ref_indices else incoming_frame_end

        if frame_indices.size >= 2:
            diffs = np.diff(frame_indices.astype(np.int64))
            positive_diffs = diffs[diffs > 0]
            if positive_diffs.size > 0:
                self._stream_frame_step = max(1, int(np.min(positive_diffs)))

        for i, frame_idx_raw in enumerate(frame_indices):
            frame_idx = int(frame_idx_raw)
            entry = self._stream_ref_frames.get(frame_idx, {})
            if smpl_joints is not None and i < smpl_joints.shape[0]:
                entry["smpl_joints"] = np.asarray(smpl_joints[i], dtype=np.float32).copy()
            if body_quat_w is not None and i < body_quat_w.shape[0]:
                entry["body_quat_w"] = np.asarray(body_quat_w[i], dtype=np.float32).copy()
            if joint_pos is not None and i < joint_pos.shape[0]:
                entry["joint_pos"] = np.asarray(joint_pos[i], dtype=np.float32).copy()
            self._stream_ref_frames[frame_idx] = entry
            sorted_insert_unique(self._stream_ref_indices, frame_idx)

        oldest = self._stream_ref_indices[0]
        newest = self._stream_ref_indices[-1]
        current_playback_frame = self._stream_current_frame
        frame_step = self._stream_frame_step

        # Approximate deploy's StreamedMotionMerger sliding-window semantics:
        # keep a small history before playback, preserve global playback position
        # across window shifts, and catch up when the stream jumps too far ahead.
        if self._stream_playback_frame_idx is None:
            self._stream_window_start = incoming_frame_start
            self._stream_current_frame = 0
        else:
            global_playback_frame = (
                self._stream_window_start
                + frame_step * max(0, current_playback_frame - self._stream_history_keep)
            )
            max_gap_frames = self._stream_max_gap_frames + self._stream_history_keep

            did_catchup = False
            if incoming_frame_start <= self._stream_window_start:
                did_catchup = True
            elif had_existing_window and incoming_frame_end <= old_window_end:
                did_catchup = True
            else:
                desired_window_start = global_playback_frame
                tentative_window_start = min(desired_window_start, incoming_frame_start)
                delta_to_incoming = incoming_frame_start - tentative_window_start
                tentative_merge_dst = delta_to_incoming // frame_step if frame_step > 0 else 0
                large_gap_from_old = incoming_frame_start > old_window_end + frame_step
                if tentative_merge_dst > max_gap_frames or large_gap_from_old:
                    did_catchup = True

            if did_catchup:
                self._stream_window_start = incoming_frame_start
                self._stream_current_frame = 0
            else:
                desired_window_start = global_playback_frame
                new_window_start = min(desired_window_start, incoming_frame_start)
                window_shift = (new_window_start - old_window_start) // frame_step if frame_step > 0 else 0
                adjusted_frame = current_playback_frame - window_shift
                self._stream_window_start = new_window_start
                if adjusted_frame < 0:
                    adjusted_frame = 0
                self._stream_current_frame = adjusted_frame

        max_cursor_global = newest - (_STEP1_FRAMES - 1) * frame_step
        if max_cursor_global < oldest:
            self._stream_window_start = oldest
            self._stream_current_frame = 0
            self._stream_playback_frame_idx = oldest
        else:
            current_global = self._stream_window_start + frame_step * self._stream_current_frame
            current_global = max(self._stream_window_start, min(current_global, max_cursor_global))
            self._stream_current_frame = max(0, (current_global - self._stream_window_start) // frame_step)
            self._stream_playback_frame_idx = current_global

        self._prune_stream_reference_timeline()

    def _stream_window_is_available(self) -> bool:
        """Whether streamed reference timeline has the fields needed by POSE encoder."""
        if self._stream_playback_frame_idx is None or not self._stream_ref_indices:
            return False
        required = ("smpl_joints", "body_quat_w", "joint_pos")
        for frame_idx in self._stream_ref_indices:
            entry = self._stream_ref_frames.get(frame_idx, {})
            if all(key in entry for key in required):
                return True
        return False

    def _resolve_stream_frame_idx(self, desired_frame_idx: int) -> int:
        """Resolve desired frame index to the closest available frame in the merged timeline."""
        if not self._stream_ref_indices:
            raise RuntimeError("stream reference timeline is empty")

        pos = bisect_right(self._stream_ref_indices, desired_frame_idx)
        if pos == 0:
            return self._stream_ref_indices[0]
        if pos >= len(self._stream_ref_indices):
            return self._stream_ref_indices[-1]
        prev_idx = self._stream_ref_indices[pos - 1]
        next_idx = self._stream_ref_indices[pos]
        if prev_idx == desired_frame_idx:
            return prev_idx
        if next_idx == desired_frame_idx:
            return next_idx
        return prev_idx

    def _gather_stream_reference_window(self, num_frames: int, step_size: int = 1):
        """Gather a current->future reference window from the merged streamed timeline."""
        if not self._stream_window_is_available():
            return None, None, None

        current_frame_idx = self._get_stream_global_playback_frame()
        assert current_frame_idx is not None
        smpl_window = np.zeros((num_frames, _N_SMPL_JOINTS, 3), dtype=np.float32)
        quat_window = np.zeros((num_frames, 4), dtype=np.float32)
        joint_window = np.zeros((num_frames, 29), dtype=np.float32)

        for i in range(num_frames):
            desired_frame_idx = current_frame_idx + i * step_size * self._stream_frame_step
            resolved_frame_idx = self._resolve_stream_frame_idx(desired_frame_idx)
            entry = self._stream_ref_frames[resolved_frame_idx]
            smpl_window[i] = entry["smpl_joints"]
            quat_window[i] = entry["body_quat_w"]
            joint_window[i] = entry["joint_pos"]

        return smpl_window, quat_window, joint_window

    def _advance_stream_playback_cursor(self):
        """Advance streamed-motion playback cursor by one control tick when future context exists."""
        if self._stream_playback_frame_idx is None or not self._stream_ref_indices:
            return

        oldest = self._stream_ref_indices[0]
        newest = self._stream_ref_indices[-1]
        max_cursor = newest - (_STEP1_FRAMES - 1) * self._stream_frame_step

        if max_cursor < oldest:
            self._stream_window_start = oldest
            self._stream_current_frame = 0
            self._stream_playback_frame_idx = oldest
            return

        next_cursor = self._stream_playback_frame_idx + self._stream_frame_step
        next_cursor = min(next_cursor, max_cursor)
        self._stream_playback_frame_idx = next_cursor
        if next_cursor >= self._stream_window_start:
            self._stream_current_frame = (next_cursor - self._stream_window_start) // max(1, self._stream_frame_step)

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
            joint_pos_sonic = joint_pos_sonic - self._sonic_default_np

            self._robot_joint_pos_hist = np.roll(self._robot_joint_pos_hist, -1, axis=0)
            self._robot_joint_pos_hist[-1] = joint_pos_sonic
            self._robot_joint_vel_hist = np.roll(self._robot_joint_vel_hist, -1, axis=0)
            self._robot_joint_vel_hist[-1] = joint_vel_sonic

        except Exception:
            pass

    def _bootstrap_pose_histories(
        self,
        smpl_joints_frame: np.ndarray | None,
        smpl_pose_frame: np.ndarray | None,
        anchor_rot6d_frame: np.ndarray | None,
        root_z_value: float | None,
        joint_pos_frame: np.ndarray | None,
        joint_vel_frame: np.ndarray | None,
    ) -> None:
        # Fill the startup history with the first valid teleop/reference frame so the
        # policy sees a static standing window instead of zeros or partially-filled data.
        if smpl_joints_frame is not None:
            self._smpl_joints_buf[:] = smpl_joints_frame.astype(np.float32)
            self._ref_smpl_joints_window[:] = smpl_joints_frame.astype(np.float32)
        if smpl_pose_frame is not None:
            self._smpl_pose_buf[:] = smpl_pose_frame.astype(np.float32)
        if anchor_rot6d_frame is not None:
            self._body_rot6d_buf[:] = anchor_rot6d_frame.astype(np.float32)
            self._motion_anchor_rot6d_hist[:] = anchor_rot6d_frame.astype(np.float32)
        if root_z_value is not None:
            self._motion_root_z_hist[:] = float(root_z_value)
        if joint_pos_frame is not None:
            joint_pos_frame = joint_pos_frame.astype(np.float32)
            self._motion_joint_pos_hist[:] = joint_pos_frame
            self._ref_joint_pos_window[:] = joint_pos_frame
        if joint_vel_frame is not None:
            joint_vel_frame = joint_vel_frame.astype(np.float32)
            self._motion_joint_vel_hist[:] = joint_vel_frame
        self._ref_window_valid = True
        self._smpl_history_fill = _STEP1_FRAMES

    # ------------------------------------------------------------------
    # Per-step: 读取 POSE（ZMQ 或 Redis）
    # ------------------------------------------------------------------

    def _fetch_zmq_pose(self):
        """从 ZMQ 读取最新 POSE 消息，更新 SMPL 历史缓冲。"""
        if self._zmq_poller is None:
            if self._sonic_debug and self._frame_count <= 3:
                print("[ZMQ] ERROR: _zmq_poller is None!")
            return
        raw = self._zmq_poller.get_data()
        if raw is None:
            if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % 50 == 0):
                print(f"[ZMQ] No data available (frame={self._frame_count})")
            return
        if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % 50 == 0):
            print(f"[ZMQ] Received raw data, size={len(raw)} bytes (frame={self._frame_count})")
        data = _parse_zmq_pose(raw)
        if data is None:
            print(f"[ZMQ] ERROR: Failed to parse ZMQ data (frame={self._frame_count})")
            return
        if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % 50 == 0):
            print(f"[ZMQ] Successfully parsed data, calling _apply_pose_data (frame={self._frame_count})")
        self._apply_pose_data(data, "zmq")

    def _fetch_redis_pose(self):
        """从 Redis 读取遥操 pose（human_smplx_data_unitree_g1_with_hands），转成与 ZMQ 同格式后更新缓冲。"""
        if self._redis_client is None:
            return
        try:
            (
                raw_smplx,
                raw_left,
                raw_right,
                controller_raw,
                recording_control_raw,
                ready_guard_raw,
            ) = self._redis_client.mget(
                [
                    "human_smplx_data_unitree_g1_with_hands",
                    "action_hand_left_unitree_g1_with_hands",
                    "action_hand_right_unitree_g1_with_hands",
                    "controller_data",
                    "recording_control_unitree_g1_with_hands",
                    self._input_ready_key,
                ]
            )
        except Exception as e:
            print(f"[REDIS] Failed to read from Redis: {e}")
            return
        self._latest_controller_data = None
        if controller_raw is not None:
            try:
                payload = controller_raw.decode("utf-8") if isinstance(controller_raw, bytes) else controller_raw
                self._latest_controller_data = json.loads(payload)
            except Exception:
                self._latest_controller_data = None
        self._update_input_ready_guard(ready_guard_raw)
        if self._is_stale_controller_payload(self._latest_controller_data):
            if self._input_ready_epoch_id != self._stale_input_drop_logged_epoch:
                controller_ts = "n/a"
                if isinstance(self._latest_controller_data, dict):
                    controller_ts = self._latest_controller_data.get("timestamp_realtime", "n/a")
                print(
                    f"[SonicActionProvider] Ignoring stale SONIC input: "
                    f"controller_ts={controller_ts}, "
                    f"ready_realtime={self._input_ready_timestamp_realtime:.6f}, "
                    f"ready_monotonic={self._input_ready_timestamp_monotonic:.6f}"
                )
                self._stale_input_drop_logged_epoch = self._input_ready_epoch_id
            self._latest_human_smplx_frame = None
            self._latest_recording_control = None
            self._recording_command = "none"
            return
        if raw_smplx is None:
            return
        try:
            frame = json.loads(raw_smplx.decode("utf-8") if isinstance(raw_smplx, bytes) else raw_smplx)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[REDIS] JSON decode failed: {e}")
            return
        if not isinstance(frame, dict):
            return
        try:
            from tools.sonic_pose_npz_replay_server import SMPL_JOINT_ORDER_24
        except ImportError as e:
            print(f"[REDIS] Failed to import SMPL_JOINT_ORDER_24: {e}")
            return

        missing = [name for name in SMPL_JOINT_ORDER_24 if name not in frame]
        if missing:
            return

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
        except Exception as e:
            print(f"[REDIS][SMPL_FRAME_BAD_FORMAT] {e}", flush=True)
            return

        self._latest_human_smplx_frame = frame
        self._latest_recording_control = None
        if recording_control_raw is not None:
            try:
                payload = (
                    recording_control_raw.decode("utf-8")
                    if isinstance(recording_control_raw, bytes)
                    else recording_control_raw
                )
                self._latest_recording_control = json.loads(payload)
                sequence = int(self._latest_recording_control.get("sequence", -1))
                command = str(self._latest_recording_control.get("command", "none"))
                self._recording_active = bool(self._latest_recording_control.get("active", False))
                if (
                    not self._waiting_for_reset_complete
                    and command != "none"
                    and sequence != self._latest_recording_control_sequence
                ):
                    self._latest_recording_control_sequence = sequence
                    self._command_edge_this_frame = command
                    self._recording_command = command
            except Exception:
                self._latest_recording_control = None

        left_hand = right_hand = None
        for raw_h, which in [(raw_left, "left"), (raw_right, "right")]:
            try:
                if raw_h is None:
                    continue
                sh = raw_h.decode("utf-8") if isinstance(raw_h, bytes) else raw_h
                arr = np.asarray(json.loads(sh), dtype=np.float32)
                if arr.size == 7:
                    if which == "left":
                        left_hand = arr
                    else:
                        right_hand = arr
            except Exception:
                pass

        try:
            body_poses_24x7 = np.zeros((24, 7), dtype=np.float32)
            for i, joint_name in enumerate(SMPL_JOINT_ORDER_24):
                joint_data = frame[joint_name]
                pos3 = joint_data[0]
                quat4 = joint_data[1]
                body_poses_24x7[i, 0:3] = pos3
                body_poses_24x7[i, 3:7] = quat4
        except Exception as e:
            print(f"[REDIS] Failed to convert frame dict to body_poses_24x7: {e}")
            return

        self._redis_frame_index += 1
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
            data = _pico_single_frame_from_body_poses(
                device=self.device,
                body_poses_24x7=body_poses_24x7,
                frame_index=self._redis_frame_index,
                left_hand=left_hand,
                right_hand=right_hand,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
            )
        except Exception as e:
            print(f"[REDIS] Failed to convert frame to pose fields: {e}")
            return
        self._latest_frame_index = int(np.asarray(data.get("frame_index", [self._redis_frame_index])).reshape(-1)[-1])
        if "timestamp_realtime" in data:
            self._latest_timestamp_realtime = float(np.asarray(data["timestamp_realtime"]).reshape(-1)[-1])
        if "timestamp_monotonic" in data:
            self._latest_timestamp_monotonic = float(np.asarray(data["timestamp_monotonic"]).reshape(-1)[-1])
        if "heading_increment" in data:
            self._latest_heading_increment = float(np.asarray(data["heading_increment"]).reshape(-1)[-1])
        self._apply_pose_data(data, "redis")

    def _apply_pose_data(self, data: dict, source: str = "zmq"):
        debug_log = self._sonic_debug and (
            self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0
        )
        # 在函数开头添加
        if "timestamp_realtime" in data:
            current_ts = float(data["timestamp_realtime"][0] if data["timestamp_realtime"].ndim > 0
                               else data["timestamp_realtime"])
            self._latest_timestamp_realtime = current_ts

            if not hasattr(self, "_smpl_ts_history"):
                self._smpl_ts_history = []

            self._smpl_ts_history.append(current_ts)
            if len(self._smpl_ts_history) > 10:
                self._smpl_ts_history.pop(0)
            # 每 25 帧打印一次时间间隔
            if debug_log and len(self._smpl_ts_history) >= 2:
                intervals = [
                    (self._smpl_ts_history[i] - self._smpl_ts_history[i - 1]) * 1000
                    for i in range(1, len(self._smpl_ts_history))
                ]
                print(f"[SMPL_INTERVAL_TEST] frame={self._frame_count} "
                      f"intervals_ms={[f'{x:.1f}' for x in intervals]} "
                      f"mean={np.mean(intervals):.1f}ms")
        if "timestamp_monotonic" in data:
            self._latest_timestamp_monotonic = float(
                data["timestamp_monotonic"][0]
                if np.asarray(data["timestamp_monotonic"]).ndim > 0
                else data["timestamp_monotonic"]
            )

        raw_frame_indices = None
        if "frame_index" in data:
            raw_frame_indices = np.asarray(data["frame_index"], dtype=np.int64).reshape(-1)
            if raw_frame_indices.size > 0:
                self._latest_frame_index = int(raw_frame_indices[-1])
            current_idx = int(data["frame_index"][0] if hasattr(data["frame_index"], '__len__')
                              else data["frame_index"])

            if not hasattr(self, "_last_frame_index"):
                self._last_frame_index = current_idx - 1

            expected_idx = self._last_frame_index + 1
            skip_count = current_idx - expected_idx

            if debug_log and skip_count > 0:
                print(f"[FRAME_SKIP_TEST] frame={self._frame_count} "
                      f"expected={expected_idx} got={current_idx} skipped={skip_count}")

            self._last_frame_index = current_idx



        """用解析后的 pose 字典更新 SMPL/机器人/手部缓冲。data 格式与 ZMQ v3 一致。"""
        tag = source.upper()
        # print(f"[{tag}] Received data keys: {list(data.keys())}")
        got_pose_frame = False
        raw_smpl_joints_window = None
        raw_body_quat_window = None
        raw_joint_pos_window = None
        root_z_value = None
        root_z_source = None
        latest_smpl_joints_frame = None
        latest_smpl_pose_frame = None
        latest_anchor_rot6d_frame = None
        latest_joint_pos_frame = None
        latest_joint_vel_frame = None

        # smpl_joints: (N, 24, 3) - 本地 provider 采用最新帧滚动历史
        if "smpl_joints" in data:
            sj = data["smpl_joints"].astype(np.float32)  # (N, 24, 3)
            raw_smpl_joints_window = sj if sj.ndim > 2 else sj[np.newaxis, ...]
            frame = sj[-1] if sj.ndim > 2 else sj
            if np.abs(frame).sum() > 0.01:
                self._smpl_data_valid = True
                if debug_log:
                    print(f"[{tag}] SMPL data marked as VALID")
                latest_smpl_joints_frame = frame.copy()
            self._smpl_joints_buf = np.roll(self._smpl_joints_buf, -1, axis=0)
            self._smpl_joints_buf[-1] = frame
            if sj.ndim > 2 and sj.shape[0] > 0:
                self._ref_smpl_joints_window[:] = build_future_window(sj, _STEP1_FRAMES)
                self._ref_window_valid = True
            got_pose_frame = True

        # smpl_pose: (N, 21, 3) - 本地 provider 采用最新帧滚动历史
        if "smpl_pose" in data:
            sp = data["smpl_pose"].astype(np.float32)    # (N, 21, 3)
            frame = sp[-1] if sp.ndim > 2 else sp
            latest_smpl_pose_frame = frame.copy()
            self._smpl_pose_buf = np.roll(self._smpl_pose_buf, -1, axis=0)
            self._smpl_pose_buf[-1] = frame

        # body_quat_w: (N, 4) → 转换为6D旋转表示
        # 本地 provider 采用最新参考帧滚动历史；直接覆盖整窗会破坏本地时间基准
        # motion_anchor_orientation 是机器人局部坐标系下的相对旋转
        # 公式: base_to_ref = base^(-1) * ref
        if "body_quat_w" in data:
            bq = data["body_quat_w"].astype(np.float32)  # (N, 4) wxyz
            raw_body_quat_window = bq if bq.ndim > 1 else bq[np.newaxis, ...]
            ref_quat_wxyz = quat_normalize_wxyz(bq[-1] if bq.ndim > 1 else bq)
            if debug_log:
                print(f"[{tag}] body_quat_w shape: {bq.shape}, latest: {ref_quat_wxyz}")
            got_pose_frame = True

            # 获取机器人当前朝向（从Isaac Lab）
            robot = self.env.scene["robot"].data
            base_quat_wxyz = robot.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)  # [w,x,y,z]

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
                # print(
                #     f"[{tag}][ANCHOR_INIT] "
                #     f"raw_init_angle_deg={raw_init_angle_deg:.2f} "
                #     f"aligned_init_angle_deg={aligned_init_angle_deg:.2f} "
                #     f"use_heading_align={self._anchor_use_heading_align}"
                # )

            aligned_ref_quat_wxyz = ref_quat_wxyz.copy()
            if self._anchor_use_heading_align:
                aligned_ref_quat_wxyz = quat_mul_wxyz(
                    self._anchor_heading_align_quat_wxyz,
                    ref_quat_wxyz,
                )
            rel_quat_wxyz = quat_mul_wxyz(
                quat_conjugate_wxyz(base_quat_wxyz),
                aligned_ref_quat_wxyz,
            )
            rot6d_latest = quat_to_rotation_6d(rel_quat_wxyz.reshape(1, 4))[0]

            if debug_log:
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

            if debug_log:
                print(f"[{tag}] converted to rot6d latest: {rot6d_latest}")
            latest_anchor_rot6d_frame = rot6d_latest.copy()
            self._body_rot6d_buf = np.roll(self._body_rot6d_buf, -1, axis=0)
            self._body_rot6d_buf[-1] = rot6d_latest
            self._motion_anchor_rot6d_hist = np.roll(self._motion_anchor_rot6d_hist, -1, axis=0)
            self._motion_anchor_rot6d_hist[-1] = rot6d_latest
            if bq.ndim > 1 and bq.shape[0] > 0:
                self._ref_body_quat_window[:] = build_future_window(bq, _STEP1_FRAMES)
                self._ref_window_valid = True

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
            if debug_log:
                print(
                    f"[{tag}][ROOT_Z] "
                    f"source={root_z_source} "
                    f"value={root_z_value:.4f}"
                )

        if got_pose_frame:
            self._smpl_history_fill = min(_STEP1_FRAMES, self._smpl_history_fill + 1)
            if debug_log:
                print(
                    f"[{tag}][HISTORY] "
                    f"smpl_history_fill={self._smpl_history_fill}/{_STEP1_FRAMES} "
                    f"smpl_valid={self._smpl_data_valid}"
                )

        # 机器人关节状态（来自 ZMQ，用于 obs 构建）
        if "joint_pos" in data:
            jp = data["joint_pos"].astype(np.float32)
            raw_joint_pos_window = jp if jp.ndim > 1 else jp[np.newaxis, ...]
            self._robot_joint_pos = jp[-1] if jp.ndim > 1 else jp
            latest_joint_pos_frame = self._robot_joint_pos.copy()
            self._motion_joint_pos_hist = np.roll(self._motion_joint_pos_hist, -1, axis=0)
            self._motion_joint_pos_hist[-1] = self._robot_joint_pos
            if jp.ndim > 1 and jp.shape[0] > 0:
                self._ref_joint_pos_window[:] = build_future_window(jp, _STEP1_FRAMES)
                self._ref_window_valid = True
            if debug_log:
                wrist_ref = self._motion_joint_pos_hist[-1, OFFICIAL_WRIST_INDICES]
                print(
                    f"[{tag}][REF_JOINT_POS] "
                    f"range={array_range_str(self._robot_joint_pos)} "
                    f"wrist_range={array_range_str(wrist_ref)} "
                    f"heading_init={'YES' if self._anchor_heading_initialized else 'NO'}"
                )
        if "joint_vel" in data:
            jv = data["joint_vel"].astype(np.float32)
            self._robot_joint_vel = jv[-1] if jv.ndim > 1 else jv
            latest_joint_vel_frame = self._robot_joint_vel.copy()
            self._motion_joint_vel_hist = np.roll(self._motion_joint_vel_hist, -1, axis=0)
            self._motion_joint_vel_hist[-1] = self._robot_joint_vel
            if debug_log:
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
            if debug_log:
                print(
                    f"[{tag}][VR_3PT_POS] "
                    f"range={array_range_str(self._vr_3pt_position)}"
                )
        if "vr_orientation" in data:
            vr_orn = data["vr_orientation"].astype(np.float32)  # (12,)
            self._vr_3pt_orientation = vr_orn.flatten()
            if debug_log:
                print(
                    f"[{tag}][VR_3PT_ORN] "
                    f"range={array_range_str(self._vr_3pt_orientation)}"
                )
        if "heading_increment" in data:
            self._latest_heading_increment = float(np.asarray(data["heading_increment"]).reshape(-1)[-1])

        if (
            raw_frame_indices is not None
            and raw_smpl_joints_window is not None
            and raw_body_quat_window is not None
            and raw_joint_pos_window is not None
        ):
            self._merge_stream_reference_data(
                frame_indices=raw_frame_indices,
                smpl_joints=raw_smpl_joints_window,
                body_quat_w=raw_body_quat_window,
                joint_pos=raw_joint_pos_window,
            )
            if debug_log and self._stream_playback_frame_idx is not None and self._stream_ref_indices:
                print(
                    f"[{tag}][STREAM_REF] "
                    f"cursor={self._stream_playback_frame_idx} "
                    f"oldest={self._stream_ref_indices[0]} "
                    f"newest={self._stream_ref_indices[-1]} "
                    f"count={len(self._stream_ref_indices)} "
                    f"step={self._stream_frame_step}"
                )

        if (
            got_pose_frame
            and self._smpl_data_valid
            and self._smpl_history_fill == 0
            and latest_smpl_joints_frame is not None
        ):
            self._bootstrap_pose_histories(
                smpl_joints_frame=latest_smpl_joints_frame,
                smpl_pose_frame=latest_smpl_pose_frame,
                anchor_rot6d_frame=latest_anchor_rot6d_frame,
                root_z_value=root_z_value,
                joint_pos_frame=latest_joint_pos_frame,
                joint_vel_frame=latest_joint_vel_frame,
            )
            if debug_log:
                print(f"[{tag}][HISTORY_BOOTSTRAP] filled startup pose history with first valid frame")

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
            joint_pos_sonic = joint_pos_sonic - self._sonic_default_np

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

            # ============================================================================
            # SMPL Mode (mode_id=2) Encoder Input Construction
            # ============================================================================
            # According to observation_config.yaml, SMPL mode only requires 4 observation blocks:
            # 1. encoder_mode_4
            # 2. smpl_joints_10frame_step1
            # 3. smpl_anchor_orientation_10frame_step1
            # 4. motion_joint_positions_wrists_10frame_step1
            #
            # All other observations must be ZERO to match C++ implementation.
            # Reference: gear_sonic_deploy/policy/release/observation_config.yaml:74-80
            # Reference: g1_deploy_onnx_ref.cpp:1920-1942 (mode filtering logic)
            # ============================================================================

             # These observations are NOT required in SMPL mode → set to ZERO
            motion_joint_pos_step5_full = np.zeros(290, dtype=np.float32)
            motion_joint_vel_step5_full = np.zeros(290, dtype=np.float32)
            motion_root_z_step5 = np.zeros(10, dtype=np.float32)
            motion_root_z = np.zeros(1, dtype=np.float32)
            motion_anchor_orient = np.zeros(6, dtype=np.float32)
            motion_anchor_orient_step5_full = np.zeros(60, dtype=np.float32)
            motion_joint_pos_lowerbody_full = np.zeros(120, dtype=np.float32)
            motion_joint_vel_lowerbody_full = np.zeros(120, dtype=np.float32)
            vr_3pt_pos = np.zeros(9, dtype=np.float32)
            vr_3pt_orn = np.zeros(12, dtype=np.float32)

            wrist_indices = OFFICIAL_WRIST_INDICES
            stream_window = self._gather_stream_reference_window(_STEP1_FRAMES, 1)
            if stream_window is not None:
                smpl_joint_window, ref_quat_window, full_joint_window = stream_window
                wrist_window = full_joint_window[:, wrist_indices]

                base_quat_wxyz = robot.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
                anchor_window = np.zeros((_STEP1_FRAMES, 6), dtype=np.float32)
                for i in range(_STEP1_FRAMES):
                    ref_quat_wxyz = quat_normalize_wxyz(ref_quat_window[i])
                    aligned_ref_quat_wxyz = ref_quat_wxyz.copy()
                    if self._anchor_use_heading_align:
                        aligned_ref_quat_wxyz = quat_mul_wxyz(
                            self._anchor_heading_align_quat_wxyz,
                            ref_quat_wxyz,
                        )
                    rel_quat_wxyz = quat_mul_wxyz(
                        quat_conjugate_wxyz(base_quat_wxyz),
                        aligned_ref_quat_wxyz,
                    )
                    anchor_window[i] = quat_to_rotation_6d(rel_quat_wxyz.reshape(1, 4))[0]
            elif self._ref_window_valid:
                smpl_joint_window = self._ref_smpl_joints_window
                ref_quat_window = self._ref_body_quat_window
                wrist_window = self._ref_joint_pos_window[:, wrist_indices]

                base_quat_wxyz = robot.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
                anchor_window = np.zeros((_STEP1_FRAMES, 6), dtype=np.float32)
                for i in range(_STEP1_FRAMES):
                    ref_quat_wxyz = quat_normalize_wxyz(ref_quat_window[i])
                    aligned_ref_quat_wxyz = ref_quat_wxyz.copy()
                    if self._anchor_use_heading_align:
                        aligned_ref_quat_wxyz = quat_mul_wxyz(
                            self._anchor_heading_align_quat_wxyz,
                            ref_quat_wxyz,
                        )
                    rel_quat_wxyz = quat_mul_wxyz(
                        quat_conjugate_wxyz(base_quat_wxyz),
                        aligned_ref_quat_wxyz,
                    )
                    anchor_window[i] = quat_to_rotation_6d(rel_quat_wxyz.reshape(1, 4))[0]
            else:
                smpl_joint_window = self._smpl_joints_buf
                anchor_window = self._body_rot6d_buf
                motion_wrist_window = gather_temporal_window(self._motion_joint_pos_hist, _STEP1_FRAMES, 1)
                wrist_window = motion_wrist_window[:, wrist_indices]

            smpl_joints_flat = smpl_joint_window.reshape(-1)  # (10, 24, 3) → (720,)
            smpl_anchor_orient_flat = anchor_window.reshape(-1)  # (10, 6) → (60,)
            motion_wrist_pos = wrist_window.reshape(-1)

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
            self._latest_encoder_input = encoder_input[0].astype(np.float32, copy=True)
            self._latest_smpl_joint_window = smpl_joint_window.astype(np.float32, copy=True)
            self._latest_anchor_window = anchor_window.astype(np.float32, copy=True)
            self._latest_wrist_window = wrist_window.astype(np.float32, copy=True)

            if do_log:
                print(f"[SONIC] Encoder input shape: {encoder_input.shape}, expected: (1, 1762)")
                print(f"[SONIC] Encoder input dtype: {encoder_input.dtype}")
                print(f"[SONIC] Encoder input range: [{encoder_input.min():.4f}, {encoder_input.max():.4f}]")
                print(f"[SONIC] SMPL joints sum: {np.abs(smpl_joints_flat).sum():.4f}")

            if do_log or (self._sonic_debug and np.max(np.abs(encoder_input)) > 8.0):
                print(
                    "[SONIC][SMPL_MODE] "
                    f"encoder_mode_vec={encoder_mode.tolist()} "
                    f"active={SMPL_MODE_ACTIVE_BLOCKS}"
                )
                print(
                    "[SONIC][SMPL_MODE_ACTIVE_BLOCKS] "
                    f"smpl_joints={array_range_str(smpl_joints_flat)} "
                    f"smpl_anchor={array_range_str(smpl_anchor_orient_flat)} "
                    f"wrist_pos={array_range_str(motion_wrist_pos)}"
                )
                print(
                    "[SONIC][SMPL_MODE_ZEROED_BLOCKS] "
                    f"All zeroed blocks (918 dims) are correctly set to 0.0 "
                    f"to match C++ implementation"
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
            ang_vel = robot.root_ang_vel_b[0].cpu().numpy()    # (3,)
            base_quat_wxyz = robot.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
            proj_grav = gravity_dir_from_base_quat_wxyz(base_quat_wxyz)

            # 更新 decoder 历史缓冲区
            self._ang_vel_hist   = np.roll(self._ang_vel_hist,   -1, axis=0)
            self._ang_vel_hist[-1] = ang_vel
            self._grav_dir_hist  = np.roll(self._grav_dir_hist,  -1, axis=0)
            self._grav_dir_hist[-1] = proj_grav
            # NOTE: last_action_hist will be updated AFTER decoder inference with raw action

            # 构建 994 维 decoder 输入
            dec_obs = np.concatenate([
                latent.flatten(),                          # token_state: 64
                self._ang_vel_hist.flatten(),              # his_base_angular_velocity
                self._robot_joint_pos_hist.flatten(),      # his_body_joint_positions
                self._robot_joint_vel_hist.flatten(),      # his_body_joint_velocities
                self._last_action_hist.flatten(),          # his_last_actions
                self._grav_dir_hist.flatten(),             # his_gravity_dir
            ])[np.newaxis].astype(np.float32)  # (1, 994)
            self._latest_decoder_obs = dec_obs[0].astype(np.float32, copy=True)

            if do_log:
                print(f"[SONIC] Decoder input shape: {dec_obs.shape}, expected: (1, 994)")
            dec_inputs = {self._decoder.get_inputs()[0].name: dec_obs}
            t_dec0 = time.perf_counter()
            action_sonic = self._decoder.run(None, dec_inputs)[0]
            t_dec1 = time.perf_counter()
            raw_sonic_unclipped = action_sonic.flatten()[:29].astype(np.float32, copy=False)
            if do_log:
                print(f"[SONIC] ✓ Decoder output shape: {action_sonic.shape}")
                print(f"[SONIC] Raw sonic range (before clip): [{raw_sonic_unclipped.min():.4f}, {raw_sonic_unclipped.max():.4f}]")

            # Match deploy semantics: do not clip the raw policy output locally.
            raw_sonic = raw_sonic_unclipped
            self._latest_decoder_raw_action = raw_sonic_unclipped.astype(np.float32, copy=True)
            if do_log:
                print(f"[SONIC] Raw sonic range (after clip): [{raw_sonic.min():.4f}, {raw_sonic.max():.4f}]")

            # deploy 记录的是 policy 原始输出，不做本地 clip；同时它出现在下一拍历史里。
            # 本地这里保持相同相位，但写入 unclipped raw action 以匹配 C++。
            self._last_action_hist = np.roll(self._last_action_hist, -1, axis=0)
            self._last_action_hist[-1] = raw_sonic_unclipped

            # 后处理：per-joint action_scale + default（SONIC IsaacLab order）
            # 参考 GR00T g1_deploy_onnx_ref.cpp:2824
            # target = action * action_scale + default_angle
            target_sonic = raw_sonic * G1_ACTION_SCALE_ISAACLAB + self._sonic_default_np
            self._latest_decoder_target = target_sonic.astype(np.float32, copy=True)
            if do_log:
                print(f"[SONIC] ✓ Final target range (before safety clip): [{target_sonic.min():.4f}, {target_sonic.max():.4f}]")

            if do_log:
                print(f"[SONIC] ✓ Final target range (after safety clip): [{target_sonic.min():.4f}, {target_sonic.max():.4f}]")

            if do_log:
                q_hist_latest = self._robot_joint_pos_hist[-1]
                dq_hist_latest = self._robot_joint_vel_hist[-1]
                last_action_latest = self._last_action_hist[-1]
                wrist_window_latest = wrist_window[-1]
                wrist_window_prev = wrist_window[-2] if wrist_window.shape[0] >= 2 else wrist_window[-1]
                wrist_delta = wrist_window_latest - wrist_window_prev
                smpl_joint_latest = smpl_joint_window[-1]
                smpl_joint_prev = smpl_joint_window[-2] if smpl_joint_window.shape[0] >= 2 else smpl_joint_window[-1]
                smpl_delta = smpl_joint_latest - smpl_joint_prev

                ref_is_static = (
                    np.max(np.abs(wrist_delta)) < 0.01
                    and np.max(np.abs(smpl_delta)) < 0.01
                    and np.max(np.abs(ang_vel)) < 0.3
                )

                print(
                    "[SONIC][STAND_DIAG] "
                    f"ref_is_static={ref_is_static} "
                    f"ang_vel={np.array2string(ang_vel, precision=4, separator=', ')} "
                    f"grav={np.array2string(proj_grav, precision=4, separator=', ')}"
                )
                print(
                    "[SONIC][STAND_DIAG_RANGES] "
                    f"q_def={array_range_str(q_hist_latest)} "
                    f"dq={array_range_str(dq_hist_latest)} "
                    f"last_action={array_range_str(last_action_latest)} "
                    f"raw={array_range_str(raw_sonic_unclipped)} "
                    f"target_delta={array_range_str(target_sonic - self._sonic_default_np)} "
                    f"wrist_delta={array_range_str(wrist_delta)} "
                    f"smpl_delta={array_range_str(smpl_delta)}"
                )
                print(
                    "[SONIC][STAND_DIAG_TOPK] "
                    f"raw={topk_joint_abs_str(raw_sonic_unclipped, SONIC_ISAACLAB_JOINT_ORDER, k=8)} "
                    f"target_delta={topk_joint_abs_str(target_sonic - self._sonic_default_np, SONIC_ISAACLAB_JOINT_ORDER, k=8)} "
                    f"q_def={topk_joint_abs_str(q_hist_latest, SONIC_ISAACLAB_JOINT_ORDER, k=6)} "
                    f"dq={topk_joint_abs_str(dq_hist_latest, SONIC_ISAACLAB_JOINT_ORDER, k=6)} "
                    f"last_action={topk_joint_abs_str(last_action_latest, SONIC_ISAACLAB_JOINT_ORDER, k=6)}"
                )

            # Store encoder/decoder timing in performance buffers
            enc_ms = (t_enc1 - t_enc0) * 1000.0
            dec_ms = (t_dec1 - t_dec0) * 1000.0
            self._perf_encoder_ms.append(enc_ms)
            self._perf_decoder_ms.append(dec_ms)
            if len(self._perf_encoder_ms) > self._perf_buffer_size:
                self._perf_encoder_ms.pop(0)
            if len(self._perf_decoder_ms) > self._perf_buffer_size:
                self._perf_decoder_ms.pop(0)

            if self._sonic_debug and (self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0):
                t1 = time.perf_counter()
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
            self._command_edge_this_frame = "none"
            debug_log = self._sonic_debug and (
                self._frame_count <= 3 or self._frame_count % self._sonic_log_every == 0
            )
            if self._waiting_for_reset_complete:
                if self._check_reset_complete():
                    print("[SONIC] reset complete received")
                    self._waiting_for_reset_complete = False
                    self._reset_complete_received = True
                    self.on_env_reset()
                    self._episode_id += 1
                    if not self._replay_enabled:
                        # Align post-reset episode boundaries with startup recording:
                        # start the new segment immediately after reset, before any
                        # fresh live pose frames can advance the provider state.
                        self._begin_episode_recording()
                    else:
                        self._recording_active = False
                        self._recording_display_state = "idle"
                        self._recording_display_counter = 0
                else:
                    return self._default_pos.clone().squeeze(0)

            # RTF monitoring initialization
            if self._enable_rtf_monitor and self._rtf_wall_time_start is None:
                self._rtf_wall_time_start = time.perf_counter()
                self._rtf_sim_time_start = env.sim.current_time

            # 1. 读取 POSE（ZMQ 或 Redis）
            t_step0 = time.perf_counter()
            t_fetch0 = time.perf_counter()
            if self._replay_enabled:
                replay_frame_idx = self._next_replay_frame_idx()
                if replay_frame_idx is None:
                    return self._default_pos.clone().squeeze(0)
                replay_direct_targets = self._prepare_replay_frame(replay_frame_idx)
                if debug_log:
                    print(f"replay pose frame={replay_frame_idx} mode={self._replay_mode}")
            elif self._pose_source == "redis":
                self._fetch_redis_pose()
                if debug_log:
                    print("redis pose")
            else:
                self._fetch_zmq_pose()
                if debug_log:
                    print("zmq pose")
            t_fetch1 = time.perf_counter()
            fetch_ms = (t_fetch1 - t_fetch0) * 1000.0
            self._perf_fetch_pose_ms.append(fetch_ms)
            if len(self._perf_fetch_pose_ms) > self._perf_buffer_size:
                self._perf_fetch_pose_ms.pop(0)

            if self._replay_enabled and self._replay_mode == "direct_replay":
                if replay_direct_targets is None:
                    return self._default_pos.clone().squeeze(0)
                sonic_targets = replay_direct_targets
            elif self._replay_enabled and self._replay_mode == "inference_replay":
                sonic_targets = self._run_gear_sonic_replay_inference(replay_frame_idx)
            else:
                sonic_targets = self._run_gear_sonic()

            if debug_log:
                sonic_targets_str = np.array2string(
                    sonic_targets,
                    precision=6,
                    separator=", ",
                    suppress_small=False,
                    max_line_width=2000
                )
                # print("\n" + "=" * 120)
                # print(
                #     f"[SONIC_29_QPOS] frame={self._frame_count}  "
                #     f"history_ready={history_ready}"
                # )
                # print(f"[SONIC_29_QPOS_VALUES] {sonic_targets_str}")
                # print(f"[SONIC_29_QPOS_RANGE] min={sonic_targets.min():.6f}, max={sonic_targets.max():.6f}")
                # print("=" * 120)

            # 3. 构建完整 Isaac 动作
            full_action = self._default_pos.clone().squeeze(0)
            sonic_t = torch.tensor(sonic_targets, dtype=torch.float32, device=self.device)
            full_action.index_copy_(0, self._sonic_idx, sonic_t)

            use_body_effort = self._use_effort_control
            body_effort_preview = (
                self._compute_sonic_effort(sonic_targets).detach().cpu().numpy().astype(np.float32)
                if use_body_effort
                else np.zeros((29,), dtype=np.float32)
            )

            # 4. 手部关节
            self._apply_hand_targets(full_action)
            if use_body_effort:
                self._ensure_effort_mode_runtime_config(env)
            else:
                self._ensure_position_mode_runtime_config(env)

            if not self._replay_enabled and self._recording_command != "none":
                self._handle_recording_command()
                if self._waiting_for_reset_complete:
                    return self._default_pos.clone().squeeze(0)
            self._latest_decoder_body_effort = body_effort_preview.copy()
            if not self._replay_enabled and self.recording_manager.is_recording:
                self.recording_manager.add_frame(
                    self._collect_recording_data(
                        full_action=full_action,
                        sonic_targets=sonic_targets,
                        body_effort_target=body_effort_preview,
                    )
                )
                if self._reset_complete_received:
                    self._reset_complete_received = False
            if not self._replay_enabled:
                self._update_recording_display_state()

            # 5. 步进仿真（decimation）
            t_sim0 = time.perf_counter()
            for _ in range(self._decimation):
                if self.enable_dex3:
                    if hasattr(self, "_left_hand_idx") and self._left_hand_idx.numel() > 0:
                        env.scene["robot"].set_joint_position_target(
                            full_action[self._left_hand_idx], joint_ids=self._left_hand_idx
                        )
                    if hasattr(self, "_right_hand_idx") and self._right_hand_idx.numel() > 0:
                        env.scene["robot"].set_joint_position_target(
                            full_action[self._right_hand_idx], joint_ids=self._right_hand_idx
                        )
                if use_body_effort:
                    # Recompute PD torque at every physics step to match MuJoCo's low-level
                    # motor loop semantics instead of holding one constant torque over decimation.
                    body_effort = self._compute_sonic_effort(sonic_targets)
                    env.scene["robot"].set_joint_effort_target(body_effort, joint_ids=self._sonic_idx)
                else:
                    env.scene["robot"].set_joint_position_target(full_action)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)
            t_sim1 = time.perf_counter()
            sim_ms = (t_sim1 - t_sim0) * 1000.0
            self._perf_sim_step_ms.append(sim_ms)
            if len(self._perf_sim_step_ms) > self._perf_buffer_size:
                self._perf_sim_step_ms.pop(0)

            # Joint tracking comparison (for PD coefficient tuning)
            # Store targets for next frame comparison
            self._tracking_target_buffer = sonic_targets.copy()

            current_pos = env.scene["robot"].data.joint_pos[0, self._sonic_idx].cpu().numpy()
            current_vel = env.scene["robot"].data.joint_vel[0, self._sonic_idx].cpu().numpy()
            pos_error = current_pos - sonic_targets
            if self._replay_enabled and replay_frame_idx is not None:
                self._log_replay_joint_position_error(current_pos, replay_frame_idx)
                self._log_replay_object_state_error(replay_frame_idx)
            support_joint_indices = [0, 1, 9, 10, 13, 14, 17, 18]

            if debug_log:
                print(
                    "[SONIC][TRACKING_TOPK] "
                    f"error={topk_joint_abs_str(pos_error, SONIC_ISAACLAB_JOINT_ORDER, k=8)} "
                    f"target={topk_joint_abs_str(sonic_targets, SONIC_ISAACLAB_JOINT_ORDER, k=8)} "
                    f"actual={topk_joint_abs_str(current_pos, SONIC_ISAACLAB_JOINT_ORDER, k=8)} "
                    f"dq={topk_joint_abs_str(current_vel, SONIC_ISAACLAB_JOINT_ORDER, k=8)}"
                )
                print(
                    "[SONIC][TRACKING_SUPPORT] "
                    f"target={joint_slice_str(sonic_targets, SONIC_ISAACLAB_JOINT_ORDER, support_joint_indices)} "
                    f"actual={joint_slice_str(current_pos, SONIC_ISAACLAB_JOINT_ORDER, support_joint_indices)} "
                    f"error={joint_slice_str(pos_error, SONIC_ISAACLAB_JOINT_ORDER, support_joint_indices)} "
                    f"dq={joint_slice_str(current_vel, SONIC_ISAACLAB_JOINT_ORDER, support_joint_indices)}"
                )

            if self._frame_count % self._tracking_log_interval == 0:
                pos_error_abs = np.abs(pos_error)

                # print(f"\n[SONIC_TRACKING] Frame {self._frame_count} - Joint Position Tracking")
                # print("=" * 80)
                # print(f"{'Joint':<25} {'Target':>10} {'Current':>10} {'Error':>10} {'|Error|':>10}")
                # print("-" * 80)

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

                # for joint_name, idx in key_joints.items():
                #     print(f"{joint_name:<25} {sonic_targets[idx]:>10.4f} {current_pos[idx]:>10.4f} "
                #           f"{pos_error[idx]:>10.4f} {pos_error_abs[idx]:>10.4f}")

                # print("-" * 80)
                # print(f"Max absolute error: {pos_error_abs.max():.4f} rad ({np.degrees(pos_error_abs.max()):.2f}°)")
                # print(f"Mean absolute error: {pos_error_abs.mean():.4f} rad ({np.degrees(pos_error_abs.mean()):.2f}°)")
                # print(f"RMS error: {np.sqrt(np.mean(pos_error**2)):.4f} rad ({np.degrees(np.sqrt(np.mean(pos_error**2))):.2f}°)")
                # print("=" * 80)

            self._advance_stream_playback_cursor()

            t_render0 = time.perf_counter()
            t0 = time.perf_counter()
            env.sim.render()
            t1 = time.perf_counter()
            env.observation_manager.compute()
            t2 = time.perf_counter()
            if self._sonic_debug or self._frame_count % self._perf_report_interval == 0:
                print(f"render: {(t1 - t0) * 1000:.3f} ms")
                print(f"obs: {(t2 - t1) * 1000:.3f} ms")

            t_render1 = time.perf_counter()
            render_ms = (t_render1 - t_render0) * 1000.0
            self._perf_render_ms.append(render_ms)
            if len(self._perf_render_ms) > self._perf_buffer_size:
                self._perf_render_ms.pop(0)

            # Total step time
            t_step1 = time.perf_counter()
            total_ms = (t_step1 - t_step0) * 1000.0
            self._perf_total_ms.append(total_ms)
            if len(self._perf_total_ms) > self._perf_buffer_size:
                self._perf_total_ms.pop(0)

            # Performance report
            if self._frame_count % self._perf_report_interval == 0 and len(self._perf_total_ms) > 0:
                print("\n" + "=" * 80)
                print(f"[PERF_REPORT] Frame {self._frame_count} - RTF Performance Breakdown")
                print("=" * 80)
                print(f"{'Component':<30} {'Mean(ms)':>10} {'Max(ms)':>10} {'%Total':>10}")
                print("-" * 80)

                mean_total = np.mean(self._perf_total_ms)
                mean_fetch = np.mean(self._perf_fetch_pose_ms) if self._perf_fetch_pose_ms else 0
                mean_enc = np.mean(self._perf_encoder_ms) if self._perf_encoder_ms else 0
                mean_dec = np.mean(self._perf_decoder_ms) if self._perf_decoder_ms else 0
                mean_sim = np.mean(self._perf_sim_step_ms) if self._perf_sim_step_ms else 0
                mean_render = np.mean(self._perf_render_ms) if self._perf_render_ms else 0

                max_fetch = np.max(self._perf_fetch_pose_ms) if self._perf_fetch_pose_ms else 0
                max_enc = np.max(self._perf_encoder_ms) if self._perf_encoder_ms else 0
                max_dec = np.max(self._perf_decoder_ms) if self._perf_decoder_ms else 0
                max_sim = np.max(self._perf_sim_step_ms) if self._perf_sim_step_ms else 0
                max_render = np.max(self._perf_render_ms) if self._perf_render_ms else 0

                print(f"{'Fetch Pose (ZMQ/Redis)':<30} {mean_fetch:>10.2f} {max_fetch:>10.2f} {mean_fetch/mean_total*100:>9.1f}%")
                print(f"{'SONIC Encoder':<30} {mean_enc:>10.2f} {max_enc:>10.2f} {mean_enc/mean_total*100:>9.1f}%")
                print(f"{'SONIC Decoder':<30} {mean_dec:>10.2f} {max_dec:>10.2f} {mean_dec/mean_total*100:>9.1f}%")
                print(f"{'Sim Step (x{self._decimation})':<30} {mean_sim:>10.2f} {max_sim:>10.2f} {mean_sim/mean_total*100:>9.1f}%")
                print(f"{'Render + Obs':<30} {mean_render:>10.2f} {max_render:>10.2f} {mean_render/mean_total*100:>9.1f}%")
                print("-" * 80)
                print(f"{'TOTAL':<30} {mean_total:>10.2f} {np.max(self._perf_total_ms):>10.2f} {'100.0':>9}%")
                print(f"{'Target FPS (RTF=1.0)':<30} {1000.0/(env.physics_dt*self._decimation):>10.1f}")
                print(f"{'Actual FPS':<30} {1000.0/mean_total:>10.1f}")
                print(f"{'RTF Estimate':<30} {(env.physics_dt*self._decimation)/(mean_total/1000.0):>10.2f}")
                print("=" * 80 + "\n")

            # RTF monitoring
            if self._enable_rtf_monitor and self._frame_count % self._rtf_log_interval == 0:
                wall_time_now = time.perf_counter()
                sim_time_now = env.sim.current_time
                wall_elapsed = wall_time_now - self._rtf_wall_time_start
                sim_elapsed = sim_time_now - self._rtf_sim_time_start
                rtf = sim_elapsed / wall_elapsed if wall_elapsed > 0 else 0.0
                print(f"[RTF] Frame {self._frame_count}: RTF={rtf:.3f} "
                      f"(sim={sim_elapsed:.2f}s, wall={wall_elapsed:.2f}s, fps={self._frame_count/wall_elapsed:.1f})")

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
        try:
            self.recording_manager.shutdown()
        except Exception:
            pass
        if self._zmq_poller is not None:
            try:
                self._zmq_poller.close()
            except Exception:
                pass
        if getattr(self, "_redis_client", None) is not None:
            try:
                self._redis_client.close()
            except Exception:
                pass
        if getattr(self, "_redis_control_client", None) is not None:
            try:
                self._redis_control_client.close()
            except Exception:
                pass
