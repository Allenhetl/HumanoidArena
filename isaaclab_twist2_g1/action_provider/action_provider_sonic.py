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

from action_provider.action_base import ActionProvider

# ---------------------------------------------------------------------------
# Resolve gear_sonic package path
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TWIST2_ROOT = os.path.dirname(_THIS_DIR)
_GROOT_ROOT = os.path.join(os.path.dirname(_TWIST2_ROOT), "GR00T-WholeBodyControl")
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

try:
    from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
    _HAS_ZMQ_POLLER = True
except ImportError:
    _HAS_ZMQ_POLLER = False
    print("[SonicActionProvider] WARNING: gear_sonic ZMQPoller not found.")

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

        self.enable_dex3    = getattr(args_cli, "enable_dex3_dds",   False)
        self.enable_gripper = getattr(args_cli, "enable_dex1_dds",   False)
        self.zmq_host       = getattr(args_cli, "sonic_zmq_host",    "localhost")
        self.zmq_port       = getattr(args_cli, "sonic_zmq_port",    5556)
        self.encoder_path   = getattr(args_cli, "sonic_encoder_path", "")
        self.decoder_path   = getattr(args_cli, "sonic_decoder_path", "")
        cfg = getattr(env, "cfg", None)
        self._decimation    = int(getattr(cfg, "decimation", 4))

        self._setup_joint_mapping()
        self._setup_zmq()
        self._setup_policy()
        self._setup_buffers()
        self._setup_hand_dds(args_cli)

        print(f"[SonicActionProvider] POSE mode ready  "
              f"zmq={self.zmq_host}:{self.zmq_port}  "
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
            return
        try:
            self._zmq_poller = ZMQPoller(
                host=self.zmq_host, port=self.zmq_port, topic="pose")
            print(f"[SonicActionProvider] ZMQ connected "
                  f"tcp://{self.zmq_host}:{self.zmq_port} topic=pose")
        except Exception as e:
            print(f"[SonicActionProvider] ZMQ init failed: {e}")

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
        providers = []
        if str(self.device).startswith("cuda"):
            avail = ort.get_available_providers()
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
            (10, _N_SMPL_JOINTS, 3), dtype=np.float32)   # (10, 24, 3)
        self._smpl_pose_buf   = np.zeros(
            (10, _N_SMPL_POSES,  3), dtype=np.float32)   # (10, 21, 3)
        self._body_rot6d_buf  = np.tile(
            np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32), (10, 1))  # (10, 6)

        # 机器人状态历史缓冲（SONIC IsaacLab order）
        # 用于 encoder 输入（step5 采样）和 decoder 输入（step1 连续10帧）
        self._robot_joint_pos_hist = np.tile(
            self._sonic_default_np[np.newaxis], (10, 1))  # (10, 29)
        self._robot_joint_vel_hist = np.zeros((10, 29), dtype=np.float32)

        # Decoder 需要的额外历史缓冲
        # his_base_angular_velocity_10frame_step1: (10, 3)
        self._ang_vel_hist    = np.zeros((10, 3),  dtype=np.float32)
        # his_gravity_dir_10frame_step1: (10, 3)
        self._grav_dir_hist   = np.zeros((10, 3),  dtype=np.float32)
        # his_last_actions_10frame_step1: (10, 29)
        self._last_action_hist = np.tile(
            self._sonic_default_np[np.newaxis], (10, 1))  # (10, 29)

        # 手部关节目标
        self._left_hand_target  = np.zeros(7, dtype=np.float32)
        self._right_hand_target = np.zeros(7, dtype=np.float32)

        # encoder 输出的 latent（首次推理前为零）
        self._latent = None

        # SMPL数据有效性标志（用于检测是否接收到有效的遥操数据）
        self._smpl_data_valid = False

    def _setup_hand_dds(self, args_cli):
        self._dex3_dds = None
        try:
            from dds.dds_master import dds_manager
            if self.enable_dex3:
                self._dex3_dds = dds_manager.get_object("dex3")
        except Exception as e:
            print(f"[SonicActionProvider] hand DDS init skipped: {e}")

    # ------------------------------------------------------------------
    # Per-step: 读取 ZMQ POSE 消息
    # ------------------------------------------------------------------

    def _fetch_zmq_pose(self):
        """从 ZMQ 读取最新 POSE 消息，更新 SMPL 历史缓冲。"""
        if self._zmq_poller is None:
            return
        raw = self._zmq_poller.get_data()
        if raw is None:
            return
        data = _parse_zmq_pose(raw)
        if data is None:
            return

        print(f"[ZMQ] Received data keys: {list(data.keys())}")

        # smpl_joints: (N, 24, 3) — 取最新一帧
        if "smpl_joints" in data:
            sj = data["smpl_joints"].astype(np.float32)  # (N, 24, 3)
            frame = sj[-1]  # (24, 3)
            print(f"[ZMQ] smpl_joints shape: {sj.shape}, latest frame sum: {np.abs(frame).sum():.4f}")
            print(f"[ZMQ] smpl_joints latest frame:\n{frame}")
            # 检查是否为有效数据（非全0）
            if np.abs(frame).sum() > 0.01:
                self._smpl_data_valid = True
                print(f"[ZMQ] SMPL data marked as VALID")
            self._smpl_joints_buf = np.roll(self._smpl_joints_buf, -1, axis=0)
            self._smpl_joints_buf[-1] = frame

        # smpl_pose: (N, 21, 3)
        if "smpl_pose" in data:
            sp = data["smpl_pose"].astype(np.float32)    # (N, 21, 3)
            print(f"[ZMQ] smpl_pose shape: {sp.shape}, latest frame:\n{sp[-1]}")
            self._smpl_pose_buf = np.roll(self._smpl_pose_buf, -1, axis=0)
            self._smpl_pose_buf[-1] = sp[-1]

        # body_quat_w: (N, 4) → 转换为6D旋转表示
        if "body_quat_w" in data:
            bq = data["body_quat_w"].astype(np.float32)  # (N, 4)
            print(f"[ZMQ] body_quat_w shape: {bq.shape}, latest: {bq[-1]}")
            # 转换为6D旋转表示
            rot6d = quat_to_rotation_6d(bq)  # (N, 6)
            print(f"[ZMQ] converted to rot6d shape: {rot6d.shape}, latest: {rot6d[-1]}")
            self._body_rot6d_buf = np.roll(self._body_rot6d_buf, -1, axis=0)
            self._body_rot6d_buf[-1] = rot6d[-1]  # (6,)

        # 机器人关节状态（来自 ZMQ，用于 obs 构建）
        if "joint_pos" in data:
            jp = data["joint_pos"].astype(np.float32)
            self._robot_joint_pos = jp[-1]  # (29,)
        if "joint_vel" in data:
            jv = data["joint_vel"].astype(np.float32)
            self._robot_joint_vel = jv[-1]  # (29,)

        # 手部关节
        if "left_hand_joints" in data:
            lh = data["left_hand_joints"].flatten().astype(np.float32)
            self._left_hand_target[:len(lh)] = lh[:7]
        if "right_hand_joints" in data:
            rh = data["right_hand_joints"].flatten().astype(np.float32)
            self._right_hand_target[:len(rh)] = rh[:7]

    # ------------------------------------------------------------------
    # GEAR-SONIC encoder + decoder 推理
    # ------------------------------------------------------------------

    def _run_gear_sonic(self) -> np.ndarray:
        """运行 GEAR-SONIC encoder+decoder，返回 SONIC IsaacLab 顺序的 29 DOF 关节目标。

        Encoder输入: 1762维，包含所有启用的观察值
        - encoder_mode_4: 4
        - motion_joint_positions_10frame_step5: 290
        - motion_joint_velocities_10frame_step5: 290
        - motion_root_z_position_10frame_step5: 10
        - motion_root_z_position: 1
        - motion_anchor_orientation: 6
        - motion_anchor_orientation_10frame_step5: 60
        - motion_joint_positions_lowerbody_10frame_step5: 120
        - motion_joint_velocities_lowerbody_10frame_step5: 120
        - vr_3point_local_target: 9
        - vr_3point_local_orn_target: 12
        - smpl_joints_10frame_step1: 720
        - smpl_anchor_orientation_10frame_step1: 60
        - motion_joint_positions_wrists_10frame_step1: 60
        """
        print(f"[SONIC] _run_gear_sonic called")
        print(f"[SONIC] encoder={self._encoder is not None}, decoder={self._decoder is not None}")
        print(f"[SONIC] _smpl_data_valid={self._smpl_data_valid}")

        if self._encoder is None or self._decoder is None:
            print(f"[SONIC] Encoder/Decoder not loaded, returning default pose")
            return self._sonic_default_np.copy()

        # 检查历史缓冲区是否有有效数据（即使ZMQ暂时没有新数据）
        smpl_joints_sum = np.abs(self._smpl_joints_buf).sum()
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

            # 1. encoder_mode_4 (4) - SMPL模式 (mode_id=2)
            encoder_mode = np.array([0., 0., 1., 0.], dtype=np.float32)

            # 2. motion_joint_positions_10frame_step5 (290) - 每隔5帧采样
            motion_joint_pos_step5 = self._robot_joint_pos_hist[::5].reshape(-1)  # (2, 29) → (58,)
            # 需要10帧，但我们只有2帧，用零填充
            motion_joint_pos_step5_full = np.zeros(290, dtype=np.float32)
            motion_joint_pos_step5_full[:58] = motion_joint_pos_step5

            # 3. motion_joint_velocities_10frame_step5 (290)
            motion_joint_vel_step5 = self._robot_joint_vel_hist[::5].reshape(-1)
            motion_joint_vel_step5_full = np.zeros(290, dtype=np.float32)
            motion_joint_vel_step5_full[:58] = motion_joint_vel_step5

            # 4. motion_root_z_position_10frame_step5 (10) - 用零填充（没有motion数据）
            motion_root_z_step5 = np.zeros(10, dtype=np.float32)

            # 5. motion_root_z_position (1)
            motion_root_z = np.zeros(1, dtype=np.float32)

            # 6. motion_anchor_orientation (6) - 当前帧的6D旋转
            motion_anchor_orient = self._body_rot6d_buf[-1]  # (6,)

            # 7. motion_anchor_orientation_10frame_step5 (60)
            motion_anchor_orient_step5 = self._body_rot6d_buf[::5].reshape(-1)  # (2, 6) → (12,)
            motion_anchor_orient_step5_full = np.zeros(60, dtype=np.float32)
            motion_anchor_orient_step5_full[:12] = motion_anchor_orient_step5

            # 8. motion_joint_positions_lowerbody_10frame_step5 (120) - 12个下半身关节
            # 下半身关节索引（前12个关节）
            lowerbody_indices = list(range(12))
            motion_joint_pos_lowerbody = self._robot_joint_pos_hist[::5, :12].reshape(-1)  # (2, 12) → (24,)
            motion_joint_pos_lowerbody_full = np.zeros(120, dtype=np.float32)
            motion_joint_pos_lowerbody_full[:24] = motion_joint_pos_lowerbody

            # 9. motion_joint_velocities_lowerbody_10frame_step5 (120)
            motion_joint_vel_lowerbody = self._robot_joint_vel_hist[::5, :12].reshape(-1)
            motion_joint_vel_lowerbody_full = np.zeros(120, dtype=np.float32)
            motion_joint_vel_lowerbody_full[:24] = motion_joint_vel_lowerbody

            # 10. vr_3point_local_target (9) - 用零填充（SMPL模式不需要）
            vr_3pt_pos = np.zeros(9, dtype=np.float32)

            # 11. vr_3point_local_orn_target (12)
            vr_3pt_orn = np.zeros(12, dtype=np.float32)

            # 12. smpl_joints_10frame_step1 (720) - 10帧连续SMPL关节
            smpl_joints_flat = self._smpl_joints_buf.reshape(-1)  # (10, 24, 3) → (720,)

            # 13. smpl_anchor_orientation_10frame_step1 (60) - 10帧连续6D旋转
            smpl_anchor_orient_flat = self._body_rot6d_buf.reshape(-1)  # (10, 6) → (60,)

            # 14. motion_joint_positions_wrists_10frame_step1 (60) - 6个手腕关节
            # 手腕关节索引（肩部3个关节 × 2）
            wrist_indices = [12, 13, 14, 15, 16, 17]  # left_shoulder_*, right_shoulder_*
            motion_wrist_pos = self._robot_joint_pos_hist[:, wrist_indices].reshape(-1)  # (10, 6) → (60,)

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

            print(f"[SONIC] Encoder input shape: {encoder_input.shape}, expected: (1, 1762)")
            print(f"[SONIC] Encoder input dtype: {encoder_input.dtype}")
            print(f"[SONIC] Encoder input range: [{encoder_input.min():.4f}, {encoder_input.max():.4f}]")
            print(f"[SONIC] SMPL joints sum: {np.abs(smpl_joints_flat).sum():.4f}")

            # Encoder推理
            enc_inputs = {
                self._encoder.get_inputs()[0].name: encoder_input
            }
            print(f"[SONIC] Running encoder inference...")
            latent = self._encoder.run(None, enc_inputs)[0]
            self._latent = latent
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

            print(f"[SONIC] Decoder input shape: {dec_obs.shape}, expected: (1, 994)")
            dec_inputs = {self._decoder.get_inputs()[0].name: dec_obs}
            print(f"[SONIC] Running decoder inference...")
            action_sonic = self._decoder.run(None, dec_inputs)[0]
            raw_sonic = action_sonic.flatten()[:29]
            print(f"[SONIC] ✓ Decoder output shape: {action_sonic.shape}")
            print(f"[SONIC] Raw sonic range (before clip): [{raw_sonic.min():.4f}, {raw_sonic.max():.4f}]")

            # ✨ CRITICAL FIX: Clip raw action to reasonable range
            # Normal decoder output should be in [-2, 2] range
            raw_sonic = np.clip(raw_sonic, -2.0, 2.0)
            print(f"[SONIC] Raw sonic range (after clip): [{raw_sonic.min():.4f}, {raw_sonic.max():.4f}]")

            # 更新 last_action_hist with raw action (before scaling)
            # 参考 GR00T g1_deploy_onnx_ref.cpp:269, 2825
            self._last_action_hist = np.roll(self._last_action_hist, -1, axis=0)
            self._last_action_hist[-1] = raw_sonic

            # 后处理：per-joint action_scale + default（SONIC IsaacLab order）
            # 参考 GR00T g1_deploy_onnx_ref.cpp:2824
            # target = action * action_scale + default_angle
            target_sonic = raw_sonic * G1_ACTION_SCALE_ISAACLAB + self._sonic_default_np
            print(f"[SONIC] ✓ Final target range (before safety clip): [{target_sonic.min():.4f}, {target_sonic.max():.4f}]")

            # ✨ Safety clip: Ensure final targets are within reasonable joint limits
            # Typical G1 joint limits are around [-3.14, 3.14] rad
            target_sonic = np.clip(target_sonic, -3.0, 3.0)
            print(f"[SONIC] ✓ Final target range (after safety clip): [{target_sonic.min():.4f}, {target_sonic.max():.4f}]")

            return target_sonic.astype(np.float32)

        except Exception as e:
            print(f"[SonicActionProvider] GEAR-SONIC inference error: {e}")
            import traceback
            traceback.print_exc()
            return self._sonic_default_np.copy()

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    # def get_action(self, env) -> Optional[torch.Tensor]:
    #     try:
    #         # 1. 读取 ZMQ POSE 消息（完整 SMPL 全身数据）
    #         self._fetch_zmq_pose()
    #         print(f"_smpl_joints_buf:{self._smpl_joints_buf[-1]}")
    #         # 2. GEAR-SONIC 全身 retargeting → 29 DOF（SONIC IsaacLab order）
    #         sonic_targets = self._run_gear_sonic()  # np (29,) SONIC IsaacLab order
    #         print(f'sonic_targets:{sonic_targets}')
    #         # 3. 构建完整 Isaac 动作
    #         full_action = self._default_pos.clone().squeeze(0)  # (N,)
    #         sonic_t = torch.tensor(sonic_targets, dtype=torch.float32,
    #                                device=self.device)
    #         full_action.index_copy_(0, self._sonic_idx, sonic_t)
    #
    #         # 4. 手部关节
    #         self._apply_hand_targets(full_action)
    #
    #         # 5. 步进仿真（decimation）
    #         for _ in range(self._decimation):
    #             env.scene["robot"].set_joint_position_target(full_action)
    #             env.scene.write_data_to_sim()
    #             env.sim.step(render=False)
    #             env.scene.update(dt=env.physics_dt)
    #
    #         env.sim.render()
    #         env.observation_manager.compute()
    #         return full_action
    #
    #     except Exception as e:
    #         print(f"[SonicActionProvider] get_action error: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None

    def get_action(self, env) -> Optional[torch.Tensor]:
        """测试：持续设置 SONIC 默认姿态，让 PD 控制器维持平衡"""
        if not hasattr(self, '_test_logged'):
            self._test_logged = True
            self._frame_count = 0
            print(f"\n[TEST] Setting joint targets to SONIC default pose")
            print(f"[TEST] SONIC_DEFAULT_POS shape: {self._sonic_default_np.shape}")
            print(f"[TEST] Target joint values:")
            print(f"  left_shoulder_pitch (idx 11): {self._sonic_default_np[11]}")
            print(f"  right_shoulder_pitch (idx 12): {self._sonic_default_np[12]}")
            print(f"  left_shoulder_roll (idx 15): {self._sonic_default_np[15]}")
            print(f"  right_shoulder_roll (idx 16): {self._sonic_default_np[16]}")
            print(f"  left_elbow (idx 21): {self._sonic_default_np[21]}")
            print(f"  right_elbow (idx 22): {self._sonic_default_np[22]}")

            # 读取当前关节位置
            current_pos = env.scene["robot"].data.joint_pos[0, self._sonic_idx].cpu().numpy()
            print(f"\n[TEST] Current joint positions:")
            print(f"  left_shoulder_pitch (idx 11): {current_pos[11]:.3f}")
            print(f"  right_shoulder_pitch (idx 12): {current_pos[12]:.3f}")
            print(f"  left_shoulder_roll (idx 15): {current_pos[15]:.3f}")
            print(f"  right_shoulder_roll (idx 16): {current_pos[16]:.3f}")
            print(f"  left_elbow (idx 21): {current_pos[21]:.3f}")
            print(f"  right_elbow (idx 22): {current_pos[22]:.3f}")

        try:
            self._frame_count += 1

            # 使用 SONIC 默认姿态作为关节目标
            sonic_targets = self._sonic_default_np.copy()

            # 构建完整 Isaac 动作
            full_action = self._default_pos.clone().squeeze(0)  # (N,)
            sonic_t = torch.tensor(sonic_targets, dtype=torch.float32, device=self.device)
            full_action.index_copy_(0, self._sonic_idx, sonic_t)

            # 步进仿真（decimation）
            for _ in range(self._decimation):
                env.scene["robot"].set_joint_position_target(full_action)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

            # 每50帧打印一次当前位置
            if self._frame_count % 50 == 0:
                current_pos = env.scene["robot"].data.joint_pos[0, self._sonic_idx].cpu().numpy()
                print(f"\n[TEST] Frame {self._frame_count} - Current positions:")
                print(f"  left_elbow: {current_pos[21]:.3f} (target: {sonic_targets[21]:.3f})")
                print(f"  right_elbow: {current_pos[22]:.3f} (target: {sonic_targets[22]:.3f})")
                print(f"  left_shoulder_roll: {current_pos[15]:.3f} (target: {sonic_targets[15]:.3f})")
                print(f"  right_shoulder_roll: {current_pos[16]:.3f} (target: {sonic_targets[16]:.3f})")

            env.sim.render()
            env.observation_manager.compute()

            return full_action

        except Exception as e:
            print(f"[TEST] Error: {e}")
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