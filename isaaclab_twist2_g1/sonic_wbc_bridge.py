#!/usr/bin/env python3
"""sonic_wbc_bridge.py - POSE 模式 ZMQ → DDS 桥接（可选方案）

本脚本提供 SONIC POSE 模式的 **独立进程** 集成方案：
  Pico VR (ZMQ "pose" topic)
    → 本脚本读取完整 SMPL 数据
    → GEAR-SONIC encoder+decoder ONNX 推理
    → 发布 G1 关节命令到 DDS rt/lowcmd
    → Isaac Lab 通过 action_provider_wh_dds.py 读取

优势：
  - 零代码修改：复用 twist2 现有 DDS 基础设施
  - 进程隔离：SONIC 策略独立运行，不影响 Isaac Lab 主循环

劣势：
  - 额外进程：需要同时运行 3 个进程（Pico + Bridge + Isaac Lab）
  - 通信开销：ZMQ → ONNX → DDS 链路较长

推荐方案：
  - 若需要快速验证，使用本脚本（run_sonic_dds.sh）
  - 若需要最佳性能，使用 action_provider_sonic.py（run_sonic.sh）

Usage:
    # Terminal 1: Pico VR
    cd GR00T-WholeBodyControl
    python gear_sonic/scripts/pico_manager_thread_server.py --manager --port 5556

    # Terminal 2: 本脚本
    python sonic_wbc_bridge.py \\
        --zmq_port 5556 \\
        --encoder /path/to/model_encoder.onnx \\
        --decoder /path/to/model_decoder.onnx \\
        --domain_id 1

    # Terminal 3: Isaac Lab
    cd isaaclab_twist2_g1
    bash run_sonic_dds.sh
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── 添加 gear_sonic 到路径 ────────────────────────────────────────────
GROOT_ROOT = Path(__file__).parent.parent / "GR00T-WholeBodyControl"
sys.path.insert(0, str(GROOT_ROOT))

try:
    from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    import onnxruntime as ort
except ImportError as e:
    print(f"Import error: {e}")
    print("请确保已安装 gear_sonic 和 unitree_sdk2py。")
    sys.exit(1)


# ── ZMQ 消息解析 ──────────────────────────────────────────────────────
_HEADER_SIZE = 1280


def parse_zmq_pose(raw: bytes) -> Optional[dict]:
    """解析 ZMQ pose 消息（Protocol v3）。"""
    try:
        topic_end = raw.index(b"{")
        header_bytes = raw[topic_end: topic_end + _HEADER_SIZE]
        payload = raw[topic_end + _HEADER_SIZE:]
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))

        dtype_map = {
            "f32": (np.float32, 4), "f64": (np.float64, 8),
            "i32": (np.int32, 4),   "i64": (np.int64, 8),
            "u8":  (np.uint8,  1),  "bool": (np.bool_, 1),
        }
        result, offset = {}, 0
        for f in header.get("fields", []):
            np_dtype, itemsize = dtype_map.get(f["dtype"], (np.float32, 4))
            n = int(np.prod(f["shape"]))
            arr = np.frombuffer(payload[offset: offset + n * itemsize],
                                dtype=np_dtype).reshape(f["shape"])
            result[f["name"]] = arr
            offset += n * itemsize
        return result
    except Exception as e:
        print(f"[Bridge] parse error: {e}")
        return None


# ── SONIC 默认姿态 ────────────────────────────────────────────────────
SONIC_DEFAULT_POS = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,          # 左腿
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,          # 右腿
     0.0, 0.0, 0.0,                           # 腰部
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # 左臂
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # 右臂
], dtype=np.float32)


class SonicWBCBridge:
    """POSE 模式 ZMQ → GEAR-SONIC → DDS 桥接。"""

    def __init__(self, zmq_port: int, encoder_path: str, decoder_path: str,
                 domain_id: int):
        self.zmq_port = zmq_port

        # DDS 初始化
        ChannelFactoryInitialize(domain_id, "lo")  # loopback for sim
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.crc = CRC()

        # ZMQ 初始化
        self.zmq_poller = ZMQPoller(host="localhost", port=zmq_port, topic="pose")

        # 加载 ONNX 模型
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)
        print(f"[Bridge] encoder loaded: {encoder_path}")
        print(f"[Bridge] decoder loaded: {decoder_path}")
        print(f"[Bridge] providers: {self.encoder.get_providers()}")

        # SMPL 历史缓冲（10 帧）
        self.smpl_joints_buf = np.zeros((10, 24, 3), dtype=np.float32)
        self.smpl_pose_buf   = np.zeros((10, 21, 3), dtype=np.float32)
        self.body_quat_buf   = np.tile(np.array([1., 0., 0., 0.], dtype=np.float32),
                                       (10, 1))
        self.robot_joint_pos = SONIC_DEFAULT_POS.copy()
        self.robot_joint_vel = np.zeros(29, dtype=np.float32)
        self.latent = None

        print(f"[Bridge] Ready. ZMQ port={zmq_port}, DDS domain={domain_id}")

    def fetch_vr_pose(self):
        """从 ZMQ 读取最新 POSE 消息，更新 SMPL 缓冲。"""
        raw = self.zmq_poller.get_data()
        if raw is None:
            return
        data = parse_zmq_pose(raw)
        if data is None:
            return

        if "smpl_joints" in data:
            sj = data["smpl_joints"].astype(np.float32)  # (N, 24, 3)
            self.smpl_joints_buf = np.roll(self.smpl_joints_buf, -1, axis=0)
            self.smpl_joints_buf[-1] = sj[-1]

        if "smpl_pose" in data:
            sp = data["smpl_pose"].astype(np.float32)    # (N, 21, 3)
            self.smpl_pose_buf = np.roll(self.smpl_pose_buf, -1, axis=0)
            self.smpl_pose_buf[-1] = sp[-1]

        if "body_quat_w" in data:
            bq = data["body_quat_w"].astype(np.float32)  # (N, 4)
            self.body_quat_buf = np.roll(self.body_quat_buf, -1, axis=0)
            self.body_quat_buf[-1] = bq[-1]

        if "joint_pos" in data:
            self.robot_joint_pos = data["joint_pos"][-1].astype(np.float32)
        if "joint_vel" in data:
            self.robot_joint_vel = data["joint_vel"][-1].astype(np.float32)

    def run_gear_sonic(self) -> np.ndarray:
        """运行 GEAR-SONIC encoder+decoder，返回 29 DOF 关节目标。"""
        try:
            # encoder 输入
            smpl_joints_in = self.smpl_joints_buf[np.newaxis]          # (1,10,24,3)
            anchor_orient  = self.body_quat_buf[np.newaxis]            # (1,10,4)
            joint_pos_hist = np.tile(
                self.robot_joint_pos[np.newaxis, np.newaxis],
                (1, 10, 1)).astype(np.float32)                          # (1,10,29)

            enc_inputs = {
                self.encoder.get_inputs()[0].name: smpl_joints_in,
                self.encoder.get_inputs()[1].name: anchor_orient,
                self.encoder.get_inputs()[2].name: joint_pos_hist,
            }
            latent = self.encoder.run(None, enc_inputs)[0]             # (1, latent_dim)
            self.latent = latent

            # decoder 输入（简化：使用零 proprio，实际应从机器人状态读取）
            ang_vel   = np.zeros(3, dtype=np.float32)
            proj_grav = np.array([0., 0., -9.81], dtype=np.float32)
            dof_delta = self.robot_joint_pos - SONIC_DEFAULT_POS
            dof_vel   = self.robot_joint_vel

            proprio = np.concatenate([
                ang_vel * 0.25,
                proj_grav,
                dof_delta,
                dof_vel * 0.05,
            ]).astype(np.float32)[np.newaxis]                           # (1, 64)

            dec_inputs = {
                self.decoder.get_inputs()[0].name: latent,
                self.decoder.get_inputs()[1].name: proprio,
            }
            action = self.decoder.run(None, dec_inputs)[0]             # (1, 29)
            raw = action.flatten()[:29]

            target = raw * 0.25 + SONIC_DEFAULT_POS
            return target.astype(np.float32)

        except Exception as e:
            print(f"[Bridge] GEAR-SONIC inference error: {e}")
            return SONIC_DEFAULT_POS.copy()

    def publish_dds(self, joint_targets: np.ndarray):
        """发布关节目标到 DDS rt/lowcmd。"""
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr = 0
        msg.mode_machine = 5

        for i in range(min(29, len(joint_targets))):
            msg.motor_cmd[i].q = float(joint_targets[i])
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = 100.0
            msg.motor_cmd[i].kd = 2.0

        msg.crc = self.crc.Crc(msg)
        self.publisher.Write(msg)

    def run(self):
        """主循环。"""
        rate = 50  # Hz
        dt = 1.0 / rate

        print(f"[Bridge] Running at {rate} Hz. Press Ctrl+C to stop.")
        try:
            while True:
                t0 = time.time()

                # 1. 读取 VR pose
                self.fetch_vr_pose()

                # 2. GEAR-SONIC 推理
                targets = self.run_gear_sonic()

                # 3. 发布到 DDS
                self.publish_dds(targets)

                # 4. 保持频率
                elapsed = time.time() - t0
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Bridge] Stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description="SONIC POSE 模式 ZMQ → DDS 桥接")
    parser.add_argument("--zmq_port", type=int, default=5556,
                       help="ZMQ pose topic 端口")
    parser.add_argument("--encoder", type=str, required=True,
                       help="GEAR-SONIC encoder ONNX 路径")
    parser.add_argument("--decoder", type=str, required=True,
                       help="GEAR-SONIC decoder ONNX 路径")
    parser.add_argument("--domain_id", type=int, default=1,
                       help="DDS domain ID")
    args = parser.parse_args()

    bridge = SonicWBCBridge(
        zmq_port=args.zmq_port,
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        domain_id=args.domain_id
    )
    bridge.run()


if __name__ == "__main__":
    main()
