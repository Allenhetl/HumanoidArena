#!/usr/bin/env python3
"""
SONIC数据流诊断工具

用法：
1. 启动pico_server
2. 运行此脚本：python diagnose_sonic_dataflow.py
3. 在VR中做一个简单动作（比如举起右手）
4. 查看输出，找出哪个环节出了问题

诊断点：
- [1] ZMQ数据接收
- [2] SMPL数据解析
- [3] Encoder输入构建
- [4] Encoder推理
- [5] Decoder输入构建
- [6] Decoder推理
- [7] 输出后处理
"""

import json
import sys
import time
import numpy as np
import zmq

# 添加路径
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TWIST2_ROOT = os.path.dirname(_THIS_DIR)
_GROOT_ROOT = os.path.join(os.path.dirname(_TWIST2_ROOT), "GR00T-WholeBodyControl")
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("⚠️  onnxruntime not found, will skip model inference tests")

# ZMQ消息解析
_HEADER_SIZE = 1280

def parse_zmq_pose(raw: bytes):
    """解析ZMQ pose消息"""
    try:
        topic_end = raw.index(b"{")
        header_bytes = raw[topic_end: topic_end + _HEADER_SIZE]
        payload = raw[topic_end + _HEADER_SIZE:]
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))

        _DTYPE_MAP = {
            "f32": (np.float32, 4), "f64": (np.float64, 8),
            "i32": (np.int32, 4), "i64": (np.int64, 8),
            "u8": (np.uint8, 1), "bool": (np.bool_, 1),
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
        print(f"❌ [1] ZMQ解析失败: {e}")
        return None

def quat_to_rotation_6d(quat_wxyz):
    """四元数转6D旋转表示"""
    quat_wxyz = quat_wxyz.reshape(-1, 4)
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]

    # 转旋转矩阵
    R = np.zeros((quat_wxyz.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    # 取前两列
    rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)
    return rot6d

def build_encoder_input_smpl_mode(smpl_joints_hist, body_rot6d_hist, wrist_pos_hist):
    """构建SMPL模式的encoder输入（1762维）"""
    # encoder_mode_4: [0, 0, 1, 0]
    encoder_mode = np.array([0, 0, 1, 0], dtype=np.float32)

    # smpl_joints_10frame_step1: 10帧 x 24关节 x 3坐标 = 720
    smpl_joints_flat = smpl_joints_hist.reshape(-1).astype(np.float32)

    # smpl_anchor_orientation_10frame_step1: 10帧 x 6D = 60
    anchor_rot6d_flat = body_rot6d_hist.reshape(-1).astype(np.float32)

    # motion_joint_positions_wrists_10frame_step1: 10帧 x 6个wrist关节 = 60
    wrist_flat = wrist_pos_hist.reshape(-1).astype(np.float32)

    # 其他字段全部置零
    zeros_motion_pos = np.zeros(290, dtype=np.float32)  # 10x29
    zeros_motion_vel = np.zeros(290, dtype=np.float32)  # 10x29
    zeros_root_z_hist = np.zeros(10, dtype=np.float32)
    zeros_root_z = np.zeros(1, dtype=np.float32)
    zeros_anchor_single = np.zeros(6, dtype=np.float32)
    zeros_anchor_hist = np.zeros(60, dtype=np.float32)
    zeros_lowerbody_pos = np.zeros(120, dtype=np.float32)  # 10x12
    zeros_lowerbody_vel = np.zeros(120, dtype=np.float32)  # 10x12
    zeros_vr3pt_pos = np.zeros(9, dtype=np.float32)  # 3点 x 3坐标
    zeros_vr3pt_orn = np.zeros(12, dtype=np.float32)  # 3点 x 4四元数（不是6D！）

    # 拼接（顺序必须与observation_config.yaml一致）
    encoder_input = np.concatenate([
        encoder_mode,           # 4
        zeros_motion_pos,       # 290
        zeros_motion_vel,       # 290
        zeros_root_z_hist,      # 10
        zeros_root_z,           # 1
        zeros_anchor_single,    # 6
        zeros_anchor_hist,      # 60
        zeros_lowerbody_pos,    # 120
        zeros_lowerbody_vel,    # 120
        zeros_vr3pt_pos,        # 9
        zeros_vr3pt_orn,        # 18
        smpl_joints_flat,       # 720
        anchor_rot6d_flat,      # 60
        wrist_flat,             # 60
    ], axis=0)

    return encoder_input.reshape(1, -1)

def diagnose():
    """运行完整诊断"""
    print("=" * 80)
    print("SONIC数据流诊断工具")
    print("=" * 80)

    # 配置
    ZMQ_HOST = "localhost"
    ZMQ_PORT = 5556
    ENCODER_PATH = "/home/dreams/Users/Alyssa/HumanoidArena_V1/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
    DECODER_PATH = "/home/dreams/Users/Alyssa/HumanoidArena_V1/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"

    # 检查模型文件
    print(f"\n📁 检查模型文件...")
    if not os.path.exists(ENCODER_PATH):
        print(f"❌ Encoder模型不存在: {ENCODER_PATH}")
        print(f"   请检查路径或使用 --encoder_path 参数指定")
        return
    else:
        print(f"✅ Encoder: {ENCODER_PATH}")

    if not os.path.exists(DECODER_PATH):
        print(f"❌ Decoder模型不存在: {DECODER_PATH}")
        print(f"   请检查路径或使用 --decoder_path 参数指定")
        return
    else:
        print(f"✅ Decoder: {DECODER_PATH}")

    # [1] 连接ZMQ
    print(f"\n[1] 连接ZMQ: {ZMQ_HOST}:{ZMQ_PORT}")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时

    print("   等待接收数据...")
    try:
        raw = socket.recv()
        print(f"✅ 接收到 {len(raw)} 字节")
    except zmq.error.Again:
        print("❌ 5秒内未收到数据")
        print("   请确认：")
        print("   1. pico_server正在运行")
        print("   2. Pico设备已连接并开始追踪")
        print("   3. ZMQ端口正确（默认5556）")
        return

    # [2] 解析SMPL数据
    print(f"\n[2] 解析SMPL数据")
    data = parse_zmq_pose(raw)
    if data is None:
        return

    print(f"✅ 解析成功，包含字段: {list(data.keys())}")

    # 检查关键字段
    required_fields = ["smpl_joints", "smpl_pose", "body_quat_w", "joint_pos"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        print(f"❌ 缺少必需字段: {missing}")
        return

    smpl_joints = data["smpl_joints"].astype(np.float32)  # (N, 24, 3)
    smpl_pose = data["smpl_pose"].astype(np.float32)      # (N, 21, 3)
    body_quat_w = data["body_quat_w"].astype(np.float32)  # (N, 4)
    joint_pos = data["joint_pos"].astype(np.float32)      # (N, 29)

    print(f"   smpl_joints: {smpl_joints.shape}, 范围: [{smpl_joints.min():.3f}, {smpl_joints.max():.3f}]")
    print(f"   smpl_pose: {smpl_pose.shape}, 范围: [{smpl_pose.min():.3f}, {smpl_pose.max():.3f}]")
    print(f"   body_quat_w: {body_quat_w.shape}, 最新: {body_quat_w[-1]}")
    print(f"   joint_pos: {joint_pos.shape}, 范围: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]")

    # 检查数据有效性
    if np.abs(smpl_joints).sum() < 0.01:
        print("⚠️  smpl_joints接近全0，可能没有有效的追踪数据")
        print("   请确认：")
        print("   1. Pico的5个追踪点（头+双手腕+双脚踝）都在工作")
        print("   2. 在VR中做一个明显的动作（比如举手）")
        return
    else:
        print("✅ SMPL数据有效（非全0）")

    # [3] 构建Encoder输入
    print(f"\n[3] 构建Encoder输入")

    # 准备10帧历史（这里简化为重复最新帧）
    smpl_joints_hist = np.tile(smpl_joints[-1:], (10, 1, 1))  # (10, 24, 3)
    body_rot6d = quat_to_rotation_6d(body_quat_w[-1:])  # (1, 6)
    body_rot6d_hist = np.tile(body_rot6d, (10, 1))  # (10, 6)

    # Wrist关节位置（索引23-28）
    WRIST_INDICES = [23, 24, 25, 26, 27, 28]
    wrist_pos = joint_pos[-1, WRIST_INDICES]  # (6,)
    wrist_pos_hist = np.tile(wrist_pos, (10, 1))  # (10, 6)

    encoder_input = build_encoder_input_smpl_mode(
        smpl_joints_hist, body_rot6d_hist, wrist_pos_hist
    )

    print(f"✅ Encoder输入: {encoder_input.shape}")
    print(f"   期望: (1, 1762)")
    if encoder_input.shape[1] != 1762:
        print(f"❌ 维度不匹配！")
        return
    print(f"   数值范围: [{encoder_input.min():.3f}, {encoder_input.max():.3f}]")
    print(f"   前10维（encoder_mode等）: {encoder_input[0, :10]}")

    # [4] Encoder推理
    if not HAS_ORT:
        print(f"\n⚠️  跳过模型推理测试（onnxruntime未安装）")
        return

    print(f"\n[4] Encoder推理")
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        encoder_session = ort.InferenceSession(ENCODER_PATH, providers=providers)
        print(f"✅ 加载Encoder模型")
        print(f"   输入: {encoder_session.get_inputs()[0].name}, shape: {encoder_session.get_inputs()[0].shape}")
        print(f"   输出: {encoder_session.get_outputs()[0].name}, shape: {encoder_session.get_outputs()[0].shape}")
    except Exception as e:
        print(f"❌ 加载Encoder失败: {e}")
        return

    try:
        latent = encoder_session.run(None, {encoder_session.get_inputs()[0].name: encoder_input})[0]
        print(f"✅ Encoder推理成功")
        print(f"   输出latent: {latent.shape}, 范围: [{latent.min():.3f}, {latent.max():.3f}]")
        print(f"   latent前10维: {latent[0, :10]}")
    except Exception as e:
        print(f"❌ Encoder推理失败: {e}")
        return

    # [5] 构建Decoder输入
    print(f"\n[5] 构建Decoder输入")

    # Decoder需要994维输入
    # token_state(64) + ang_vel_hist(30) + joint_pos_hist(290) + joint_vel_hist(290) + last_action_hist(290) + grav_dir_hist(30)
    ang_vel_hist = np.zeros((10, 3), dtype=np.float32)
    joint_pos_hist = np.tile(joint_pos[-1:], (10, 1))  # (10, 29)
    joint_vel_hist = np.zeros((10, 29), dtype=np.float32)
    last_action_hist = np.zeros((10, 29), dtype=np.float32)
    grav_dir_hist = np.tile(np.array([0, 0, -1], dtype=np.float32), (10, 1))  # (10, 3)

    decoder_input = np.concatenate([
        latent.reshape(-1),              # 64
        ang_vel_hist.reshape(-1),        # 30
        joint_pos_hist.reshape(-1),      # 290
        joint_vel_hist.reshape(-1),      # 290
        last_action_hist.reshape(-1),    # 290
        grav_dir_hist.reshape(-1),       # 30
    ], axis=0).reshape(1, -1)

    print(f"✅ Decoder输入: {decoder_input.shape}")
    print(f"   期望: (1, 994)")
    if decoder_input.shape[1] != 994:
        print(f"❌ 维度不匹配！")
        return
    print(f"   数值范围: [{decoder_input.min():.3f}, {decoder_input.max():.3f}]")

    # [6] Decoder推理
    print(f"\n[6] Decoder推理")
    try:
        decoder_session = ort.InferenceSession(DECODER_PATH, providers=providers)
        print(f"✅ 加载Decoder模型")
        print(f"   输入: {decoder_session.get_inputs()[0].name}, shape: {decoder_session.get_inputs()[0].shape}")
        print(f"   输出: {decoder_session.get_outputs()[0].name}, shape: {decoder_session.get_outputs()[0].shape}")
    except Exception as e:
        print(f"❌ 加载Decoder失败: {e}")
        return

    try:
        action_raw = decoder_session.run(None, {decoder_session.get_inputs()[0].name: decoder_input})[0]
        print(f"✅ Decoder推理成功")
        print(f"   输出action: {action_raw.shape}, 范围: [{action_raw.min():.3f}, {action_raw.max():.3f}]")
        print(f"   action前10维: {action_raw[0, :10]}")
    except Exception as e:
        print(f"❌ Decoder推理失败: {e}")
        return

    # [7] 输出后处理
    print(f"\n[7] 输出后处理")

    # 动作缩放
    G1_ACTION_SCALE = np.array([
        0.3506614566, 0.3506614566, 0.5475464463, 0.3506614566, 0.3506614566,
        0.4385773242, 0.5475464463, 0.5475464463, 0.4385773242, 0.3506614566,
        0.3506614566, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
        0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
        0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
        0.0745008737, 0.0745008737, 0.0745008737, 0.0745008737,
    ], dtype=np.float32)

    SONIC_DEFAULT_POS = np.array([
        -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4,
        0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)

    action_scaled = action_raw[0] * G1_ACTION_SCALE
    joint_targets = SONIC_DEFAULT_POS + action_scaled

    print(f"✅ 后处理完成")
    print(f"   action_scaled范围: [{action_scaled.min():.3f}, {action_scaled.max():.3f}]")
    print(f"   joint_targets范围: [{joint_targets.min():.3f}, {joint_targets.max():.3f}]")
    print(f"   joint_targets前10维: {joint_targets[:10]}")

    # 最终总结
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    print("✅ 所有环节测试通过！")
    print("\n如果IsaacLab中仍然没有动作，可能的原因：")
    print("1. 历史缓冲未填满：需要等待10帧数据（约0.2秒）")
    print("2. 动作平滑：smooth_steps=20会导致约0.4秒的延迟")
    print("3. 关节映射：检查SONIC joint order是否与IsaacLab一致")
    print("4. PD控制器参数：stiffness/damping可能需要调整")
    print("\n建议：")
    print("- 在VR中做一个大幅度、持续的动作（比如持续举手5秒）")
    print("- 查看IsaacLab日志中的 [SONIC] 输出")
    print("- 检查joint_targets是否随时间变化")

if __name__ == "__main__":
    diagnose()
