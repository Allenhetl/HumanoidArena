#!/usr/bin/env bash
# run_sonic.sh – SONIC POSE 模式全身遥操作（Isaac Lab 仿真）
#
# POSE 模式：Pico 头显 + 手腕控制器 + 脚踝 tracker → 完整 SMPL 全身姿态
#           → GEAR-SONIC encoder+decoder → G1 机器人 29 DOF（含腿部跟踪）
#
# 前置条件（需在独立终端运行）：
#   Terminal 1: Pico VR 数据采集
#     cd GR00T-WholeBodyControl
#     python gear_sonic/scripts/pico_manager_thread_server.py \
#         --manager --port 5556 --wbc_version sonic_model12
#
#   Terminal 2: 本脚本（Isaac Lab 仿真）
#     cd isaaclab_twist2_g1
#     bash run_sonic.sh
#
# 硬件要求：
#   - Pico 头显（支持 POSE 模式）
#   - 2 个手腕控制器
#   - 2 个脚踝 tracker（必须，用于下半身跟踪）
#   - 穿紧身裤以保证脚踝 tracker 视线
#
# 使用方法：
#   bash run_sonic.sh [--encoder /path/to/encoder.onnx] [--decoder /path/to/decoder.onnx]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── GEAR-SONIC 模型路径 ────────────────────────────────────────────────
# 默认路径（根据实际部署修改）
GROOT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)/GR00T-WholeBodyControl"
ENCODER_PATH="${SONIC_ENCODER_PATH:-${GROOT_ROOT}/ckpts/policy/release/model_encoder.onnx}"
DECODER_PATH="${SONIC_DECODER_PATH:-${GROOT_ROOT}/ckpts/policy/release/model_decoder.onnx}"

# 命令行参数覆盖
while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder)
            ENCODER_PATH="$2"
            shift 2
            ;;
        --decoder)
            DECODER_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ZMQ 端点（pico_manager_thread_server 发布 pose 数据）
SONIC_ZMQ_HOST="${SONIC_ZMQ_HOST:-localhost}"
SONIC_ZMQ_PORT="${SONIC_ZMQ_PORT:-5556}"

# ── 清理 Redis 缓存（twist2 惯例）─────────────────────────────────────
redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action

# ── 启动 Isaac Lab 仿真 ───────────────────────────────────────────────
cd "$SCRIPT_DIR"

python sim_main.py \
    --device cuda \
    --enable_cameras \
    --task Isaac-Move-Cylinder-G129-Dex3-Wholebody \
    --robot_type g129 \
    --enable_dex3_dds \
    --action_source sonic_wholebody \
    --sonic_zmq_host "$SONIC_ZMQ_HOST" \
    --sonic_zmq_port "$SONIC_ZMQ_PORT" \
    --sonic_encoder_path "$ENCODER_PATH" \
    --sonic_decoder_path "$DECODER_PATH" \
    --image_transport xrobot \
    --image_xrobot_host 10.42.0.35 \
    --image_xrobot_port 12345 \
    --image_xrobot_width 640 \
    --image_xrobot_height 480 \
    --image_xrobot_bitrate 4194304 \
    --image_fps 30 \
    --image_xrobot_ffmpeg /usr/bin/ffmpeg \
    --enable_world_camera

# 注意：
# 1. POSE 模式需要 Pico 脚踝 tracker，否则腿部跟踪不可用
# 2. pico_manager_thread_server.py 必须先启动并发布 ZMQ "pose" topic
# 3. encoder/decoder 模型路径需根据实际部署调整
# 4. 若要使用 VR_3PT 模式（仅上半身 IK + 下半身 RL），请使用 gear_sonic 原始部署脚本