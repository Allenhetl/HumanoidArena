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

ENCODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
DECODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"

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
#Isaac-Move-ArtVIP-Livingroom-NoSofa-G129-Dex3-Wholebody
#Isaac-Move-Cylinder-G129-Dex3-Wholebody
#Isaac-Move-Football-G129-Dex3-Wholebody
#Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody

# 可切换：
#   一般沙袋: Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody
#   吊挂沙袋: Isaac-Move-Boxing-Bag-Hanging-G129-Dex3-Wholebody
#   足球: Isaac-Move-Football-G129-Dex3-Wholebody
#   双桌面拾放: Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody
#   Push-T: Isaac-Push-T-G129-Dex3-Wholebody
#   客厅交互：Isaac-Move-ArtVIP-Livingroom-G129-Dex3-Wholebody
#   客厅抓杯：Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody
#   三级台阶平台：Isaac-Move-Three-Step-Platform-G129-Dex3-Wholebody
# ready
# TASK_NAME="${TASK_NAME:-Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody}"
# TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"
# TASK_NAME="${TASK_NAME:-Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody}"
# TASK_NAME="${TASK_NAME:-Isaac-Move-Three-Step-Platform-G129-Dex3-Wholebody}"
# TASK_NAME="${TASK_NAME:-Isaac-Move-ArtVIP-Livingroom-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody}"
 TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"

# 机器人脚部碰撞版本切换：
#   fourpoints  -> temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd（四球脚部碰撞）
#   box         -> g1_29dof_with_dex3_rev_1_0.usd（长方体脚部碰撞）
ROBOT_COLLIDER_MODE="${ROBOT_COLLIDER_MODE:-box}"
if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
    export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
    export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"
fi
echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"

python sim_main.py \
    --device cuda \
    --enable_cameras \
    --task "${TASK_NAME}" \
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
