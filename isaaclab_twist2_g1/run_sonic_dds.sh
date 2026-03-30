#!/usr/bin/env bash
# run_sonic_dds.sh – SONIC POSE 模式（DDS 桥接方案）
#
# 本脚本使用 **独立进程桥接** 方案：
#   Terminal 1: Pico VR → ZMQ "pose" topic
#   Terminal 2: sonic_wbc_bridge.py → DDS rt/lowcmd
#   Terminal 3: Isaac Lab (本脚本) → 读取 DDS
#
# 优势：零代码修改，复用 twist2 现有 DDS 基础设施
# 劣势：需要运行 3 个进程，通信链路较长
#
# 推荐：若需要最佳性能，使用 run_sonic.sh（直接集成方案）
#
# 前置条件：
#   Terminal 1: Pico VR
#     cd GR00T-WholeBodyControl
#     python gear_sonic/scripts/pico_manager_thread_server.py --manager --port 5556
#
#   Terminal 2: SONIC WBC Bridge
#     cd isaaclab_twist2_g1
#     python sonic_wbc_bridge.py \
#         --zmq_port 5556 \
#         --encoder ../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx \
#         --decoder ../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
#         --domain_id 1
#
#   Terminal 3: 本脚本
#     bash run_sonic_dds.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 清理 Redis 缓存 ───────────────────────────────────────────────────
redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action

# ── 启动 Isaac Lab 仿真 ───────────────────────────────────────────────
cd "$SCRIPT_DIR"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/sonic_default.yaml}"

# 使用 action_source=dds（复用 twist2 现有 DDS 基础设施）
# sonic_wbc_bridge.py 会发布到 DDS rt/lowcmd
# action_provider_wh_dds.py line 392 会从 SharedMemory "dds_robot_cmd" 读取

python sim_main.py \
    --device cuda \
    --enable_cameras \
    --env_config_yaml "${ENV_CONFIG_YAML}" \
    --task Isaac-Move-Cylinder-G129-Dex3-Wholebody \
    --robot_type g129 \
    --enable_dex3_dds \
    --action_source dds \
    --image_transport xrobot \
    --image_xrobot_host 10.42.0.35 \
    --image_xrobot_port 12345 \
    --image_xrobot_bitrate 4194304 \
    --image_fps 30 \
    --image_xrobot_ffmpeg /usr/bin/ffmpeg \
    --enable_world_camera

# 注意：
# 1. 必须先启动 sonic_wbc_bridge.py，否则 Isaac Lab 无法读取 DDS 命令
# 2. 若要使用直接集成方案（更高性能），使用 run_sonic.sh
