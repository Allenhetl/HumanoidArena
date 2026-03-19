#!/usr/bin/env bash
# run_recreate_managed.sh - 使用父进程管理器启动 Isaac Sim

# 清理 Redis 缓存
redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action \
  isaac_reset_trigger \
  isaac_reset_trigger_manager \
  isaac_reset_complete_unitree_g1_with_hands || true

# 任务名称
TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"

# 随机种子
SEED="${SEED:-42}"

# 录制保存目录
RECORDING_DIR="${RECORDING_DIR:-/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/0316/zk}"

echo "=================================="
echo "🎮 Isaac Sim Manager"
echo "=================================="
echo "Task: ${TASK_NAME}"
echo "Seed: ${SEED}"
echo "Recording dir: ${RECORDING_DIR}"
echo "=================================="
echo ""

# 使用父进程管理器启动
python sim_main_manager.py \
  --device cuda \
  --enable_cameras \
  --task "${TASK_NAME}" \
  --robot_type g129 \
  --enable_dex3_dds \
  --image_transport xrobot \
  --image_xrobot_host 10.42.0.35 \
  --image_xrobot_port 12345 \
  --image_xrobot_width 480 \
  --image_xrobot_height 320 \
  --image_xrobot_bitrate 4194304 \
  --image_fps 30 \
  --image_xrobot_ffmpeg /usr/bin/ffmpeg \
  --recording_save_dir "${RECORDING_DIR}" \
  --seed "${SEED}"
