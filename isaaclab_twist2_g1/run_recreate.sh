#!/bin/bash

# 清除 Redis 缓存（防止启动时使用旧数据）
echo "Clearing Redis cache..."
redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action \
  isaac_reset_trigger \
  state_body_unitree_g1_with_hands \
  state_hand_left_unitree_g1_with_hands \
  state_hand_right_unitree_g1_with_hands \
  state_neck_unitree_g1_with_hands \
  t_state \
  recording_control_unitree_g1_with_hands \
  isaac_reset_complete_unitree_g1_with_hands \
  human_smplx_data_unitree_g1_with_hands \
  human_info_unitree_g1_with_hands > /dev/null 2>&1
echo "Redis cache cleared"
echo ""

# run_recreate.sh
# 使用完全重建环境的版本运行仿真

# 设置环境变量
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 任务名称（可切换）
TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"

# Random seed for reproducibility
SEED="${SEED:-42}"

echo "========================================"
echo "Starting Simulation (RECREATE VERSION)"
echo "========================================"
echo "Task: $TASK_NAME"
echo "Seed: $SEED"
echo "========================================"

# 运行仿真（参数与 run.sh 保持一致）
python3 sim_main_recreate.py \
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
  --recording_save_dir /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/0315/zk \
  --seed "${SEED}" \
  # --headless

echo ""
echo "Simulation ended"
