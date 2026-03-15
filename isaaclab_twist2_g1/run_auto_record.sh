#!/bin/bash

# Test script for auto-start recording from reset
# This ensures Frame 0 is the first state after env.reset()

echo "=========================================="
echo "Auto-Start Recording Test"
echo "=========================================="
echo ""
echo "This will:"
echo "1. Start recording immediately after env.reset()"
echo "2. Ensure Frame 0 is the first state after reset"
echo "3. When you press 'save', it will save from Frame 0"
echo ""
echo "Usage:"
echo "  1. Run this script"
echo "  2. Wait for simulation to start"
echo "  3. Control the robot (teleop)"
echo "  4. Press 'save' button to save recording"
echo ""
echo "=========================================="

# Clear Redis keys first
redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action \
  isaac_reset_trigger

# Call sim_main.py directly with all required arguments
python sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Move-Football-G129-Dex3-Wholebody \
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
  --recording_save_dir /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data \
  --seed 42 \
  --auto_start_recording
