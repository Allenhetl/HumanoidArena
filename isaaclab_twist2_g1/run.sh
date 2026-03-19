SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action \
  isaac_reset_trigger

#  python sim_main.py \
#    --device cuda \
#    --enable_cameras \
#    --task Isaac-Move-Cylinder-G129-Dex1-Wholebody \
#    --robot_type g129 \
#    --enable_dex1_dds \
#    --image_transport xrobot \
#    --image_xrobot_host 10.42.0.35 \
#    --image_xrobot_port 12345 \
#    --image_xrobot_width 640 \
#    --image_xrobot_height 480 \
#    --image_xrobot_bitrate 4194304 \
#    --image_fps 30 \
#    --image_xrobot_ffmpeg /usr/bin/ffmpeg
##    --image_xrobot_host 172.20.10.2 \

  # python sim_main.py \
  #   --device cuda \
  #   --enable_cameras \
  #   --task Isaac-Move-Football-G129-Dex3-Wholebody \
  #   --robot_type g129 \
  #   --enable_dex3_dds \
  #   --image_transport xrobot \
  #   --image_xrobot_host 10.42.0.35 \
  #   --image_xrobot_port 12345 \
  #   --image_xrobot_width 480 \
  #   --image_xrobot_height 320 \
  #   --image_xrobot_bitrate 4194304 \
  #   --image_fps 30 \
  #   --image_xrobot_ffmpeg /usr/bin/ffmpeg \
  #   --recording_save_dir /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data
#    --enable_world_camera \
#    --image_xrobot_host 172.20.10.2 \

  # 可切換：
  #   一般沙袋: Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody
  #   吊掛沙袋: Isaac-Move-Boxing-Bag-Hanging-G129-Dex3-Wholebody
  #   足球: Isaac-Move-Football-G129-Dex3-Wholebody
  #   双桌面拾放: Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody
  #   Push-T: Isaac-Push-T-G129-Dex3-Wholebody
  # TASK_NAME="${TASK_NAME:-Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody}"
  # TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"
  TASK_NAME="${TASK_NAME:-Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody}"
#  TASK_NAME="${TASK_NAME:-Isaac-Push-T-G129-Dex3-Wholebody}"
#   TASK_NAME="${TASK_NAME:-Isaac-Move-Boxing-Bag-Hanging-G129-Dex3-Wholebody}"

  # Random seed for reproducibility (set to fixed value for deterministic behavior)
  SEED="${SEED:-42}"

  python "${SCRIPT_DIR}/sim_main.py" \
    --device cuda \
    --enable_cameras \
    --task "${TASK_NAME}" \
    --robot_type g129 \
    --enable_dex3_dds \
    --image_transport xrobot \
    --image_xrobot_host 10.42.0.35 \
    --image_xrobot_port 12345 \
    --image_xrobot_width 960 \
    --image_xrobot_height 540 \
    --image_xrobot_bitrate 16777216 \
    --image_fps 30 \
    --image_xrobot_ffmpeg /usr/bin/ffmpeg \
    --recording_save_dir /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/0319/zk \
    --camera_enable_depth \
    --camera_width 960 \
    --camera_height 540 \
    --seed "${SEED}"
