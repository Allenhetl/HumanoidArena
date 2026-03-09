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

  python sim_main.py \
    --device cuda \
    --enable_cameras \
    --task Isaac-Move-Cylinder-G129-Dex3-Wholebody \
    --robot_type g129 \
    --enable_dex3_dds \
    --image_transport xrobot \
    --image_xrobot_host 10.42.0.35 \
    --image_xrobot_port 12345 \
    --image_xrobot_width 640 \
    --image_xrobot_height 480 \
    --image_xrobot_bitrate 4194304 \
    --image_fps 10 \
    --image_xrobot_ffmpeg /usr/bin/ffmpeg
#    --enable_world_camera \
#    --image_xrobot_host 172.20.10.2 \
