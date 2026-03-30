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
  #   客厅交互：Isaac-Move-ArtVIP-Livingroom-G129-Dex3-Wholebody
  #   客厅抓杯：Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody
  #   三级台阶平台：Isaac-Move-Three-Step-Platform-G129-Dex3-Wholebody
  #ready
#TASK_NAME="${TASK_NAME:-Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-Three-Step-Platform-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-ArtVIP-Livingroom-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody}"
TASK_NAME="${TASK_NAME:-Isaac-Move-Football-Single-G129-Dex3-Wholebody}"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/twist2_default.yaml}"

#TASK_NAME="${TASK_NAME:-Isaac-Push-T-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-Boxing-Bag-Hanging-G129-Dex3-Wholebody}"

  # Random seed for reproducibility (set to fixed value for deterministic behavior)
  SEED="${SEED:-42}"

  # 机器人脚部碰撞版本切换：
  #   fourpoints  -> temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd（四球脚部碰撞）
  #   box         -> g1_29dof_with_dex3_rev_1_0.usd（长方体脚部碰撞）
  ROBOT_COLLIDER_MODE="${ROBOT_COLLIDER_MODE:-box}"
  if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
    export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
  else
    export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
  fi
  echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
  echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"

  python "${SCRIPT_DIR}/sim_main.py" \
    --device cpu \
    --enable_cameras \
    --env_config_yaml "${ENV_CONFIG_YAML}" \
    --input_source pico_twist2 \
    --gmt_backend twist2 \
    --task "${TASK_NAME}" \
    --robot_type g129 \
    --enable_dex3_dds \
    --image_transport xrobot \
    --image_xrobot_host 10.42.0.35 \
    --image_xrobot_port 12345 \
    --image_xrobot_bitrate 16777216 \
    --image_fps 30 \
    --image_xrobot_ffmpeg /usr/bin/ffmpeg \
    --recording_save_dir /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/0319/tw \
    --seed "${SEED}"
