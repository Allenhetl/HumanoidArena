#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

PYTHON_BIN="python"
TASK_NAME="Isaac-Move-Football-Single-G129-Dex3-Wholebody"
ENV_CONFIG_YAML="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/common_env_config/football_single_sonic.yaml"
RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box"
SEED="42"
ENABLE_CAMERAS=1
ENABLE_DEX3_DDS=1
HEADLESS=1
SONIC_REDIS_HOST="localhost"
SONIC_REDIS_PORT="6379"
SONIC_ENCODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
SONIC_DECODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
IMAGE_TRANSPORT="xrobot"
IMAGE_XROBOT_HOST="10.42.0.35"
IMAGE_XROBOT_PORT="12345"
IMAGE_XROBOT_BITRATE="2097152"
IMAGE_FPS="30"
IMAGE_XROBOT_FFMPEG="/usr/bin/ffmpeg"
RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data/HOI_football_v2/sonic/tw"

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [ ! -f "${SONIC_ENCODER_PATH}" ]; then
  echo "Error: SONIC encoder not found: ${SONIC_ENCODER_PATH}"
  exit 1
fi
if [ ! -f "${SONIC_DECODER_PATH}" ]; then
  echo "Error: SONIC decoder not found: ${SONIC_DECODER_PATH}"
  exit 1
fi

redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  human_smplx_data_unitree_g1_with_hands \
  human_info_unitree_g1_with_hands \
  gmr_full_qpos_unitree_g1_with_hands \
  gmr_joint_pos_unitree_g1_with_hands \
  gmr_joint_vel_unitree_g1_with_hands \
  gmr_body_pos_unitree_g1_with_hands \
  gmr_body_quat_w_unitree_g1_with_hands \
  gmr_frame_index_unitree_g1_with_hands \
  recording_control_unitree_g1_with_hands \
  isaac_reset_trigger \
  isaac_reset_complete_unitree_g1_with_hands \
  isaac_input_ready_sonic_joint29_unitree_g1_with_hands \
  controller_data \
  t_action >/dev/null 2>&1 || true

if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[sonic_joint29] task=${TASK_NAME}"
echo "[sonic_joint29] env_config=${ENV_CONFIG_YAML}"
echo "[sonic_joint29] seed=${SEED}"
echo "[sonic_joint29] encoder=${SONIC_ENCODER_PATH}"
echo "[sonic_joint29] decoder=${SONIC_DECODER_PATH}"
echo "[recording] save_dir=${RECORDING_SAVE_DIR}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --task "${TASK_NAME}"
  --robot_type "${ROBOT_TYPE}"
  --input_source pico_twist2
  --gmt_backend sonic_joint29
  --sonic_pose_source redis
  --sonic_redis_host "${SONIC_REDIS_HOST}"
  --sonic_redis_port "${SONIC_REDIS_PORT}"
  --sonic_encoder_path "${SONIC_ENCODER_PATH}"
  --sonic_decoder_path "${SONIC_DECODER_PATH}"
  --image_transport "${IMAGE_TRANSPORT}"
  --image_xrobot_host "${IMAGE_XROBOT_HOST}"
  --image_xrobot_port "${IMAGE_XROBOT_PORT}"
  --image_xrobot_bitrate "${IMAGE_XROBOT_BITRATE}"
  --image_fps "${IMAGE_FPS}"
  --image_xrobot_ffmpeg "${IMAGE_XROBOT_FFMPEG}"
  --recording_save_dir "${RECORDING_SAVE_DIR}"
  --seed "${SEED}"
)

if [ "${ENABLE_CAMERAS}" = "1" ]; then
  cmd+=(--enable_cameras)
fi
if [ "${ENABLE_DEX3_DDS}" = "1" ]; then
  cmd+=(--enable_dex3_dds)
fi
if [ "${HEADLESS}" = "1" ]; then
  cmd+=(--headless)
fi

if [ "$#" -gt 0 ]; then
  cmd+=("$@")
fi

exec "${cmd[@]}"
