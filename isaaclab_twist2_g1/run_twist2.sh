#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# ------------------------------------------------------------------
# User config: edit here
# ------------------------------------------------------------------
PYTHON_BIN="python"
TASK_NAME="Isaac-Move-Football-Single-G129-Dex3-Wholebody"
ENV_CONFIG_YAML="tasks/common_env_config/twist2_default.yaml"
RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box"   # box | fourpoints
SEED="42"
ENABLE_CAMERAS=1
ENABLE_DEX3_DDS=1
HEADLESS=1
IMAGE_TRANSPORT="xrobot"
IMAGE_XROBOT_HOST="10.42.0.35"
IMAGE_XROBOT_PORT="12345"
IMAGE_XROBOT_BITRATE="16777216"
IMAGE_FPS="30"
IMAGE_XROBOT_FFMPEG="/usr/bin/ffmpeg"
RECORDING_SAVE_DIR="${SCRIPT_DIR}/recording_data//0401/twist2/zk"

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action \
  isaac_reset_trigger \
  isaac_input_ready_twist2_unitree_g1_with_hands >/dev/null 2>&1 || true

if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[twist2] task=${TASK_NAME}"
echo "[twist2] env_config=${ENV_CONFIG_YAML}"
echo "[twist2] recording_save_dir=${RECORDING_SAVE_DIR}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --input_source pico_twist2
  --gmt_backend twist2
  --task "${TASK_NAME}"
  --robot_type "${ROBOT_TYPE}"
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

exec "${cmd[@]}"


# 可切换：
#一般沙袋: Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody
#吊挂沙袋: Isaac-Move-Boxing-Bag-Hanging-G129-Dex3-Wholebody
#足球: Isaac-Move-Football-G129-Dex3-Wholebody
#双桌面拾放: Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody
#Push-T: Isaac-Push-T-G129-Dex3-Wholebody
#客厅交互：Isaac-Move-ArtVIP-Livingroom-G129-Dex3-Wholebody
#客厅抓杯：Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody
#三级台阶平台：Isaac-Move-Three-Step-Platform-G129-Dex3-Wholebody
#ready
#TASK_NAME="${TASK_NAME:-Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-Three-Step-Platform-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-ArtVIP-Livingroom-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"
#TASK_NAME="${TASK_NAME:-Isaac-Move-Football-Single-G129-Dex3-Wholebody}"