#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# ------------------------------------------------------------------
# User config: edit here
# ------------------------------------------------------------------
PYTHON_BIN="python"
DEFAULT_ISAACLAB_PY="/home/dreams/miniconda3/envs/env_isaaclab_510_yb/bin/python"
REPLAY_FILE="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1774962608663017.npz"
REPLAY_MODE="inference_replay"   # inference_replay | direct_replay
REPLAY_LOOP=0                    # 1 | 0
TASK_NAME=""                     # 留空则从 replay 文件读取
ENV_CONFIG_YAML="tasks/common_env_config/sonic_default.yaml"
RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box"        # box | fourpoints
ENABLE_CAMERAS=1
ENABLE_DEX3_DDS=1
HEADLESS=1
SONIC_ENCODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
SONIC_DECODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
IMAGE_TRANSPORT="zmq"
IMAGE_ZMQ_PORT="5555"
IMAGE_FPS="30"
LOG_DIR="${SCRIPT_DIR}/logs/sonic_replay"

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [ ! -f "${REPLAY_FILE}" ]; then
  echo "Error: replay file not found: ${REPLAY_FILE}"
  exit 1
fi
if [ "${REPLAY_MODE}" != "direct_replay" ] && [ "${REPLAY_MODE}" != "inference_replay" ]; then
  echo "Error: replay mode must be direct_replay or inference_replay, got: ${REPLAY_MODE}"
  exit 1
fi
if [ ! -f "${SONIC_ENCODER_PATH}" ]; then
  echo "Error: SONIC encoder not found: ${SONIC_ENCODER_PATH}"
  exit 1
fi
if [ ! -f "${SONIC_DECODER_PATH}" ]; then
  echo "Error: SONIC decoder not found: ${SONIC_DECODER_PATH}"
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import onnxruntime" >/dev/null 2>&1; then
  if [ -x "${DEFAULT_ISAACLAB_PY}" ] && "${DEFAULT_ISAACLAB_PY}" -c "import onnxruntime" >/dev/null 2>&1; then
    PYTHON_BIN="${DEFAULT_ISAACLAB_PY}"
  fi
fi
if ! "${PYTHON_BIN}" -c "import onnxruntime" >/dev/null 2>&1; then
  echo "Error: usable python with onnxruntime not found"
  exit 1
fi

REPLAY_FILE="$(realpath "${REPLAY_FILE}")"
if [ -z "${TASK_NAME}" ]; then
  TASK_NAME="$("${PYTHON_BIN}" - "${REPLAY_FILE}" <<'PY'
import sys
import numpy as np

replay_file = sys.argv[1]
with np.load(replay_file, allow_pickle=True) as data:
    task = data.get("task")
    if task is None:
        raise SystemExit("Replay file missing 'task' metadata")
    if hasattr(task, "item"):
        task = task.item()
    print(task)
PY
)"
fi

if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi

redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  isaac_input_ready_sonic_unitree_g1_with_hands \
  controller_data \
  t_action >/dev/null 2>&1 || true

mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SIM_LOG="${LOG_DIR}/sim_main_${RUN_TS}.log"

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[sonic replay] file=${REPLAY_FILE}"
echo "[sonic replay] mode=${REPLAY_MODE}"
echo "[sonic replay] task=${TASK_NAME}"
echo "[sonic replay] encoder=${SONIC_ENCODER_PATH}"
echo "[sonic replay] decoder=${SONIC_DECODER_PATH}"
echo "[sonic replay] python=${PYTHON_BIN}"
echo "[sonic replay] log=${SIM_LOG}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --task "${TASK_NAME}"
  --robot_type "${ROBOT_TYPE}"
  --input_source replay
  --gmt_backend sonic
  --sonic_encoder_path "${SONIC_ENCODER_PATH}"
  --sonic_decoder_path "${SONIC_DECODER_PATH}"
  --replay_file "${REPLAY_FILE}"
  --replay_mode "${REPLAY_MODE}"
  --image_transport "${IMAGE_TRANSPORT}"
  --image_fps "${IMAGE_FPS}"
  --image_zmq_port "${IMAGE_ZMQ_PORT}"
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
if [ "${REPLAY_LOOP}" = "1" ]; then
  cmd+=(--replay_loop)
fi

{
  echo "[$(date '+%F %T')] Starting SONIC replay"
  echo "[$(date '+%F %T')] file=${REPLAY_FILE}"
  echo "[$(date '+%F %T')] mode=${REPLAY_MODE}"
  echo "[$(date '+%F %T')] task=${TASK_NAME}"
  echo "[$(date '+%F %T')] python=${PYTHON_BIN}"
} | tee -a "${SIM_LOG}"

"${cmd[@]}" 2>&1 | tee -a "${SIM_LOG}"
