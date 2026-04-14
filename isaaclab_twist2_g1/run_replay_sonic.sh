#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# ------------------------------------------------------------------
# User config: edit here
# ------------------------------------------------------------------
PYTHON_BIN="python"
DEFAULT_ISAACLAB_PY="/home/dreams/miniconda3/envs/unitree_sim_env_isaaclab5_0/bin/python"
# Double-desk HOI recording (edit REPLAY_FILE for other runs)
REPLAY_FILE="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic_v3/zk/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1776167470033014.npz"
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775894660832965.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775894789682027.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775894841888126.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775894979774767.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775895024789299.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775912992795677.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775913318734262.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775913369100656.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775913586857113.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775914283717499.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775914526205379.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775914801537965.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775914895501671.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/zk/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775907447485753.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/zk/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775908196051364.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/zk/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775908368386789.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/zk/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775908558332072.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/zk/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775908638445975.npz
  # - isaaclab_twist2_g1/recording_data/HOI_football_v2/sonic/zk/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1775909094537531.npz
# REPLAY_FILE="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/HOI_open_door/sonic/zk/Isaac-Move-Open-Door-G129-Dex3-Wholebody_sonic_1775975894226796.npz"
REPLAY_MODE="direct_replay"   # inference_replay | direct_replay
REPLAY_LOOP=0                    # 1 | 0
# ENV_CONFIG_YAML="tasks/common_env_config/opendoor_sonic.yaml"
# ENV_CONFIG_YAML="tasks/common_env_config/doubledesk_sonic.yaml"
ENV_CONFIG_YAML="tasks/common_env_config/football_single_sonic.yaml"
RUN_DEVICE="cpu"
ROBOT_TYPE="g129"
ROBOT_COLLIDER_MODE="box"        # box | fourpoints
ENABLE_CAMERAS=1
ENABLE_DEX3_DDS=1
HEADLESS=0
SONIC_ENCODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
SONIC_DECODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
IMAGE_TRANSPORT="zmq"
IMAGE_ZMQ_PORT="5555"
IMAGE_FPS="30"
LOG_DIR="${SCRIPT_DIR}/logs/sonic_replay"
SEED="42"

export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

load_task_name_from_yaml() {
  "${PYTHON_BIN}" - "${SCRIPT_DIR}/tasks/common_env_config/loader.py" "${1}" <<'PY'
import importlib.util
import pathlib
import sys

loader_path = pathlib.Path(sys.argv[1])
config_path = sys.argv[2]
spec = importlib.util.spec_from_file_location("common_env_config_loader", loader_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

task_name = module.get_env_config_task_name(config_path)
if not task_name:
    raise SystemExit(
        f"Error: env config YAML must define a top-level 'task_name': {config_path}"
    )
print(task_name)
PY
}

TASK_NAME="${TASK_NAME:-$(load_task_name_from_yaml "${ENV_CONFIG_YAML}")}"

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


export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"


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
echo "[sonic replay] seed=${SEED}"
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
