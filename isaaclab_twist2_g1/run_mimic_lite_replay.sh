#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAACLAB_CONDA_ENV_NAME="${ISAACLAB_CONDA_ENV_NAME:-unitree_sim_env_isaaclab5_0}"
source "${SCRIPT_DIR}/script/common/runtime_paths.sh"
cd "${SCRIPT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ISAACLAB_PYTHON}}"

ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-${ISAACLAB_ROOT}/tasks/common_env_config/opendoor_sonic.yaml}"
RUN_DEVICE="${RUN_DEVICE:-cpu}"
ROBOT_TYPE="${ROBOT_TYPE:-g129}"
ROBOT_COLLIDER_MODE="${ROBOT_COLLIDER_MODE:-box}"
SEED="${SEED:-42}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
ENABLE_DEX3_DDS="${ENABLE_DEX3_DDS:-1}"
HEADLESS="${HEADLESS:-0}"
DRY_RUN="${DRY_RUN:-0}"
VIEWPORT_CAMERA="${VIEWPORT_CAMERA:-front}"
MIMIC_LITE_REDIS_HOST="${MIMIC_LITE_REDIS_HOST:-localhost}"
MIMIC_LITE_REDIS_PORT="${MIMIC_LITE_REDIS_PORT:-6379}"
export MIMIC_LITE_DEBUG="${MIMIC_LITE_DEBUG:-1}"
export MIMIC_LITE_LOG_EVERY="${MIMIC_LITE_LOG_EVERY:-50}"
export MIMIC_LITE_RENDER_INTERVAL="${MIMIC_LITE_RENDER_INTERVAL:-1}"
export MIMIC_LITE_STARTUP_BLEND_SEC="${MIMIC_LITE_STARTUP_BLEND_SEC:-0}"
export MIMIC_LITE_ROBOT_CFG="${MIMIC_LITE_ROBOT_CFG:-1}"
export MIMIC_LITE_USE_SELF_TORQUE="${MIMIC_LITE_USE_SELF_TORQUE:-0}"
MIMIC_LITE_ONNX_PATH="${MIMIC_LITE_ONNX_PATH:-${SCRIPT_DIR}/assets/checkpoints/mimic_lite/policy-xua2csee-4000.onnx}"
MIMIC_LITE_YAML_PATH="${MIMIC_LITE_YAML_PATH:-${SCRIPT_DIR}/assets/checkpoints/mimic_lite/policy-xua2csee-4000.yaml}"
IMAGE_TRANSPORT="${IMAGE_TRANSPORT:-xrobot}"
IMAGE_XROBOT_HOST="${IMAGE_XROBOT_HOST:-10.42.0.23}"
IMAGE_XROBOT_PORT="${IMAGE_XROBOT_PORT:-12345}"
IMAGE_XROBOT_BITRATE="${IMAGE_XROBOT_BITRATE:-2097152}"
IMAGE_FPS="${IMAGE_FPS:-30}"
IMAGE_XROBOT_FFMPEG="${IMAGE_XROBOT_FFMPEG:-/usr/bin/ffmpeg}"
RECORDING_SAVE_DIR="${RECORDING_SAVE_DIR:-${SCRIPT_DIR}/recording_data/HSI_open_door/mimic_lite/replay_rerecord}"
LOG_DIR="${SCRIPT_DIR}/logs"

# Replay-specific configuration
REPLAY_FILE="${REPLAY_FILE:-${SCRIPT_DIR}/recording_data/HSI_open_door/mimic_lite/zz/Isaac-Move-Open-Door-G129-Dex3-Wholebody_mimic_lite_1784969894028271.npz}"
REPLAY_MODE="${REPLAY_MODE:-direct_replay}"
REPLAY_LOOP="${REPLAY_LOOP:-0}"
RECORD_DURING_REPLAY="${RECORD_DURING_REPLAY:-0}"
EXIT_WHEN_REPLAY_COMPLETE="${EXIT_WHEN_REPLAY_COMPLETE:-1}"

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

if [ ! -f "${MIMIC_LITE_ONNX_PATH}" ]; then
  echo "Error: MimicLite ONNX not found: ${MIMIC_LITE_ONNX_PATH}"
  exit 1
fi
if [ ! -f "${MIMIC_LITE_YAML_PATH}" ]; then
  echo "Error: MimicLite YAML not found: ${MIMIC_LITE_YAML_PATH}"
  exit 1
fi
if [ ! -f "${REPLAY_FILE}" ]; then
  echo "Error: Replay file not found: ${REPLAY_FILE}"
  exit 1
fi

if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
  export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi

echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"
echo "[mimic_lite] task=${TASK_NAME}"
echo "[mimic_lite] env_config=${ENV_CONFIG_YAML}"
echo "[mimic_lite] onnx=${MIMIC_LITE_ONNX_PATH}"
echo "[mimic_lite] yaml=${MIMIC_LITE_YAML_PATH}"
echo "[mimic_lite] debug=${MIMIC_LITE_DEBUG} log_every=${MIMIC_LITE_LOG_EVERY} render_interval=${MIMIC_LITE_RENDER_INTERVAL} robot_cfg=${MIMIC_LITE_ROBOT_CFG} self_torque=${MIMIC_LITE_USE_SELF_TORQUE}"
echo "[replay] file=${REPLAY_FILE}"
echo "[replay] mode=${REPLAY_MODE} loop=${REPLAY_LOOP} record_during_replay=${RECORD_DURING_REPLAY} exit_when_complete=${EXIT_WHEN_REPLAY_COMPLETE}"
echo "[recording] save_dir=${RECORDING_SAVE_DIR}"

mkdir -p "${RECORDING_SAVE_DIR}"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SIM_LOG="${LOG_DIR}/sim_main_mimic_lite_replay_${RUN_TS}.log"
echo "[log] file=${SIM_LOG}"

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sim_main.py"
  --device "${RUN_DEVICE}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --task "${TASK_NAME}"
  --robot_type "${ROBOT_TYPE}"
  --action_source replay
  --input_source replay
  --gmt_backend mimic_lite
  --replay_file "${REPLAY_FILE}"
  --replay_mode "${REPLAY_MODE}"
  --mimic_lite_redis_host "${MIMIC_LITE_REDIS_HOST}"
  --mimic_lite_redis_port "${MIMIC_LITE_REDIS_PORT}"
  --mimic_lite_onnx_path "${MIMIC_LITE_ONNX_PATH}"
  --mimic_lite_yaml_path "${MIMIC_LITE_YAML_PATH}"
  --image_transport "${IMAGE_TRANSPORT}"
  --image_xrobot_host "${IMAGE_XROBOT_HOST}"
  --image_xrobot_port "${IMAGE_XROBOT_PORT}"
  --image_xrobot_bitrate "${IMAGE_XROBOT_BITRATE}"
  --image_fps "${IMAGE_FPS}"
  --image_xrobot_ffmpeg "${IMAGE_XROBOT_FFMPEG}"
  --recording_save_dir "${RECORDING_SAVE_DIR}"
  --viewport_camera "${VIEWPORT_CAMERA}"
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
if [ "${RECORD_DURING_REPLAY}" = "1" ]; then
  cmd+=(--record_during_replay)
fi
if [ "${EXIT_WHEN_REPLAY_COMPLETE}" = "1" ]; then
  cmd+=(--exit_when_replay_complete)
fi

if [ "$#" -gt 0 ]; then
  cmd+=("$@")
fi

if [ "${DRY_RUN}" = "1" ]; then
  printf '[dry_run] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

{
  echo "[$(date '+%F %T')] Starting MimicLite replay"
  echo "[$(date '+%F %T')] task=${TASK_NAME}"
  echo "[$(date '+%F %T')] env_config=${ENV_CONFIG_YAML}"
  echo "[$(date '+%F %T')] replay_file=${REPLAY_FILE}"
  echo "[$(date '+%F %T')] replay_mode=${REPLAY_MODE}"
  echo "[$(date '+%F %T')] recording_save_dir=${RECORDING_SAVE_DIR}"
  echo "[$(date '+%F %T')] log_file=${SIM_LOG}"
} | tee -a "${SIM_LOG}"

exec "${cmd[@]}" 2>&1 | tee -a "${SIM_LOG}"

# Usage examples:
#   Direct replay (use recorded 29-DoF targets, no ONNX inference):
#     ./run_mimic_lite_replay.sh
#
#   Inference replay (rerun ONNX with recorded command/policy inputs):
#     REPLAY_MODE=inference_replay ./run_mimic_lite_replay.sh
#
#   Replay a different file:
#     REPLAY_FILE=/path/to/other.npz ./run_mimic_lite_replay.sh
#
#   Replay with rerecord (save new npz during replay):
#     RECORD_DURING_REPLAY=1 ./run_mimic_lite_replay.sh
#
#   Loop replay (don't exit at end):
#     REPLAY_LOOP=1 EXIT_WHEN_REPLAY_COMPLETE=0 ./run_mimic_lite_replay.sh
