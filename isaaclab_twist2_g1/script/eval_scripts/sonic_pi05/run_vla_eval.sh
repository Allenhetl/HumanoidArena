#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROBOT_USD_OVERRIDE="${ISAACLAB_ROOT}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/football_single_sonic.yaml}"
ISAAC_DEVICE="cpu"
HEADLESS=1
MAX_STEPS=1000
VIDEO_FPS=30
POST_TERMINATION_RECORD_STEPS=50
ROBOT_TYPE="unitree_g1_refpose_v3_1"
SONIC_VLA_ROOT_ROT6D_LAYOUT="row"
SONIC_VLA_ROOT_MAX_DELTA_DEG="26.0"
LEROBOT_VLA_RECORD_OUTPUTS="${LEROBOT_VLA_RECORD_OUTPUTS:-1}"

SONIC_ENCODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
SONIC_DECODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"

SERVER_PYTHON="${SERVER_PYTHON:-/ai/Yichi/0_Systems/miniconda3/envs/lerobot2/bin/python}"
SERVER_SCRIPT="${ISAACLAB_ROOT}/../lerobot/scripts/serve_lerobot_vla_http.py"
SERVER_DEVICE="cuda:0"
SERVER_HOST="127.0.0.1"
SERVER_PORT=8443
SERVER_SCHEME="http"
SERVER_READY_TIMEOUT=60
LEROBOT_SERVER_TIMEOUT=5.0
LEROBOT_VERIFY_SSL=0
TLS_CERT_FILE=""
TLS_KEY_FILE=""

REPEATS_PER_SEED=1
SEEDS=($(seq 0 99))
echo "${SEEDS[@]}"

load_task_name_from_yaml() {
  python - "${ISAACLAB_ROOT}/tasks/common_env_config/loader.py" "${1}" <<'PY'
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


MODEL_PATHS=(
  "/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/train/act_sonic_football_rand_0414_64_40/checkpoints/last/pretrained_model"
)

RESULTS_DIR="${SCRIPT_DIR}/eval_results/act_sonic_football_batchtest_0418$(date +%Y%m%d_%H%M%S)"

ARGS=(
  --task "${TASK_NAME}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --repeats_per_seed "${REPEATS_PER_SEED}"
  --max_steps "${MAX_STEPS}"
  --video_fps "${VIDEO_FPS}"
  --post_termination_record_steps "${POST_TERMINATION_RECORD_STEPS}"
  --robot_type "${ROBOT_TYPE}"
  --sonic_encoder_path "${SONIC_ENCODER_PATH}"
  --sonic_decoder_path "${SONIC_DECODER_PATH}"
  --sonic_vla_root_rot6d_layout "${SONIC_VLA_ROOT_ROT6D_LAYOUT}"
  --sonic_vla_root_max_delta_deg "${SONIC_VLA_ROOT_MAX_DELTA_DEG}"
  --results_dir "${RESULTS_DIR}"
  --isaac_device "${ISAAC_DEVICE}"
  --server_python "${SERVER_PYTHON}"
  --server_script "${SERVER_SCRIPT}"
  --server_device "${SERVER_DEVICE}"
  --server_host "${SERVER_HOST}"
  --server_port "${SERVER_PORT}"
  --server_scheme "${SERVER_SCHEME}"
  --server_ready_timeout "${SERVER_READY_TIMEOUT}"
  --lerobot_server_timeout "${LEROBOT_SERVER_TIMEOUT}"
)

if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi

if [[ "${LEROBOT_VERIFY_SSL}" == "1" ]]; then
  ARGS+=(--lerobot_server_verify_ssl)
fi

if [[ -n "${TLS_CERT_FILE}" ]]; then
  ARGS+=(--tls_cert_file "${TLS_CERT_FILE}")
fi

if [[ -n "${TLS_KEY_FILE}" ]]; then
  ARGS+=(--tls_key_file "${TLS_KEY_FILE}")
fi

for seed in "${SEEDS[@]}"; do
  ARGS+=(--seed "${seed}")
done

for model_path in "${MODEL_PATHS[@]}"; do
  ARGS+=(--model-path "${model_path}")
done

cd "${SCRIPT_DIR}"
  
if [[ "${LEROBOT_VLA_RECORD_OUTPUTS}" == "1" ]]; then
  export LEROBOT_VLA_RECORD_OUTPUTS=1
fi

#SONIC_VLA_ROOT_DEBUG=1 \
#SONIC_VLA_ROOT_JUMP_L2_THRESHOLD=0.20 \
#SONIC_VLA_ROOT_JUMP_DEG_THRESHOLD=15.0 \
#SONIC_VLA_USE_HEADING_ALIGN=1 \
#SONIC_OUTPUT_DELAY_STEPS=0 \
python eval_vla_suite.py "${ARGS[@]}"
