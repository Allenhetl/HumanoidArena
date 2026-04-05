#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export ROBOT_USD_OVERRIDE="${ISAACLAB_ROOT}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
TASK_NAME="Isaac-Move-Football-Single-G129-Dex3-Wholebody"
ENV_CONFIG_YAML="tasks/common_env_config/football_single_twist2_vla.yaml"
ISAAC_DEVICE="cpu"
HEADLESS=1
MAX_STEPS=500
VIDEO_FPS=30
ROBOT_TYPE="g129"

TWIST2_MODEL_PATH="${ISAACLAB_ROOT}/../TWIST2/assets/ckpts/twist2_1017_20k.onnx"

SERVER_PYTHON="/home/dreams/miniconda3/envs/lerobot/bin/python"
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

REPEATS_PER_SEED=3
SEEDS=(42 43 44 45 46 47 48 49 50 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41)

MODEL_PATHS=(
  "/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/train/diffusion_twist2_0401/checkpoints/last/pretrained_model"
)

RESULTS_DIR="${SCRIPT_DIR}/eval_results/dp_rand_$(date +%Y%m%d_%H%M%S)"

ARGS=(
  --task "${TASK_NAME}"
  --env_config_yaml "${ENV_CONFIG_YAML}"
  --repeats_per_seed "${REPEATS_PER_SEED}"
  --max_steps "${MAX_STEPS}"
  --video_fps "${VIDEO_FPS}"
  --robot_type "${ROBOT_TYPE}"
  --twist2_model_path "${TWIST2_MODEL_PATH}"
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
python eval_vla_suite.py "${ARGS[@]}"
