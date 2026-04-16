#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LEROBOT_SERVER_URL="${LEROBOT_SERVER_URL:-http://127.0.0.1:8443}"
LEROBOT_SERVER_TIMEOUT="${LEROBOT_SERVER_TIMEOUT:-5.0}"
LEROBOT_VERIFY_SSL="${LEROBOT_VERIFY_SSL:-0}"
LEROBOT_POLICY_DEVICE="${LEROBOT_POLICY_DEVICE:-cuda:0}"
TWIST2_MODEL_PATH="${TWIST2_MODEL_PATH:-${SCRIPT_DIR}/../TWIST2/assets/ckpts/twist2_1017_20k.onnx}"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/football_single_twist2_vla.yaml}"
SEED="${SEED:-42}"
#export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"

load_task_name_from_yaml() {
  python - "${SCRIPT_DIR}/tasks/common_env_config/loader.py" "${1}" <<'PY'
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

if [[ -z "${LEROBOT_SERVER_URL}" ]]; then
  echo "LEROBOT_SERVER_URL is required"
  echo "Example:"
  echo "  LEROBOT_SERVER_URL=http://127.0.0.1:8443 bash run_vla.sh"
  exit 1
fi

cd "${SCRIPT_DIR}"
SSL_ARGS=()
if [[ "${LEROBOT_VERIFY_SSL}" == "1" ]]; then
  SSL_ARGS+=(--lerobot_server_verify_ssl)
fi

python sim_main.py \
  --input_source vla \
  --gmt_backend twist2 \
  --headless \
  --env_config_yaml "${ENV_CONFIG_YAML}" \
  --task "${TASK_NAME}" \
  --robot_type g129 \
  --seed "${SEED}" \
  --enable_cameras \
  --model_path "${TWIST2_MODEL_PATH}" \
  --lerobot_server_url "${LEROBOT_SERVER_URL}" \
  --lerobot_server_timeout "${LEROBOT_SERVER_TIMEOUT}" \
  --lerobot_policy_device "${LEROBOT_POLICY_DEVICE}" \
  "${SSL_ARGS[@]}" \
  "$@"
