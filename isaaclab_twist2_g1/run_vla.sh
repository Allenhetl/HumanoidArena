#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-}"
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-}"
TWIST2_MODEL_PATH="${TWIST2_MODEL_PATH:-${SCRIPT_DIR}/../TWIST2/assets/ckpts/twist2_1017_20k.onnx}"
TASK_NAME="${TASK_NAME:-Isaac-Move-Football-G129-Dex3-Wholebody}"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/twist2_default.yaml}"

if [[ -z "${OPENPI_CHECKPOINT}" ]]; then
  echo "OPENPI_CHECKPOINT is required"
  exit 1
fi

if [[ -z "${LANGUAGE_INSTRUCTION}" ]]; then
  echo "LANGUAGE_INSTRUCTION is required"
  exit 1
fi

cd "${SCRIPT_DIR}"
python sim_main.py \
  --input_source vla \
  --gmt_backend twist2 \
  --env_config_yaml "${ENV_CONFIG_YAML}" \
  --task "${TASK_NAME}" \
  --robot_type g129 \
  --enable_dex3_dds \
  --openpi_checkpoint "${OPENPI_CHECKPOINT}" \
  --language_instruction "${LANGUAGE_INSTRUCTION}" \
  --twist2_model_path "${TWIST2_MODEL_PATH}" \
  "$@"
