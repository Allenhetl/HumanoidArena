#!/usr/bin/env bash
set -euo pipefail

# Sonic-only cross-GMT evaluation:
#   sonic_open_door -> refpose/open_door 100000 VLA
#   sonic_football  -> HOI_football 100000 VLA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPEN_DOOR_MODEL_PATH="${OPEN_DOOR_MODEL_PATH:-/ai/Yichi/zikang/mimiclite_v3.1/0726/diffusion_mimic_lite_refpose_v3_1/checkpoints/100000/pretrained_model}"
FOOTBALL_MODEL_PATH="${FOOTBALL_MODEL_PATH:-/ai/Yichi/zikang/mimiclite_v3.1/0726/diffusion_HOI_football/checkpoints/100000/pretrained_model}"
for model_path in "${OPEN_DOOR_MODEL_PATH}" "${FOOTBALL_MODEL_PATH}"; do
  for required_file in config.json model.safetensors; do
    [[ -s "${model_path}/${required_file}" ]] || {
      echo "Error: incomplete VLA model: ${model_path}/${required_file}" >&2
      exit 1
    }
  done
done

# This is the Sonic GMT backend, never the MimicLite backend configuration.
unset MIMIC_LITE_ROBOT_CFG
unset VLA_MIMICLITE_YAW_CALIB_DEG
export NUM_WORKERS="${NUM_WORKERS:-2}"
export RESUME_LATEST="${RESUME_LATEST:-0}"
export DRY_RUN="${DRY_RUN:-0}"
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then export SEEDS_OVERRIDE; fi
if [[ -n "${REPEATS_PER_SEED:-}" ]]; then export REPEATS_PER_SEED; fi
if [[ -n "${MAX_STEPS:-}" ]]; then export MAX_STEPS; fi

RESULTS_ROOT="${RESULTS_ROOT:-${ISAACLAB_ROOT}/script/eval_results_cross_gmt_refpose_v3_1}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
UNIFIED_RESULTS_DIR="${UNIFIED_RESULTS_DIR:-${RESULTS_ROOT}/cross_gmt_refpose_v3_1_100000_sonic_only_${RUN_TS}}"
mkdir -p "${UNIFIED_RESULTS_DIR}"

run_stage() {
  local task="$1" launcher="$2" model_path="$3"
  local stage_name="sonic_${task}"
  local stage_results_dir="${UNIFIED_RESULTS_DIR}/${stage_name}"

  echo "[cross_gmt_sonic_only] stage=${stage_name}"
  echo "[cross_gmt_sonic_only] VLA=${model_path}"
  echo "[cross_gmt_sonic_only] GMT backend=sonic"
  echo "[cross_gmt_sonic_only] results=${stage_results_dir}"
  env \
    MODEL_PATHS_CSV="${model_path}" \
    RESULTS_TAG="${stage_name}" \
    RESULTS_DIR="${stage_results_dir}" \
    bash "${launcher}"
}

run_stage "open_door" \
  "${ISAACLAB_ROOT}/script/eval_scripts/sonic/HSI_open_door_run_vla_eval_parallel.sh" \
  "${OPEN_DOOR_MODEL_PATH}"
run_stage "football" \
  "${ISAACLAB_ROOT}/script/eval_scripts/sonic/HOI_football_run_vla_eval_parallel.sh" \
  "${FOOTBALL_MODEL_PATH}"

echo "[cross_gmt_sonic_only] completed: ${UNIFIED_RESULTS_DIR}"
