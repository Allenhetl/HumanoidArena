#!/usr/bin/env bash
set -euo pipefail

# Cross-GMT evaluation for task-specific corrected MimicLite VLA checkpoints.
#
# Task-specific VLA models are evaluated in four conditions, sequentially:
#   1. refpose 100000       -> twist2 GMT backend -> open_door
#   2. HOI_football 100000  -> twist2 GMT backend -> football
#   3. refpose 100000       -> sonic  GMT backend -> open_door
#   4. HOI_football 100000  -> sonic  GMT backend -> football
#
# The backend-specific launcher determines both its GMT route and matching task
# YAML. Do not use the mimic_lite launchers here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPEN_DOOR_MODEL_PATH="${OPEN_DOOR_MODEL_PATH:-/ai/Yichi/zikang/mimiclite_v3.1/0726/diffusion_mimic_lite_refpose_v3_1/checkpoints/100000/pretrained_model}"
FOOTBALL_MODEL_PATH="${FOOTBALL_MODEL_PATH:-/ai/Yichi/zikang/mimiclite_v3.1/0726/diffusion_HOI_football/checkpoints/100000/pretrained_model}"
for model_path in "${OPEN_DOOR_MODEL_PATH}" "${FOOTBALL_MODEL_PATH}"; do
  for required_file in config.json model.safetensors; do
    if [[ ! -s "${model_path}/${required_file}" ]]; then
      echo "Error: incomplete VLA model; missing/empty ${model_path}/${required_file}" >&2
      exit 1
    fi
  done
done

# This is deliberately NOT the MimicLite GMT. Ensure a caller's prior in-GMT
# environment cannot bind BeyondMimic PD parameters to twist2 or sonic runs.
unset MIMIC_LITE_ROBOT_CFG
unset VLA_MIMICLITE_YAW_CALIB_DEG

export NUM_WORKERS="${NUM_WORKERS:-2}"
export RESUME_LATEST="${RESUME_LATEST:-0}"
export DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then export SEEDS_OVERRIDE; fi
if [[ -n "${REPEATS_PER_SEED:-}" ]]; then export REPEATS_PER_SEED; fi
if [[ -n "${MAX_STEPS:-}" ]]; then export MAX_STEPS; fi

BATCH_RESULTS_ROOT="${BATCH_RESULTS_ROOT:-${ISAACLAB_ROOT}/script/eval_results_cross_gmt_refpose_v3_1}"
RESULTS_TAG_PREFIX="${RESULTS_TAG_PREFIX:-cross_gmt_refpose_v3_1_100000_twist2_sonic}"
BATCH_RUN_TS="${BATCH_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
UNIFIED_RESULTS_DIR="${BATCH_RESULTS_ROOT}/${RESULTS_TAG_PREFIX}_${BATCH_RUN_TS}"
mkdir -p "${UNIFIED_RESULTS_DIR}"

run_stage() {
  local backend="$1"
  local task="$2"
  local launcher="$3"
  local model_path="$4"
  local stage_name="${backend}_${task}"
  local stage_results_dir="${UNIFIED_RESULTS_DIR}/${stage_name}"

  echo ""
  echo "============================================================"
  echo "[cross_gmt_refpose_v31_100000] stage=${stage_name}"
  echo "[cross_gmt_refpose_v31_100000] VLA=${model_path}"
  echo "[cross_gmt_refpose_v31_100000] GMT backend=${backend}"
  echo "[cross_gmt_refpose_v31_100000] launcher=${launcher}"
  echo "[cross_gmt_refpose_v31_100000] results=${stage_results_dir}"
  echo "============================================================"

  # RESULTS_DIR prevents accidental merging/resuming with another backend.
  env \
    MODEL_PATHS_CSV="${model_path}" \
    RESULTS_TAG="${stage_name}" \
    RESULTS_DIR="${stage_results_dir}" \
    bash "${launcher}"
}

echo "[cross_gmt_refpose_v31_100000] open_door model=${OPEN_DOOR_MODEL_PATH}"
echo "[cross_gmt_refpose_v31_100000] football model=${FOOTBALL_MODEL_PATH}"
echo "[cross_gmt_refpose_v31_100000] results_root=${UNIFIED_RESULTS_DIR}"
echo "[cross_gmt_refpose_v31_100000] NUM_WORKERS=${NUM_WORKERS}, DRY_RUN=${DRY_RUN}"

run_stage "twist2" "open_door" \
  "${ISAACLAB_ROOT}/script/eval_scripts/twist2/HSI_open_door_run_vla_eval_parallel.sh" \
  "${OPEN_DOOR_MODEL_PATH}"
run_stage "twist2" "football" \
  "${ISAACLAB_ROOT}/script/eval_scripts/twist2/HOI_football_run_vla_eval_parallel.sh" \
  "${FOOTBALL_MODEL_PATH}"
run_stage "sonic" "open_door" \
  "${ISAACLAB_ROOT}/script/eval_scripts/sonic/HSI_open_door_run_vla_eval_parallel.sh" \
  "${OPEN_DOOR_MODEL_PATH}"
run_stage "sonic" "football" \
  "${ISAACLAB_ROOT}/script/eval_scripts/sonic/HOI_football_run_vla_eval_parallel.sh" \
  "${FOOTBALL_MODEL_PATH}"

echo ""
echo "[cross_gmt_refpose_v31_100000] all four stages completed"
echo "[cross_gmt_refpose_v31_100000] results=${UNIFIED_RESULTS_DIR}"
