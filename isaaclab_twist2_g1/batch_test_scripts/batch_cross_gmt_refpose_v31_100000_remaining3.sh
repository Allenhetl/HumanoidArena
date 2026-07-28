#!/usr/bin/env bash
set -euo pipefail

# Continue the 20260727_124624 cross-GMT run without repeating its completed
# twist2_open_door stage. Runs only:
#   twist2_football, sonic_open_door, sonic_football.

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

# These runs use twist2 / sonic GMTs, never MimicLite GMT PD or yaw calibration.
unset MIMIC_LITE_ROBOT_CFG
unset VLA_MIMICLITE_YAW_CALIB_DEG
export NUM_WORKERS="${NUM_WORKERS:-2}"
export RESUME_LATEST="${RESUME_LATEST:-0}"
export DRY_RUN="${DRY_RUN:-0}"
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then export SEEDS_OVERRIDE; fi
if [[ -n "${REPEATS_PER_SEED:-}" ]]; then export REPEATS_PER_SEED; fi
if [[ -n "${MAX_STEPS:-}" ]]; then export MAX_STEPS; fi

# Default: add the three missing stages alongside completed twist2_open_door.
UNIFIED_RESULTS_DIR="${UNIFIED_RESULTS_DIR:-${ISAACLAB_ROOT}/script/eval_results_cross_gmt_refpose_v3_1/cross_gmt_refpose_v3_1_100000_twist2_sonic_20260727_124624}"
mkdir -p "${UNIFIED_RESULTS_DIR}"

run_stage() {
  local backend="$1" task="$2" launcher="$3" model_path="$4"
  local stage_name="${backend}_${task}"
  local stage_results_dir="${UNIFIED_RESULTS_DIR}/${stage_name}"

  echo ""
  echo "============================================================"
  echo "[cross_gmt_remaining3] stage=${stage_name}"
  echo "[cross_gmt_remaining3] VLA=${model_path}"
  echo "[cross_gmt_remaining3] GMT backend=${backend}"
  echo "[cross_gmt_remaining3] results=${stage_results_dir}"
  echo "============================================================"

  env \
    MODEL_PATHS_CSV="${model_path}" \
    RESULTS_TAG="${stage_name}" \
    RESULTS_DIR="${stage_results_dir}" \
    bash "${launcher}"
}

echo "[cross_gmt_remaining3] existing run root=${UNIFIED_RESULTS_DIR}"
echo "[cross_gmt_remaining3] skip completed stage=twist2_open_door"
echo "[cross_gmt_remaining3] open_door model=${OPEN_DOOR_MODEL_PATH}"
echo "[cross_gmt_remaining3] football model=${FOOTBALL_MODEL_PATH}"

run_stage "twist2" "football" \
  "${ISAACLAB_ROOT}/script/eval_scripts/twist2/HOI_football_run_vla_eval_parallel.sh" \
  "${FOOTBALL_MODEL_PATH}"
run_stage "sonic" "open_door" \
  "${ISAACLAB_ROOT}/script/eval_scripts/sonic/HSI_open_door_run_vla_eval_parallel.sh" \
  "${OPEN_DOOR_MODEL_PATH}"
run_stage "sonic" "football" \
  "${ISAACLAB_ROOT}/script/eval_scripts/sonic/HOI_football_run_vla_eval_parallel.sh" \
  "${FOOTBALL_MODEL_PATH}"

echo "[cross_gmt_remaining3] all remaining stages completed: ${UNIFIED_RESULTS_DIR}"
