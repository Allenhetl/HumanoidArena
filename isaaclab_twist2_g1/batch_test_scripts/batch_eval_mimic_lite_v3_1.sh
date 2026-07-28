#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Campaign evaluation for task-specific MimicLite VLA checkpoints (0726 series).
#
# Supports four GMT backends via EVAL_BACKENDS:
#   mimic_lite         (in-GMT; default)
#   sonic_low_latency  (cross-GMT; 1247-dim SONIC encoder)
#   sonic              (cross-GMT; 1762-dim SONIC encoder)
#   twist2             (cross-GMT)
#
# Two task-specific policies are evaluated (separately trained):
#   - open_door  <- diffusion_mimic_lite_refpose_v3_1
#   - football   <- diffusion_HOI_football
#
# Usage:
#   bash isaaclab_twist2_g1/batch_test_scripts/batch_eval_mimic_lite_v3_1.sh
#
# Common overrides:
#   EVAL_BACKENDS      - space-separated backends (default: "mimic_lite")
#   EVAL_TASKS         - space-separated tasks (default: "open_door football")
#   OPEN_DOOR_MODEL_PATH / FOOTBALL_MODEL_PATH
#   OPEN_DOOR_CHECKPOINTS_DIR / FOOTBALL_CHECKPOINTS_DIR
#   NUM_WORKERS, SEEDS_OVERRIDE, REPEATS_PER_SEED, MAX_STEPS
#   RUN_LABEL          - short label appended to campaign_id (optional)
#   RUN_TIMESTAMP      - fixed timestamp for cross-backend run_id sharing (optional)
#   RESUME_LATEST      - resume latest matching run (default: 0)
#   DRY_RUN            - print commands without launching (default: 0)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common/eval_batch_utils.sh"

# -----------------------------------------------------------------------------
# Model resolution
# -----------------------------------------------------------------------------
OPEN_DOOR_CHECKPOINTS_DIR="${OPEN_DOOR_CHECKPOINTS_DIR:-/ai/Yichi/zikang/mimiclite_v3.1/0726/diffusion_mimic_lite_refpose_v3_1/checkpoints}"
FOOTBALL_CHECKPOINTS_DIR="${FOOTBALL_CHECKPOINTS_DIR:-/ai/Yichi/zikang/mimiclite_v3.1/0726/diffusion_HOI_football/checkpoints}"

OPEN_DOOR_MODEL_PATH="${OPEN_DOOR_MODEL_PATH:-$(eval_batch_find_latest_complete_model "${OPEN_DOOR_CHECKPOINTS_DIR}")}"
FOOTBALL_MODEL_PATH="${FOOTBALL_MODEL_PATH:-$(eval_batch_find_latest_complete_model "${FOOTBALL_CHECKPOINTS_DIR}")}"

eval_batch_validate_model_complete "${OPEN_DOOR_MODEL_PATH}"
eval_batch_validate_model_complete "${FOOTBALL_MODEL_PATH}"

OPEN_DOOR_POLICY_ID="$(eval_batch_extract_policy_id "${OPEN_DOOR_MODEL_PATH}")"
OPEN_DOOR_STEP="$(eval_batch_extract_checkpoint_step "${OPEN_DOOR_MODEL_PATH}")"
FOOTBALL_POLICY_ID="$(eval_batch_extract_policy_id "${FOOTBALL_MODEL_PATH}")"
FOOTBALL_STEP="$(eval_batch_extract_checkpoint_step "${FOOTBALL_MODEL_PATH}")"

# -----------------------------------------------------------------------------
# Campaign configuration
# -----------------------------------------------------------------------------
EVAL_BACKENDS="${EVAL_BACKENDS:-mimic_lite}"
EVAL_TASKS="${EVAL_TASKS:-open_door football}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RESUME_LATEST="${RESUME_LATEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_LABEL="${RUN_LABEL:-}"

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then export SEEDS_OVERRIDE; fi
if [[ -n "${REPEATS_PER_SEED:-}" ]]; then export REPEATS_PER_SEED; fi
if [[ -n "${MAX_STEPS:-}" ]]; then export MAX_STEPS; fi

CAMPAIGN_ID="mimic_lite_v3_1"
[[ -n "${RUN_LABEL}" ]] && CAMPAIGN_ID="${CAMPAIGN_ID}__${RUN_LABEL}"
RUN_ID="$(eval_batch_compute_run_id "${CAMPAIGN_ID}")"
export RUN_TIMESTAMP="${RUN_ID##*__}"
export NUM_WORKERS RESUME_LATEST DRY_RUN
export EVAL_BATCH_CAMPAIGN_ID="${CAMPAIGN_ID}"
export EVAL_BATCH_RUN_ID="${RUN_ID}"

BATCH_START_TS="$(date +%s)"
BATCH_START_HUMAN="$(date '+%F %T %Z')"

format_duration() {
  local t="$1" h m s
  h=$(( t / 3600 )); m=$(( (t % 3600) / 60 )); s=$(( t % 60 ))
  printf '%02d:%02d:%02d' "${h}" "${m}" "${s}"
}

print_batch_summary() {
  local code="$1"
  local end_ts end_human elapsed
  end_ts="$(date +%s)"; end_human="$(date '+%F %T %Z')"
  elapsed=$(( end_ts - BATCH_START_TS ))
  echo ""
  echo "============================================="
  echo "[batch_eval_mimic_lite_v3_1] finished_at=${end_human} elapsed=$(format_duration "${elapsed}") exit_code=${code}"
  echo "[batch_eval_mimic_lite_v3_1] run_id=${RUN_ID}"
  echo "============================================="
}
trap 'print_batch_summary "$?"' EXIT

# Map task -> (model_path, policy_id, step)
task_model_info() {
  case "$1" in
    open_door) printf '%s\001%s\001%s' "${OPEN_DOOR_MODEL_PATH}" "${OPEN_DOOR_POLICY_ID}" "${OPEN_DOOR_STEP}" ;;
    football)  printf '%s\001%s\001%s' "${FOOTBALL_MODEL_PATH}"  "${FOOTBALL_POLICY_ID}"  "${FOOTBALL_STEP}"  ;;
    *) return 1 ;;
  esac
}

# Task -> default env config yaml (backend-agnostic base; launcher may override)
task_default_env_yaml() {
  case "$1" in
    open_door) printf '%s' "tasks/common_test_config/base_test/open_door_sonic_test.yaml" ;;
    football)  printf '%s' "tasks/common_test_config/base_test/football_single_sonic_test.yaml" ;;
    *) return 1 ;;
  esac
}

echo "[batch_eval_mimic_lite_v3_1] started_at=${BATCH_START_HUMAN}"
echo "[batch_eval_mimic_lite_v3_1] run_id=${RUN_ID}"
echo "[batch_eval_mimic_lite_v3_1] backends=${EVAL_BACKENDS}"
echo "[batch_eval_mimic_lite_v3_1] tasks=${EVAL_TASKS}"
echo "[batch_eval_mimic_lite_v3_1] open_door=${OPEN_DOOR_POLICY_ID} step=${OPEN_DOOR_STEP}"
echo "[batch_eval_mimic_lite_v3_1] football=${FOOTBALL_POLICY_ID} step=${FOOTBALL_STEP}"
echo "[batch_eval_mimic_lite_v3_1] NUM_WORKERS=${NUM_WORKERS} DRY_RUN=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Stage runner
# -----------------------------------------------------------------------------
MANIFEST_STAGES=()

run_stage() {
  local backend="$1" task="$2"
  eval_batch_setup_backend_env "${backend}"

  local info model_path policy_id step
  info="$(task_model_info "${task}")"
  IFS=$'\001' read -r model_path policy_id step <<<"${info}"

  local launcher stage_dir backend_results_root stage_results_dir manifest_path
  launcher="$(eval_batch_launcher_path "${backend}" "${task}")"
  stage_dir="$(eval_batch_compute_stage_dir "${task}" "${policy_id}" "${step}")"
  backend_results_root="$(eval_batch_backend_results_root "${backend}" "${RUN_ID}")"
  stage_results_dir="${backend_results_root}/${stage_dir}"
  manifest_path="${backend_results_root}/campaign_manifest.json"

  local env_yaml
  env_yaml="${task^^}_ENV_CONFIG_YAML"
  env_yaml="${!env_yaml:-$(task_default_env_yaml "${task}")}"

  echo ""
  echo "============================================================"
  echo "[batch_eval_mimic_lite_v3_1] backend=${backend} task=${task} gmt=${EVAL_BATCH_GMT_RELATION}"
  echo "[batch_eval_mimic_lite_v3_1] VLA=${model_path}"
  echo "[batch_eval_mimic_lite_v3_1] results=${stage_results_dir}"
  echo "============================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[batch_eval_mimic_lite_v3_1] DRY_RUN=1, skipping launch."
    MANIFEST_STAGES+=("$(eval_batch_manifest_stage_fields "${backend}" "${EVAL_BATCH_GMT_RELATION}" "${task}" "${policy_id}" "${step}" "${model_path}" "${stage_results_dir}")")
    return 0
  fi

  mkdir -p "${backend_results_root}"
  env \
    MODEL_PATHS_CSV="${model_path}" \
    RESULTS_TAG="${stage_dir}" \
    RESULTS_DIR="${stage_results_dir}" \
    ENV_CONFIG_YAML="${env_yaml}" \
    bash "${launcher}"

  MANIFEST_STAGES+=("$(eval_batch_manifest_stage_fields "${backend}" "${EVAL_BATCH_GMT_RELATION}" "${task}" "${policy_id}" "${step}" "${model_path}" "${stage_results_dir}")")
}

# -----------------------------------------------------------------------------
# Execute all backend x task combinations
# -----------------------------------------------------------------------------
for backend in ${EVAL_BACKENDS}; do
  for task in ${EVAL_TASKS}; do
    run_stage "${backend}" "${task}"
  done
  # Write manifest per backend after all its tasks finish.
  local_results_root="$(eval_batch_backend_results_root "${backend}" "${RUN_ID}")"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${local_results_root}"
  fi
  # Collect only this backend's stages and pipe to manifest writer.
  local_manifest_lines=()
  for fields in "${MANIFEST_STAGES[@]}"; do
    if [[ "${fields}" == "${backend}"$'\t'* ]]; then
      local_manifest_lines+=("${fields}")
    fi
  done
  if [[ ${#local_manifest_lines[@]} -gt 0 ]]; then
    printf '%s\n' "${local_manifest_lines[@]}" | eval_batch_write_manifest "${local_results_root}/campaign_manifest.json"
    echo "[batch_eval_mimic_lite_v3_1] manifest=${local_results_root}/campaign_manifest.json"
  fi
done

echo ""
echo "[batch_eval_mimic_lite_v3_1] all stages completed. run_id=${RUN_ID}"
