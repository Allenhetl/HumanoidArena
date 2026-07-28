#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Campaign evaluation for 0529 v3-1 VLA checkpoints.
#
# These policies were trained on twist2 / sonic / mimic_lite data. This script
# evaluates them on a *different* GMT backend (cross-GMT). Supported backends:
#   sonic_low_latency  (default; 1247-dim SONIC encoder)
#   mimic_lite
#   sonic
#   twist2
#
# Policies (auto-discovered via MODEL_GLOB under CKPT_ROOT):
#   open_door:   *mtp_twist2_opendoor_0529*  |  *diffusion_sonic_opendoor_0529*
#   football:    *diffusion_twist2_football_0529*  |  *diffusion_sonic_football_0529*
#
# Usage:
#   bash isaaclab_twist2_g1/batch_test_scripts/batch_eval_0529_v3_1.sh
#
# Common overrides:
#   EVAL_BACKENDS      - space-separated backends (default: "sonic_low_latency")
#   CKPT_ROOT          - checkpoint root (default: /ai/Yichi/taowen/ckpts/0529_v3-1_infer_ckpts/small)
#   OPEN_DOOR_GLOBS    - space-separated globs for open_door models
#   FOOTBALL_GLOBS     - space-separated globs for football models
#   NUM_WORKERS, SEEDS_OVERRIDE, REPEATS_PER_SEED, MAX_STEPS
#   RUN_LABEL, RUN_TIMESTAMP, RESUME_LATEST, DRY_RUN
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common/eval_batch_utils.sh"

# -----------------------------------------------------------------------------
# Model discovery
# -----------------------------------------------------------------------------
CKPT_ROOT="${CKPT_ROOT:-/ai/Yichi/taowen/ckpts/0529_v3-1_infer_ckpts/small}"
OPEN_DOOR_GLOBS="${OPEN_DOOR_GLOBS:-*mtp_twist2_opendoor_0529* *diffusion_sonic_opendoor_0529*}"
FOOTBALL_GLOBS="${FOOTBALL_GLOBS:-*diffusion_twist2_football_0529* *diffusion_sonic_football_0529*}"

# Discover model paths for each task.
discover_models_for_task() {
  local task_root="$1"; shift
  local globs=("$@")
  local found=() p
  # Use model_batch_utils discover via env, then collect.
  for glob in "${globs[@]}"; do
    while IFS= read -r p; do
      [[ -n "${p}" ]] && found+=("${p}")
    done < <(
      {
        find "${task_root}" -path '*/checkpoints/*/pretrained_model' -type d 2>/dev/null
        find "${task_root}" -mindepth 2 -maxdepth 2 -name pretrained_model -type d 2>/dev/null
      } | sort -u | while IFS= read -r cand; do
        if [[ "${cand}" == ${glob} || "${cand%/pretrained_model}" == ${glob} ]]; then
          printf '%s\n' "${cand}"
        fi
      done
    )
  done
  printf '%s\n' "${found[@]}" | sort -u
}

mapfile -t OPEN_DOOR_MODELS < <(discover_models_for_task "${CKPT_ROOT}/HSI_open_door" ${OPEN_DOOR_GLOBS})
mapfile -t FOOTBALL_MODELS  < <(discover_models_for_task "${CKPT_ROOT}/HOI_football"  ${FOOTBALL_GLOBS})

if [[ ${#OPEN_DOOR_MODELS[@]} -eq 0 && ${#FOOTBALL_MODELS[@]} -eq 0 ]]; then
  echo "Error: no 0529 models found under ${CKPT_ROOT}" >&2
  exit 1
fi

# Validate all discovered models.
for m in "${OPEN_DOOR_MODELS[@]}" "${FOOTBALL_MODELS[@]}"; do
  [[ -n "${m}" ]] && eval_batch_validate_model_complete "${m}"
done

# -----------------------------------------------------------------------------
# Campaign configuration
# -----------------------------------------------------------------------------
EVAL_BACKENDS="${EVAL_BACKENDS:-sonic_low_latency}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RESUME_LATEST="${RESUME_LATEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_LABEL="${RUN_LABEL:-}"

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then export SEEDS_OVERRIDE; fi
if [[ -n "${REPEATS_PER_SEED:-}" ]]; then export REPEATS_PER_SEED; fi
if [[ -n "${MAX_STEPS:-}" ]]; then export MAX_STEPS; fi

CAMPAIGN_ID="0529_v3_1"
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
  echo "[batch_eval_0529_v3_1] finished_at=${end_human} elapsed=$(format_duration "${elapsed}") exit_code=${code}"
  echo "[batch_eval_0529_v3_1] run_id=${RUN_ID}"
  echo "============================================="
}
trap 'print_batch_summary "$?"' EXIT

task_default_env_yaml() {
  case "$1" in
    open_door) printf '%s' "tasks/common_test_config/base_test/open_door_sonic_test.yaml" ;;
    football)  printf '%s' "tasks/common_test_config/base_test/football_single_sonic_test.yaml" ;;
    *) return 1 ;;
  esac
}

echo "[batch_eval_0529_v3_1] started_at=${BATCH_START_HUMAN}"
echo "[batch_eval_0529_v3_1] run_id=${RUN_ID}"
echo "[batch_eval_0529_v3_1] backends=${EVAL_BACKENDS}"
echo "[batch_eval_0529_v3_1] CKPT_ROOT=${CKPT_ROOT}"
echo "[batch_eval_0529_v3_1] open_door_models=${#OPEN_DOOR_MODELS[@]}"
echo "[batch_eval_0529_v3_1] football_models=${#FOOTBALL_MODELS[@]}"
echo "[batch_eval_0529_v3_1] NUM_WORKERS=${NUM_WORKERS} DRY_RUN=${DRY_RUN}"

MANIFEST_STAGES=()

run_stage() {
  local backend="$1" task="$2" model_path="$3"
  eval_batch_setup_backend_env "${backend}"

  local policy_id step launcher stage_dir backend_results_root stage_results_dir
  policy_id="$(eval_batch_extract_policy_id "${model_path}")"
  step="$(eval_batch_extract_checkpoint_step "${model_path}")"
  launcher="$(eval_batch_launcher_path "${backend}" "${task}")"
  stage_dir="$(eval_batch_compute_stage_dir "${task}" "${policy_id}" "${step}")"
  backend_results_root="$(eval_batch_backend_results_root "${backend}" "${RUN_ID}")"
  stage_results_dir="${backend_results_root}/${stage_dir}"

  local env_yaml
  env_yaml="${task^^}_ENV_CONFIG_YAML"
  env_yaml="${!env_yaml:-$(task_default_env_yaml "${task}")}"

  echo ""
  echo "============================================================"
  echo "[batch_eval_0529_v3_1] backend=${backend} task=${task} gmt=${EVAL_BATCH_GMT_RELATION}"
  echo "[batch_eval_0529_v3_1] VLA=${model_path} (${policy_id} step=${step})"
  echo "[batch_eval_0529_v3_1] results=${stage_results_dir}"
  echo "============================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[batch_eval_0529_v3_1] DRY_RUN=1, skipping launch."
    MANIFEST_STAGES+=("$(eval_batch_manifest_stage_fields "${backend}" "${EVAL_BATCH_GMT_RELATION}" "${task}" "${policy_id}" "${step}" "${model_path}" "${stage_results_dir}")")
    return 0
  fi

  mkdir -p "${backend_results_root}"
  env \
    MODEL_ROOT="$(dirname "$(dirname "${model_path}")")" \
    MODEL_GLOB="${model_path}" \
    RESULTS_TAG="${stage_dir}" \
    RESULTS_DIR="${stage_results_dir}" \
    ENV_CONFIG_YAML="${env_yaml}" \
    bash "${launcher}"

  MANIFEST_STAGES+=("$(eval_batch_manifest_stage_fields "${backend}" "${EVAL_BATCH_GMT_RELATION}" "${task}" "${policy_id}" "${step}" "${model_path}" "${stage_results_dir}")")
}

# -----------------------------------------------------------------------------
# Execute all backend x task x model combinations
# -----------------------------------------------------------------------------
for backend in ${EVAL_BACKENDS}; do
  backend_stages=()
  for model_path in "${OPEN_DOOR_MODELS[@]}"; do
    [[ -n "${model_path}" ]] || continue
    run_stage "${backend}" "open_door" "${model_path}"
    backend_stages+=("${MANIFEST_STAGES[-1]}")
  done
  for model_path in "${FOOTBALL_MODELS[@]}"; do
    [[ -n "${model_path}" ]] || continue
    run_stage "${backend}" "football" "${model_path}"
    backend_stages+=("${MANIFEST_STAGES[-1]}")
  done
  # Write manifest per backend
  if [[ ${#backend_stages[@]} -gt 0 ]]; then
    local_results_root="$(eval_batch_backend_results_root "${backend}" "${RUN_ID}")"
    [[ "${DRY_RUN}" != "1" ]] && mkdir -p "${local_results_root}"
    printf '%s\n' "${backend_stages[@]}" | eval_batch_write_manifest "${local_results_root}/campaign_manifest.json"
    echo "[batch_eval_0529_v3_1] manifest=${local_results_root}/campaign_manifest.json"
  fi
done

echo ""
echo "[batch_eval_0529_v3_1] all stages completed. run_id=${RUN_ID}"
