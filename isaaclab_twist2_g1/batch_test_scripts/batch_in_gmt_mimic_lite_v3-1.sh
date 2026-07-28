#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# In-GMT batch evaluation for 0726 mimic_lite diffusion checkpoints.
#
# Both VLA models are trained on mimic_lite data and evaluated with the SAME
# gmt backend: mimic_lite. This is the in-GMT test.
#
# Runs two VLA evaluations sequentially:
#   1. diffusion_mimic_lite_opendoor_0726  -> HSI_open_door  (mimic_lite backend)
#   2. diffusion_mimic_lite_football_0726  -> HOI_football   (mimic_lite backend)
#
# Usage:
#   bash isaaclab_twist2_g1/batch_test_scripts/batch_in_gmt_mimic_lite_v3-1.sh
#
# Optional env overrides:
#   CKPT_ROOT           - base checkpoint root (default: /ai/Yichi/taowen/ckpts/0726/small)
#   SEEDS_OVERRIDE      - override seeds (default: from yaml, e.g. "0 1 2")
#   REPEATS_PER_SEED    - override repeats per seed (default: from yaml)
#   NUM_WORKERS         - parallel workers (default: 2)
#   MAX_STEPS           - override max steps (default: from each task script)
#   RESULTS_TAG_PREFIX  - prefix for the unified results dir name (default: in_gmt_mimic_lite_0726_v3-1)
#   BATCH_RESULTS_ROOT  - parent dir for unified results (default: isaaclab_twist2_g1/script/eval_results_in_gmt_mimic_lite)
#   DRY_RUN             - set to 1 to skip actual evaluation launch
#
# All 2 stages write under a single unified directory:
#   <BATCH_RESULTS_ROOT>/<RESULTS_TAG_PREFIX>_<timestamp>/{open_door_diffusion_mimic_lite, football_diffusion_mimic_lite}
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BATCH_START_TS="$(date +%s)"
BATCH_START_HUMAN="$(date '+%F %T %Z')"

CKPT_ROOT="${CKPT_ROOT:-/ai/Yichi/taowen/ckpts/0726/small}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RESULTS_TAG_PREFIX="${RESULTS_TAG_PREFIX:-in_gmt_mimic_lite_0726_v3-1}"
DRY_RUN="${DRY_RUN:-0}"

# Unified results root: all 2 stages write under a single directory for easy review.
BATCH_RESULTS_ROOT="${BATCH_RESULTS_ROOT:-${ISAACLAB_ROOT}/script/eval_results_in_gmt_mimic_lite}"
BATCH_RUN_TS="${BATCH_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
UNIFIED_RESULTS_DIR="${BATCH_RESULTS_ROOT}/${RESULTS_TAG_PREFIX}_${BATCH_RUN_TS}"
mkdir -p "${UNIFIED_RESULTS_DIR}"

# Common exports
export NUM_WORKERS
export RESUME_LATEST="${RESUME_LATEST:-0}"
# NOTE: Do NOT export RESULTS_TAG_PREFIX here. Child scripts would re-prepend it
# onto RESULTS_TAG, producing a duplicated prefix (e.g. in_gmt_..._in_gmt_..._).
# We pass a clean RESULTS_TAG per-stage instead, plus an explicit RESULTS_DIR.
# Propagate optional seed/repeat overrides to child scripts
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  export SEEDS_OVERRIDE
fi
if [[ -n "${REPEATS_PER_SEED:-}" ]]; then
  export REPEATS_PER_SEED
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  export MAX_STEPS
fi

format_duration() {
  local total_seconds="$1"
  local hours=$(( total_seconds / 3600 ))
  local minutes=$(( (total_seconds % 3600) / 60 ))
  local seconds=$(( total_seconds % 60 ))
  printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

print_batch_summary() {
  local exit_code="$1"
  local batch_end_ts="$(date +%s)"
  local batch_end_human="$(date '+%F %T %Z')"
  local batch_elapsed=$(( batch_end_ts - BATCH_START_TS ))
  echo ""
  echo "============================================="
  echo "[in_gmt_mimic_lite_batch] finished_at=${batch_end_human}"
  echo "[in_gmt_mimic_lite_batch] total_elapsed=$(format_duration "${batch_elapsed}")"
  echo "[in_gmt_mimic_lite_batch] exit_code=${exit_code}"
  echo "============================================="
}

trap 'print_batch_summary "$?"' EXIT

# -----------------------------------------------------------------------------
# Helper: run a single in-GMT evaluation.
#   $1 = model root directory (parent of pretrained_model)
#   $2 = model glob pattern (must include * wildcards for full-path matching)
#   $3 = results tag suffix (used for RESULTS_TAG and the per-stage subdir name)
#   $4 = eval script path
# -----------------------------------------------------------------------------
run_in_gmt_eval() {
  local model_root="$1"
  local model_glob="$2"
  local tag_suffix="$3"
  local script_path="$4"

  local stage_results_dir="${UNIFIED_RESULTS_DIR}/${tag_suffix}"
  mkdir -p "${stage_results_dir}"

  local stage_start_ts="$(date +%s)"
  local stage_start_human="$(date '+%F %T %Z')"

  echo ""
  echo "============================================="
  echo "[in_gmt_mimic_lite_batch] stage: ${tag_suffix}"
  echo "[in_gmt_mimic_lite_batch]   model_root:   ${model_root}"
  echo "[in_gmt_mimic_lite_batch]   model_glob:   ${model_glob}"
  echo "[in_gmt_mimic_lite_batch]   script:       ${script_path}"
  echo "[in_gmt_mimic_lite_batch]   results_dir:  ${stage_results_dir}"
  echo "[in_gmt_mimic_lite_batch]   started_at:   ${stage_start_human}"
  echo "============================================="

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[in_gmt_mimic_lite_batch] DRY_RUN=1, skipping evaluation launch."
    return 0
  fi

  # RESULTS_DIR has the highest priority in child scripts (resolve_results_dir
  # returns it immediately when set), so all stages land under UNIFIED_RESULTS_DIR.
  # RESULTS_TAG is set to the clean suffix so resume-matching (if any) uses the
  # per-stage name without a duplicated prefix.
  env \
    MODEL_ROOT="${model_root}" \
    MODEL_GLOB="${model_glob}" \
    RESULTS_TAG="${tag_suffix}" \
    RESULTS_DIR="${stage_results_dir}" \
    bash "${script_path}"

  local stage_end_ts="$(date +%s)"
  local stage_end_human="$(date '+%F %T %Z')"
  local stage_elapsed=$(( stage_end_ts - stage_start_ts ))
  echo "[in_gmt_mimic_lite_batch] stage: ${tag_suffix} finished_at=${stage_end_human} elapsed=$(format_duration "${stage_elapsed}")"
}

echo "[in_gmt_mimic_lite_batch] started_at=${BATCH_START_HUMAN}"
echo "[in_gmt_mimic_lite_batch] CKPT_ROOT=${CKPT_ROOT}"
echo "[in_gmt_mimic_lite_batch] NUM_WORKERS=${NUM_WORKERS}"
echo "[in_gmt_mimic_lite_batch] RESULTS_TAG_PREFIX=${RESULTS_TAG_PREFIX}"
echo "[in_gmt_mimic_lite_batch] UNIFIED_RESULTS_DIR=${UNIFIED_RESULTS_DIR}"
echo "[in_gmt_mimic_lite_batch] DRY_RUN=${DRY_RUN}"

# -----------------------------------------------------------------------------
# Stage 1: diffusion_mimic_lite_opendoor_0726 -> HSI_open_door (mimic_lite backend)
# -----------------------------------------------------------------------------
run_in_gmt_eval \
  "${CKPT_ROOT}/HSI_open_door" \
  "*diffusion_mimic_lite_opendoor_0726*" \
  "open_door_diffusion_mimic_lite" \
  "${ISAACLAB_ROOT}/script/eval_scripts/mimic_lite/HSI_open_door_run_vla_eval_parallel.sh"

# -----------------------------------------------------------------------------
# Stage 2: diffusion_mimic_lite_football_0726 -> HOI_football (mimic_lite backend)
# -----------------------------------------------------------------------------
run_in_gmt_eval \
  "${CKPT_ROOT}/HOI_football" \
  "*diffusion_mimic_lite_football_0726*" \
  "football_diffusion_mimic_lite" \
  "${ISAACLAB_ROOT}/script/eval_scripts/mimic_lite/HOI_football_run_vla_eval_parallel.sh"

echo ""
echo "[in_gmt_mimic_lite_batch] All 2 stages completed successfully."
echo "[in_gmt_mimic_lite_batch] Unified results: ${UNIFIED_RESULTS_DIR}"
echo "[in_gmt_mimic_lite_batch]   Stage 1 (open_door_diffusion_mimic_lite):   ${UNIFIED_RESULTS_DIR}/open_door_diffusion_mimic_lite"
echo "[in_gmt_mimic_lite_batch]   Stage 2 (football_diffusion_mimic_lite):   ${UNIFIED_RESULTS_DIR}/football_diffusion_mimic_lite"
