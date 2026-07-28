#!/usr/bin/env bash
set -euo pipefail

# Evaluate task-specific MimicLite-trained v3.1 checkpoints on one or more GMTs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common/eval_batch_utils.sh"

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

export EVAL_BACKENDS="${EVAL_BACKENDS:-mimic_lite}"
EVAL_TASKS="${EVAL_TASKS:-open_door football}"
export NUM_WORKERS="${NUM_WORKERS:-2}" RESUME_LATEST="${RESUME_LATEST:-0}" DRY_RUN="${DRY_RUN:-0}"
export RUN_LABEL="${RUN_LABEL:-}"
CAMPAIGN_ID=mimic_lite_v3_1
[[ -n "${RUN_LABEL}" ]] && CAMPAIGN_ID+="__${RUN_LABEL}"
RUN_ID="$(eval_batch_resolve_run_id "${CAMPAIGN_ID}" "${EVAL_BACKENDS}")"
export RUN_TIMESTAMP="${RUN_ID#${CAMPAIGN_ID}__}"
export EVAL_BATCH_CAMPAIGN_ID="${CAMPAIGN_ID}" EVAL_BATCH_RUN_ID="${RUN_ID}"
export EVAL_BATCH_SCRIPT="${BASH_SOURCE[0]}"
export EVAL_BATCH_MODEL_SOURCE="open_door=${OPEN_DOOR_CHECKPOINTS_DIR}; football=${FOOTBALL_CHECKPOINTS_DIR}"
export EVAL_BATCH_ACTUAL_SEEDS="${SEEDS_OVERRIDE:-0 1 2}"
export EVAL_BATCH_ACTUAL_REPEATS="${REPEATS_PER_SEED:-20}"
if [[ -n "${MAX_STEPS:-}" ]]; then
  export EVAL_BATCH_ACTUAL_MAX_STEPS_JSON="{\"open_door\":${MAX_STEPS},\"football\":${MAX_STEPS}}"
else
  export EVAL_BATCH_ACTUAL_MAX_STEPS_JSON='{"open_door":1800,"football":2000}'
fi
export EVAL_BATCH_MIMIC_YAW_CALIB_DEG="${VLA_MIMICLITE_YAW_CALIB_DEG:-0.0}"
[[ -n "${SEEDS_OVERRIDE:-}" ]] && export SEEDS_OVERRIDE
[[ -n "${REPEATS_PER_SEED:-}" ]] && export REPEATS_PER_SEED
[[ -n "${MAX_STEPS:-}" ]] && export MAX_STEPS

task_model_info() {
  case "$1" in
    open_door) printf '%s\t%s\t%s' "$OPEN_DOOR_MODEL_PATH" "$OPEN_DOOR_POLICY_ID" "$OPEN_DOOR_STEP" ;;
    football) printf '%s\t%s\t%s' "$FOOTBALL_MODEL_PATH" "$FOOTBALL_POLICY_ID" "$FOOTBALL_STEP" ;;
    *) echo "Error: unsupported task '$1'" >&2; return 1 ;;
  esac
}

MANIFEST_STAGES=()
for backend in ${EVAL_BACKENDS}; do
  eval_batch_backend_script_dir "$backend" >/dev/null
  execution_gmt="$(eval_batch_execution_gmt "$backend")"
  relation="$(eval_batch_gmt_relation mimic_lite "$execution_gmt")"
  for task in ${EVAL_TASKS}; do
    IFS=$'\t' read -r model policy step <<<"$(task_model_info "$task")"
    yaml="$(eval_batch_env_config_yaml "$backend" "$task")"
    stage="$(eval_batch_compute_stage_dir "$task" "$policy" "$step")"
    results="$(eval_batch_backend_results_root "$backend" "$RUN_ID")/$stage"
    MANIFEST_STAGES+=("$(eval_batch_manifest_stage_fields "$backend" "$execution_gmt" mimic_lite "$relation" "$task" "$policy" "$step" "$model" "$results" "$yaml" pending)")
  done
done

write_all_manifests() {
  local backend root
  for backend in ${EVAL_BACKENDS}; do
    root="$(eval_batch_backend_results_root "$backend" "$RUN_ID")"
    printf '%s\n' "${MANIFEST_STAGES[@]}" | eval_batch_write_manifest "$root/campaign_manifest.json"
  done
}

update_status() {
  local index="$1" status="$2" b eg tg rel task policy step model results yaml old
  old="${MANIFEST_STAGES[$index]}"
  IFS=$'\t' read -r b eg tg rel task policy step model results yaml _ <<<"$old"
  MANIFEST_STAGES[$index]="$(eval_batch_manifest_stage_fields "$b" "$eg" "$tg" "$rel" "$task" "$policy" "$step" "$model" "$results" "$yaml" "$status")"
}

echo "[batch_eval_mimic_lite_v3_1] run_id=${RUN_ID} backends=${EVAL_BACKENDS} tasks=${EVAL_TASKS} DRY_RUN=${DRY_RUN}"
if [[ "$DRY_RUN" == 1 ]]; then
  for row in "${MANIFEST_STAGES[@]}"; do
    IFS=$'\t' read -r b eg tg rel task policy step model results yaml status <<<"$row"
    echo "[plan] backend=$b execution_gmt=$eg training_gmt=$tg gmt_relation=$rel task=$task env_config_yaml=$yaml results=$results"
  done
  echo "[manifest preview]"
  printf '%s\n' "${MANIFEST_STAGES[@]}" | eval_batch_write_manifest -
  exit 0
fi

# Every backend root receives the same complete plan before any evaluator starts.
write_all_manifests
for i in "${!MANIFEST_STAGES[@]}"; do
  IFS=$'\t' read -r backend execution_gmt training_gmt relation task policy step model results yaml status <<<"${MANIFEST_STAGES[$i]}"
  launcher="$(eval_batch_launcher_path "$backend" "$task")"
  eval_batch_setup_backend_env "$backend"
  echo "[run] backend=$backend task=$task training_gmt=$training_gmt execution_gmt=$execution_gmt gmt_relation=$relation env_config_yaml=$yaml"
  if env MODEL_PATHS_CSV="$model" RESULTS_TAG="$(basename "$results")" RESULTS_DIR="$results" ENV_CONFIG_YAML="$yaml" bash "$launcher"; then
    update_status "$i" completed
    write_all_manifests
  else
    code=$?
    update_status "$i" failed
    write_all_manifests
    exit "$code"
  fi
done
echo "[batch_eval_mimic_lite_v3_1] all stages completed; run_id=${RUN_ID}"
