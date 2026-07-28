#!/usr/bin/env bash
set -euo pipefail

# Evaluate the mixed Twist2/Sonic 0529 policy set on one or more GMT backends.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common/eval_batch_utils.sh"

CKPT_ROOT="${CKPT_ROOT:-/ai/Yichi/taowen/ckpts/0529_v3-1_infer_ckpts/small}"
OPEN_DOOR_GLOBS="${OPEN_DOOR_GLOBS:-*mtp_twist2_opendoor_0529* *diffusion_sonic_opendoor_0529*}"
FOOTBALL_GLOBS="${FOOTBALL_GLOBS:-*diffusion_twist2_football_0529* *diffusion_sonic_football_0529*}"

discover_models_for_task() {
  local root="$1"; shift
  local glob candidate
  for glob in "$@"; do
    while IFS= read -r candidate; do
      [[ "$candidate" == $glob || "${candidate%/pretrained_model}" == $glob ]] && printf '%s\n' "$candidate"
    done < <({ find "$root" -path '*/checkpoints/*/pretrained_model' -type d 2>/dev/null; find "$root" -mindepth 2 -maxdepth 2 -name pretrained_model -type d 2>/dev/null; } | sort -u)
  done | sort -u
}

mapfile -t OPEN_DOOR_MODELS < <(discover_models_for_task "${CKPT_ROOT}/HSI_open_door" ${OPEN_DOOR_GLOBS})
mapfile -t FOOTBALL_MODELS < <(discover_models_for_task "${CKPT_ROOT}/HOI_football" ${FOOTBALL_GLOBS})
if [[ ${#OPEN_DOOR_MODELS[@]} -eq 0 && ${#FOOTBALL_MODELS[@]} -eq 0 ]]; then
  echo "Error: no 0529 models found under ${CKPT_ROOT}" >&2; exit 1
fi
for model in "${OPEN_DOOR_MODELS[@]}" "${FOOTBALL_MODELS[@]}"; do [[ -z "$model" ]] || eval_batch_validate_model_complete "$model"; done

training_gmt_from_policy_id() {
  local id="${1,,}" twist=0 sonic=0
  [[ "$id" == *twist2* ]] && twist=1
  [[ "$id" == *sonic* ]] && sonic=1
  if (( twist + sonic != 1 )); then
    echo "Error: cannot deterministically classify training GMT from policy_id '$1'" >&2; return 1
  fi
  (( twist )) && printf '%s' twist2 || printf '%s' sonic
}

export EVAL_BACKENDS="${EVAL_BACKENDS:-sonic_low_latency}"
export NUM_WORKERS="${NUM_WORKERS:-2}" RESUME_LATEST="${RESUME_LATEST:-0}" DRY_RUN="${DRY_RUN:-0}"
export RUN_LABEL="${RUN_LABEL:-}"
CAMPAIGN_ID=0529_v3_1
[[ -n "$RUN_LABEL" ]] && CAMPAIGN_ID+="__${RUN_LABEL}"
RUN_ID="$(eval_batch_resolve_run_id "$CAMPAIGN_ID" "$EVAL_BACKENDS")"
export RUN_TIMESTAMP="${RUN_ID#${CAMPAIGN_ID}__}"
export EVAL_BATCH_CAMPAIGN_ID="$CAMPAIGN_ID" EVAL_BATCH_RUN_ID="$RUN_ID"
export EVAL_BATCH_SCRIPT="${BASH_SOURCE[0]}"
export EVAL_BATCH_MODEL_SOURCE="CKPT_ROOT=${CKPT_ROOT}; OPEN_DOOR_GLOBS=${OPEN_DOOR_GLOBS}; FOOTBALL_GLOBS=${FOOTBALL_GLOBS}"
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

MANIFEST_STAGES=()
plan_model() {
  local backend="$1" task="$2" model="$3" policy step training execution relation yaml stage results
  policy="$(eval_batch_extract_policy_id "$model")"
  step="$(eval_batch_extract_checkpoint_step "$model")"
  training="$(training_gmt_from_policy_id "$policy")"
  execution="$(eval_batch_execution_gmt "$backend")"
  relation="$(eval_batch_gmt_relation "$training" "$execution")"
  yaml="$(eval_batch_env_config_yaml "$backend" "$task")"
  stage="$(eval_batch_compute_stage_dir "$task" "$policy" "$step")"
  results="$(eval_batch_backend_results_root "$backend" "$RUN_ID")/$stage"
  MANIFEST_STAGES+=("$(eval_batch_manifest_stage_fields "$backend" "$execution" "$training" "$relation" "$task" "$policy" "$step" "$model" "$results" "$yaml" pending)")
}

for backend in ${EVAL_BACKENDS}; do
  eval_batch_backend_script_dir "$backend" >/dev/null
  for model in "${OPEN_DOOR_MODELS[@]}"; do [[ -z "$model" ]] || plan_model "$backend" open_door "$model"; done
  for model in "${FOOTBALL_MODELS[@]}"; do [[ -z "$model" ]] || plan_model "$backend" football "$model"; done
done

write_all_manifests() {
  local backend root
  for backend in ${EVAL_BACKENDS}; do
    root="$(eval_batch_backend_results_root "$backend" "$RUN_ID")"
    printf '%s\n' "${MANIFEST_STAGES[@]}" | eval_batch_write_manifest "$root/campaign_manifest.json"
  done
}

update_status() {
  local index="$1" status="$2" b eg tg rel task policy step model results yaml
  IFS=$'\t' read -r b eg tg rel task policy step model results yaml _ <<<"${MANIFEST_STAGES[$index]}"
  MANIFEST_STAGES[$index]="$(eval_batch_manifest_stage_fields "$b" "$eg" "$tg" "$rel" "$task" "$policy" "$step" "$model" "$results" "$yaml" "$status")"
}

echo "[batch_eval_0529_v3_1] run_id=$RUN_ID backends=$EVAL_BACKENDS models=$((${#OPEN_DOOR_MODELS[@]} + ${#FOOTBALL_MODELS[@]})) DRY_RUN=$DRY_RUN"
if [[ "$DRY_RUN" == 1 ]]; then
  for row in "${MANIFEST_STAGES[@]}"; do
    IFS=$'\t' read -r b eg tg rel task policy step model results yaml status <<<"$row"
    echo "[plan] backend=$b execution_gmt=$eg training_gmt=$tg gmt_relation=$rel task=$task policy_id=$policy env_config_yaml=$yaml results=$results"
  done
  echo "[manifest preview]"
  printf '%s\n' "${MANIFEST_STAGES[@]}" | eval_batch_write_manifest -
  exit 0
fi

write_all_manifests
for i in "${!MANIFEST_STAGES[@]}"; do
  IFS=$'\t' read -r backend execution training relation task policy step model results yaml status <<<"${MANIFEST_STAGES[$i]}"
  launcher="$(eval_batch_launcher_path "$backend" "$task")"
  eval_batch_setup_backend_env "$backend"
  echo "[run] backend=$backend task=$task policy_id=$policy training_gmt=$training execution_gmt=$execution gmt_relation=$relation env_config_yaml=$yaml"
  if env MODEL_ROOT="$(dirname "$(dirname "$model")")" MODEL_GLOB="$model" RESULTS_TAG="$(basename "$results")" RESULTS_DIR="$results" ENV_CONFIG_YAML="$yaml" bash "$launcher"; then
    update_status "$i" completed; write_all_manifests
  else
    code=$?; update_status "$i" failed; write_all_manifests; exit "$code"
  fi
done
echo "[batch_eval_0529_v3_1] all stages completed; run_id=$RUN_ID"
