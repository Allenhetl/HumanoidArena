#!/usr/bin/env bash
# Shared helpers for campaign-style batch evaluation scripts.
#
# Conventions:
#   - Results always land under <backend>/eval_results/<run_id>/<stage_dir>
#   - run_id  = <campaign_id>__<timestamp>   (shared across backends in one batch)
#   - stage_dir = <task>__<policy_id>__step_<step>
#   - A campaign_manifest.json is written to each backend's run_id dir.
#
# Source this file:  source "${SCRIPT_DIR}/common/eval_batch_utils.sh"

if [[ -n "${_EVAL_BATCH_UTILS_SH:-}" ]]; then return 0; fi
readonly _EVAL_BATCH_UTILS_SH=1

_EVAL_BATCH_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_EVAL_BATCH_BATCH_DIR="$(cd "${_EVAL_BATCH_UTILS_DIR}/.." && pwd)"
_EVAL_BATCH_ISAACLAB_ROOT="$(cd "${_EVAL_BATCH_BATCH_DIR}/.." && pwd)"

# -----------------------------------------------------------------------------
# Backend -> launcher directory mapping.
#   $1 = backend (twist2 | sonic | sonic_low_latency | mimic_lite)
# -----------------------------------------------------------------------------
eval_batch_backend_script_dir() {
  local backend="$1"
  case "${backend}" in
    twist2)            printf '%s' "${_EVAL_BATCH_ISAACLAB_ROOT}/script/eval_scripts/twist2" ;;
    sonic)             printf '%s' "${_EVAL_BATCH_ISAACLAB_ROOT}/script/eval_scripts/sonic" ;;
    sonic_low_latency) printf '%s' "${_EVAL_BATCH_ISAACLAB_ROOT}/script/eval_scripts/sonic_low_latency" ;;
    mimic_lite)        printf '%s' "${_EVAL_BATCH_ISAACLAB_ROOT}/script/eval_scripts/mimic_lite" ;;
    *)
      echo "Error: unknown backend '${backend}'" >&2
      return 1
      ;;
  esac
}

# Task -> launcher filename within the backend script dir.
#   $1 = task (open_door | football | ...)
eval_batch_task_launcher() {
  local task="$1"
  case "${task}" in
    open_door)   printf '%s' "HSI_open_door_run_vla_eval_parallel.sh" ;;
    football)    printf '%s' "HOI_football_run_vla_eval_parallel.sh" ;;
    doubledesk)  printf '%s' "HOI_double_desk_run_vla_eval_parallel.sh" ;;
    pp_box)      printf '%s' "HOI_pp_box_run_vla_eval_parallel.sh" ;;
    boxing)      printf '%s' "HSI_boxing_run_vla_eval_parallel.sh" ;;
    sit_sofa)    printf '%s' "HSI_sit_sofa_run_vla_eval_parallel.sh" ;;
    vision_navi) printf '%s' "HSI_vision_navi_run_vla_eval_parallel.sh" ;;
    *)
      echo "Error: unknown task '${task}'" >&2
      return 1
      ;;
  esac
}

# Full path to the backend+task launcher.
#   $1 = backend, $2 = task
eval_batch_launcher_path() {
  local backend="$1" task="$2"
  local script_dir launcher
  script_dir="$(eval_batch_backend_script_dir "${backend}")" || return 1
  launcher="$(eval_batch_task_launcher "${task}")" || return 1
  printf '%s/%s' "${script_dir}" "${launcher}"
}

# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

# Find the latest complete numeric checkpoint under a checkpoints dir.
#   $1 = checkpoints dir (containing <step>/pretrained_model/{config.json,model.safetensors})
# Prints the pretrained_model path.
eval_batch_find_latest_complete_model() {
  local checkpoints_dir="$1"
  local latest_step=-1 latest_model_path=""
  local checkpoint_dir step candidate step_value
  for checkpoint_dir in "${checkpoints_dir}"/*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    step="$(basename "${checkpoint_dir}")"
    [[ "${step}" =~ ^[0-9]+$ ]] || continue
    candidate="${checkpoint_dir}/pretrained_model"
    [[ -s "${candidate}/config.json" && -s "${candidate}/model.safetensors" ]] || continue
    step_value=$((10#${step}))
    if (( step_value > latest_step )); then
      latest_step=${step_value}
      latest_model_path="${candidate}"
    fi
  done
  [[ -n "${latest_model_path}" ]] || {
    echo "Error: no complete numeric checkpoint found under ${checkpoints_dir}" >&2
    return 1
  }
  printf '%s' "${latest_model_path}"
}

# Validate that a pretrained_model dir has config.json + model.safetensors.
#   $1 = model path (pretrained_model dir)
eval_batch_validate_model_complete() {
  local model_path="$1"
  local f
  for f in config.json model.safetensors; do
    [[ -s "${model_path}/${f}" ]] || {
      echo "Error: incomplete VLA model: ${model_path}/${f}" >&2
      return 1
    }
  done
}

# Extract policy_id from a model path.
# Supports two layouts:
#   Layout A (mimic_lite): .../<policy_id>/checkpoints/<step>/pretrained_model
#   Layout B (0529):       .../<task>/<policy_id>/pretrained_model
#   Fallback:              parent dir name of pretrained_model
#   $1 = model path (.../pretrained_model)
eval_batch_extract_policy_id() {
  local model_path="$1"
  local step_dir checkpoints_dir job_dir parent_dir
  step_dir="$(dirname "${model_path}")"                              # .../<step> or .../<policy_id>
  parent_dir="$(basename "${step_dir}")"                             # <step> or <policy_id>
  checkpoints_dir="$(dirname "${step_dir}")"                          # .../checkpoints or .../<task>
  # If parent of step_dir is "checkpoints", layout A: go up one more.
  if [[ "$(basename "${checkpoints_dir}")" == "checkpoints" ]]; then
    job_dir="$(basename "$(dirname "${checkpoints_dir}")")"           # <policy_id>
  else
    job_dir="${parent_dir}"                                           # <policy_id> (layout B)
  fi
  [[ -n "${job_dir}" && "${job_dir}" != "/" ]] || {
    echo "Error: cannot extract policy_id from ${model_path}" >&2
    return 1
  }
  printf '%s' "${job_dir}"
}

# Extract checkpoint step from a model path.
#   Layout A: .../<policy_id>/checkpoints/<step>/pretrained_model  -> <step>
#   Layout B: .../<policy_id>/pretrained_model                     -> "final"
#   $1 = model path (.../pretrained_model)
eval_batch_extract_checkpoint_step() {
  local model_path="$1"
  local step_dir step checkpoints_dir
  step_dir="$(dirname "${model_path}")"
  step="$(basename "${step_dir}")"
  checkpoints_dir="$(dirname "${step_dir}")"
  if [[ "$(basename "${checkpoints_dir}")" == "checkpoints" && "${step}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${step}"
  else
    # Layout B: no numeric step directory; use "final".
    printf '%s' "final"
  fi
}

# -----------------------------------------------------------------------------
# Naming helpers
# -----------------------------------------------------------------------------

# Sanitize a string for use in a directory name: lowercase, replace non-alnum with _.
#   $1 = raw string
eval_batch_sanitize_component() {
  local raw="$1"
  local s
  s="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  s="$(printf '%s' "${s}" | tr -c '[:alnum:]' '_' | sed 's/^_//; s/_$//; s/__*/_/g')"
  printf '%s' "${s}"
}

# Compute run_id = <campaign_id>__<timestamp>
# Uses RUN_TIMESTAMP env if set (for cross-backend consistency).
eval_batch_compute_run_id() {
  local campaign_id="$1"
  local ts="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
  printf '%s__%s' "${campaign_id}" "${ts}"
}

# Compute stage dir name = <task>__<policy_id>__step_<step>
eval_batch_compute_stage_dir() {
  local task="$1" policy_id="$2" step="$3"
  local task_s policy_s
  task_s="$(eval_batch_sanitize_component "${task}")"
  policy_s="$(eval_batch_sanitize_component "${policy_id}")"
  printf '%s__%s__step_%s' "${task_s}" "${policy_s}" "${step}"
}

# Compute the full results root for a backend:
#   <backend_script_dir>/eval_results/<run_id>
eval_batch_backend_results_root() {
  local backend="$1" run_id="$2"
  local script_dir
  script_dir="$(eval_batch_backend_script_dir "${backend}")" || return 1
  printf '%s/eval_results/%s' "${script_dir}" "${run_id}"
}

# -----------------------------------------------------------------------------
# Backend environment isolation
# -----------------------------------------------------------------------------

# Set up backend-specific environment variables.
# MimicLite gets MIMIC_LITE_ROBOT_CFG=1 and yaw calibration.
# All other backends unset those to prevent contamination.
#   $1 = backend
# Sets globals: EVAL_BATCH_GMT_RELATION (in_gmt | cross_gmt)
eval_batch_setup_backend_env() {
  local backend="$1"
  case "${backend}" in
    mimic_lite)
      export MIMIC_LITE_ROBOT_CFG=1
      export VLA_MIMICLITE_YAW_CALIB_DEG="${VLA_MIMICLITE_YAW_CALIB_DEG:-0.0}"
      EVAL_BATCH_GMT_RELATION="in_gmt"
      ;;
    twist2|sonic|sonic_low_latency)
      unset MIMIC_LITE_ROBOT_CFG || true
      unset VLA_MIMICLITE_YAW_CALIB_DEG || true
      EVAL_BATCH_GMT_RELATION="cross_gmt"
      ;;
    *)
      echo "Error: unknown backend '${backend}'" >&2
      return 1
      ;;
  esac
}

# -----------------------------------------------------------------------------
# Manifest generation (Python, writes JSON)
# Reads stage records from stdin, one per line, tab-separated:
#   backend\tgmt_relation\ttask\tpolicy_id\tstep\tmodel_path\tresults_dir
#   $1 = manifest path
# -----------------------------------------------------------------------------
eval_batch_write_manifest() {
  local manifest_path="$1"
  python3 -c '
import json, os, sys, subprocess

manifest_path = sys.argv[1]
stages = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    parts = line.split("\t")
    if len(parts) != 7:
        continue
    backend, gmt_relation, task, policy_id, step, model_path, results_dir = parts
    try:
        step_val = int(step)
    except ValueError:
        step_val = step
    stages.append({
        "backend": backend,
        "gmt_relation": gmt_relation,
        "task": task,
        "policy_id": policy_id,
        "checkpoint_step": step_val,
        "model_path": model_path,
        "results_dir": results_dir,
    })

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(manifest_path)))),
        text=True,
    ).strip()
except Exception:
    commit = "unknown"

manifest = {
    "schema_version": 1,
    "campaign_id": os.environ.get("EVAL_BATCH_CAMPAIGN_ID", ""),
    "run_id": os.environ.get("EVAL_BATCH_RUN_ID", ""),
    "run_timestamp": os.environ.get("RUN_TIMESTAMP", ""),
    "run_label": os.environ.get("RUN_LABEL", ""),
    "git_commit": commit,
    "num_workers": os.environ.get("NUM_WORKERS", ""),
    "seeds": os.environ.get("SEEDS_OVERRIDE", ""),
    "repeats_per_seed": os.environ.get("REPEATS_PER_SEED", ""),
    "max_steps": os.environ.get("MAX_STEPS", ""),
    "stages": stages,
}

os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
' "${manifest_path}"
}

# Format a single stage record as a tab-separated line for stdin piping.
#   $1=backend $2=gmt_relation $3=task $4=policy_id $5=step $6=model_path $7=results_dir
eval_batch_manifest_stage_fields() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s' "$1" "$2" "$3" "$4" "$5" "$6" "$7"
}
