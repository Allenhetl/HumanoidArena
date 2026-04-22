#!/usr/bin/env bash
set -euo pipefail

BATCH_START_TS="$(date +%s)"
BATCH_START_HUMAN="$(date '+%F %T %Z')"

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
  echo "[batch_test] finished_at=${batch_end_human} total_elapsed=$(format_duration "${batch_elapsed}") exit_code=${exit_code}"
}

run_task() {
  local task_name="$1"
  local script_path="$2"
  local task_start_ts="$(date +%s)"
  local task_start_human="$(date '+%F %T %Z')"

  echo "[batch_test] task=${task_name} started_at=${task_start_human}"
  bash "${script_path}"

  local task_end_ts="$(date +%s)"
  local task_end_human="$(date '+%F %T %Z')"
  local task_elapsed=$(( task_end_ts - task_start_ts ))
  echo "[batch_test] task=${task_name} finished_at=${task_end_human} elapsed=$(format_duration "${task_elapsed}")"
}

trap 'print_batch_summary "$?"' EXIT

echo "[batch_test] started_at=${BATCH_START_HUMAN}"

RESUME_LATEST=1
export RESUME_LATEST

run_task sonic script/eval_scripts/sonic/run_vla_eval_parallel.sh
run_task twist2 script/eval_scripts/twist2/run_vla_eval_parallel.sh
