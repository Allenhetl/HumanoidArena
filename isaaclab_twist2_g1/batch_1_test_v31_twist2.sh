#!/usr/bin/env bash
set -euo pipefail

export RESUME_LATEST=0
SELECTED_EVAL_BACKEND="${EVAL_BACKEND:-${BATCH_TEST_BACKEND:-all}}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export VLA_MAX_ROOT_DELTA_DEG="${VLA_MAX_ROOT_DELTA_DEG:-0}"
if [[ "${VLA_MAX_ROOT_DELTA_DEG}" != "0" && "${VLA_MAX_ROOT_DELTA_DEG}" != "0.0" ]]; then
  echo "[batch_1_test] VLA_MAX_ROOT_DELTA_DEG=${VLA_MAX_ROOT_DELTA_DEG}"
fi
TEST_MODE="${TEST_MODE:-base_test}"
export TEST_MODE
export RESULTS_TAG_PREFIX="${RESULTS_TAG_PREFIX:-${BATCH_RESULTS_PREFIX:-${1:-}}}"
if [[ -n "${RESULTS_TAG_PREFIX}" ]]; then
  export RESULTS_TAG_PREFIX
  echo "[batch_1_test] RESULTS_TAG_PREFIX=${RESULTS_TAG_PREFIX}"
fi
case "${SELECTED_EVAL_BACKEND}" in
  all|sonic|twist2) ;;
  *)
    echo "[batch_1_test] invalid EVAL_BACKEND=${SELECTED_EVAL_BACKEND}; expected all, sonic, or twist2" >&2
    exit 2
    ;;
esac
echo "[batch_1_test] EVAL_BACKEND=${SELECTED_EVAL_BACKEND}"
if [[ -n "${MODEL_GLOB:-}" ]]; then
  export MODEL_GLOB
  echo "[batch_1_test] MODEL_GLOB=${MODEL_GLOB}"
fi

run_batch_backend() {
  local backend="$1"
  local model_root="$2"
  local script_path="$3"

  case "${SELECTED_EVAL_BACKEND}" in
    all|"${backend}") ;;
    *)
      echo "[batch_1_test] backend=${backend} skipped_by_EVAL_BACKEND=${SELECTED_EVAL_BACKEND} script=${script_path}"
      return 0
      ;;
  esac

  echo "[batch_1_test] backend=${backend} MODEL_GLOB=${MODEL_GLOB:-<unset>} MODEL_ROOT=${model_root} script=${script_path}"
  EVAL_BACKEND="${backend}" MODEL_ROOT="${model_root}" bash "${script_path}"
}

run_batch() {
  local model_root="$1"
  local script_path="$2"
  run_batch_backend sonic "${model_root}" "${script_path}"
  run_batch_backend twist2 "${model_root}" "${script_path}"
}

run_batch "/ai/Yichi/taowen/ckpts/0529/small/HSI_open_door" /ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/batch_test_open_door.sh
run_batch "/ai/Yichi/taowen/ckpts/0529/small/HOI_pp_box" /ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/batch_test_pp_box.sh
run_batch "/ai/Yichi/taowen/ckpts/0529/small/HSI_boxing" /ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/batch_test_boxing.sh
run_batch "/ai/Yichi/taowen/ckpts/0529/small/HOI_football" /ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/batch_test_football.sh
run_batch "/ai/Yichi/taowen/ckpts/0529/small/HOI_double_desk" /ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/batch_test_doubledesk.sh
run_batch "/ai/Yichi/taowen/ckpts/0529/small/HSI_sit_sofa" /ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/batch_test_sit_sofa.sh
run_batch "/ai/Yichi/taowen/ckpts/0529/small/HSI_vision_navi" /ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1/batch_test_vision_navi.sh
