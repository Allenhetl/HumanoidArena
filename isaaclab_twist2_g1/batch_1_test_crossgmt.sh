#!/usr/bin/env bash
set -euo pipefail

export RESUME_LATEST=1
export NUM_WORKERS="${NUM_WORKERS:-2}"
TEST_MODE="${TEST_MODE:-base}"
export TEST_MODE

run_cross_batch() {
  local label="$1"
  local eval_backend="$2"
  local model_glob="$3"
  local results_prefix="$4"

  echo "========================================="
  echo " Cross-GMT: ${label}"
  echo "   MODEL_GLOB=${model_glob}"
  echo "   EVAL_BACKEND=${eval_backend}"
  echo "   RESULTS_TAG_PREFIX=${results_prefix}"
  echo "========================================="

  export EVAL_BACKEND="${eval_backend}"
  export MODEL_GLOB="${model_glob}"
  export RESULTS_TAG_PREFIX="${results_prefix}"

  bash batch_test_football.sh
  bash batch_test_doubledesk.sh
  bash batch_test_sit_sofa.sh
  bash batch_test_vision_navi.sh
}

# Mode A: sonic ckpts → twist2 eval
run_cross_batch "sonic→twist2" "twist2" "*sonic*" "test1w_cross_s2t"

# Mode B: twist2 ckpts → sonic eval
run_cross_batch "twist2→sonic" "sonic" "*twist2*" "test1w_cross_t2s"
