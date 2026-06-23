#!/usr/bin/env bash
# TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_GLOB="*sonic*" RESULTS_TAG_PREFIX=v3_1-sonic-merage  bash batch_1_test_v31_merage.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_GLOB="*sonic*" RESULTS_TAG_PREFIX=v3_1-sonic-merage  bash batch_1_test_v31_merage.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_GLOB="*sonic*" RESULTS_TAG_PREFIX=v3_1-sonic-merage  bash batch_1_test_v31_merage.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_GLOB="*sonic*" RESULTS_TAG_PREFIX=v3_1-sonic-merage  bash batch_1_test_v31_merage.sh
