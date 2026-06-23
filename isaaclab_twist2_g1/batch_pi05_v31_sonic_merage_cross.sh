#!/usr/bin/env bash
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/merage/pi05_twist28_merage_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_merage_cross bash pi05_batch_test_boxing.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/merage/pi05_twist28_merage_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_merage_cross bash pi05_batch_test_football.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/merage/pi05_twist28_merage_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_merage_cross bash pi05_batch_test_pp_box.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/merage/pi05_twist28_merage_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_merage_cross bash pi05_batch_test_doubledesk.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/merage/pi05_twist28_merage_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_merage_cross bash pi05_batch_test_open_door.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/merage/pi05_twist28_merage_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_merage_cross bash pi05_batch_test_sit_sofa.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/merage/pi05_twist28_merage_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_merage_cross bash pi05_batch_test_vision_navi.sh
