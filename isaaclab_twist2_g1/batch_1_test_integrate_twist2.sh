#!/usr/bin/env bash
#in-gmt
TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-in bash batch_test_football.sh 
TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-in bash batch_test_doubledesk.sh 
TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-in bash batch_test_open_door.sh 
TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-in bash batch_test_pp_box.sh 
TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-in bash batch_test_sit_sofa.sh 
TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-in bash batch_test_vision_navi.sh 

#cross-gmt
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-cross bash batch_test_football.sh 
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-cross bash batch_test_doubledesk.sh 
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-cross bash batch_test_open_door.sh 
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-cross bash batch_test_pp_box.sh 
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-cross bash batch_test_sit_sofa.sh 
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/merge_all_small_100000" MODEL_GLOB="*twist2*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=test1w-merage-cross bash batch_test_vision_navi.sh 