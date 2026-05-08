#!/usr/bin/env bash

#cross gmt
# TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_cross bash pi05_batch_test_boxing.sh
# TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_cross bash pi05_batch_test_football.sh
# TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_cross bash pi05_batch_test_pp_box.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_doubledesk_0418_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_cross bash pi05_batch_test_doubledesk.sh
# TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_opendoor_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_cross bash pi05_batch_test_open_door.sh
# TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_sitsofa_0423_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_cross bash pi05_batch_test_sit_sofa.sh
# TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_visionnavi_0419_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_cross bash pi05_batch_test_vision_navi.sh