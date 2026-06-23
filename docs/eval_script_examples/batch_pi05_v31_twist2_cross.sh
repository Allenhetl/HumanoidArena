#!/usr/bin/env bash
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_boxing/pi05_twist2_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base_cross bash pi05_batch_test_boxing.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_double_desk/pi05_twist2_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base_cross bash pi05_batch_test_doubledesk.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_football/pi05_twist2_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base_cross bash pi05_batch_test_football.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_pp_box/pi05_twist2_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base_cross bash pi05_batch_test_pp_box.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_open_door/pi05_twist2_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base_cross bash pi05_batch_test_open_door.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_sit_sofa/pi05_twist2_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base_cross bash pi05_batch_test_sit_sofa.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_vision_navi/pi05_twist2_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base_cross bash pi05_batch_test_vision_navi.sh
