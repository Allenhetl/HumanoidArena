#!/usr/bin/env bash
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_boxing/pi05_sonic_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash pi05_batch_test_boxing.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_boxing/pi05_sonic_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash pi05_batch_test_boxing.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_boxing/pi05_sonic_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash pi05_batch_test_boxing.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_boxing/pi05_sonic_boxing_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash pi05_batch_test_boxing.sh


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_football/pi05_sonic_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash pi05_batch_test_football.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_football/pi05_sonic_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash pi05_batch_test_football.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_football/pi05_sonic_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash pi05_batch_test_football.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_football/pi05_sonic_football_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash pi05_batch_test_football.sh


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_pp_box/pi05_sonic_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash pi05_batch_test_pp_box.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_pp_box/pi05_sonic_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash pi05_batch_test_pp_box.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_pp_box/pi05_sonic_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash pi05_batch_test_pp_box.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_pp_box/pi05_sonic_ppbox_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash pi05_batch_test_pp_box.sh


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_double_desk/pi05_sonic_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash pi05_batch_test_doubledesk.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_double_desk/pi05_sonic_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash pi05_batch_test_doubledesk.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_double_desk/pi05_sonic_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash pi05_batch_test_doubledesk.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HOI_double_desk/pi05_sonic_doubledesk_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash pi05_batch_test_doubledesk.sh


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_open_door/pi05_sonic_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash pi05_batch_test_open_door.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_open_door/pi05_sonic_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash pi05_batch_test_open_door.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_open_door/pi05_sonic_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash pi05_batch_test_open_door.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_open_door/pi05_sonic_opendoor_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash pi05_batch_test_open_door.sh


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_sit_sofa/pi05_sonic_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash pi05_batch_test_sit_sofa.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_sit_sofa/pi05_sonic_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash pi05_batch_test_sit_sofa.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_sit_sofa/pi05_sonic_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash pi05_batch_test_sit_sofa.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_sit_sofa/pi05_sonic_sitsofa_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash pi05_batch_test_sit_sofa.sh


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_vision_navi/pi05_sonic_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_base bash pi05_batch_test_vision_navi.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_vision_navi/pi05_sonic_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_semantic bash pi05_batch_test_vision_navi.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_vision_navi/pi05_sonic_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_vision bash pi05_batch_test_vision_navi.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0529/pi/HSI_vision_navi/pi05_sonic_visionnavi_0529" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=v31_pi05_execution bash pi05_batch_test_vision_navi.sh

