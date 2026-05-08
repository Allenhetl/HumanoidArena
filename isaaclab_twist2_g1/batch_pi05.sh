#!/usr/bin/env bash
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base bash pi05_batch_test_boxing.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash pi05_batch_test_boxing.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_vision bash pi05_batch_test_boxing.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_execution bash pi05_batch_test_boxing.sh


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base bash pi05_batch_test_football.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash pi05_batch_test_football.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_vision bash pi05_batch_test_football.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_execution bash pi05_batch_test_football.sh


TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base bash pi05_batch_test_pp_box.sh
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash pi05_batch_test_pp_box.sh
TEST_MODE=vision EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_vision bash pi05_batch_test_pp_box.sh
TEST_MODE=execution EVAL_BACKEND=sonic MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_sonic_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_execution bash pi05_batch_test_pp_box.sh


TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base bash pi05_batch_test_boxing.sh
TEST_MODE=semantic EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash pi05_batch_test_boxing.sh
TEST_MODE=vision EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_vision bash pi05_batch_test_boxing.sh
TEST_MODE=execution EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_boxing_0424_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_execution bash pi05_batch_test_boxing.sh

TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_doubledesk_0418_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base bash pi05_batch_test_doubledesk.sh
TEST_MODE=semantic EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_doubledesk_0418_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash pi05_batch_test_doubledesk.sh
TEST_MODE=vision EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_doubledesk_0418_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_vision bash pi05_batch_test_doubledesk.sh
TEST_MODE=execution EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_doubledesk_0418_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_execution bash pi05_batch_test_doubledesk.sh


TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base bash pi05_batch_test_football.sh
TEST_MODE=semantic EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash pi05_batch_test_football.sh
TEST_MODE=vision EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_vision bash pi05_batch_test_football.sh
TEST_MODE=execution EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_football_0416_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_execution bash pi05_batch_test_football.sh


TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_base bash pi05_batch_test_pp_box.sh
TEST_MODE=semantic EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash pi05_batch_test_pp_box.sh
TEST_MODE=vision EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_vision bash pi05_batch_test_pp_box.sh
TEST_MODE=execution EVAL_BACKEND=twist2 MODEL_ROOT="/ai/Yichi/taowen/ckpts/0424_new_100000/pi0.5_100000/Pi0.5_twist2_ppbox_0421_100000" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_execution bash pi05_batch_test_pp_box.sh
