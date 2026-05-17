TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_GLOB="*sonic*" RESULTS_TAG_PREFIX=merage-v3-sonic bash batch_1_test_v3_sonic.sh
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_GLOB="*all*" RESULTS_TAG_PREFIX=merage-v3-all-sonic bash batch_1_test_v3_all.sh
TEST_MODE=base_test EVAL_BACKEND=twist2 MODEL_GLOB="*all*" RESULTS_TAG_PREFIX=merage-v3-all-twist2 bash batch_1_test_v3_all.sh