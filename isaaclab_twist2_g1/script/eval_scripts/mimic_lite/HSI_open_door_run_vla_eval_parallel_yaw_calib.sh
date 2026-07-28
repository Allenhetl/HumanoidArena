#!/usr/bin/env bash
set -euo pipefail

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${RUN_SCRIPT_DIR}/../../.." && pwd)"

export ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_test_config/base_test/open_door_sonic_test.yaml}"

# Yaw calibration for mimic_lite VLA data convention mismatch.
# mimic_lite training data has state[0:6] and action[3:9]/[0:2] with a fixed
# +yaw(robot_reset) offset (raw world frame) instead of heading-canonicalized (yaw=0).
# This injects +calib to state and -calib to action to match the training distribution.
# Default 90.0 = robot reset world yaw in IsaacLab.
export VLA_MIMICLITE_YAW_CALIB_DEG="${VLA_MIMICLITE_YAW_CALIB_DEG:-90.0}"

# Model: 0726 mimic_lite diffusion (in-GMT: VLA=mimic_lite, GMT=mimic_lite)
export MODEL_PATHS_CSV="${MODEL_PATHS_CSV:-/ai/Yichi/taowen/ckpts/0726/small/HSI_open_door/diffusion_mimic_lite_opendoor_0726/pretrained_model}"

# Results dir: in-GMT with yaw calibration
RESULTS_TAG_BASE="${RESULTS_TAG:-in_gmt_mimic_lite_0726_yawcalib}"
if [[ -n "${RESULTS_TAG_PREFIX:-}" ]]; then
  export RESULTS_TAG="${RESULTS_TAG_PREFIX}_${RESULTS_TAG_BASE}"
else
  export RESULTS_TAG="${RESULTS_TAG_BASE}"
fi

export MAX_STEPS="${MAX_STEPS:-1800}"

# GPU and port config (override via env if needed)
export SERVER_GPU_IDS="${SERVER_GPU_IDS:-0,1,2,3,4,5,6,7}"
export SERVER_PORT_BASE="${SERVER_PORT_BASE:-10000}"
export SERVER_PORT_MAX="${SERVER_PORT_MAX:-15000}"
export NUM_WORKERS="${NUM_WORKERS:-2}"

exec "${RUN_SCRIPT_DIR}/run_vla_eval_parallel.sh"
