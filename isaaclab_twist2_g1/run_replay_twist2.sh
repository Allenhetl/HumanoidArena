#!/bin/bash
# run_replay.sh - Launch TWIST2 replay simulation through sim_main.py
# Usage:
#   ./run_replay.sh <replay_file> [inference|direct] [--loop]
#
# Examples:
#   ./run_replay.sh ./recording_data/recording_20260311_123456.npz inference
#   ./run_replay.sh ./recording_data/recording_20260311_123456.npz direct --loop

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# Check if replay file is provided
if [ -z "$1" ]; then
    echo "Error: Replay file path is required"
    echo "Usage: $0 <replay_file> [inference|direct] [--loop]"
    echo ""
    echo "Examples:"
    echo "  $0 ./recording_data/recording_20260311_123456.npz inference"
    echo "  $0 ./recording_data/recording_20260311_123456.npz direct --loop"
    exit 1
fi

REPLAY_FILE="$1"
REPLAY_MODE="${2:-inference}"  # Default to inference mode
LOOP_FLAG=""
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/twist2_default.yaml}"
REPLAY_DEVICE="${REPLAY_DEVICE:-cpu}"
# Check for --loop flag
if [ "$3" == "--loop" ] || [ "$2" == "--loop" ]; then
    LOOP_FLAG="--loop"
fi

# Validate replay mode
if [ "$REPLAY_MODE" != "inference" ] && [ "$REPLAY_MODE" != "direct" ]; then
    echo "Error: Invalid replay mode '$REPLAY_MODE'. Must be 'inference' or 'direct'"
    exit 1
fi

# Check if replay file exists
if [ ! -f "$REPLAY_FILE" ]; then
    echo "Error: Replay file not found: $REPLAY_FILE"
    exit 1
fi

echo "=========================================="
echo "Starting Replay Simulation"
echo "=========================================="
echo "Replay file: $REPLAY_FILE"
echo "Task: ${TASK_NAME:-will-resolve-from-file}"
echo "Replay mode: $REPLAY_MODE"
echo "Loop: ${LOOP_FLAG:-disabled}"
echo "Env config YAML: $ENV_CONFIG_YAML"
echo "Device: $REPLAY_DEVICE"
echo "=========================================="

# Random seed for reproducibility (set to fixed value for deterministic behavior)
SEED="${SEED:-42}"
echo "Random seed: $SEED"
echo "=========================================="

# Set environment variables
export PROJECT_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

if [ -z "${TASK_NAME:-}" ]; then
    TASK_NAME="$(python - "$REPLAY_FILE" <<'PY'
import sys
import numpy as np

replay_file = sys.argv[1]
with np.load(replay_file, allow_pickle=True) as data:
    task = data.get("task")
    if task is None:
        raise SystemExit("Replay file missing 'task' metadata")
    if hasattr(task, "item"):
        task = task.item()
    print(task)
PY
)"
fi

if [ -z "${TASK_NAME}" ]; then
    echo "Error: Failed to resolve task name from replay file"
    exit 1
fi

echo "Detected replay task: $TASK_NAME"

# Match the robot USD override used by run_twist2.sh.
ROBOT_COLLIDER_MODE="${ROBOT_COLLIDER_MODE:-box}"
if [ "${ROBOT_COLLIDER_MODE}" = "fourpoints" ]; then
    export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
else
    export ROBOT_USD_OVERRIDE="${SCRIPT_DIR}/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
fi
echo "[robot_usd] mode=${ROBOT_COLLIDER_MODE}"
echo "[robot_usd] path=${ROBOT_USD_OVERRIDE}"

# Isaac Lab environment setup
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/.local/share/ov/pkg/isaac-sim-4.2.0}"
if [ -d "$ISAACLAB_PATH" ]; then
    export ISAACLAB_PATH
    echo "Using Isaac Lab: $ISAACLAB_PATH"
else
    echo "Warning: Isaac Lab path not found at $ISAACLAB_PATH"
fi

# Launch replay simulation through the normal sim_main entry.
python "${SCRIPT_DIR}/sim_main.py" \
    --device "${REPLAY_DEVICE}" \
    --enable_cameras \
    --task "${TASK_NAME}" \
    --env_config_yaml "${ENV_CONFIG_YAML}" \
    --robot_type g129 \
    --input_source replay \
    --gmt_backend twist2 \
    --replay_file "$REPLAY_FILE" \
    --replay_mode "$REPLAY_MODE" \
    ${LOOP_FLAG:+--replay_loop} \
    --model_path /home/dreams/Users/taowen/HumanoidArena/TWIST2/assets/ckpts/twist2_1017_20k.onnx \
    --image_transport zmq \
    --image_fps 30 \
    --image_zmq_port 5555 \
    --stats_interval 10.0 \
    --step_hz 50 \
    --video_fps 30 \
    --seed "${SEED}"

echo ""
echo "Replay simulation finished"
