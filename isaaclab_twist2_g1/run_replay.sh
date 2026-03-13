#!/bin/bash
# run_replay.sh - Launch replay simulation
# Usage:
#   ./run_replay.sh <replay_file> [inference|direct] [--loop]
#
# Examples:
#   ./run_replay.sh ./recording_data/recording_20260311_123456.npz inference
#   ./run_replay.sh ./recording_data/recording_20260311_123456.npz direct --loop

set -e

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
echo "Replay mode: $REPLAY_MODE"
echo "Loop: ${LOOP_FLAG:-disabled}"
echo "=========================================="

# Random seed for reproducibility (set to fixed value for deterministic behavior)
SEED="${SEED:-42}"
echo "Random seed: $SEED"
echo "=========================================="

# Set environment variables
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Isaac Lab environment setup
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/.local/share/ov/pkg/isaac-sim-4.2.0}"
if [ -d "$ISAACLAB_PATH" ]; then
    export ISAACLAB_PATH
    echo "Using Isaac Lab: $ISAACLAB_PATH"
else
    echo "Warning: Isaac Lab path not found at $ISAACLAB_PATH"
fi

# Launch replay simulation
python sim_main_replay.py \
    --device cuda:0 \
    --enable_cameras \
    --task Isaac-Move-Football-G129-Dex3-Wholebody \
    --robot_type g129 \
    --replay_file "$REPLAY_FILE" \
    --replay_mode "$REPLAY_MODE" \
    ${LOOP_FLAG:+--replay_loop} \
    --model_path /home/dreams/Users/taowen/HumanoidArena/TWIST2/assets/ckpts/twist2_1017_20k.onnx \
    --image_transport zmq \
    --image_fps 30 \
    --image_zmq_port 5555 \
    --stats_interval 10.0 \
    --step_hz 50 \
    --gravity_z -9.8 \
    --seed "${SEED}"

echo ""
echo "Replay simulation finished"
