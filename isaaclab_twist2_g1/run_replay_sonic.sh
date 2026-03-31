#!/usr/bin/env bash
# run_replay_sonic_model.sh - 在 IsaacLab 中直接用 SONIC 录制数据回放
#
# 说明：
# - direct_replay：直接执行录制里保存的 SONIC 29DOF target
# - inference_replay：从录制的 SONIC pose 数据重跑 encoder+decoder，再执行新推理结果
# - 整个 replay 流程都在对应 action provider 内部完成，不再启动外部 ZMQ/Redis replay server
#
# 用法：
#   bash run_replay_sonic_model.sh [npz_path] [direct_replay|inference_replay] [--loop]
#
# 示例：
#   bash run_replay_sonic_model.sh
#   bash run_replay_sonic_model.sh ./recording_data/sonic/tw/foo.npz direct_replay
#   bash run_replay_sonic_model.sh ./recording_data/sonic/tw/foo.npz inference_replay --loop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SONIC_ENCODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
SONIC_DECODER_PATH="/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
# 选择 npz（默认使用固定的调试 npz 文件，也可通过参数覆盖）
# DEFAULT_NPZ="/home/dreams/Users/Alyssa/HumanoidArena/isaaclab_twist2_g1/recording_data_for_debug/Isaac-Move-Football-G129-Dex3-Wholebody_smpl_Left_Shoulder_global_0_to_-3_to_0_aggressive.npz"
DEFAULT_NPZ="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/sonic/tw/Isaac-Move-Football-Single-G129-Dex3-Wholebody_sonic_1774946564919039.npz"
NPZ="${1:-$DEFAULT_NPZ}"
REPLAY_MODE="${2:-${SONIC_REPLAY_MODE:-inference_replay}}"
LOOP_FLAG=""
if [ -z "$NPZ" ]; then
  echo "Error: NPZ path is empty"
  exit 1
fi
if [ ! -f "$NPZ" ]; then
  echo "Error: NPZ not found: $NPZ"
  exit 1
fi
if [ "$REPLAY_MODE" != "direct_replay" ] && [ "$REPLAY_MODE" != "inference_replay" ]; then
  echo "Error: replay mode must be direct_replay or inference_replay, got: $REPLAY_MODE"
  exit 1
fi
if [ "${3:-}" = "--loop" ] || [ "${SONIC_REPLAY_LOOP:-}" = "1" ]; then
  LOOP_FLAG="--replay_loop"
fi

# 先把输入路径固化为绝对路径，避免工作目录变化带来路径歧义。
NPZ="$(realpath "$NPZ")"

# SONIC encoder/decoder：默认使用 GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/
# 若未下载过，请到 GR00T-WholeBodyControl 下执行: python download_from_hf.py
GROOT_ROOT="/home/dreams/Users/Alyssa/HumanoidArena_V1/GR00T-WholeBodyControl"
SONIC_POLICY_DIR="${GROOT_ROOT}/gear_sonic_deploy/policy/release"
ENCODER_PATH="${SONIC_ENCODER_PATH:-${SONIC_POLICY_DIR}/model_encoder.onnx}"
DECODER_PATH="${SONIC_DECODER_PATH:-${SONIC_POLICY_DIR}/model_decoder.onnx}"

if [ ! -f "$ENCODER_PATH" ]; then
  echo "Error: SONIC encoder not found: $ENCODER_PATH"
  echo "Download: cd ${GROOT_ROOT} && python download_from_hf.py"
  echo "Or set: SONIC_ENCODER_PATH=/path/to/model_encoder.onnx"
  exit 1
fi
if [ ! -f "$DECODER_PATH" ]; then
  echo "Error: SONIC decoder not found: $DECODER_PATH"
  echo "Download: cd ${GROOT_ROOT} && python download_from_hf.py"
  echo "Or set: SONIC_DECODER_PATH=/path/to/model_decoder.onnx"
  exit 1
fi

# 选择能够运行 IsaacLab + onnxruntime 的 Python
# 优先级：
# 1. 用户显式指定的 SONIC_PYTHON_BIN
# 2. 当前环境里的 python
# 3. 常用 IsaacLab conda 环境
PYTHON_BIN="${SONIC_PYTHON_BIN:-python}"
DEFAULT_ISAACLAB_PY="/home/dreams/miniconda3/envs/env_isaaclab_510_yb/bin/python"

if ! "$PYTHON_BIN" -c "import onnxruntime" >/dev/null 2>&1; then
  if [ -x "$DEFAULT_ISAACLAB_PY" ] && "$DEFAULT_ISAACLAB_PY" -c "import onnxruntime" >/dev/null 2>&1; then
    PYTHON_BIN="$DEFAULT_ISAACLAB_PY"
  fi
fi

if ! "$PYTHON_BIN" -c "import onnxruntime" >/dev/null 2>&1; then
  echo "Error: usable python not found for SONIC replay"
  echo "Tried: ${PYTHON_BIN}"
  echo "Need a python with 'onnxruntime' installed."
  echo "You can set: SONIC_PYTHON_BIN=/path/to/python"
  exit 1
fi

export PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

LOG_DIR="${SCRIPT_DIR}/logs/sonic_replay"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
SIM_LOG="${LOG_DIR}/sim_main_${RUN_TS}.log"

echo "=========================================="
echo "SONIC model replay from NPZ"
echo "NPZ: $NPZ"
echo "Replay mode: $REPLAY_MODE"
echo "Encoder: $ENCODER_PATH"
echo "Decoder: $DECODER_PATH"
echo "Loop: ${LOOP_FLAG:-disabled}"
echo "Python: $PYTHON_BIN"
echo "Sim log: $SIM_LOG"
echo "=========================================="

# 清理 Redis 缓存（twist2 惯例）
redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action || true

# 启动 Isaac Lab（通过统一 replay 入口路由到 sonic_wholebody）
cd "$SCRIPT_DIR"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-tasks/common_env_config/sonic_default.yaml}"

{
  echo "[$(date '+%F %T')] Starting sim_main.py"
  echo "[$(date '+%F %T')] Working directory: $SCRIPT_DIR"
  echo "[$(date '+%F %T')] NPZ: $NPZ"
  echo "[$(date '+%F %T')] Replay mode: $REPLAY_MODE"
  echo "[$(date '+%F %T')] Loop: ${LOOP_FLAG:-disabled}"
  echo "[$(date '+%F %T')] Python: $PYTHON_BIN"
} | tee -a "$SIM_LOG"

"$PYTHON_BIN" sim_main.py \
  --device cpu \
  --enable_cameras \
  --env_config_yaml "$ENV_CONFIG_YAML" \
  --task Isaac-Move-Football-Single-G129-Dex3-Wholebody \
  --robot_type g129 \
  --enable_dex3_dds \
  --input_source replay \
  --gmt_backend sonic \
  --sonic_encoder_path "$ENCODER_PATH" \
  --sonic_decoder_path "$DECODER_PATH" \
  --replay_file "$NPZ" \
  --replay_mode "$REPLAY_MODE" \
  ${LOOP_FLAG} \
  --image_transport zmq \
  --image_fps 30 \
  --image_zmq_port 5555 \
  --enable_world_camera 2>&1 | tee -a "$SIM_LOG"
