#!/usr/bin/env bash
# run_replay_sonic_model.sh - 用 SONIC encoder/decoder 在 IsaacLab 中回放（从 recording_data_for_debug/*.npz 发布 pose）
#
# 说明：
# - 这条路径是真正“跟 SONIC 模型”的：action_source=sonic_wholebody（同 run_sonic.sh）
# - 我们用 `tools/sonic_pose_npz_replay_server.py` 把 .npz 里的 human_smplx_data 发布成 ZMQ "pose" (Protocol v3)
# - 然后让 `SonicActionProvider` 读取 ZMQ pose，跑 encoder+decoder，输出 29DOF target
#
# 用法：
#   bash run_replay_sonic_model.sh [npz_path]
#
# 示例：
#   bash run_replay_sonic_model.sh
#   bash run_replay_sonic_model.sh ./recording_data_for_debug/Isaac-Move-Football-G129-Dex3-Wholebody_1773215051410272.npz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDING_DIR="${SCRIPT_DIR}/recording_data_for_debug"

# 选择 npz（默认使用固定的调试 npz 文件，也可通过参数覆盖）
# DEFAULT_NPZ="/home/dreams/Users/Alyssa/HumanoidArena/isaaclab_twist2_g1/recording_data_for_debug/Isaac-Move-Football-G129-Dex3-Wholebody_smpl_Left_Shoulder_global_0_to_-3_to_0_aggressive.npz"
DEFAULT_NPZ="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/former/Isaac-Move-Football-G129-Dex3-Wholebody_1773495310633878.npz"
NPZ="${1:-$DEFAULT_NPZ}"
if [ -z "$NPZ" ]; then
  echo "Error: NPZ path is empty"
  exit 1
fi
if [ ! -f "$NPZ" ]; then
  echo "Error: NPZ not found: $NPZ"
  exit 1
fi

# 后续会切换到 tools/ 目录启动 replay server，因此这里先把输入路径固化为绝对路径，
# 避免像 ./recording_data_for_debug/foo.npz 这样的相对路径在切目录后失效。
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

# 本地 pose 发布端口（避免和真实 pico 冲突）
SONIC_ZMQ_HOST="${SONIC_ZMQ_HOST:-localhost}"
SONIC_ZMQ_PORT="${SONIC_ZMQ_PORT:-5566}"

# 选择能够同时运行 replay server 和 IsaacLab 的 Python
# 优先级：
# 1. 用户显式指定的 SONIC_PYTHON_BIN
# 2. 当前环境里的 python
# 3. 常用 IsaacLab conda 环境
PYTHON_BIN="${SONIC_PYTHON_BIN:-python}"
DEFAULT_ISAACLAB_PY="/home/dreams/miniconda3/envs/env_isaaclab_510_yb/bin/python"

if ! "$PYTHON_BIN" -c "import zmq, onnxruntime" >/dev/null 2>&1; then
  if [ -x "$DEFAULT_ISAACLAB_PY" ] && "$DEFAULT_ISAACLAB_PY" -c "import zmq, onnxruntime" >/dev/null 2>&1; then
    PYTHON_BIN="$DEFAULT_ISAACLAB_PY"
  fi
fi

if ! "$PYTHON_BIN" -c "import zmq, onnxruntime" >/dev/null 2>&1; then
  echo "Error: usable python not found for SONIC replay"
  echo "Tried: ${PYTHON_BIN}"
  echo "Need a python with both 'pyzmq' and 'onnxruntime' installed."
  echo "You can set: SONIC_PYTHON_BIN=/path/to/python"
  exit 1
fi

export PROJECT_ROOT="$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

LOG_DIR="${SCRIPT_DIR}/logs/sonic_replay"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
REPLAY_LOG="${LOG_DIR}/replay_server_${RUN_TS}.log"
SIM_LOG="${LOG_DIR}/sim_main_${RUN_TS}.log"

echo "=========================================="
echo "SONIC model replay from NPZ"
echo "NPZ: $NPZ"
echo "ZMQ: tcp://${SONIC_ZMQ_HOST}:${SONIC_ZMQ_PORT} topic=pose (v3)"
echo "Encoder: $ENCODER_PATH"
echo "Decoder: $DECODER_PATH"
echo "Python: $PYTHON_BIN"
echo "Replay log: $REPLAY_LOG"
echo "Sim log: $SIM_LOG"
echo "=========================================="

# 启动 pose replay server（后台）
# 调用方式与 sonic_readme 中 pico_server_pose_only.py 一致：进入脚本所在目录后 python script.py --port ... [--vis_vr3pt] [--vis_smpl]
REPLAY_TOOLS_DIR="${SCRIPT_DIR}/tools"
VIS_FLAGS=""
if [ -n "${REPLAY_VIS:-}" ]; then
  VIS_FLAGS="--vis_vr3pt --vis_smpl"
fi
cd "$REPLAY_TOOLS_DIR"
"$PYTHON_BIN" sonic_pose_npz_replay_server.py \
  --npz "$NPZ" \
  --port "$SONIC_ZMQ_PORT" \
  --fps 30 \
  --loop \
  $VIS_FLAGS > "$REPLAY_LOG" 2>&1 &
cd "$SCRIPT_DIR"
REPLAY_PID=$!
cleanup() {
  if kill -0 "$REPLAY_PID" >/dev/null 2>&1; then
    kill "$REPLAY_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# 清理 Redis 缓存（twist2 惯例）
redis-cli DEL \
  action_body_unitree_g1_with_hands \
  action_hand_left_unitree_g1_with_hands \
  action_hand_right_unitree_g1_with_hands \
  action_neck_unitree_g1_with_hands \
  controller_data \
  t_action || true

# 启动 Isaac Lab（真正走 sonic_wholebody）
cd "$SCRIPT_DIR"

{
  echo "[$(date '+%F %T')] Starting sim_main.py"
  echo "[$(date '+%F %T')] Working directory: $SCRIPT_DIR"
  echo "[$(date '+%F %T')] NPZ: $NPZ"
  echo "[$(date '+%F %T')] Python: $PYTHON_BIN"
  echo "[$(date '+%F %T')] Replay server PID: $REPLAY_PID"
} | tee -a "$SIM_LOG"

"$PYTHON_BIN" sim_main.py \
  --device cuda \
  --enable_cameras \
  --task Isaac-Move-Football-G129-Dex3-Wholebody \
  --robot_type g129 \
  --enable_dex3_dds \
  --action_source sonic_wholebody \
  --sonic_zmq_host "$SONIC_ZMQ_HOST" \
  --sonic_zmq_port "$SONIC_ZMQ_PORT" \
  --sonic_encoder_path "$ENCODER_PATH" \
  --sonic_decoder_path "$DECODER_PATH" \
  --image_transport zmq \
  --image_fps 30 \
  --image_zmq_port 5555 \
  --enable_world_camera 2>&1 | tee -a "$SIM_LOG"

