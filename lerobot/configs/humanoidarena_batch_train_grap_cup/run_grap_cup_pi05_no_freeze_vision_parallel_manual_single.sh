#!/usr/bin/env bash
set -euo pipefail

cd /ai/Yichi/taowen/HumanoidArena/lerobot
export PYTHONPATH=src
export TORCH_HOME=/ai/Yichi/taowen/ckpts/checkpoints/resnet
export LD_LIBRARY_PATH=/ai/Yichi/0_Systems/miniconda3/envs/lerobot/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=0

LOG_ROOT=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_pi05_no_freeze_vision_8gpu_serial_train_logs
mkdir -p "$LOG_ROOT"

# 8 卡串行版本：三个数据集按 sonic -> twist2 -> merge 顺序依次训练。
# 每个 job 都使用全部 8 卡。需要改卡号时只改这里。
GPUS=0,1,2,3,4,5,6,7
NPROC=8
GPU_MEMORY_THRESHOLD_MB=2048
GPU_CHECK_INTERVAL_SECONDS=300

wait_for_idle_gpus() {
  local job="$1"
  while true; do
    local busy=0
    local status
    status=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    echo "[gpu-check] $(date '+%F %T') before $job"
    while IFS=',' read -r raw_idx raw_mem; do
      local idx="${raw_idx//[[:space:]]/}"
      local mem="${raw_mem//[[:space:]]/}"
      if [[ ",$GPUS," == *",$idx,"* ]]; then
        echo "  GPU $idx: ${mem} MiB"
        if (( mem >= GPU_MEMORY_THRESHOLD_MB )); then
          busy=1
        fi
      fi
    done <<< "$status"

    if (( busy == 0 )); then
      echo "[gpu-check] all selected GPUs are below ${GPU_MEMORY_THRESHOLD_MB} MiB; starting $job"
      return 0
    fi

    echo "[gpu-check] selected GPUs are busy; wait ${GPU_CHECK_INTERVAL_SECONDS}s before rechecking"
    sleep "$GPU_CHECK_INTERVAL_SECONDS"
  done
}

wait_for_idle_gpus pi05_no_freeze_vision_HOI_grap_cup_sonic
echo "[start] pi05_no_freeze_vision_HOI_grap_cup_sonic on GPUs $GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NPROC \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.image_transforms.enable=false \
  --policy.type=pi05 \
  --policy.pretrained_path=/ai/Yichi/taowen/ckpts/checkpoints/pi05_base \
  --policy.device=cuda \
  --policy.max_state_dim=64 \
  --policy.max_action_dim=40 \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.optimizer_lr=2.5e-05 \
  --policy.push_to_hub=false \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --wandb.enable=true \
  --wandb.mode=online \
  --seed=42 \
  --batch_size=8 \
  --steps=50000 \
  --dataset.repo_id=local/sonic_grapcup_0423 \
  --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/sonic_grapcup_0423 \
  --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_pi05_no_freeze_vision_8gpu_serial_train/pi05_no_freeze_vision_HOI_grap_cup_sonic \
  --job_name=pi05_no_freeze_vision_HOI_grap_cup_sonic \
  --wandb.project=HumanoidArena \
  --wandb.notes=task=HOI_grap_cup,dataset_kind=sonic,dataset=sonic_grapcup_0423,model=pi05_no_freeze_vision,manual=true,8gpu_serial=true \
  2>&1 | tee "$LOG_ROOT/pi05_no_freeze_vision_HOI_grap_cup_sonic.log"
echo "[ok] pi05_no_freeze_vision_HOI_grap_cup_sonic"

wait_for_idle_gpus pi05_no_freeze_vision_HOI_grap_cup_twist2
echo "[start] pi05_no_freeze_vision_HOI_grap_cup_twist2 on GPUs $GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NPROC \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.image_transforms.enable=false \
  --policy.type=pi05 \
  --policy.pretrained_path=/ai/Yichi/taowen/ckpts/checkpoints/pi05_base \
  --policy.device=cuda \
  --policy.max_state_dim=64 \
  --policy.max_action_dim=40 \
  --policy.n_obs_steps=1 \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.optimizer_lr=2.5e-05 \
  --policy.push_to_hub=false \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --wandb.enable=true \
  --wandb.mode=online \
  --seed=42 \
  --batch_size=8 \
  --steps=50000 \
  --dataset.repo_id=local/twist2_grapcup_0423 \
  --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/twist2_grapcup_0423 \
  --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_pi05_no_freeze_vision_8gpu_serial_train/pi05_no_freeze_vision_HOI_grap_cup_twist2 \
  --job_name=pi05_no_freeze_vision_HOI_grap_cup_twist2 \
  --wandb.project=HumanoidArena \
  --wandb.notes=task=HOI_grap_cup,dataset_kind=twist2,dataset=twist2_grapcup_0423,model=pi05_no_freeze_vision,manual=true,8gpu_serial=true \
  2>&1 | tee "$LOG_ROOT/pi05_no_freeze_vision_HOI_grap_cup_twist2.log"
echo "[ok] pi05_no_freeze_vision_HOI_grap_cup_twist2"

# wait_for_idle_gpus pi05_no_freeze_vision_HOI_grap_cup_merge
# echo "[start] pi05_no_freeze_vision_HOI_grap_cup_merge on GPUs $GPUS"
# CUDA_VISIBLE_DEVICES="$GPUS" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python -m torch.distributed.run \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=$NPROC \
#   src/lerobot/scripts/lerobot_train.py \
#   --dataset.image_transforms.enable=false \
#   --policy.type=pi05 \
#   --policy.pretrained_path=/ai/Yichi/taowen/ckpts/checkpoints/pi05_base \
#   --policy.device=cuda \
#   --policy.max_state_dim=64 \
#   --policy.max_action_dim=40 \
#   --policy.n_obs_steps=1 \
#   --policy.chunk_size=20 \
#   --policy.n_action_steps=20 \
#   --policy.optimizer_lr=2.5e-05 \
#   --policy.push_to_hub=false \
#   --policy.compile_model=true \
#   --policy.gradient_checkpointing=true \
#   --policy.dtype=bfloat16 \
#   --policy.freeze_vision_encoder=false \
#   --policy.train_expert_only=false \
#   --wandb.enable=true \
#   --wandb.mode=online \
#   --seed=42 \
#   --batch_size=8 \
#   --steps=50000 \
#   --dataset.repo_id=local/merged_grabcup \
#   --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/merged_grabcup \
#   --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_pi05_no_freeze_vision_8gpu_serial_train/pi05_no_freeze_vision_HOI_grap_cup_merge \
#   --job_name=pi05_no_freeze_vision_HOI_grap_cup_merge \
#   --wandb.project=HumanoidArena \
#   --wandb.notes=task=HOI_grap_cup,dataset_kind=merge,dataset=merged_grabcup,model=pi05_no_freeze_vision,manual=true,8gpu_serial=true \
#   2>&1 | tee "$LOG_ROOT/pi05_no_freeze_vision_HOI_grap_cup_merge.log"
# echo "[ok] pi05_no_freeze_vision_HOI_grap_cup_merge"
