#!/usr/bin/env bash
set -euo pipefail

cd /ai/Yichi/taowen/HumanoidArena/lerobot
export PYTHONPATH=src
export TORCH_HOME=/ai/Yichi/taowen/ckpts/checkpoints/resnet
export LD_LIBRARY_PATH=/ai/Yichi/0_Systems/miniconda3/envs/lerobot/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=0

LOG_ROOT=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_pi05_no_freeze_vision_manual_train_logs
mkdir -p "$LOG_ROOT"

# 每个 pi05 job 使用 4 卡 torchrun。
# 8 卡机器一次最多同时跑两个；三个数据集全并行需要 12 卡。
# 如果你确实有 12 卡，把 merge 改成 8,9,10,11；否则先跑前两个，跑完后再把 merge 改成 0,1,2,3。
# 不想跑某个 job 就填 skip。
declare -A GPU_MAP=(
  [pi05_no_freeze_vision_HOI_grap_cup_sonic]=0,1,2,3
  [pi05_no_freeze_vision_HOI_grap_cup_twist2]=4,5,6,7
  [pi05_no_freeze_vision_HOI_grap_cup_merge]=skip
)

count_gpus() {
  local gpus="$1"
  if [[ "$gpus" == "skip" || -z "$gpus" ]]; then
    echo 0
    return
  fi
  local no_commas="${gpus//,/}"
  echo $(( ${#gpus} - ${#no_commas} + 1 ))
}

pids=()
start_job() {
  local job="$1"
  local gpus="${GPU_MAP[$job]}"
  if [[ "$gpus" == "skip" ]]; then
    echo "[skip] $job"
    return 0
  fi
  local nproc
  nproc=$(count_gpus "$gpus")
  if [[ "$nproc" -ne 4 ]]; then
    echo "[error] $job requires exactly 4 GPUs, got $gpus" >&2
    return 2
  fi
  echo "[start] $job on GPUs $gpus"
  (

    if [[ "$job" == "pi05_no_freeze_vision_HOI_grap_cup_sonic" ]]; then
      CUDA_VISIBLE_DEVICES="$gpus" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=4 \
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
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_pi05_no_freeze_vision_manual_train/pi05_no_freeze_vision_HOI_grap_cup_sonic \
        --job_name=pi05_no_freeze_vision_HOI_grap_cup_sonic \
        --wandb.project=HumanoidArena \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=sonic,dataset=sonic_grapcup_0423,model=pi05_no_freeze_vision,manual=true,4gpu=true
      return $?
    fi

    if [[ "$job" == "pi05_no_freeze_vision_HOI_grap_cup_twist2" ]]; then
      CUDA_VISIBLE_DEVICES="$gpus" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=4 \
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
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_pi05_no_freeze_vision_manual_train/pi05_no_freeze_vision_HOI_grap_cup_twist2 \
        --job_name=pi05_no_freeze_vision_HOI_grap_cup_twist2 \
        --wandb.project=HumanoidArena \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=twist2,dataset=twist2_grapcup_0423,model=pi05_no_freeze_vision,manual=true,4gpu=true
      return $?
    fi

    if [[ "$job" == "pi05_no_freeze_vision_HOI_grap_cup_merge" ]]; then
      CUDA_VISIBLE_DEVICES="$gpus" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=4 \
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
        --dataset.repo_id=local/merged_grabcup \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/merged_grabcup \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_pi05_no_freeze_vision_manual_train/pi05_no_freeze_vision_HOI_grap_cup_merge \
        --job_name=pi05_no_freeze_vision_HOI_grap_cup_merge \
        --wandb.project=HumanoidArena \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=merge,dataset=merged_grabcup,model=pi05_no_freeze_vision,manual=true,4gpu=true
      return $?
    fi

    echo "unknown job: $job" >&2
    return 2
  ) >"$LOG_ROOT/${job}.log" 2>&1 &
  pids+=("$!:$job")
}

start_job pi05_no_freeze_vision_HOI_grap_cup_sonic
start_job pi05_no_freeze_vision_HOI_grap_cup_twist2
start_job pi05_no_freeze_vision_HOI_grap_cup_merge

failed=0
for item in "${pids[@]}"; do
  pid="${item%%:*}"
  job="${item#*:}"
  if wait "$pid"; then
    echo "[ok] $job"
  else
    rc=$?
    echo "[failed] $job rc=$rc log=$LOG_ROOT/${job}.log" >&2
    failed=1
  fi
done

exit $failed
