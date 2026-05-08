#!/usr/bin/env bash
set -euo pipefail

cd /ai/Yichi/taowen/HumanoidArena/lerobot
export PYTHONPATH=src
export TORCH_HOME=/ai/Yichi/taowen/ckpts/checkpoints/resnet
export LD_LIBRARY_PATH=/ai/Yichi/0_Systems/miniconda3/envs/lerobot/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=0

LOG_ROOT=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2_logs
mkdir -p "$LOG_ROOT"

# 改这里就行：每个 job 填你想用的 GPU 编号。
# 如果不想跑某个 job，把对应值改成 skip。
declare -A GPU_MAP=(
  [act_HOI_grap_cup_sonic]=6
  [dp_HOI_grap_cup_sonic]=3
  [mtp_HOI_grap_cup_sonic]=0
  [act_HOI_grap_cup_twist2]=7
  [dp_HOI_grap_cup_twist2]=4
  [mtp_HOI_grap_cup_twist2]=1
  [act_HOI_grap_cup_merge]=6
  [dp_HOI_grap_cup_merge]=5
  [mtp_HOI_grap_cup_merge]=2
)

pids=()
start_job() {
  local job="$1"
  local gpu="${GPU_MAP[$job]}"
  if [[ "$gpu" == "skip" ]]; then
    echo "[skip] $job"
    return 0
  fi
  echo "[start] $job on GPU $gpu"
  (

    if [[ "$job" == "act_HOI_grap_cup_sonic" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=act \
        --policy.vision_backbone=resnet18 \
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.chunk_size=20 \
        --policy.n_action_steps=20 \
        --policy.optimizer_lr=1e-05 \
        --policy.push_to_hub=false \
        --dataset.repo_id=local/sonic_grapcup_0423 \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/sonic_grapcup_0423 \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/act_HOI_grap_cup_sonic \
        --job_name=act_HOI_grap_cup_sonic \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=sonic,dataset=sonic_grapcup_0423,model=act,manual=true,v2=true
      return $?
    fi

    if [[ "$job" == "dp_HOI_grap_cup_sonic" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=diffusion \
        --policy.vision_backbone=resnet18 \
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
        --policy.use_group_norm=false \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.horizon=24 \
        --policy.n_action_steps=20 \
        --policy.resize_shape='[224,224]' \
        --policy.optimizer_lr=0.0001 \
        --policy.push_to_hub=false \
        --dataset.repo_id=local/sonic_grapcup_0423 \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/sonic_grapcup_0423 \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/dp_HOI_grap_cup_sonic \
        --job_name=dp_HOI_grap_cup_sonic \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=sonic,dataset=sonic_grapcup_0423,model=dp,manual=true,v2=true
      return $?
    fi

    if [[ "$job" == "mtp_HOI_grap_cup_sonic" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=multi_task_dit \
        --policy.objective=flow_matching \
        --policy.vision_encoder_name=/ai/Yichi/taowen/ckpts/checkpoints/clip-vit-base-patch16 \
        --policy.text_encoder_name=/ai/Yichi/taowen/ckpts/checkpoints/clip-vit-base-patch16 \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.horizon=40 \
        --policy.n_action_steps=20 \
        --policy.image_resize_shape='[224,224]' \
        --policy.optimizer_lr=2e-05 \
        --policy.push_to_hub=false \
        --rename_map='{}' \
        --dataset.repo_id=local/sonic_grapcup_0423 \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/sonic_grapcup_0423 \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/mtp_HOI_grap_cup_sonic \
        --job_name=mtp_HOI_grap_cup_sonic \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=sonic,dataset=sonic_grapcup_0423,model=mtp,manual=true,v2=true
      return $?
    fi

    if [[ "$job" == "act_HOI_grap_cup_twist2" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=act \
        --policy.vision_backbone=resnet18 \
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.chunk_size=20 \
        --policy.n_action_steps=20 \
        --policy.optimizer_lr=1e-05 \
        --policy.push_to_hub=false \
        --dataset.repo_id=local/twist2_grapcup_0423 \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/twist2_grapcup_0423 \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/act_HOI_grap_cup_twist2 \
        --job_name=act_HOI_grap_cup_twist2 \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=twist2,dataset=twist2_grapcup_0423,model=act,manual=true,v2=true
      return $?
    fi

    if [[ "$job" == "dp_HOI_grap_cup_twist2" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=diffusion \
        --policy.vision_backbone=resnet18 \
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
        --policy.use_group_norm=false \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.horizon=24 \
        --policy.n_action_steps=20 \
        --policy.resize_shape='[224,224]' \
        --policy.optimizer_lr=0.0001 \
        --policy.push_to_hub=false \
        --dataset.repo_id=local/twist2_grapcup_0423 \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/twist2_grapcup_0423 \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/dp_HOI_grap_cup_twist2 \
        --job_name=dp_HOI_grap_cup_twist2 \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=twist2,dataset=twist2_grapcup_0423,model=dp,manual=true,v2=true
      return $?
    fi

    if [[ "$job" == "mtp_HOI_grap_cup_twist2" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=multi_task_dit \
        --policy.objective=flow_matching \
        --policy.vision_encoder_name=/ai/Yichi/taowen/ckpts/checkpoints/clip-vit-base-patch16 \
        --policy.text_encoder_name=/ai/Yichi/taowen/ckpts/checkpoints/clip-vit-base-patch16 \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.horizon=40 \
        --policy.n_action_steps=20 \
        --policy.image_resize_shape='[224,224]' \
        --policy.optimizer_lr=2e-05 \
        --policy.push_to_hub=false \
        --rename_map='{}' \
        --dataset.repo_id=local/twist2_grapcup_0423 \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/twist2_grapcup_0423 \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/mtp_HOI_grap_cup_twist2 \
        --job_name=mtp_HOI_grap_cup_twist2 \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=twist2,dataset=twist2_grapcup_0423,model=mtp,manual=true,v2=true
      return $?
    fi

    if [[ "$job" == "act_HOI_grap_cup_merge" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=act \
        --policy.vision_backbone=resnet18 \
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.chunk_size=20 \
        --policy.n_action_steps=20 \
        --policy.optimizer_lr=1e-05 \
        --policy.push_to_hub=false \
        --dataset.repo_id=local/merged_grabcup \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/merged_grabcup \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/act_HOI_grap_cup_merge \
        --job_name=act_HOI_grap_cup_merge \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=merge,dataset=merged_grabcup,model=act,manual=true,v2=true
      return $?
    fi

    if [[ "$job" == "dp_HOI_grap_cup_merge" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=diffusion \
        --policy.vision_backbone=resnet18 \
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
        --policy.use_group_norm=false \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.horizon=24 \
        --policy.n_action_steps=20 \
        --policy.resize_shape='[224,224]' \
        --policy.optimizer_lr=0.0001 \
        --policy.push_to_hub=false \
        --dataset.repo_id=local/merged_grabcup \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/merged_grabcup \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/dp_HOI_grap_cup_merge \
        --job_name=dp_HOI_grap_cup_merge \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=merge,dataset=merged_grabcup,model=dp,manual=true,v2=true
      return $?
    fi

    if [[ "$job" == "mtp_HOI_grap_cup_merge" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
        --dataset.image_transforms.enable=true \
        --wandb.enable=true \
        --wandb.mode=online \
        --wandb.project=HumanoidArena \
        --seed=42 \
        --batch_size=64 \
        --steps=50000 \
        --dataset.image_transforms.max_num_transforms=2 \
        --dataset.image_transforms.random_order=false \
        --dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[168,224]}},"pad":{"weight":1.0,"type":"Pad","kwargs":{"padding":[0,28,0,28],"fill":0}}}' \
        --policy.type=multi_task_dit \
        --policy.objective=flow_matching \
        --policy.vision_encoder_name=/ai/Yichi/taowen/ckpts/checkpoints/clip-vit-base-patch16 \
        --policy.text_encoder_name=/ai/Yichi/taowen/ckpts/checkpoints/clip-vit-base-patch16 \
        --policy.device=cuda \
        --policy.n_obs_steps=1 \
        --policy.horizon=40 \
        --policy.n_action_steps=20 \
        --policy.image_resize_shape='[224,224]' \
        --policy.optimizer_lr=2e-05 \
        --policy.push_to_hub=false \
        --rename_map='{}' \
        --dataset.repo_id=local/merged_grabcup \
        --dataset.root=/ai/Yichi/taowen/dataset/HOI_grap_cup/merged_grabcup \
        --output_dir=/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2/mtp_HOI_grap_cup_merge \
        --job_name=mtp_HOI_grap_cup_merge \
        --wandb.notes=task=HOI_grap_cup,dataset_kind=merge,dataset=merged_grabcup,model=mtp,manual=true,v2=true
      return $?
    fi

    echo "unknown job: $job" >&2
    return 2
  ) >"$LOG_ROOT/${job}.log" 2>&1 &
  pids+=("$!:$job")
}

start_job act_HOI_grap_cup_sonic
start_job dp_HOI_grap_cup_sonic
start_job mtp_HOI_grap_cup_sonic
start_job act_HOI_grap_cup_twist2
start_job dp_HOI_grap_cup_twist2
start_job mtp_HOI_grap_cup_twist2
start_job act_HOI_grap_cup_merge
start_job dp_HOI_grap_cup_merge
start_job mtp_HOI_grap_cup_merge

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
