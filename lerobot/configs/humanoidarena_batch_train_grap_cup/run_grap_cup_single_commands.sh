#!/usr/bin/env bash
set -euo pipefail

cd /ai/Yichi/taowen/HumanoidArena/lerobot
export PYTHONPATH=src
export TORCH_HOME=/ai/Yichi/taowen/ckpts/checkpoints/resnet
export LD_LIBRARY_PATH=/ai/Yichi/0_Systems/miniconda3/envs/lerobot/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=0

# 用法：复制下面任意一个命令单独执行；按需修改 CUDA_VISIBLE_DEVICES。
# 输出根目录：/ai/Yichi/taowen/HumanoidArena/lerobot/results/humanoidarena_grap_cup_manual_train_v2
# DP/MTP 显式设置 224x224 policy resize，用来匹配 dataset transform 后的真实输入尺寸。
# 如果你想覆盖旧结果，请先手动确认并清理对应 output_dir。

# ===== act_HOI_grap_cup_sonic =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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

# ===== dp_HOI_grap_cup_sonic =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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

# ===== mtp_HOI_grap_cup_sonic =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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

# ===== act_HOI_grap_cup_twist2 =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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

# ===== dp_HOI_grap_cup_twist2 =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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

# ===== mtp_HOI_grap_cup_twist2 =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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

# ===== act_HOI_grap_cup_merge =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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

# ===== dp_HOI_grap_cup_merge =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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

# ===== mtp_HOI_grap_cup_merge =====
CUDA_VISIBLE_DEVICES=0 /ai/Yichi/0_Systems/miniconda3/envs/lerobot/bin/python src/lerobot/scripts/lerobot_train.py \
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
