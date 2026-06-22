# IsaacLab 命令速查

本文集中整理 `isaaclab_twist2_g1/` 当前常用的运行命令，覆盖：

- 遥操作录制
- replay
- rerecord
- LeRobot / VLA 推理评测

这些命令大多依赖脚本顶部的本地路径和模型路径配置。运行前先检查对应 `.sh` 文件里的变量。

## 1. 遥操作录制

### 1.1 TWIST2 teleop 上游

```bash
cd TWIST2
bash teleop.sh
```

### 1.2 SONIC / TWIST2 在 IsaacLab 中启动 live 录制

在仓库根目录执行：

```bash
bash isaaclab_twist2_g1/run_twist2.sh
```

```bash
bash isaaclab_twist2_g1/run_sonic.sh
```

### 1.3 OpenDoor 的 SONIC joint29 live 推理 / 录制

```bash
bash isaaclab_twist2_g1/run_sonic_joint29.sh
```

这条链现在会自动把 open-door 解析为 `live_inference -> inference` profile。

## 2. Replay

### 2.1 脚本入口

```bash
bash isaaclab_twist2_g1/run_replay_twist2.sh
```

```bash
bash isaaclab_twist2_g1/run_replay_sonic.sh
```

### 2.2 直接调用 `sim_main.py` 做 SONIC replay

把 `replay_file` 换成目标 `.npz`：

```bash
python isaaclab_twist2_g1/sim_main.py \
  --device cpu \
  --env_config_yaml isaaclab_twist2_g1/tasks/common_env_config/opendoor_sonic.yaml \
  --task Isaac-Move-Open-Door-G129-Dex3-Wholebody \
  --robot_type g129 \
  --input_source replay \
  --gmt_backend sonic \
  --sonic_encoder_path ${GROOT_ROOT}/gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder_path ${GROOT_ROOT}/gear_sonic_deploy/policy/release/model_decoder.onnx \
  --replay_file /path/to/open_door_sonic_recording.npz \
  --replay_mode direct_replay \
  --enable_cameras \
  --enable_dex3_dds \
  --seed 42
```

`replay_mode` 可切换为：

- `direct_replay`
- `inference_replay`

当前 open-door 会自动把：

- `direct_replay`
- `inference_replay`

都解析到 `replay_compat` profile。

## 3. Rerecord

### 3.1 并行 rerecord 包装脚本

```bash
bash isaaclab_twist2_g1/run_rerecord.sh
```

可通过环境变量控制输入目录和并发，例如：

```bash
TWIST2_INPUT_ROOT=/path/to/twist2 \
SONIC_SOURCE_ROOT=/path/to/sonic \
TWIST2_PARALLEL_JOBS=2 \
SONIC_PARALLEL_JOBS=2 \
bash isaaclab_twist2_g1/run_rerecord.sh
```

### 3.2 SONIC rerecord

```bash
python isaaclab_twist2_g1/tools/data_tools/rerecord_sonic_recordings_to_multicam.py \
  /path/to/sonic_source_root \
  --parallel-jobs 1
```

### 3.3 OpenDoor SONIC rerecord

这是已经验证过的 open-door 自动 profile 版本：

```bash
python isaaclab_twist2_g1/tools/data_tools/rerecord_sonic_recordings_to_multicam.py \
  ${ISAACLAB_ROOT}/recording_data/perspective-use/ \
  --enable-perspective-camera \
  --disable-front-camera \
  --disable-wrist-cameras \
  --parallel-jobs 1 \
  --force
```

这条链现在会自动把 open-door 解析为：

- `context=rerecord`
- `profile=replay_compat`

等价于旧的：

```bash
OPEN_DOOR_LATCH_DISABLE=1 OPEN_DOOR_SCENE_AS_ARTICULATION=0 ...
```

如果要显式指定：

```bash
python isaaclab_twist2_g1/tools/data_tools/rerecord_sonic_recordings_to_multicam.py \
  ${ISAACLAB_ROOT}/recording_data/perspective-use/ \
  --task-runtime-profile replay_compat \
  --enable-perspective-camera \
  --disable-front-camera \
  --disable-wrist-cameras \
  --parallel-jobs 1 \
  --force
```

### 3.4 TWIST2 rerecord

```bash
python isaaclab_twist2_g1/tools/data_tools/rerecord_twist2_recordings_to_multicam.py \
  /path/to/twist2_input_root \
  --parallel-jobs 1
```

## 4. LeRobot / VLA 评测

### 4.1 单任务运行脚本

例如：

```bash
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval.sh
```

```bash
bash isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval.sh
```

并行版本：

```bash
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval_parallel.sh
```

```bash
bash isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval_parallel.sh
```

### 4.2 批量任务评测

按任务分的 batch 脚本例如：

```bash
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/path/to/pi0.5_sonic_doubledesk_checkpoint" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash isaaclab_twist2_g1/pi05_batch_test_doubledesk.sh
```

OpenDoor：

```bash
TEST_MODE=semantic EVAL_BACKEND=sonic MODEL_ROOT="/path/to/pi0.5_sonic_opendoor_checkpoint" MODEL_GLOB="*" NUM_WORKERS=2 RESULTS_TAG_PREFIX=pi05_semantic bash isaaclab_twist2_g1/pi05_batch_test_open_door.sh
```

### 4.3 你提到的这类入口

仓库当前没有搜到 `batch_1_test_v31_sonic.sh`，但同类入口是有的，比如：

```bash
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_GLOB="*sonic*" RESULTS_TAG_PREFIX=merage-v3-sonic bash isaaclab_twist2_g1/batch_1_test_v3_sonic.sh
```

以及更通用的：

```bash
TEST_MODE=base_test EVAL_BACKEND=sonic MODEL_GLOB="*all*" RESULTS_TAG_PREFIX=merage-v3-all-sonic bash isaaclab_twist2_g1/batch_1_test_v3_all.sh
```

如果你后续新增了 `batch_1_test_v31_sonic.sh`，建议也按同样格式补进这里。

## 5. 相关说明文档

- [TWIST2_DATA_FORMAT.md](docs/TWIST2_DATA_FORMAT.md)
- [SONIC_DATA_FORMAT.md](docs/SONIC_DATA_FORMAT.md)
- [OPEN_DOOR_LATCH_EVENT_AND_ASSET_NOTES.md](docs/OPEN_DOOR_LATCH_EVENT_AND_ASSET_NOTES.md)
- [SCENE_RANDOMIZATION_SEED_RULES.md](docs/SCENE_RANDOMIZATION_SEED_RULES.md)
