# Evaluation

This guide collects the maintained evaluation entrypoints for LeRobot, vision execution, semantic evaluation, and batch testing.

## 1. Evaluation Modes

HumanoidArena evaluation scripts commonly use these environment variables:

```text
TEST_MODE
EVAL_BACKEND
MODEL_ROOT
MODEL_GLOB
NUM_WORKERS
RESULTS_TAG_PREFIX
EVAL_BACKENDS
EVAL_TASKS
RUN_TIMESTAMP
RESUME_LATEST
DRY_RUN
```

Typical `TEST_MODE` values include:

| Mode | Meaning |
| --- | --- |
| `base_test` | <small><span style="color: #666;">Matches teleoperation data conditions: object randomization range, asset textures, and lighting stay consistent with the demonstration setup.</span></small> |
| `semantic` | <small><span style="color: #666;">Applies semantic changes around task-relevant objects, such as texture/material replacement or semantically similar distractors in the task area.</span></small> |
| `vision` | <small><span style="color: #666;">Randomizes lighting conditions relative to the original teleoperation data.</span></small> |
| `execution` | <small><span style="color: #666;">Expands the randomization range for task-relevant assets beyond the range used during teleoperation.</span></small> |

Typical `EVAL_BACKEND` values include:

```text
twist2
sonic
mimic_lite
sonic_low_latency
```

`sonic_low_latency` is a SONIC execution variant that uses the low-latency encoder observation layout. It is currently supported for evaluation only. MimicLite supports both teleoperation/data collection and evaluation.

## 2. Checkpoint Layout

Download the released model checkpoints from Hugging Face into a local directory and keep the release folder layout:

```bash
huggingface-cli download HumanoidArena/<model-repo> \
  --local-dir /path/to/humanoidarena_checkpoints
```

Batch wrappers use `MODEL_ROOT_BASE` and derive task-specific checkpoint paths from it:

```text
/path/to/humanoidarena_checkpoints/
  small/
    HSI_open_door/
    HOI_pp_box/
    HSI_boxing/
    HOI_football/
    HOI_double_desk/
    HSI_sit_sofa/
    HSI_vision_navi/
  small_merge/
  pi/
    HSI_boxing/
    HOI_football/
    HOI_pp_box/
    HOI_double_desk/
    HSI_open_door/
    HSI_sit_sofa/
    HSI_vision_navi/
```

Set `MODEL_ROOT` directly only for single-task scripts. Set `MODEL_ROOT_BASE` for the batch wrappers listed below.

## 3. Single-Task VLA Evaluation

SONIC:

```bash
MODEL_PATH=/path/to/checkpoint/pretrained_model \
EVAL_SEEDS="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval.sh
```

TWIST2:

```bash
MODEL_PATH=/path/to/checkpoint/pretrained_model \
EVAL_SEEDS="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval.sh
```

MimicLite:

```bash
MODEL_PATH=/path/to/checkpoint/pretrained_model \
EVAL_SEEDS="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/mimic_lite/run_vla_eval.sh
```

SONIC low-latency:

```bash
MODEL_PATH=/path/to/checkpoint/pretrained_model \
EVAL_SEEDS="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/sonic_low_latency/run_vla_eval.sh
```

The low-latency runner loads `GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_encoder.onnx` and `model_decoder.onnx` by default. Override `SONIC_ENCODER_PATH` and `SONIC_DECODER_PATH` when the artifacts live elsewhere.

`MODEL_PATH` can also be the parent checkpoint directory when it contains a `pretrained_model/` subdirectory. Use `RESULTS_DIR` to override the output directory and `REPEATS_PER_SEED` to repeat each seed. For PI0.5 checkpoints, use `script/eval_scripts/sonic_pi05/run_vla_eval.sh` or `script/eval_scripts/twist2_pi05/run_vla_eval.sh` with the same variables.

Parallel variants:

```bash
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval_parallel.sh
bash isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval_parallel.sh
bash isaaclab_twist2_g1/script/eval_scripts/mimic_lite/run_vla_eval_parallel.sh
bash isaaclab_twist2_g1/script/eval_scripts/sonic_low_latency/run_vla_eval_parallel.sh
```

PI0.5 variants are under:

```text
isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/
isaaclab_twist2_g1/script/eval_scripts/twist2_pi05/
```

### SONIC latent VLA policies

For SONIC policies trained to output encoder latents, set `SONIC_VLA_ACTION_FORMAT=latent64` and use the normal live VLA path with `--gmt_backend sonic`. The runtime accepts a 64-D latent action, or a 66-D action with left/right hand binary commands appended.

This interface is experimental and intended for testing latent-output VLA policies. It has not yet been validated for exact replay equivalence with direct replay.

HTTP inference servers may return `latent64`, `latent64_chunk`, `encoder_latent`, or `encoder_latent_chunk`:

```bash
SONIC_VLA_ACTION_FORMAT=latent64 \
python isaaclab_twist2_g1/sim_main.py \
  --input_source vla \
  --gmt_backend sonic \
  --lerobot_server_url http://127.0.0.1:18080 \
  --sonic_decoder_path /path/to/sonic_decoder.onnx \
  --task Isaac-Move-Open-Door-G129-Dex3-Wholebody \
  --robot_type g129 \
  --enable_cameras
```

Keep `SONIC_VLA_ACTION_FORMAT` unset, or set it to `semantic_v3`, for the default 40-D semantic VLA action interface.

## 4. Vision Execution Evaluation

Vision execution uses the same simulator-side evaluation workers, with model paths and camera settings configured in the selected shell script. Before launching, check:

- `MODEL_ROOT_BASE` points to the downloaded checkpoint root for batch wrappers.
- Camera flags match the dataset used for training.
- The selected backend matches the training data backend.
- Result paths point to a local output directory, not a tracked git path.

A standard SONIC launch shape is:

```bash
TEST_MODE=base_test \
EVAL_BACKEND=sonic \
MODEL_ROOT_BASE="/path/to/humanoidarena_checkpoints" \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
RESULTS_TAG_PREFIX=vision_exec \
bash isaaclab_twist2_g1/batch_test_scripts/batch_1_test_v31_sonic.sh
```

TWIST2:

```bash
TEST_MODE=base_test \
EVAL_BACKEND=twist2 \
MODEL_ROOT_BASE="/path/to/humanoidarena_checkpoints" \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
RESULTS_TAG_PREFIX=vision_exec \
bash isaaclab_twist2_g1/batch_test_scripts/batch_1_test_v31_twist2.sh
```

## 5. Semantic Evaluation

Semantic evaluation uses `TEST_MODE=semantic`.

SONIC example:

```bash
TEST_MODE=semantic \
EVAL_BACKEND=sonic \
MODEL_ROOT="/path/to/pi0.5_sonic_checkpoint" \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
RESULTS_TAG_PREFIX=pi05_semantic \
bash isaaclab_twist2_g1/batch_test_scripts/task/pi05_batch_test_doubledesk.sh
```

Open-door example:

```bash
TEST_MODE=semantic \
EVAL_BACKEND=sonic \
MODEL_ROOT="/path/to/pi0.5_sonic_opendoor_checkpoint" \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
RESULTS_TAG_PREFIX=pi05_semantic \
bash isaaclab_twist2_g1/batch_test_scripts/task/pi05_batch_test_open_door.sh
```

## 6. Batch Evaluation

Primary maintained batch entrypoints live under `isaaclab_twist2_g1/`:

```text
batch_test_scripts/batch_1_test_v31_sonic.sh
batch_test_scripts/batch_1_test_v31_twist2.sh
batch_test_scripts/batch_1_test_v31_merage.sh
batch_test_scripts/batch_pi05_v31_sonic.sh
batch_test_scripts/batch_pi05_v31_twist2.sh
batch_test_scripts/batch_eval_mimic_lite_v3_1.sh
batch_test_scripts/batch_eval_0529_v3_1.sh
```

Use these maintained entrypoints as launch templates and pass the downloaded checkpoint root through `MODEL_ROOT_BASE`. Override `SMALL_MODEL_ROOT`, `MERGE_MODEL_ROOT`, or `PI05_MODEL_ROOT` only when using a custom layout.

Example:

```bash
MODEL_ROOT_BASE=/path/to/humanoidarena_checkpoints \
TEST_MODE=semantic \
EVAL_BACKEND=sonic \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
bash isaaclab_twist2_g1/batch_test_scripts/batch_pi05_v31_sonic.sh
```

### MimicLite V3.1 campaign

`batch_eval_mimic_lite_v3_1.sh` evaluates the task-specific MimicLite open-door and football checkpoints. Its default backend is `mimic_lite`; set `EVAL_BACKENDS` to run the same campaign across multiple execution backends:

```bash
OPEN_DOOR_CHECKPOINTS_DIR=/path/to/open_door/checkpoints \
FOOTBALL_CHECKPOINTS_DIR=/path/to/football/checkpoints \
EVAL_BACKENDS="mimic_lite sonic_low_latency sonic twist2" \
EVAL_TASKS="open_door football" \
NUM_WORKERS=2 \
bash isaaclab_twist2_g1/batch_test_scripts/batch_eval_mimic_lite_v3_1.sh
```

Preview the model selection, backend-specific YAML, GMT relation, and output paths without creating result directories:

```bash
DRY_RUN=1 \
EVAL_BACKENDS="mimic_lite sonic_low_latency" \
bash isaaclab_twist2_g1/batch_test_scripts/batch_eval_mimic_lite_v3_1.sh
```

### 0529 cross-backend campaign

`batch_eval_0529_v3_1.sh` discovers the configured Twist2- and SONIC-trained 0529 policies and evaluates them on one or more backends. SONIC low-latency is the default execution backend:

```bash
CKPT_ROOT=/path/to/0529_v3-1_infer_ckpts/small \
EVAL_BACKENDS="sonic_low_latency mimic_lite twist2" \
NUM_WORKERS=2 \
bash isaaclab_twist2_g1/batch_test_scripts/batch_eval_0529_v3_1.sh
```

The campaign manifest records both `training_gmt` and normalized `execution_gmt`. `sonic_low_latency` is normalized to `sonic` when deriving `in_gmt` versus `cross_gmt`, while `execution_backend` retains the low-latency variant name.

### Campaign identity and resume

- Set `RUN_TIMESTAMP` to use an exact shared run ID across all selected backends.
- Set `RESUME_LATEST=1` without `RUN_TIMESTAMP` to resume the latest campaign only when every selected backend has the same latest run ID.
- Keep `RESUME_LATEST=0` for a new timestamped campaign.
- Use `SEEDS_OVERRIDE`, `REPEATS_PER_SEED`, and `MAX_STEPS` to override the base-test defaults.
- Use `OPEN_DOOR_ENV_CONFIG_YAML` or `FOOTBALL_ENV_CONFIG_YAML` only for deliberate task-specific configuration overrides.

## 7. Results

Evaluation outputs are local runtime artifacts and should stay out of git:

```text
isaaclab_twist2_g1/script/eval_scripts/*/eval_results/
lerobot/results/
lerobot/outputs/
```

Campaign outputs use the backend-first layout:

```text
isaaclab_twist2_g1/script/eval_scripts/<execution_backend>/eval_results/
  <campaign_id>__<timestamp>/
    campaign_manifest.json
    <task>__<policy_id>__step_<checkpoint>/
```

Every backend root in a multi-backend run receives the same complete `campaign_manifest.json`. It records all planned stages, model paths, backend-specific test YAMLs, training/execution GMTs, controller resources, seeds, repeats, max steps, Git commit, and per-stage status.

Before reporting benchmark results, keep the following together:

- Git commit hash.
- Model checkpoint identifier.
- Dataset identifier.
- Backend and test mode.
- Number of workers and random seed.
- Result directory path or uploaded artifact link.
