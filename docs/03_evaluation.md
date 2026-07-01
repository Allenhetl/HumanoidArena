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
```

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

`MODEL_PATH` can also be the parent checkpoint directory when it contains a `pretrained_model/` subdirectory. Use `RESULTS_DIR` to override the output directory and `REPEATS_PER_SEED` to repeat each seed. For PI0.5 checkpoints, use `script/eval_scripts/sonic_pi05/run_vla_eval.sh` or `script/eval_scripts/twist2_pi05/run_vla_eval.sh` with the same variables.

Parallel variants:

```bash
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval_parallel.sh
bash isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval_parallel.sh
```

PI0.5 variants are under:

```text
isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/
isaaclab_twist2_g1/script/eval_scripts/twist2_pi05/
```

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
bash isaaclab_twist2_g1/batch_1_test_v31_sonic.sh
```

TWIST2:

```bash
TEST_MODE=base_test \
EVAL_BACKEND=twist2 \
MODEL_ROOT_BASE="/path/to/humanoidarena_checkpoints" \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
RESULTS_TAG_PREFIX=vision_exec \
bash isaaclab_twist2_g1/batch_1_test_v31_twist2.sh
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
bash isaaclab_twist2_g1/pi05_batch_test_doubledesk.sh
```

Open-door example:

```bash
TEST_MODE=semantic \
EVAL_BACKEND=sonic \
MODEL_ROOT="/path/to/pi0.5_sonic_opendoor_checkpoint" \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
RESULTS_TAG_PREFIX=pi05_semantic \
bash isaaclab_twist2_g1/pi05_batch_test_open_door.sh
```

## 6. Batch Evaluation

Primary maintained batch entrypoints live under `isaaclab_twist2_g1/`:

```text
batch_1_test_v31_sonic.sh
batch_1_test_v31_twist2.sh
batch_1_test_v31_merage.sh
batch_pi05_v31_sonic.sh
batch_pi05_v31_twist2.sh
```

Use these maintained entrypoints as launch templates and pass the downloaded checkpoint root through `MODEL_ROOT_BASE`. Override `SMALL_MODEL_ROOT`, `MERGE_MODEL_ROOT`, or `PI05_MODEL_ROOT` only when using a custom layout.

Example:

```bash
MODEL_ROOT_BASE=/path/to/humanoidarena_checkpoints \
TEST_MODE=semantic \
EVAL_BACKEND=sonic \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
bash isaaclab_twist2_g1/batch_pi05_v31_sonic.sh
```

## 7. Results

Evaluation outputs are local runtime artifacts and should stay out of git:

```text
isaaclab_twist2_g1/script/eval_scripts/*/eval_results/
lerobot/results/
lerobot/outputs/
```

Before reporting benchmark results, keep the following together:

- Git commit hash.
- Model checkpoint identifier.
- Dataset identifier.
- Backend and test mode.
- Number of workers and random seed.
- Result directory path or uploaded artifact link.
