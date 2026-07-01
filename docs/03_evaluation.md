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

## 2. Single-Task VLA Evaluation

SONIC:

```bash
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval.sh
```

TWIST2:

```bash
bash isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval.sh
```

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

## 3. Vision Execution Evaluation

Vision execution uses the same simulator-side evaluation workers, with model paths and camera settings configured in the selected shell script. Before launching, check:

- `MODEL_ROOT` points to the intended checkpoint directory.
- Camera flags match the dataset used for training.
- The selected backend matches the training data backend.
- Result paths point to a local output directory, not a tracked git path.

A standard SONIC launch shape is:

```bash
TEST_MODE=base_test \
EVAL_BACKEND=sonic \
MODEL_ROOT="/path/to/checkpoints" \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
RESULTS_TAG_PREFIX=vision_exec \
bash isaaclab_twist2_g1/batch_1_test_v31_sonic.sh
```

TWIST2:

```bash
TEST_MODE=base_test \
EVAL_BACKEND=twist2 \
MODEL_ROOT="/path/to/checkpoints" \
MODEL_GLOB="*" \
NUM_WORKERS=2 \
RESULTS_TAG_PREFIX=vision_exec \
bash isaaclab_twist2_g1/batch_1_test_v31_twist2.sh
```

## 4. Semantic Evaluation

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

## 5. Batch Evaluation

Primary maintained batch entrypoints live under `isaaclab_twist2_g1/`:

```text
batch_1_test_v31_sonic.sh
batch_1_test_v31_twist2.sh
batch_1_test_v31_merage.sh
batch_pi05_v31_sonic.sh
batch_pi05_v31_twist2.sh
```

Historical or comparison-only examples live under:

```text
docs/eval_script_examples/
```

Use those examples as launch templates only; hard-coded checkpoint paths should be updated before reuse.

## 6. Results

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
