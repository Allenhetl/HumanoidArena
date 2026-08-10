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
PERSISTENT_SIM
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

Use `MODEL_PATH` for single-checkpoint `run_vla_eval.sh`, `MODEL_ROOT` for the parallel/task wrappers, and `MODEL_ROOT_BASE` for the top-level batch wrappers.

## 3. Batch Evaluation

Primary maintained batch entrypoints live under `isaaclab_twist2_g1/`:

```text
batch_test_scripts/batch_1_test_v31_sonic.sh
batch_test_scripts/batch_1_test_v31_twist2.sh
batch_test_scripts/batch_1_test_v31_merage.sh
batch_test_scripts/batch_pi05_v31_sonic.sh
batch_test_scripts/batch_pi05_v31_twist2.sh
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

### Persistent simulation (`PERSISTENT_SIM`)

The `persist_sim` option controls whether a single Isaac Sim instance is reused across episodes or restarted for each episode:

| Setting | Behavior | Speed | Result |
| --- | --- | --- | --- |
| `persist_sim=1` (enabled) | One Isaac Sim process runs all repeats for a given `(model, seed)` in a single launch (`--episode_batch_json`). The environment is reset (`env.sim.reset()` + `env.reset()`) between episodes, while the process and runtime objects are retained, avoiding repeated startup overhead. | Fast | Unreset process-level state may persist across episodes and could affect results; observed performance may be slightly lower. |
| `persist_sim=0` (disabled) | Isaac Lab is restarted for every episode. | Slow | Each episode starts from a clean process and environment; performance may be better. |

Parallel/batch evaluation enables `persist_sim` by default. The defaults come from `test_defaults.persistent_sim: true` in `tasks/common_test_config/{base_test,vision,semantic,execution}/*.yaml`, with the shell scripts falling back to `1` when the YAML field is absent. The benchmark results reported in the paper were produced with the maintained batch scripts (`batch_test_scripts/`, which dispatch to the `*_run_vla_eval_parallel.sh` variants), so all of them ran with `persist_sim` enabled.

You can override it either with the `PERSISTENT_SIM` environment variable (read by the parallel shell scripts) or with the `--persistent_sim` command-line argument (when calling `eval_vla_suite*.py` directly).

Example of both settings with the parallel variant. `run_vla_eval_parallel.sh` discovers models from `MODEL_ROOT` + `MODEL_GLOB` and takes seeds from `SEEDS_OVERRIDE`; the generic runner defaults to the football task config, so point `MODEL_ROOT` at a task-specific checkpoint directory (for other tasks, set `ENV_CONFIG_YAML` and a matching `MODEL_ROOT`):

```bash
# persist_sim=1 (default): one Isaac Sim process runs all repeats for (model, seed), fast
PERSISTENT_SIM=1 \
MODEL_ROOT=/path/to/humanoidarena_checkpoints/small/HOI_football \
MODEL_GLOB="*" \
SEEDS_OVERRIDE="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval_parallel.sh

# persist_sim=0: restart Isaac Lab for every episode, clean but slow
PERSISTENT_SIM=0 \
MODEL_ROOT=/path/to/humanoidarena_checkpoints/small/HOI_football \
MODEL_GLOB="*" \
SEEDS_OVERRIDE="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval_parallel.sh
```

## 4. Single-Task VLA Evaluation

> **Note:** The single-task scripts are mainly for debugging and quick effect checks of a single checkpoint. The batch scripts in [Batch Evaluation](#3-batch-evaluation) are the consistent entrypoints used for the eval experiments reported in the paper.

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

The single-task `run_vla_eval.sh` scripts do **not** pass the `--persistent_sim` argument, so `persist_sim` is **not enabled**: they fall back to the evaluator default of `0` and restart Isaac Lab for every episode. Use the parallel/batch variants if you want to control `PERSISTENT_SIM`.

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

## 5. Vision Execution Evaluation

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

## 6. Semantic Evaluation

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
