# HumanoidArena

HumanoidArena is a humanoid manipulation and whole-body control benchmark built around teleoperation, replay, simulation evaluation, and LeRobot-compatible policy training. The repository provides the TWIST2 and SONIC control pipelines, Isaac Lab environments, data conversion utilities, and evaluation scripts used by the project.

<p align="center">
  <a href="https://humanoidarena.github.io/">Project Page</a> |
  <a href="https://arxiv.org/abs/2606.17833">arXiv</a> |
  <a href="https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1">HF Dataset</a> |
  <a href="https://huggingface.co/WilliamWang16/HumanoidArena_models">HF Models</a> |
  <a href="https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing">Assets</a>
</p>

<p align="center">
  <img src="img/paper-main-figure.svg" alt="HumanoidArena system overview" width="90%">
</p>

## Overview

HumanoidArena focuses on full-body humanoid interaction tasks with reproducible data collection and policy evaluation. The current release includes:

- TWIST2 and SONIC teleoperation entrypoints.
- Isaac Lab environments for live control, replay, rerecording, and VLA evaluation.
- NPZ recording and multicam rerecording pipelines.
- LeRobot-compatible data and model release links.
- Batch evaluation scripts for vision execution and semantic tests.

## Release Plan

- [x] LeRobot data released
- [x] Models released
- [ ] Raw data release
- [ ] Multicam data release

## Quick Links

| Resource | Link |
| --- | --- |
| Project page | https://humanoidarena.github.io/ |
| Paper | https://arxiv.org/abs/2606.17833 |
| LeRobot dataset | https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1 |
| Model checkpoints | https://huggingface.co/WilliamWang16/HumanoidArena_models |
| Simulation assets | https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing |

## Documentation

Start with the release-facing guides:

- [Environment setup](docs/04_environment_setup.md)
- [Teleoperation and recording](docs/01_teleoperation.md)
- [Data pipeline](docs/02_data_pipeline.md)
- [Evaluation](docs/03_evaluation.md)

Additional implementation references:

- [TWIST2 raw NPZ format](isaaclab_twist2_g1/docs/TWIST2_DATA_FORMAT.md)
- [SONIC raw NPZ format](isaaclab_twist2_g1/docs/SONIC_DATA_FORMAT.md)
- [LeRobot V3.1 format](isaaclab_twist2_g1/docs/UNITREE_G1_GMT_REFPOSE_V3_1_DATA_PROTOCOL.md)
- [Scene randomization seed rules](isaaclab_twist2_g1/docs/SCENE_RANDOMIZATION_SEED_RULES.md)

## Repository Layout

```text
TWIST2/                 TWIST2 control, assets, checkpoints, and robot-side utilities
isaaclab_twist2_g1/     Isaac Lab tasks, replay, rerecording, and evaluation entrypoints
lerobot/                LeRobot fork/integration for training and policy serving
docs/                   Release-facing documentation and evaluation examples
```

## Getting Started

Set up the simulation and LeRobot environments first:

```bash
bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --dry-run
```

After reviewing the generated commands, follow [Environment setup](docs/04_environment_setup.md) for the full installation path.

For live teleoperation and recording:

```bash
cd TWIST2
bash teleop.sh
```

Then launch the simulator-side backend from the repository root:

```bash
bash isaaclab_twist2_g1/run_twist2.sh
# or
bash isaaclab_twist2_g1/run_sonic.sh
```

For replay:

```bash
bash isaaclab_twist2_g1/run_replay_twist2.sh
bash isaaclab_twist2_g1/run_replay_sonic.sh
```

## Data and Models

The git repository should contain source code, small examples, and required lightweight runtime assets. Large data and model artifacts are released separately:

| Artifact | Status | Location |
| --- | --- | --- |
| LeRobot dataset | Released | https://huggingface.co/datasets/WilliamWang16/HumanoidArena_dataset_v3_1 |
| Model checkpoints | Released | https://huggingface.co/WilliamWang16/HumanoidArena_models |
| Simulation assets | Released | https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view?usp=sharing |
| Raw data | Planned | To be announced |
| Multicam data | Planned | To be announced |

Download released model checkpoints from the Hugging Face model repository into any local artifact directory and keep the published folder layout:

```bash
huggingface-cli download HumanoidArena/<model-repo> \
  --local-dir /path/to/humanoidarena_checkpoints
```

Batch evaluation scripts read that directory through `MODEL_ROOT_BASE`:

```text
/path/to/humanoidarena_checkpoints/
  small/
  small_merge/
  pi/
```

Download the simulation asset package separately from the release asset link and restore it under the Isaac Lab package:

```text
isaaclab_twist2_g1/assets/
  objects/
  robots/
```

The asset package is intentionally distributed outside git and released through the simulation assets link above.

The TWIST2 ONNX checkpoints required by the current runtime are kept in git:

```text
TWIST2/assets/ckpts/twist2_1017_20k.onnx
TWIST2/assets/ckpts/twist2_1017_25k.onnx
```

## Evaluation

Single-checkpoint VLA evaluation:

```bash
MODEL_PATH=/path/to/checkpoint/pretrained_model \
EVAL_SEEDS="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval.sh

MODEL_PATH=/path/to/checkpoint/pretrained_model \
EVAL_SEEDS="0 1 2" \
bash isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval.sh
```

`MODEL_PATH` may also point to a checkpoint directory that contains `pretrained_model/`. For PI0.5 checkpoints, use the matching scripts under `script/eval_scripts/sonic_pi05/` or `script/eval_scripts/twist2_pi05/`.

Batch evaluation entrypoints include:

```text
isaaclab_twist2_g1/batch_test_scripts/batch_1_test_v31_sonic.sh
isaaclab_twist2_g1/batch_test_scripts/batch_1_test_v31_twist2.sh
isaaclab_twist2_g1/batch_test_scripts/batch_1_test_v31_merage.sh
isaaclab_twist2_g1/batch_test_scripts/batch_pi05_v31_sonic.sh
isaaclab_twist2_g1/batch_test_scripts/batch_pi05_v31_twist2.sh
```

See [Evaluation](docs/03_evaluation.md) for vision execution, semantic evaluation, and batch launch examples.

## Citation

If you use HumanoidArena in your research, please cite:

```bibtex
@article{wang2026humanoidarena,
  title={HumanoidArena: Benchmarking Egocentric Hierarchical Whole-body Learning},
  author={Wang, Taowen and Xie, Zikang and Yang, Bin and others},
  journal={arXiv preprint arXiv:2606.17833},
  year={2026}
}
```

## Acknowledgements

HumanoidArena builds on and interoperates with the following open-source projects and datasets:

| Project | Role in HumanoidArena | Upstream | License / terms |
| --- | --- | --- | --- |
| TWIST2 | Whole-body teleoperation and motion/control pipeline | https://github.com/YanjieZe/TWIST | MIT |
| SONIC / GR00T Whole-Body Control | SONIC controller, policy artifacts, and deployment workflow | https://github.com/NVlabs/GR00T-WholeBodyControl | Source: Apache-2.0; model weights: NVIDIA Open Model License |
| LeRobot | Dataset format, training, and VLA policy serving integration | https://github.com/huggingface/lerobot | Apache-2.0 |
| Unitree Sim IsaacLab | Isaac Lab simulator foundation and Unitree task patterns | https://github.com/unitreerobotics/unitree_sim_isaaclab | Apache-2.0 |
| ArtVIP | Articulated-object assets and digital-twin dataset reference | https://huggingface.co/datasets/X-Humanoid/ArtVIP | Apache-2.0 |

## License

HumanoidArena includes code derived from or integrated with the projects above. Please review this repository's license files, upstream project licenses, and model/data artifact terms before redistribution or commercial use. Third-party simulator dependencies, robot assets, datasets, and model weights may be governed by separate terms.
