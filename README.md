# HumanoidArena

HumanoidArena is a humanoid manipulation and whole-body control benchmark built around teleoperation, replay, simulation evaluation, and LeRobot-compatible policy training. The repository provides the TWIST2 and SONIC control pipelines, Isaac Lab environments, data conversion utilities, and evaluation scripts used by the project.

<p align="center">
  <a href="https://humanoidarena.github.io/">Project Page</a> |
  <a href="https://arxiv.org/abs/XXXX.XXXXX">arXiv</a> |
  <a href="https://huggingface.co/datasets/HumanoidArena">HF Dataset</a> |
  <a href="https://huggingface.co/HumanoidArena">HF Models</a>
</p>

<p align="center">
  <img src="xrobotoolkit.png" alt="HumanoidArena system overview" width="80%">
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
- [ ] Full benchmark suite release
- [ ] Project page update with final videos and task gallery
- [ ] arXiv paper link update

## Quick Links

| Resource | Link |
| --- | --- |
| Project page | https://humanoidarena.github.io/ |
| Paper | https://arxiv.org/abs/XXXX.XXXXX |
| LeRobot dataset | https://huggingface.co/datasets/HumanoidArena |
| Model checkpoints | https://huggingface.co/HumanoidArena |

Update the placeholder project, paper, dataset, and model URLs before tagging the public release if the final links differ.

## Documentation

Start with the release-facing guides:

- [Environment setup](docs/04_environment_setup.md)
- [Teleoperation and recording](docs/01_teleoperation.md)
- [Data pipeline](docs/02_data_pipeline.md)
- [Evaluation](docs/03_evaluation.md)

Additional implementation references:

- [IsaacLab command quickstart](isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md)
- [TWIST2 data format](isaaclab_twist2_g1/docs/TWIST2_DATA_FORMAT.md)
- [SONIC data format](isaaclab_twist2_g1/docs/SONIC_DATA_FORMAT.md)
- [Environment setup details](isaaclab_twist2_g1/docs/ENVIRONMENT_SETUP.md)

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
| LeRobot dataset | Released | https://huggingface.co/datasets/HumanoidArena |
| Model checkpoints | Released | https://huggingface.co/HumanoidArena |
| Raw data | Planned | To be announced |
| Multicam data | Planned | To be announced |

The TWIST2 ONNX checkpoints required by the current runtime are kept in:

```text
TWIST2/assets/ckpts/twist2_1017_20k.onnx
TWIST2/assets/ckpts/twist2_1017_25k.onnx
```

## Evaluation

Single-task VLA evaluation:

```bash
bash isaaclab_twist2_g1/script/eval_scripts/sonic/run_vla_eval.sh
bash isaaclab_twist2_g1/script/eval_scripts/twist2/run_vla_eval.sh
```

Batch evaluation entrypoints include:

```text
isaaclab_twist2_g1/batch_1_test_v31_sonic.sh
isaaclab_twist2_g1/batch_1_test_v31_twist2.sh
isaaclab_twist2_g1/batch_pi05_v31_sonic.sh
isaaclab_twist2_g1/batch_pi05_v31_twist2.sh
```

See [Evaluation](docs/03_evaluation.md) for vision execution, semantic evaluation, and batch launch examples.

## Citation

If you use HumanoidArena in your research, please cite:

```bibtex
@article{humanoidarena2026,
  title   = {HumanoidArena: A Benchmark for Whole-Body Humanoid Teleoperation, Learning, and Evaluation},
  author  = {Wang, Taowen and Contributors},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

See the repository license files and third-party dependency licenses before redistribution. Some assets, simulator dependencies, and model artifacts may have separate license terms.
