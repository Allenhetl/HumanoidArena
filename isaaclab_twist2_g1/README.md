# IsaacLab TWIST2 G1

`isaaclab_twist2_g1/` contains the Isaac Lab environments, teleoperation/replay entrypoints, rerecording utilities, and evaluation scripts used by HumanoidArena. TWIST2 and SONIC remain the baseline backends; MimicLite supports teleoperation/data collection and evaluation, while SONIC low-latency is currently evaluation-only.

## Start Here

- [Environment setup](../docs/04_environment_setup.md)
- [Teleoperation and recording](../docs/01_teleoperation.md)
- [Data processing](../docs/02_data_pipeline.md)
- [Evaluation](../docs/03_evaluation.md)

## Main Directories

- `action_provider/`: TWIST2, SONIC, and MimicLite control, recording, and replay logic.
- `tasks/`: Isaac Lab task definitions and environment configuration.
- `pico_server/`: Pico bridge and Redis/ZMQ data publishing.
- `image_server/`: camera streaming helpers.
- `tools/data_tools/`: rerecording and LeRobot conversion tools.
- `script/eval_scripts/`: single-run and batch evaluation launch scripts.

## Common Entrypoints

- `run_twist2.sh`: launch TWIST2 teleoperation/recording.
- `run_sonic.sh`: launch SONIC teleoperation/recording.
- `run_sonic_joint29.sh`: launch the canonical SONIC 29-joint recording route.
- `run_mimic_lite.sh`: launch MimicLite teleoperation/recording.
- `run_replay_twist2.sh`: replay TWIST2 recordings.
- `run_replay_sonic.sh`: replay SONIC recordings.
- `run_rerecord.sh`: run rerecording jobs.
- `run_sonic_teleop_server.sh` / `run_twist2_teleop_server.sh`: teleoperation server launchers.

For current canonical SONIC V3.1 data, start the GMR server with:

```bash
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh \
  --target_backend sonic_joint29
```

and pair it with `run_sonic_joint29.sh`. `run_sonic_teleop_server.sh` is the normal pose-only SONIC server and does not accept `--target_backend`.

Maintained MimicLite/low-latency campaign evaluation entrypoints are:

```text
batch_test_scripts/batch_eval_mimic_lite_v3_1.sh
batch_test_scripts/batch_eval_0529_v3_1.sh
```

See the evaluation guide for backend selection, dry runs, result layout, and campaign manifests. SONIC low-latency should currently be used for evaluation, not as a released collection format.

Most scripts keep their task, checkpoint, and path parameters near the top of the file. Update those values for your local environment before launching.

## Format And Protocol References

- [Scene randomization seed rules](docs/SCENE_RANDOMIZATION_SEED_RULES.md)
- [TWIST2 raw NPZ format](docs/TWIST2_DATA_FORMAT.md)
- [SONIC raw NPZ format](docs/SONIC_DATA_FORMAT.md)
- [LeRobot V3.1 dataset protocol](docs/UNITREE_G1_GMT_REFPOSE_V3_1_DATA_PROTOCOL.md)
