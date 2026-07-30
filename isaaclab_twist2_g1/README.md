# IsaacLab TWIST2 G1

`isaaclab_twist2_g1/` contains the Isaac Lab environments, teleoperation and replay entrypoints, data collection utilities, evaluation scripts, and real-scene integration used by HumanoidArena. TWIST2 and SONIC remain the baseline backends; MimicLite supports teleoperation, data collection, and evaluation, while SONIC low-latency is currently evaluation-only.

## Start Here

- [Environment setup](../docs/04_environment_setup.md)
- [Teleoperation and recording](../docs/01_teleoperation.md)
- [Data processing](../docs/02_data_pipeline.md)
- [Evaluation](../docs/03_evaluation.md)
- [Real2Sim Gaussian scenes](docs/REAL2SIM_GAUSSIAN_SCENE_GUIDE.md)
- [Real-scene release registry](real_scenes/README.md)

## Main Directories

- `action_provider/`: TWIST2, SONIC, and MimicLite control, recording, and replay logic.
- `tasks/`: Isaac Lab task definitions and environment configuration.
- `real_scenes/`: versioned scene descriptors, artifact locks, acceptance policies, and small USD wrappers.
- `assets/objects/real_scene/`: downloaded or server-local runtime payload cache; large assets are not stored in Git.
- `pico_server/`: Pico bridge and Redis/ZMQ data publishing.
- `image_server/`: camera streaming helpers.
- `tools/data_tools/`: rerecording and LeRobot conversion tools.
- `tools/real_scene/`: real-scene release validation tools.
- `script/eval_scripts/`: single-run and batch evaluation launch scripts.

## Common Entrypoints

- `run_twist2.sh`: launch TWIST2 teleoperation and recording.
- `run_sonic.sh`: launch SONIC teleoperation and recording.
- `run_sonic_joint29.sh`: launch the canonical SONIC 29-joint recording route.
- `run_mimic_lite.sh`: launch MimicLite teleoperation and recording.
- `run_replay_twist2.sh`: replay TWIST2 recordings.
- `run_replay_sonic.sh`: replay SONIC recordings.
- `run_mimic_lite_replay.sh`: replay MimicLite recordings.
- `run_rerecord.sh`: run rerecording jobs.
- `run_sonic_teleop_server.sh` / `run_twist2_teleop_server.sh`: teleoperation server launchers.

For current canonical SONIC V3.1 data, start the GMR server with:

```bash
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh \
  --target_backend sonic_joint29
```

Pair it with `run_sonic_joint29.sh`. `run_sonic_teleop_server.sh` is the normal pose-only SONIC server and does not accept `--target_backend`.

Maintained MimicLite and low-latency campaign evaluation entrypoints are:

```text
batch_test_scripts/batch_eval_mimic_lite_v3_1.sh
batch_test_scripts/batch_eval_0529_v3_1.sh
```

SONIC low-latency should currently be used for evaluation, not as a released collection format.

## Replay Modes

The backends share the `sim_main.py` replay entrypoint:

- `direct_replay`: execute targets stored in the recording.
- `inference_replay`: rerun backend inference from recorded model input or observations.

## Real-Scene Releases

Real-scene rendering and physics are intentionally separated. NuRec Gaussian assets provide visual appearance, while invisible LiDAR-derived meshes provide collision. Runtime USDZ and USDC payloads live outside Git; the repository stores portable descriptors, hashes, composition wrappers, task YAMLs, and acceptance results.

Validate a deployed scene with:

```bash
python isaaclab_twist2_g1/tools/real_scene/validate_scene_release.py \
  --scene-dir isaaclab_twist2_g1/real_scenes/scenes/odin1_colmap_independent_repaired_3dgrut_30k \
  --asset-dir isaaclab_twist2_g1/assets/objects/real_scene
```

Current Odin experiment records:

- [English](docs/ODIN1_COLMAP_INDEPENDENT_REPAIRED_3DGRUT_30K_LIDAR_COLLISION.md)
- [Chinese](docs/ODIN1_COLMAP_INDEPENDENT_REPAIRED_3DGRUT_30K_LIDAR_COLLISION_ZH.md)

## Format and Protocol References

- [Scene randomization seed rules](docs/SCENE_RANDOMIZATION_SEED_RULES.md)
- [TWIST2 raw NPZ format](docs/TWIST2_DATA_FORMAT.md)
- [SONIC raw NPZ format](docs/SONIC_DATA_FORMAT.md)
- [LeRobot V3.1 dataset protocol](docs/UNITREE_G1_GMT_REFPOSE_V3_1_DATA_PROTOCOL.md)

Most launch scripts keep task, checkpoint, and path parameters near the top of the file. Update them for the target environment before launching. Tasks with randomized scene initialization must use one authoritative episode object seed.
