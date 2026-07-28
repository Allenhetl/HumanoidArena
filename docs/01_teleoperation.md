# Teleoperation and Recording

This guide covers the live teleoperation and recording workflow for HumanoidArena. It groups the maintained TWIST2, SONIC, and MimicLite entrypoints used with Isaac Lab.

## Prerequisites

Complete the environment setup first:

- [Environment setup](04_environment_setup.md)
- [TWIST2 teleoperation notes](../TWIST2/doc/TELEOP.md)

Before recording, check that these services and devices are available:

- PICO headset streaming through XRobotToolkit.
- Redis or the configured action transport used by the teleoperation server.
- Isaac Sim / Isaac Lab environment activated with access to `isaaclab_twist2_g1/assets`.
- Required TWIST2 checkpoints under `TWIST2/assets/ckpts/`.
- SONIC policy artifacts under `GR00T-WholeBodyControl/gear_sonic_deploy/policy/release` when using SONIC.
- MimicLite ONNX/YAML artifacts under `isaaclab_twist2_g1/assets/checkpoints/mimic_lite/` when using MimicLite.

## 1. Prepare PICO/XRobotToolkit Tracking

Wear the supported whole-body motion trackers and wrist controllers, then start the PICO XRobotToolkit application. Connect it to the Linux host that runs the teleoperation publisher and start streaming whole-body and hand data. Confirm that the pose stream updates before launching any simulator process.


## 2. TWIST2 Teleoperation

Start the maintained TWIST2 teleoperation server from the repository root:

```bash
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh
```

Override the configured human height when needed:

```bash
ACTUAL_HUMAN_HEIGHT=1.79 \
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh
```

The PICO height estimate can be noisy, so the configured height is usually set slightly below the measured human height.

## 3. SONIC Teleoperation

The normal SONIC pose-streaming server is:

```bash
bash isaaclab_twist2_g1/run_sonic_teleop_server.sh
```

That wrapper publishes the pose-only Redis stream consumed by `run_sonic.sh`. The current V3.1 collection pipeline uses canonical Unitree G1 29-joint targets instead. For that data format, start the maintained GMR publisher with the SONIC target route:

```bash
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh \
  --target_backend sonic_joint29
```

`--target_backend sonic_joint29` selects the SONIC input-ready key, publishes the canonical 29-joint references, and enforces the SONIC live cadence. Pair it with `run_sonic_joint29.sh`; do not pass this option to the pose-only `run_sonic_teleop_server.sh`.

The SONIC runtime expects the encoder, decoder, and observation config to exist under the configured `SONIC_POLICY_ROOT`. See [Environment setup](04_environment_setup.md#sonic-policy-artifacts).

## 4. MimicLite Teleoperation And Recording

MimicLite uses the same GMR 29-joint retargeting data as canonical SONIC, but the teleoperation publisher must target the MimicLite readiness key. Start the simulator first so it can publish that key, then start the GMR publisher in a second terminal.

Terminal A, start the MimicLite simulator and recorder:

```bash
bash isaaclab_twist2_g1/run_mimic_lite.sh
```

Terminal B, start the MimicLite-targeted GMR publisher:

```bash
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh \
  --target_backend mimic_lite
```

The launcher defaults to the football task. Select a task YAML and a new recording directory explicitly for each collection. For open-door collection:

```bash
ENV_CONFIG_YAML=isaaclab_twist2_g1/tasks/common_env_config/opendoor_sonic.yaml \
RECORDING_SAVE_DIR=isaaclab_twist2_g1/recording_data/HSI_open_door/mimic_lite/<run_id> \
bash isaaclab_twist2_g1/run_mimic_lite.sh
```

For football collection:

```bash
ENV_CONFIG_YAML=isaaclab_twist2_g1/tasks/common_env_config/football_single_sonic.yaml \
RECORDING_SAVE_DIR=isaaclab_twist2_g1/recording_data/HOI_football/mimic_lite/<run_id> \
bash isaaclab_twist2_g1/run_mimic_lite.sh
```

`run_mimic_lite.sh` requires a matching exported policy pair. Its defaults are under `isaaclab_twist2_g1/assets/checkpoints/mimic_lite/`; override both paths when using another export:

```bash
MIMIC_LITE_ONNX_PATH=/path/to/policy.onnx \
MIMIC_LITE_YAML_PATH=/path/to/policy.yaml \
bash isaaclab_twist2_g1/run_mimic_lite.sh
```

Set `MIMIC_LITE_REDIS_HOST` and `MIMIC_LITE_REDIS_PORT` when the simulator and publisher do not use local Redis. Set `IMAGE_TRANSPORT`, `IMAGE_XROBOT_HOST`, and `IMAGE_XROBOT_PORT` for the configured camera transport. Use `DRY_RUN=1` to print the simulator command after validating the MimicLite ONNX/YAML files.

The publisher reports `Teleop target backend: mimic_lite` after connecting to Redis. The simulator reports its MimicLite input-ready key, selected task, policy paths, recording directory, and log path. Do not use `--target_backend sonic_joint29` for MimicLite: that route waits for SONIC's readiness key.

## 5. Start Isaac Lab Live Recording

From the repository root, launch the simulator-side live recording process.

TWIST2:

```bash
bash isaaclab_twist2_g1/run_twist2.sh
```

SONIC:

```bash
bash isaaclab_twist2_g1/run_sonic.sh
```

Open-door SONIC joint29 live inference / recording:

```bash
bash isaaclab_twist2_g1/run_sonic_joint29.sh
```

The server/runtime pairing for canonical SONIC V3.1 data is therefore:

```text
run_twist2_teleop_server.sh --target_backend sonic_joint29
  -> run_sonic_joint29.sh
```

SONIC low-latency is currently exposed as an evaluation backend only. Do not use it as a released data-collection format; use SONIC canonical joint29 or MimicLite for new supported collections.

Most scripts keep task names, robot assets, recording directories, image streaming addresses, and replay paths near the top of the shell file. Review those variables before long recording runs.

## 6. Recording Outputs

Recording outputs are local runtime artifacts and are not part of the git release surface. They commonly live under:

```text
isaaclab_twist2_g1/recording_data/
isaaclab_twist2_g1/recording_debug_logs/
isaaclab_twist2_g1/replay_debug_logs/
```

Keep release-ready data in the Hugging Face dataset repository or another artifact store, not in this git repository.

## 7. Troubleshooting

If the robot does not move, verify the upstream teleoperation terminal first, then the Isaac Lab terminal:

- The headset pose stream is updating in XRobotToolkit.
- The TWIST2, SONIC, or MimicLite-targeted GMR publisher is publishing actions.
- The Isaac Lab script uses the matching backend and server route (`twist2`, `sonic`, `sonic_joint29`, or `mimic_lite`).
- The task and robot type match the recorded or live control profile.
- Required assets and ONNX policy files exist at the paths printed by the script.

Reference formats:

- [TWIST2 raw NPZ format](../isaaclab_twist2_g1/docs/TWIST2_DATA_FORMAT.md)
- [SONIC raw NPZ format](../isaaclab_twist2_g1/docs/SONIC_DATA_FORMAT.md)
- [Scene randomization seed rules](../isaaclab_twist2_g1/docs/SCENE_RANDOMIZATION_SEED_RULES.md)
