# Teleoperation and Recording

This guide covers the maintained PICO/XRobotToolkit teleoperation and recording workflows for TWIST2, SONIC, and MimicLite in Isaac Lab.

## 1. Prerequisites

Complete [Environment setup](04_environment_setup.md) first. Before teleoperation, verify that the following components are available:

- PICO headset, supported whole-body trackers, and wrist controllers.
- XRobotToolkit streaming whole-body and hand data to the Linux host.
- Redis on the host used by the tracker publisher and Isaac Lab runtime.
- Isaac Sim / Isaac Lab with the assets under `isaaclab_twist2_g1/assets/`.
- TWIST2 checkpoints under `TWIST2/assets/ckpts/` when using TWIST2.
- SONIC encoder and decoder artifacts under `GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/` when using SONIC.
- MimicLite ONNX and YAML artifacts under `isaaclab_twist2_g1/assets/checkpoints/mimic_lite/` when using MimicLite.

Wear and calibrate the trackers, connect XRobotToolkit to the Linux host, and confirm that the body and hand streams update before starting the commands below. See [TWIST2 teleoperation notes](../TWIST2/doc/TELEOP.md) for the tracker controls.

## 2. Configure the Experiment

Each workflow has two processes:

- The **runtime launcher** starts Isaac Lab, loads the task and control policy, streams camera images, and records episodes.
- The **tracker publisher** receives XRobotToolkit data and publishes the backend-specific Redis inputs.

Task, recording, image, and model settings belong to the runtime launcher, not the tracker publisher. Review the configuration block near the top of the selected runtime script before every collection:

| Workflow | Runtime launcher | Configuration method |
| --- | --- | --- |
| TWIST2 | `isaaclab_twist2_g1/run_twist2.sh` | Edit the `User config` block. |
| SONIC pose | `isaaclab_twist2_g1/run_sonic.sh` | Edit the `User config` block. SONIC model paths also accept environment overrides. |
| SONIC canonical joint29 | `isaaclab_twist2_g1/run_sonic_joint29.sh` | Edit the configuration block. `ENV_CONFIG_YAML` and SONIC model paths accept environment overrides. |
| MimicLite | `isaaclab_twist2_g1/run_mimic_lite.sh` | Edit the configuration block or override its settings with environment variables. |

Check these settings in the selected runtime launcher:

| Setting | Purpose |
| --- | --- |
| `ENV_CONFIG_YAML` | Task and scene configuration. Use the YAML for the same backend family as the runtime. |
| `RECORDING_SAVE_DIR` | Output directory for the current task, backend, operator, and run. Use a new directory for each collection. |
| `IMAGE_XROBOT_HOST`, `IMAGE_XROBOT_PORT` | XRobotToolkit image receiver address. |
| `IMAGE_TRANSPORT`, `IMAGE_XROBOT_BITRATE`, `IMAGE_FPS` | Camera transport and stream quality. |
| `SEED`, `ROBOT_COLLIDER_MODE` | Scene seed and robot collision configuration. |
| `SONIC_ENCODER_PATH`, `SONIC_DECODER_PATH` | SONIC policy artifacts. |
| `MIMIC_LITE_ONNX_PATH`, `MIMIC_LITE_YAML_PATH` | Matching MimicLite exported policy pair. |

Paths written directly in a runtime script are resolved after the script enters `isaaclab_twist2_g1/`. Use paths such as `tasks/common_env_config/opendoor_sonic.yaml` and `recording_data/HSI_open_door/mimic_lite/<run_id>`, or use absolute paths. Do not prefix these relative paths with `isaaclab_twist2_g1/`.

The GMR tracker publisher used by TWIST2, SONIC joint29, and MimicLite supports these environment variables:

| Variable | Purpose |
| --- | --- |
| `ACTUAL_HUMAN_HEIGHT` | Operator height used by retargeting. |
| `GMR_PYTHON` | Python executable for the GMR environment. |
| `GMR_ROOT` | GMR repository root. |
| `REDIS_IP` | Redis host used by the GMR publisher. |

The SONIC pose publisher uses `REDIS_HOST` and `REDIS_PORT`. The quick-start commands below assume local Redis. Keep the publisher and runtime on the same Redis instance.

## 3. Start Teleoperation

Run all commands from the repository root. Use the exact publisher/runtime pairing for the selected workflow.

### 3.1 TWIST2

Terminal A starts the TWIST2 Isaac Lab runtime and recorder:

```bash
bash isaaclab_twist2_g1/run_twist2.sh
```

Terminal B starts the GMR tracker publisher on the default `twist2` route:

```bash
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh
```

Set the operator height when needed:

```bash
ACTUAL_HUMAN_HEIGHT=1.79 \
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh
```

### 3.2 SONIC Pose

Terminal A starts the pose-only SONIC tracker publisher. Start this publisher before the runtime so it receives the next readiness epoch:

```bash
bash isaaclab_twist2_g1/run_sonic_teleop_server.sh
```

Terminal B starts the SONIC pose-based Isaac Lab runtime and recorder:

```bash
bash isaaclab_twist2_g1/run_sonic.sh
```

This route publishes SMPL-X pose and hand/control inputs. It does not publish the canonical 29-joint GMR stream.

### 3.3 SONIC Canonical Joint29

Terminal A starts the canonical SONIC 29-joint Isaac Lab runtime and recorder:

```bash
bash isaaclab_twist2_g1/run_sonic_joint29.sh
```

Terminal B starts the GMR tracker publisher on the `sonic_joint29` route:

```bash
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh \
  --target_backend sonic_joint29
```

Use this pairing for the current V3.1 canonical SONIC data format. Do not pass `--target_backend` to `run_sonic_teleop_server.sh`; that script is the pose-only publisher used by Section 3.2.

### 3.4 MimicLite

Terminal A starts the MimicLite Isaac Lab runtime and recorder:

```bash
bash isaaclab_twist2_g1/run_mimic_lite.sh
```

Terminal B starts the GMR tracker publisher on the `mimic_lite` route:

```bash
bash isaaclab_twist2_g1/run_twist2_teleop_server.sh \
  --target_backend mimic_lite
```

MimicLite consumes the canonical 29-joint GMR stream but uses its own readiness route and control policy. Do not substitute `sonic_joint29` for the target backend.

SONIC low-latency is currently supported for evaluation only and is not a released data-collection workflow.

## 4. Recording Outputs

Recording and runtime logs are local artifacts. They commonly live under:

```text
isaaclab_twist2_g1/recording_data/
isaaclab_twist2_g1/recording_debug_logs/
isaaclab_twist2_g1/replay_debug_logs/
isaaclab_twist2_g1/logs/
```

Keep release-ready datasets in the project dataset repository or another artifact store, not in this Git repository.

## 5. Troubleshooting

If the robot does not move or recording does not start, check the two terminals in this order:

- XRobotToolkit body and hand tracking is updating.
- The runtime launcher and tracker publisher match one row in Section 3.
- The publisher reports the expected target backend: `twist2`, `sonic_joint29`, or `mimic_lite`. The SONIC pose publisher does not use `--target_backend`.
- Both processes connect to the same Redis instance.
- The selected task YAML, model artifacts, and recording directory exist or are writable.
- The image receiver host and port match the XRobotToolkit configuration.

Reference formats:

- [TWIST2 raw NPZ format](../isaaclab_twist2_g1/docs/TWIST2_DATA_FORMAT.md)
- [SONIC raw NPZ format](../isaaclab_twist2_g1/docs/SONIC_DATA_FORMAT.md)
- [Scene randomization seed rules](../isaaclab_twist2_g1/docs/SCENE_RANDOMIZATION_SEED_RULES.md)
