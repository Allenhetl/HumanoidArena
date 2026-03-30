# IsaacLab TWIST2 G1 Architecture, Startup, and Smoke Checklist

## Scope

This document summarizes the current runtime architecture after the refactor, the supported startup paths for teleop and VLA testing, and the smoke checklist that was executed in the repository.

The current design target is:

- `sim_main.py` is the only long-term simulation entrypoint.
- `action_provider` owns runtime control logic, episode control, and simulation advancement.
- `tasks` owns environment configuration and task-specific runtime scene patching.
- All teleop-facing server programs are launched from `isaaclab_twist2_g1/pico_server`.
- sonic and twist2 keep their own control semantics and protocols.
- VLA is integrated as another upstream input source routed through GMT, not as a separate simulator mainline.

## Layered Architecture

### Layer 1: Entry / Routing

Primary file:

- `isaaclab_twist2_g1/sim_main.py`

Responsibilities:

- Parse CLI arguments.
- Normalize `input_source + gmt_backend` to the legacy `action_source`.
- Create Isaac Lab env and core runtime services.
- Create the selected `action_provider`.
- Coordinate cross-layer reset through Redis.

Non-responsibilities:

- No task-specific stage patching logic should remain here.
- No sonic or twist2 business-specific teleop protocol logic should live here.

Compatibility wrapper:

- `isaaclab_twist2_g1/sim_main_openpi.py`

This file is now a thin wrapper that forwards to `sim_main.py` with:

- `--input_source vla`
- `--gmt_backend twist2`

### Layer 2: Action Provider / Runtime Control

Primary files:

- `isaaclab_twist2_g1/action_provider/create_action_provider.py`
- `isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py`
- `isaaclab_twist2_g1/action_provider/action_provider_sonic.py`
- `isaaclab_twist2_g1/action_provider/action_provider_openpi.py`

Shared helpers:

- `isaaclab_twist2_g1/action_provider/reset_control.py`
- `isaaclab_twist2_g1/action_provider/recording_common.py`

Responsibilities:

- Read upstream inputs.
- Preserve sonic / twist2 / VLA-specific semantics.
- Produce and apply simulation actions.
- Manage runtime teleop state, including:
  - reset
  - save
  - episode boundaries
  - asynchronous recording

Current routes:

- `pico_twist2 -> twist2 GMT -> DDSRLActionProvider`
- `pico_sonic -> sonic GMT -> SonicActionProvider`
- `vla -> twist2 GMT -> OpenPIActionProvider`
- `replay -> replay provider`

### Layer 3: Environment / Tasks

Primary files:

- `isaaclab_twist2_g1/tasks/...`
- `isaaclab_twist2_g1/tasks/common_runtime/env_runtime_hooks.py`

Responsibilities:

- Environment configuration.
- Scene initialization.
- Task-specific runtime patching and post-reset patching.

Moved out of `sim_main.py`:

- football grass / pitch line / physics-material patching
- livingroom collision / mass tuning runtime patching
- scene deactivate / reposition rules
- generic CLI camera overrides

## Communication Model

### Shared transport

The runtime now treats Redis as the standard teleop transport for sonic and twist2 service integration.

### sonic

Standard upstream service:

- `isaaclab_twist2_g1/pico_server/pico_server_pose_only.py`

Default transport:

- Redis

Keys published for sonic:

- `human_smplx_data_unitree_g1_with_hands`
- `action_hand_left_unitree_g1_with_hands`
- `action_hand_right_unitree_g1_with_hands`
- `controller_data`
- `recording_control_unitree_g1_with_hands`

ZMQ status:

- still supported as compatibility / fallback
- no longer the default sonic startup path

### twist2

Standard upstream service:

- `isaaclab_twist2_g1/pico_server/twist2_teleop_server.py`

Wrapper:

- `isaaclab_twist2_g1/pico_server/run_twist2_teleop.sh`
- legacy `TWIST2/teleop.sh` now forwards to this wrapper

twist2 continues to publish its original Redis keys and protocol semantics.

### reset coordination

Shared Redis keys:

- `isaac_reset_trigger`
- `isaac_reset_complete_unitree_g1_with_hands`

Contract:

- providers publish reset request
- `sim_main.py` performs reset
- `sim_main.py` publishes reset complete signal

## sonic Runtime Completion

### Buttons

Published by `pico_server_pose_only.py`:

- left `X`: save current sonic episode
- left `Y`: discard current sonic episode and request full reset

Current implementation note:

- This matches the latest user instruction: reset and save are separate.
- `Y` is implemented as reset without saving the current unsaved buffer.

### sonic recording content

Saved episodes include:

- raw human SMPL frame JSON
- raw controller / recording-control payloads
- processed SMPL joints / pose / body quaternion
- anchor alignment state
- encoder input vector and active windows
- motion / robot history buffers
- encoder latent
- decoder observation
- decoder raw output and post-processed target action
- final 29-DoF body action
- full Isaac action
- optional body effort target
- robot qpos / qvel / root state
- key env object states
- vision frames when available
- frame / wall / monotonic / realtime timestamps
- save / reset / episode markers

This is intended to support both:

- offline VLA training
- direct replay / action reconstruction

## VLA Integration

### Unified route

VLA now uses the same simulator main entry:

- `sim_main.py --input_source vla --gmt_backend twist2`

`action_source=openpi` remains as the internal compatibility route selected by `sim_main.py`.

### Required VLA arguments

Supported by `sim_main.py`:

- `--openpi_checkpoint`
- `--language_instruction`
- `--smplx_model_path`
- `--human_height`
- `--twist2_model_path`
- `--video_save_dir`
- `--video_fps`
- `--enable_smpl_vis`

Fallback:

- if `--twist2_model_path` is empty, `OpenPIActionProvider` falls back to `--model_path`

### VLA startup wrapper

New script:

- `isaaclab_twist2_g1/run_vla.sh`

Usage pattern:

```bash
cd isaaclab_twist2_g1
OPENPI_CHECKPOINT=/path/to/checkpoint \
LANGUAGE_INSTRUCTION="walk forward slowly" \
bash run_vla.sh
```

Compatibility entry:

```bash
python isaaclab_twist2_g1/sim_main_openpi.py \
  --openpi_checkpoint /path/to/checkpoint \
  --language_instruction "walk forward slowly"
```

## Startup Guide

### 1. twist2 teleop service

Preferred:

```bash
cd isaaclab_twist2_g1/pico_server
bash run_twist2_teleop.sh
```

Legacy-compatible:

```bash
cd TWIST2
bash teleop.sh
```

### 2. sonic Pico pose service

Preferred:

```bash
cd isaaclab_twist2_g1/pico_server
bash run_sonic_pose_server.sh
```

Direct form:

```bash
python isaaclab_twist2_g1/pico_server/pico_server_pose_only.py \
  --transport redis \
  --redis_host localhost \
  --redis_port 6379
```

### 3. sonic simulator

```bash
cd isaaclab_twist2_g1
bash run_sonic.sh
```

This now defaults to:

- `--input_source pico_sonic`
- `--gmt_backend sonic`
- `--sonic_pose_source redis`

### 4. twist2 simulator

Preferred explicit entry:

```bash
cd isaaclab_twist2_g1
bash run_twist2_sim.sh
```

Compatible legacy entry:

```bash
cd isaaclab_twist2_g1
bash run.sh
```

This now routes explicitly to:

- `--input_source pico_twist2`
- `--gmt_backend twist2`

### 5. VLA simulator

```bash
cd isaaclab_twist2_g1
OPENPI_CHECKPOINT=/path/to/checkpoint \
LANGUAGE_INSTRUCTION="pick up the cup" \
bash run_vla.sh
```

### 6. Direct unified simulator CLI

Examples:

```bash
python isaaclab_twist2_g1/sim_main.py \
  --input_source pico_twist2 \
  --gmt_backend twist2 \
  --task Isaac-Move-Football-G129-Dex3-Wholebody
```

```bash
python isaaclab_twist2_g1/sim_main.py \
  --input_source pico_sonic \
  --gmt_backend sonic \
  --task Isaac-Move-Football-G129-Dex3-Wholebody
```

```bash
python isaaclab_twist2_g1/sim_main.py \
  --input_source vla \
  --gmt_backend twist2 \
  --openpi_checkpoint /path/to/checkpoint \
  --language_instruction "walk forward slowly"
```

## Smoke Checklist

### Automated checks executed

1. Python syntax compilation

Command:

```bash
python -m py_compile \
  isaaclab_twist2_g1/sim_main.py \
  isaaclab_twist2_g1/sim_main_openpi.py \
  isaaclab_twist2_g1/action_provider/action_provider_openpi.py \
  isaaclab_twist2_g1/action_provider/action_provider_sonic.py \
  isaaclab_twist2_g1/pico_server/pico_server_pose_only.py \
  isaaclab_twist2_g1/pico_server/twist2_teleop_server.py
```

Result:

- passed

2. Shell syntax validation

Command:

```bash
bash -n \
  isaaclab_twist2_g1/run_sonic.sh \
  isaaclab_twist2_g1/run_vla.sh \
  isaaclab_twist2_g1/pico_server/run_twist2_teleop.sh \
  isaaclab_twist2_g1/pico_server/run_sonic_pose_server.sh \
  TWIST2/teleop.sh
```

Result:

- passed

3. `pico_server_pose_only.py` import/syntax smoke test

Command:

```bash
cd isaaclab_twist2_g1/pico_server
python test_import.py
```

Result:

- passed

### Manual smoke checklist still required

These checks were not executed automatically in this session because they require Isaac runtime, OpenPI runtime, Pico hardware, Redis teleop flow, or other machine-local dependencies.

1. sonic end-to-end

```bash
cd isaaclab_twist2_g1/pico_server
bash run_sonic_pose_server.sh
```

In another terminal:

```bash
cd isaaclab_twist2_g1
bash run_sonic.sh
```

Verify:

- Redis keys update continuously
- sonic receives human SMPL frames
- left `X` saves an episode
- left `Y` triggers reset and the next episode restarts cleanly
- saved `.npz` contains episode markers and action/model-state data

2. twist2 end-to-end

```bash
cd isaaclab_twist2_g1/pico_server
bash run_twist2_teleop.sh
```

In another terminal:

```bash
cd isaaclab_twist2_g1
bash run.sh
```

Verify:

- existing twist2 Redis protocol still works unchanged
- recording and reset flows still behave as before

3. VLA / OpenPI end-to-end

```bash
cd isaaclab_twist2_g1
OPENPI_CHECKPOINT=/path/to/checkpoint \
LANGUAGE_INSTRUCTION="walk forward slowly" \
bash run_vla.sh
```

Verify:

- `sim_main.py` enters `input_source=vla`, `gmt_backend=twist2`
- `OpenPIActionProvider` loads OpenPI, SMPL-X, GMR, and TWIST2 policy
- VLA observation and action buffer refill work
- video output is produced if enabled

## Known Gaps

- `OpenPIActionProvider` still depends on external OpenPI / GMR / SMPL-X installations and hard external paths unless overridden by environment / CLI.
- No hardware-backed runtime validation was performed in this session.
- sonic reset is currently implemented as discard-and-reset on left `Y`, following the latest instruction to separate reset and save.

## Recommended Next Manual Pass

1. Run the three end-to-end manual smoke flows above.
2. Validate saved sonic `.npz` with a replay-oriented inspection script.
3. If VLA will later target sonic as well, add a dedicated `vla -> sonic` backend route instead of reusing the OpenPI/twist2 path.
