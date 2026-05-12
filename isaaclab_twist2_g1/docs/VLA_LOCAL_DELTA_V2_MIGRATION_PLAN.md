# VLA Local-Delta V2 Migration Plan

## Scope

Migrate VLA inference tests from the v1 action protocol to the v2 local-delta protocol.

This plan only changes code that consumes VLA policy actions during inference tests. It intentionally keeps legacy recording and data conversion code intact unless that code is on the active VLA inference path.

## Protocol Change

The action shape stays 40D:

```text
action[0:2]   root xy delta
action[2]     root z
action[3:9]   root rot6d
action[9:38]  canonical G1 joint positions
action[38:40] hand binary
```

The semantic change is only in `action[0:2]`:

```text
v1: action[0:2] = root_xy_delta_world
v2: action[0:2] = root_local_xy_delta
```

The observation state remains 64D, so existing `build_vla_observation_state` usage can stay.

## Current V1 Consumption

### Shared Runtime

`action_provider/vla_smpl_runtime.py` currently implements the v1 runtime:

- `UnifiedSMPLActionRuntime.step()` reads `action[0:2]` as `root_xy_delta_world`.
- It accumulates `body_xy_world += root_xy_delta_world`.
- `build_twist2_mimic_obs()` converts world velocity back to local velocity with `rotate_vector_inverse_wxyz(root_quat, root_lin_vel_world)`.
- `build_sonic_joint29_payload()` passes `body_pos_world`, root quaternion, canonical joints, and joint velocity into the SONIC joint29 path.

### SONIC VLA Path

The active VLA test path creates `SonicActionProvider` through `create_action_provider()`. The relevant path is:

1. `sim_eval_vla.py` sets `input_source=vla`, `gmt_backend=sonic`, and `action_source=sonic_wholebody`.
2. `SonicActionProvider._run_gear_sonic_from_vla()` fetches a LeRobot action.
3. `_apply_lerobot_semantic_action()` calls `self._lerobot_vla_runtime.step(action)`.
4. The runtime frame is converted into a SONIC joint29 payload and forwarded through `_apply_pose_data(data, "lerobot_vla_joint29")`.

SONIC coordinate alignment is separate from `action[0:2]`:

- The VLA runtime decodes the model's root orientation from `action[3:9]`.
- `_apply_pose_data()` uses the root quaternion to compute `base^-1 * aligned_ref` anchor rot6d for the SONIC encoder.
- `SONIC_VLA_USE_HEADING_ALIGN` controls this reference-orientation alignment. It is not the same as rotating the xy delta during VLA inference.
- `_apply_heading_align_to_vla_action()` still rotates world-frame xy deltas, but it is used by recording/canonical-action construction, not by the active VLA inference action consumption path.

### TWIST2 VLA Path

The active TWIST2 VLA test path creates `TWIST2ActionProvider` through `create_action_provider()`:

1. `sim_eval_vla.py` sets `input_source=vla`, `gmt_backend=twist2`, and `action_source=twist2_wholebody`.
2. `_infer_lerobot_high_level_command()` fetches a LeRobot action.
3. It calls `self._lerobot_vla_runtime.step(action_np)`.
4. `build_twist2_mimic_obs()` builds the 35D TWIST2 mimic observation.

TWIST2 already consumes root-local planar velocity in the 35D mimic observation. In v1 this required converting world delta back into the root frame. In v2, the model output is already root-local, so the conversion must be removed.

## Required Runtime Changes

### SONIC

Use `UnifiedLocalDeltaActionRuntimeV2` for VLA inference. It should:

1. Read `action[0:2]` as `root_local_xy_delta`.
2. Decode `root_quat_wxyz` from `action[3:9]` using the same rot6d row/col/auto layout rules as v1.
3. Rotate local delta to world delta:

   ```python
   root_xy_delta_world = R(root_quat_wxyz) @ [dx_local, dy_local, 0.0]
   ```

4. Accumulate `body_xy_world += root_xy_delta_world`.
5. Return both `root_local_xy_delta` and the derived `root_xy_delta_world`.
6. Pass `body_pos_world`, root quaternion, canonical joints, and joint velocity to SONIC through `build_sonic_joint29_payload_v2()`.

Rationale:

- SONIC's joint29 path mainly consumes root height, root orientation, joint positions, and joint velocities today, but the existing payload contract includes `body_pos`.
- Keeping accumulated `body_pos_world` preserves the old payload shape and makes debug/recording/future use coherent.
- The heading-align logic should remain on root orientation only; the v2 local xy delta must not be additionally heading-aligned as though it were world-frame motion.

### TWIST2

Use `UnifiedLocalDeltaActionRuntimeV2` and `build_twist2_mimic_obs_v2()`.

The TWIST2 planar velocity should be:

```python
xy_vel_local = root_local_xy_delta / control_dt
```

Do not do the old v1 conversion:

```python
xy_vel_local = rotate_inverse(root_quat, root_xy_delta_world / control_dt)
```

Rationale:

- TWIST2 mimic obs expects root-local planar velocity.
- v2 policy output is already root-local, so rotating it again would produce the wrong command whenever root yaw is nonzero.

## Files To Change

### Primary VLA Inference Consumers

- `isaaclab_twist2_g1/action_provider/action_provider_sonic.py`
  - Import v2 runtime/constants/payload builder.
  - Use v2 action/state dims for VLA inference shape checks.
  - Instantiate `UnifiedLocalDeltaActionRuntimeV2`.
  - Use `build_sonic_joint29_payload_v2`.
  - Update debug text so local and derived world xy deltas are distinguishable.

- `isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py`
  - Import v2 runtime/constants/TWIST2 mimic builder.
  - Instantiate `UnifiedLocalDeltaActionRuntimeV2`.
  - Use `build_twist2_mimic_obs_v2`.

- `isaaclab_twist2_g1/script/eval_scripts/twist2/sim_eval_vla_multisim.py`
  - This bypasses the normal action provider and has its own v1 runtime; switch it to v2.

### VLA Test Defaults

- `isaaclab_twist2_g1/batch_1_test.sh`
  - Default `TEST_MODE` should be `base_test`, because that is the existing config directory.
  - Default result tag prefix should include `localdelta_v2` to avoid mixing v1/v2 results.

- `isaaclab_twist2_g1/batch_test_*.sh` and `isaaclab_twist2_g1/pi05_batch_test_*.sh`
  - Default `TEST_MODE` should be `base_test`.

- `isaaclab_twist2_g1/script/eval_scripts/{sonic,twist2,sonic_pi05,twist2_pi05}/*vla*.py`
  - Default `robot_type` should be `unitree_g1_localdelta_v2`.

- `isaaclab_twist2_g1/script/eval_scripts/{sonic,twist2,sonic_pi05,twist2_pi05}/*run_vla_eval*.sh`
  - Default `ROBOT_TYPE` should be `unitree_g1_localdelta_v2`.

- `isaaclab_twist2_g1/sim_main.py`
  - Default `robot_type` and VLA schema log should mention v2.

## Files To Leave As Legacy

- `isaaclab_twist2_g1/action_provider/vla_smpl_runtime.py`
- `isaaclab_twist2_g1/action_provider/test_vla_smpl_runtime.py`
- `isaaclab_twist2_g1/tools/data_tools/smpl_lerobot_common.py`
- `isaaclab_twist2_g1/tools/data_tools/*64_40.py`
- `isaaclab_twist2_g1/tools/data_tools/verify_lerobot_smpl_vla.py`

These remain useful for historical data, recording, or explicit v1 verification. They should not be used by active VLA inference tests after this migration.

## Verification

Run targeted tests:

```bash
pytest isaaclab_twist2_g1/action_provider/test_vla_local_delta_runtime_v2.py -q
python -m py_compile \
  isaaclab_twist2_g1/action_provider/action_provider_sonic.py \
  isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py \
  isaaclab_twist2_g1/script/eval_scripts/twist2/sim_eval_vla_multisim.py
```

Then search for remaining active v1 runtime usage in VLA inference paths:

```bash
rg "UnifiedSMPLActionRuntime|build_twist2_mimic_obs\\(|build_sonic_joint29_payload\\(" \
  isaaclab_twist2_g1/action_provider \
  isaaclab_twist2_g1/script/eval_scripts
```

Remaining matches should only be in legacy runtime/tests or explicitly non-inference code.
