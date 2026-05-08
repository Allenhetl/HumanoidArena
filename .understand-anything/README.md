# HumanoidArena HOI Knowledge Graph

This graph is intentionally scoped to the HOI collection and VLA training path you described:

`GMT/GMR or Pico -> teleop server -> isaaclab_twist2_g1 runtime -> action_provider -> recording npz -> 64D state + 40D action -> LeRobot training / serving`

It does not try to cover the full repository. In particular, `GR00T-WholeBodyControl/` is left out on purpose because it is large and mostly outside the main HOI loop you asked to optimize for.

## What Is Indexed

- Live teleop entry points and docs
- GMR retargeting core
- Isaac Lab launch scripts and `sim_main.py`
- TWIST2 and SONIC action providers
- Canonical `64D` state / `40D` action runtime
- Dataset conversion, cleaning, and verification tools
- LeRobot train and HTTP serving entry points

## Layers

- `operator-docs`: workflow and schema docs
- `human-input`: GMR, TWIST2 teleop, Pico pose servers
- `simulation-runtime`: shell launchers, `sim_main.py`, image server
- `control-backends`: TWIST2 / SONIC providers and canonical runtime
- `dataset-tooling`: `*2lerobot*`, shared conversion helpers, verification tests
- `training-serving`: LeRobot train scripts and VLA HTTP server

## Fast Lookup

- Live TWIST2 teleop: `TWIST2/deploy_real/xrobot_teleop_to_robot_w_hand.py`
- Isaac Lab main loop: `isaaclab_twist2_g1/sim_main.py`
- TWIST2 backend: `isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py`
- SONIC backend: `isaaclab_twist2_g1/action_provider/action_provider_sonic.py`
- Canonical schema: `isaaclab_twist2_g1/action_provider/vla_smpl_runtime.py`
- Current TWIST2 export: `isaaclab_twist2_g1/tools/data_tools/twist2lerobot_64_40.py`
- Current SONIC export: `isaaclab_twist2_g1/tools/data_tools/sonic2lerobot_clean_64_40.py`
- Dataset verifier: `isaaclab_twist2_g1/tools/data_tools/verify_lerobot_smpl_vla.py`
- Main LeRobot training entry: `lerobot/train.sh`
- LeRobot VLA server: `lerobot/scripts/serve_lerobot_vla_http.py`

## Important Conventions

- Canonical observation is `64D`: `root_rot6d(6) + dof_pos_29 + dof_vel_29`
- Canonical action is `40D`: `root_xy_delta(2) + root_z(1) + root_rot6d(6) + joint_pos_29 + hand_binary(2)`
- TWIST2 live control still uses `mimic_obs35`; conversion into canonical form happens in `vla_smpl_runtime.py` and dataset tools
- There are older `twist22lerobot.py` converters. Prefer the `*64_40.py` scripts unless you explicitly need the legacy format

## When You Change The Project

Update this graph when you do any of the following:

- Add or remove a runtime entry script
- Change `action_provider` control routing
- Change recording NPZ field names or semantics
- Change canonical `64D` / `40D` definitions
- Move dataset conversion logic between `isaaclab_twist2_g1` and `lerobot`

The canonical source of truth is `.understand-anything/knowledge-graph.json`.
