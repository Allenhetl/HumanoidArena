# HumanoidArena HOI Knowledge Graph

This graph is intentionally scoped to the HOI collection, replay/rerecord, scene
randomization, and VLA training path:

`GMT/GMR or Pico -> teleop server -> isaaclab_twist2_g1 runtime -> action_provider -> task scene/YAML -> recording npz + video sidecars -> 64D state + 40D action -> LeRobot training / serving`

It does not try to cover the full repository. In particular,
`GR00T-WholeBodyControl/`, full LeRobot internals, large assets, generated
recordings, and vendored dependencies are left out on purpose because they are
mostly outside the main HOI loop this graph is meant to optimize for.

## What Is Indexed

- Live teleop entry points and docs
- GMR retargeting core
- Isaac Lab launch scripts and `sim_main.py`
- TWIST2, SONIC, OpenPI, and replay action providers
- Reset/input-ready barriers and shared async recording helpers
- Canonical `64D` state / `40D` action runtime and tests
- Task YAML loading, scene cfg modules, events, rewards, terminations, and
  object-randomization restore utilities
- Replay/rerecord scripts, multicamera video sidecars, and replay debug docs
- DDS / layered robot-control bridge modules used by runtime paths
- Dataset conversion, cleaning, rerecord, video extraction, merge, stats repair,
  and verification tools
- LeRobot train, HTTP serving, batch-plan, batch-run, resume, and dataset
  instruction entry points

## Layers

- `operator-docs`: workflow and schema docs
- `human-input`: GMR, TWIST2 teleop, Pico pose servers
- `simulation-runtime`: shell launchers, `sim_main.py`, image server
- `control-backends`: TWIST2 / SONIC providers and canonical runtime
- `task-scene-configs`: per-task YAMLs, scene cfgs, observations, rewards,
  terminations, runtime hooks, and object reset/replay utilities
- `robot-io-dds`: DDS objects and layered robot-control facade
- `replay-rerecording`: replay/rerecord scripts, deterministic scene restore,
  reward bucketing, and multicamera sidecar tooling
- `dataset-tooling`: `*2lerobot*`, shared conversion helpers, rerecord/video
  tools, merge/stats scripts, and verification tests
- `training-serving`: LeRobot train scripts, batch orchestration, resume tooling,
  and VLA HTTP server

## Fast Lookup

- Live TWIST2 teleop: `TWIST2/deploy_real/xrobot_teleop_to_robot_w_hand.py`
- Isaac Lab main loop: `isaaclab_twist2_g1/sim_main.py`
- Env YAML loader: `isaaclab_twist2_g1/tasks/common_env_config/loader.py`
- Env object reset/replay utilities: `isaaclab_twist2_g1/common_env_objects.py`
- TWIST2 backend: `isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py`
- SONIC backend: `isaaclab_twist2_g1/action_provider/action_provider_sonic.py`
- OpenPI backend: `isaaclab_twist2_g1/action_provider/action_provider_openpi.py`
- Reset barrier helpers: `isaaclab_twist2_g1/action_provider/reset_control.py`
- Video sidecar helpers: `isaaclab_twist2_g1/action_provider/vision_video.py`
- Canonical schema: `isaaclab_twist2_g1/action_provider/vla_smpl_runtime.py`
- Replay scripts: `isaaclab_twist2_g1/run_replay_twist2.sh`, `isaaclab_twist2_g1/run_replay_sonic.sh`
- Rerecord scripts: `isaaclab_twist2_g1/run_rerecord_twist2.sh`, `isaaclab_twist2_g1/run_rerecord_sonic.sh`
- Current TWIST2 export: `isaaclab_twist2_g1/tools/data_tools/twist2lerobot_64_40.py`
- Current SONIC export: `isaaclab_twist2_g1/tools/data_tools/sonic2lerobot_clean_64_40.py`
- Dataset verifier: `isaaclab_twist2_g1/tools/data_tools/verify_lerobot_smpl_vla.py`
- Multicam rerecord helpers: `isaaclab_twist2_g1/tools/data_tools/rerecord_parallel_utils.py`
- LeRobot sample merge: `lerobot/src/lerobot/scripts/lerobot_sample_merge.py`
- LeRobot hand stats repair: `lerobot/src/lerobot/scripts/fix_gripper_quantile_stats.py`
- HumanoidArena batch training: `lerobot/scripts/run_humanoidarena_batch_train.py`
- Main LeRobot training entry: `lerobot/train.sh`
- LeRobot VLA server: `lerobot/scripts/serve_lerobot_vla_http.py`

## Important Conventions

- Canonical observation is `64D`: `root_rot6d(6) + dof_pos_29 + dof_vel_29`
- Canonical action is `40D`: `root_xy_delta(2) + root_z(1) + root_rot6d(6) + joint_pos_29 + hand_binary(2)`
- TWIST2 live control still uses `mimic_obs35`; conversion into canonical form happens in `vla_smpl_runtime.py` and dataset tools
- There are older `twist22lerobot.py` converters. Prefer the `*64_40.py` scripts unless you explicitly need the legacy format
- Replay/rerecord scene restore depends on `episode_object_seed` plus
  `episode_init_env` object state fields.
- For task YAML and task-scene randomization work in this workspace,
  `object_reset_seed_source` should be `time` only. Existing graph entries tagged
  `seed-policy-review` mark current files that still need policy reconciliation.
- Front, world, and wrist RGB streams are recorded as MP4 sidecars with NPZ path,
  fps, and frame-index metadata.

## When You Change The Project

Update this graph when you do any of the following:

- Add or remove a runtime entry script
- Change `action_provider` control routing
- Change recording NPZ field names or semantics
- Change canonical `64D` / `40D` definitions
- Change task YAML merge behavior, object reset seeding, scene randomization, or
  replay scene restore behavior
- Add/remove a task scene config, reward, termination, or observation module used
  by the HOI/VLA path
- Change replay/rerecord scripts or multicamera sidecar storage
- Change HumanoidArena LeRobot batch training, dataset merge, or stats-repair
  scripts
- Move dataset conversion logic between `isaaclab_twist2_g1` and `lerobot`

The canonical source of truth is `.understand-anything/knowledge-graph.json`.
