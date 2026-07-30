# Odin1 Repaired 3DGRUT Visual + LiDAR Collision Hybrid Scene

## 1. Purpose

This experiment separates scene rendering from scene physics:

- Visual appearance uses an official 3DGRUT model trained for 30,000 iterations from a repaired, image-only, 3375-image COLMAP reconstruction.
- Physical contact uses the existing floor-aligned Odin LiDAR collision shell and exact floor.
- The NuRec Gaussian is visual-only and has no collision API.
- The invisible LiDAR shell and floor are the only static room collision geometry.

The COLMAP reconstruction, its repair, and 3DGRUT training did not inject Odin poses or LiDAR geometry. Odin camera poses seed the post-training metric alignment only, and floor-aligned LiDAR points refine the final global Sim(3).

## 2. Environment

```text
Host: ${REAL_SCENE_HOST}
Repository: ${HUMANOID_ARENA_ROOT}
Branch: feature/real-scene
Base commit during the experiment: e9d8b09
3DGRUT source commit: a37ef721012dea0f29c0fcfff2d525023b4e854a
Isaac Sim: 5.0 runtime from unitree_sim_env
Simulation environment: ${ISAACLAB_ENV}
3DGRUT environment: ${THREE_D_GRUT_VENV}
VLA server environment: ${LEROBOT_ENV}
Simulation GPU: cuda:0
VLA server device: cuda:0
Date: 2026-07-30 through 2026-07-31
```

The worktree already contained unrelated and earlier real-scene changes. This experiment was not committed automatically, and no baseline asset was overwritten.

## 3. Reconstruction and Repair

The initial image-only model registered all 3375 images, but `frame_000000.png` through `frame_000130.png` formed a coherent registration failure. COLMAP adjacent-camera jumps reached `1.11 m`, while the Odin reference trajectory stayed below approximately `0.08 m` per frame.

The failed interval corresponds to the initial window-side route:

```text
frame_000000-frame_000050: nearly stationary beside a blank wall and overexposed window
frame_000050-frame_000075: turns from the window toward the red punching bag
frame_000075-frame_000130: moves about 1.3 m past the bag toward the table/storage area
frame_000131 onward: coherent registration resumes
```

The repaired simulation-frame camera path for representative frames is:

```text
frame_000000: center [-0.131, -0.051, 1.124], heading  -4.8 deg
frame_000050: center [-0.137, -0.031, 1.095], heading  -2.9 deg
frame_000075: center [-0.214, -0.187, 1.161], heading -30.4 deg
frame_000100: center [-0.179, -0.546, 1.070], heading -61.9 deg
frame_000130: center [-0.067, -1.291, 1.066], heading -65.4 deg
frame_000131: center [-0.097, -1.323, 1.065], heading -68.1 deg
```

Repair procedure:

1. Copy the independent COLMAP model and delete the 131 incorrect registrations.
2. Run `image_registrator` with existing image poses fixed.
3. Triangulate new image-only tracks.
4. Repeat registration to propagate tracks backward through the complete initial segment.
5. Triangulate the final 3375-image model.

No Odin pose was written into the COLMAP model during this process.

Final repaired model:

```text
Path: ${REAL_SCENE_WORKSPACE}/datasets/odin1/colmap_independent_repair_initial/final
Registered images: 3375/3375
Points3D: 245,893
Observations: 3,551,208
Mean track length: 14.442087
Mean reprojection error: 0.801438 px
```

Model hashes:

```text
cameras.bin: 1277079b903c225a3efd0d84149e353675b11a0bbec27c84fb9519ae1efb85e1
images.bin:  28fd45dda52e2b3520abfc3f92e63691416d8ecce9455d69ef175af1a5d639ef
points3D.bin: fe79d42b546b32ce63e00e0010b2608dddefdbc01534bd4243ce7970fb2920cd
```

Alignment and repair tools retained in the repository:

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/tools/align_colmap_to_odin.py
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/tools/refine_colmap_lidar_alignment.py
```

## 4. 3DGRUT Visual Asset

Training input:

```text
${REAL_SCENE_WORKSPACE}/datasets/odin1/colmap_independent_repaired
```

Training result:

```text
Config: apps/colmap_3dgut.yaml
Iterations: 30,000
Gaussians: 1,493,826
Training time: 802.45 s
Iteration speed: 37.39 it/s
Test images: 422
PSNR: 27.2076
SSIM: 0.8730
LPIPS: 0.3236
Color-corrected PSNR: 27.8746
Color-corrected SSIM: 0.8708
Color-corrected LPIPS: 0.3221
```

The repaired model improved over the original full-3375 30k checkpoint:

```text
PSNR: 26.894 -> 27.208
SSIM: 0.864 -> 0.873
LPIPS: 0.334 -> 0.324
```

Checkpoint:

```text
${THREE_D_GRUT_WORKSPACE}/runs/odin1_colmap_independent_repaired_official_30k/odin1_colmap_independent_repaired_official_30k/colmap_independent_repaired-3007_230320/ours_30000/ckpt_30000.pt
SHA-256: b512a3fcbae3ff181878246dc046933f9b9730ccc16f743f43f9c74711fe3878
```

Export directory:

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin1_colmap_independent_repaired_3dgrut_30k
```

Exported files:

```text
odin1_colmap_independent_repaired_3dgrut_30k_raw.ply
  bytes: 370,470,380
  SHA-256: a2de33e00dfa0e0ddf9cdee8548cefa953e56c4e0d60f1300923361468c4416a

odin1_colmap_independent_repaired_3dgrut_30k_raw.usdz
  bytes: 176,290,538
  SHA-256: ab948809630e05d41fa6fa9b907e1e742f393aeb5d5e68209b6834e09370180c
```

The deployed USDZ contains:

```text
default.usda
gauss.usda
odin1_colmap_independent_repaired_3dgrut_30k_raw.nurec
```

The COLMAP-to-simulation transform is applied by the parent USD Xform. Gaussian positions, covariance, opacity, color, and SH coefficients are not re-baked during scene alignment.

## 5. Collision Assets

The static collision geometry is reused from the floor-aligned Odin LiDAR pipeline:

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin_kf800_lidar_agree_final22000_collision_shell.usdc
  bytes: 3,148,833
  SHA-256: 32f8b3879a16011afd368267a6e9042a4f3112e351dda539d1a0d5e238399002

${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin_kf800_lidar_agree_final22000_collision_floor.usdc
  bytes: 1,786,043
  SHA-256: 3b3bcab2d3a966a751bc81bce985e3ba7ad9a8b8c29d69b4171c1d43edf3adbf
```

Original collision sources:

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin_lidar_collision/odin_lidar_collision_environment_shell.ply
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin_lidar_collision/odin_lidar_collision_floor_slab.ply
```

Collision statistics and validation:

```text
Shell vertices: 132,275
Shell triangles: 263,623
Floor vertices: 144,870
Floor triangles: 289,744
Floor top surface in simulation coordinates: z = 0
Input cloud to collision surface median: 0.0213 m
Input cloud to collision surface p95: 0.0518 m
Input cloud fraction within 0.04 m: 0.8924
Input cloud fraction within 0.08 m: 0.9864
```

Validation report:

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin_lidar_collision/odin_lidar_collision_validation.json
```

Both collision meshes use triangle-mesh collision, approximation `none`, and invisible authored visibility.

## 6. Coordinate Alignment

`align_colmap_to_odin.py` estimates and audits a camera-based Sim(3). `refine_colmap_lidar_alignment.py` uses that transform as a seed, scans scale, and runs robust multi-scale point-to-plane ICP against the floor-aligned LiDAR cloud.

The authoritative floor transform is `odin_to_sim_floor_aligned` from:

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin_lidar_collision/odin_lidar_collision_poisson_report.json
Z translation: 1.1393921587273237 m
```

Final column-vector transform:

```text
p_sim = T_colmap_to_sim * p_colmap

T_colmap_to_sim =
[[ 0.8255063839646276,  0.0422368018366223,  0.1084796324025602, -2.8707523575629557],
 [ 0.1163980138691317, -0.3115620103245967, -0.7644561361724771, -1.8774184431887595],
 [ 0.0018111992993892,  0.7721125140386250, -0.3144066686061038,  1.1178267409565537],
 [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  1.0000000000000000]]
```

Structural alignment result:

```text
Uniform scale: 0.8336741378
Selected scale factor relative to camera seed: 0.995
Trimmed RMSE: 0.04872 m
Sparse-to-LiDAR median: 0.03428 m
10 cm ICP fitness: 0.72317
10 cm ICP inlier RMSE: 0.04172 m
Sparse fraction within 0.04 m: 0.55870
Sparse fraction within 0.08 m: 0.71289
```

Camera-only least-squares residuals remained larger than the original acceptance target:

```text
Train median: 0.03360 m
Train p95: 0.05409 m
Held-out median: 0.04373 m
Held-out p95: 0.07352 m
Original desired threshold: median <= 0.02 m and p95 <= 0.05 m
```

This indicates residual SLAM/SfM trajectory drift. The final physical placement therefore uses LiDAR structural alignment rather than claiming camera-center accuracy below 2 cm.

Reports:

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/colmap_independent_alignment_repaired/alignment_report.json
${REAL_SCENE_WORKSPACE}/artifacts/odin1/colmap_independent_structural_alignment/structural_alignment_report.json
```

## 7. Hybrid USD

Deployed visual asset:

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin1_colmap_independent_repaired_3dgrut_30k.usdz
```

Composition wrapper:

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin1_colmap_independent_repaired_3dgrut_30k_lidar_collision_scene.usda
SHA-256: fbddb47da0bd2cda8cf77b38d6c7fe46bd244ef248d70e36a8c8304fbc8929fb
```

Composition structure:

```text
World
|-- GaussianVisual
|   `-- odin1_colmap_independent_repaired_3dgrut_30k.usdz
|-- CollisionShell
|   `-- odin_kf800_lidar_agree_final22000_collision_shell.usdc
`-- CollisionFloor
    `-- odin_kf800_lidar_agree_final22000_collision_floor.usdc
```

Validated stage properties:

```text
metersPerUnit: 1
upAxis: Z
NuRec Volume count: 1
Collision-enabled prim count: 2
invertColorCorrection: false
invertToneMap: false
```

Validated prim state:

```text
/World/GaussianVisual/gauss/gauss
  type: Volume
  collision API: absent

/World/CollisionShell/mesh
  collision enabled: true
  visibility: invisible

/World/CollisionFloor/mesh
  collision enabled: true
  visibility: invisible
```

## 8. Task Configuration

Task YAML:

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/tasks/common_env_config/real_scene_odin1_colmap_independent_repaired_3dgrut_30k_lidar_collision_football_vla_smoke_sonic.yaml
SHA-256: b768b98d3b6365e6691ded4ef37237fcb697f921998960dee16ab3e6bf5fadea
```

Current task values:

```text
Task: Isaac-Move-Football-Single-G129-Dex3-Wholebody
Backend: sonic
Simulation dt: 0.005
Decimation: 4
Robot position: [-1.27, -2.98, 0.789]
Robot quaternion wxyz: [0.0805, 0, 0, -0.9968]
Robot yaw: approximately -170.8 deg
Football position: [-2.74, -3.25, 0.11]
Robot key light position: [-1.75, -3.10, 2.2]
Robot key light intensity: 8000
Default ground and goal assets: disabled
Automatic tube lights: disabled
```

The current spawn was selected from the support of training frame `frame_001232.png`:

```text
Reference training camera center: [-1.317, -3.010, 1.234]
Reference camera heading: approximately -170.8 deg
Estimated robot head-camera horizontal offset: < 0.01 m
Estimated robot head-camera vertical offset: approximately 0.03 m
Football distance from robot: approximately 1.50 m
```

This places the robot on the edge of the central mat, facing the football and open activity area. It replaces the earlier spawn `[-2.74, -4.45, 0.789]` with yaw `+90 deg`, whose nearest training position had an approximately 133-degree full orientation mismatch.

## 9. VLA Model and Runtime

VLA model:

```text
${HUMANOID_ARENA_ROOT}/vla_ckpts/HOI_football/diffusion_sonic_football_0529/pretrained_model
```

SONIC models:

```text
${HUMANOID_ARENA_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
${HUMANOID_ARENA_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
```

VLA server:

```text
Python: ${LEROBOT_ENV}/bin/python
Script: ${HUMANOID_ARENA_ROOT}/lerobot/scripts/serve_lerobot_vla_http.py
Device: cuda:0
```

## 10. Current-Spawn Smoke Command

The validated current-spawn run used:

```bash
ROBOT_USD_OVERRIDE=${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd \
LEROBOT_VLA_RECORD_OUTPUTS=0 \
${ISAACLAB_ENV}/bin/python \
  ${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_vla_suite_parallel.py \
  --task Isaac-Move-Football-Single-G129-Dex3-Wholebody \
  --env_config_yaml ${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/tasks/common_env_config/real_scene_odin1_colmap_independent_repaired_3dgrut_30k_lidar_collision_football_vla_smoke_sonic.yaml \
  --model-path ${HUMANOID_ARENA_ROOT}/vla_ckpts/HOI_football/diffusion_sonic_football_0529/pretrained_model \
  --seed 0 --repeats_per_seed 1 --persistent_sim 1 \
  --max_steps 20 --fixed_horizon \
  --video_fps 30 --post_termination_record_steps 0 --record_video_every_n 1 \
  --robot_type unitree_g1_refpose_v3_1 \
  --sonic_encoder_path ${HUMANOID_ARENA_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx \
  --sonic_decoder_path ${HUMANOID_ARENA_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
  --results_dir ${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_smoke20_20260731 \
  --headless --isaac_device cpu \
  --server_python ${LEROBOT_ENV}/bin/python \
  --server_script ${HUMANOID_ARENA_ROOT}/lerobot/scripts/serve_lerobot_vla_http.py \
  --server_gpu_ids 0 --server_port_base 10000 --server_port_max 15000 \
  --server_ready_timeout 60 --lerobot_server_timeout 5 --num_workers 1
```

## 11. Experiment Results

### 11.1 Original-Spawn 20-Step Smoke

```text
Robot spawn: [-2.74, -4.45, 0.789], yaw +90 deg
Steps: 20/20
Reason: fixed_horizon_complete
Duration: 4.13 s
Log errors: none
Result directory:
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_smoke20_20260730
```

### 11.2 Original-Spawn 500-Step Formal Run

```text
Robot spawn: [-2.74, -4.45, 0.789], yaw +90 deg
Seed: 0
Repeat: 1
Steps: 500/500
Reason: fixed_horizon_complete
Duration: 52.77 s
Final reward: -1.0
Log errors: none
Result directory:
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_vla_front_seed0_repeat1_20260730
```

Episode video:

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_vla_front_seed0_repeat1_20260730/videos/success/vla_ckpts__diffusion_sonic_football_0529__seed_0__repeat_0__episode_0__success.mp4
```

Local video:

```text
${LOCAL_EXPERIMENT_ARCHIVE}/odin1_colmap_independent_repaired_3dgrut_30k_vla_front_500steps.mp4
SHA-256: 778fee597c939c4f41eab92bd060360cac05fe29c0b4b518ca5014442f5080e3
```

The robot remained upright in the final frame. This run predates the current training-supported spawn and must not be presented as current-spawn validation.

### 11.3 Current Training-Supported Spawn Smoke

```text
Robot spawn: [-1.27, -2.98, 0.789], yaw -170.8 deg
Steps: 20/20
Reason: fixed_horizon_complete
Log errors: none
Result directory:
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_smoke20_20260731
```

Episode video:

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_smoke20_20260731/videos/success/vla_ckpts__diffusion_sonic_football_0529__seed_0__repeat_0__episode_0__success.mp4
```

Local inspection artifacts:

```text
${LOCAL_EXPERIMENT_ARCHIVE}/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_smoke20.mp4
SHA-256: 585617ac321d1fca74de2aa85932724dfadef5d5cd1dd385e012db925561eeca
${LOCAL_EXPERIMENT_ARCHIVE}/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_firstframe.png
${LOCAL_EXPERIMENT_ARCHIVE}/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_finalframe.png
```

Log checks found no matching:

```text
NuRec load/render error
USD unresolved/composition error
PhysX collision error
fall_detected
Traceback
RuntimeError
```

The current spawn has not yet received a 500-step formal run.

## 12. Visual Check

The current-spawn first frame directly observes the football, central mat, surrounding desks, chairs, cabinets, and exercise equipment. It is more suitable for evaluating the present Gaussian than the original spawn because the head-camera position is close to a training camera and the robot faces the central activity area.

The surrounding room structure is recognizable and stable. The floor directly below and immediately in front of the robot remains visibly blurred and smeared. This is expected because most source views are near-horizontal handheld views around the room perimeter, while the robot front camera has a substantially downward pitch and observes floor regions with weaker training support.

The pose change does not alter the Gaussian model. It only changes robot, football, and key-light task placement.

## 13. Capture Guidance from the Failed Segment

The failed `frame_000000-frame_000130` interval combines several weak-registration conditions:

- Little translation during the first approximately 50 frames.
- A blank wall occupying a large part of the image.
- Strongly overexposed windows.
- A transition from near-stationary capture into simultaneous translation and turning.
- Limited stable texture shared between the initial wall/window view and the later punching-bag/table view.

Future capture should:

1. Begin with at least `0.5-1.0 m` of translational motion rather than stationary rotation.
2. Keep textured objects, corners, shelf edges, or floor markings visible across consecutive frames.
3. Lock or reduce exposure near windows before beginning the route.
4. Slow down turns and avoid abrupt heading changes or motion blur.
5. Revisit an already stable route after each weak-texture section to create loop closure.
6. Add low and downward-looking views over the central floor where the robot camera operates.
7. Add views from the current robot spawn and along likely football-motion paths.
8. Standardize the camera/image orientation metadata; the current source images are consistently displayed with a 180-degree rotation.

## 14. Interpretation and Limitations

- `success=true` means fixed-horizon completion. It does not mean the football reward objective succeeded.
- The original-spawn 500-step final reward remained `-1.0`.
- The current training-supported spawn has passed only a 20-step smoke, not a formal 500-step run.
- Rendering comes from the NuRec Gaussian. Physics comes exclusively from the LiDAR shell and exact floor.
- Camera-center residuals do not meet the original `2 cm median / 5 cm p95` target. LiDAR structural alignment is authoritative for physical placement.
- The structural ICP value summarizes globally matched sparse surfaces; it does not guarantee centimeter-level alignment at every shelf, wall, or obstacle.
- The downward robot view exposes floor regions that were not sufficiently covered by the original perimeter-oriented capture.
- Keep the USDZ, wrapper, and both collision USDC files in the same asset directory because the wrapper uses relative references.

## 15. Manifest, Rollback, and Follow-Up

Machine-readable manifest:

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin1_colmap_independent_repaired_3dgrut_30k_manifest.json
SHA-256: 61809bf06a285402f804112d243b0f756c9ebeb13c1076911f06ed722c3eb065
```

All experiment files use additive names. Rollback requires only changing the task YAML `scene.room.spawn.usd_path` back to an earlier wrapper and restoring the desired robot/object spawn. No baseline asset was replaced.

Recommended follow-up:

1. Collect downward and oblique views over the central mat and the current robot-to-football corridor.
2. Include translation-rich passes through the original window, punching-bag, and storage route.
3. Re-run independent COLMAP and verify adjacent-camera motion before starting 3DGRUT training.
4. Compare the expanded reconstruction against this checkpoint using the same 422-image test split.
5. Run a current-spawn 500-step fixed-horizon experiment after accepting the new visual coverage.
6. Add visible collision-overlay and obstacle-contact tests before using walls, shelves, or narrow passages for task evaluation.
