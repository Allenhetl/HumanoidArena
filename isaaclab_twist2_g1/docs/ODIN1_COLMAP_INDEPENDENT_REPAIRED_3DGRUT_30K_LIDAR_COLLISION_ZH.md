# Odin1 修复版 3DGRUT 视觉 + LiDAR 碰撞混合场景

英文版文档：[`ODIN1_COLMAP_INDEPENDENT_REPAIRED_3DGRUT_30K_LIDAR_COLLISION.md`](ODIN1_COLMAP_INDEPENDENT_REPAIRED_3DGRUT_30K_LIDAR_COLLISION.md)

## 1. 实验目的

本实验将场景渲染与场景物理解耦：

- 视觉外观来自官方 3DGRUT 管线。训练输入是经过修复的、纯图像 COLMAP 3375 帧重建，训练步数为 30,000。
- 物理接触使用已有的、经过地面对齐的 Odin LiDAR 碰撞壳和精确地板。
- NuRec Gaussian 仅负责视觉渲染，不带碰撞 API。
- 不可见的 LiDAR 碰撞壳和地板是场景中仅有的静态房间碰撞几何。

COLMAP 重建、位姿修复和 3DGRUT 训练阶段均未注入 Odin 位姿或 LiDAR 几何。Odin 相机位姿只用于训练后的度量尺度对齐初值，最终全局 Sim(3) 再通过地面对齐后的 LiDAR 点云进行结构配准优化。

## 2. 实验环境

```text
远端主机: ${REAL_SCENE_HOST}
仓库: ${HUMANOID_ARENA_ROOT}
分支: feature/real-scene
实验时基础提交: e9d8b09
3DGRUT 源码提交: a37ef721012dea0f29c0fcfff2d525023b4e854a
Isaac Sim: unitree_sim_env 中的 Isaac Sim 5.0
Isaac 环境: ${ISAACLAB_ENV}
3DGRUT 环境: ${THREE_D_GRUT_VENV}
VLA 服务环境: ${LEROBOT_ENV}
仿真 GPU: cuda:0
VLA 服务设备: cuda:0
实验日期: 2026-07-30 至 2026-07-31
```

远端 worktree 在本实验开始前已包含其他真实场景相关改动。本实验没有自动提交，也没有覆盖任何基线资产。

## 3. COLMAP 重建与修复

初始纯图像 COLMAP 模型虽然注册了全部 3375 张图像，但 `frame_000000.png` 到 `frame_000130.png` 构成了一段连续的错误注册区间。

诊断结果：

```text
COLMAP 相邻相机中心最大跳变: 1.11 m
同区间 Odin 参考轨迹相邻步长: 小于约 0.08 m
错误帧段: frame_000000.png-frame_000130.png
恢复连贯的首帧: frame_000131.png
```

失败区间对应采集开始时的窗边路线：

```text
frame_000000-frame_000050:
  基本停留在窗户和白墙附近，平移很少；画面包含大面积空白墙和过曝窗户。

frame_000050-frame_000075:
  从窗边开始转向红色沙袋。

frame_000075-frame_000130:
  经过红色沙袋，向桌子、纸箱和储物区移动，总位移约 1.3 m。

frame_000131 之后:
  COLMAP 注册重新恢复连续。
```

修复后具有代表性的仿真坐标相机位置和朝向：

```text
frame_000000: 中心 [-0.131, -0.051, 1.124]，朝向  -4.8 deg
frame_000050: 中心 [-0.137, -0.031, 1.095]，朝向  -2.9 deg
frame_000075: 中心 [-0.214, -0.187, 1.161]，朝向 -30.4 deg
frame_000100: 中心 [-0.179, -0.546, 1.070]，朝向 -61.9 deg
frame_000130: 中心 [-0.067, -1.291, 1.066]，朝向 -65.4 deg
frame_000131: 中心 [-0.097, -1.323, 1.065]，朝向 -68.1 deg
```

修复流程：

1. 复制独立 COLMAP 模型，删除前 131 个错误注册。
2. 固定已有正确图像位姿，运行 `image_registrator`。
3. 对新注册图像进行纯图像轨迹三角化。
4. 重复注册，将有效轨迹逐步向前传播到完整起始区间。
5. 对最终 3375 张图像模型重新三角化。

此过程没有将 Odin 位姿写入 COLMAP 模型。

最终修复模型：

```text
路径: ${REAL_SCENE_WORKSPACE}/datasets/odin1/colmap_independent_repair_initial/final
注册图像: 3375/3375
Points3D: 245,893
Observations: 3,551,208
平均轨迹长度: 14.442087
平均重投影误差: 0.801438 px
```

模型文件哈希：

```text
cameras.bin: 1277079b903c225a3efd0d84149e353675b11a0bbec27c84fb9519ae1efb85e1
images.bin:  28fd45dda52e2b3520abfc3f92e63691416d8ecce9455d69ef175af1a5d639ef
points3D.bin: fe79d42b546b32ce63e00e0010b2608dddefdbc01534bd4243ce7970fb2920cd
```

保留的诊断和结构配准工具：

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/tools/align_colmap_to_odin.py
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/tools/refine_colmap_lidar_alignment.py
```

## 4. 3DGRUT 视觉资产

训练输入：

```text
${REAL_SCENE_WORKSPACE}/datasets/odin1/colmap_independent_repaired
```

训练结果：

```text
配置: apps/colmap_3dgut.yaml
训练步数: 30,000
Gaussian 数量: 1,493,826
训练时间: 802.45 s
训练速度: 37.39 it/s
测试图像: 422
PSNR: 27.2076
SSIM: 0.8730
LPIPS: 0.3236
颜色校正 PSNR: 27.8746
颜色校正 SSIM: 0.8708
颜色校正 LPIPS: 0.3221
```

相对于修复前的全量 3375 帧 30k checkpoint，指标有所提升：

```text
PSNR: 26.894 -> 27.208
SSIM: 0.864 -> 0.873
LPIPS: 0.334 -> 0.324
```

Checkpoint：

```text
${THREE_D_GRUT_WORKSPACE}/runs/odin1_colmap_independent_repaired_official_30k/odin1_colmap_independent_repaired_official_30k/colmap_independent_repaired-3007_230320/ours_30000/ckpt_30000.pt
SHA-256: b512a3fcbae3ff181878246dc046933f9b9730ccc16f743f43f9c74711fe3878
```

导出目录：

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin1_colmap_independent_repaired_3dgrut_30k
```

导出文件：

```text
odin1_colmap_independent_repaired_3dgrut_30k_raw.ply
  大小: 370,470,380 bytes
  SHA-256: a2de33e00dfa0e0ddf9cdee8548cefa953e56c4e0d60f1300923361468c4416a

odin1_colmap_independent_repaired_3dgrut_30k_raw.usdz
  大小: 176,290,538 bytes
  SHA-256: ab948809630e05d41fa6fa9b907e1e742f393aeb5d5e68209b6834e09370180c
```

部署后的 USDZ 包含：

```text
default.usda
gauss.usda
odin1_colmap_independent_repaired_3dgrut_30k_raw.nurec
```

COLMAP 到仿真世界的变换由 USD 父级 Xform 统一施加。对齐阶段没有重新烘焙 Gaussian 的位置、协方差、透明度、颜色或 SH 参数。

## 5. LiDAR 碰撞资产

静态碰撞几何复用 Odin LiDAR 地面对齐管线的结果：

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin_kf800_lidar_agree_final22000_collision_shell.usdc
  大小: 3,148,833 bytes
  SHA-256: 32f8b3879a16011afd368267a6e9042a4f3112e351dda539d1a0d5e238399002

${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin_kf800_lidar_agree_final22000_collision_floor.usdc
  大小: 1,786,043 bytes
  SHA-256: 3b3bcab2d3a966a751bc81bce985e3ba7ad9a8b8c29d69b4171c1d43edf3adbf
```

碰撞源文件：

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin_lidar_collision/odin_lidar_collision_environment_shell.ply
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin_lidar_collision/odin_lidar_collision_floor_slab.ply
```

碰撞统计和验证结果：

```text
碰撞壳顶点: 132,275
碰撞壳三角形: 263,623
地板顶点: 144,870
地板三角形: 289,744
仿真坐标地板顶面: z = 0
输入点云到碰撞面的中位距离: 0.0213 m
输入点云到碰撞面的 p95: 0.0518 m
输入点云在 0.04 m 内的比例: 0.8924
输入点云在 0.08 m 内的比例: 0.9864
```

验证报告：

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin_lidar_collision/odin_lidar_collision_validation.json
```

两个碰撞网格均使用三角网格碰撞，`approximation=none`，并设置为不可见。

## 6. 坐标对齐

`align_colmap_to_odin.py` 负责相机 Sim(3) 初始估计和残差审计。`refine_colmap_lidar_alignment.py` 以此为初值进行尺度扫描，并对地面对齐的 LiDAR 点云执行多尺度鲁棒 point-to-plane ICP。

权威地板变换来自以下报告中的 `odin_to_sim_floor_aligned`：

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/exports/odin_lidar_collision/odin_lidar_collision_poisson_report.json
Z 平移: 1.1393921587273237 m
```

最终列向量变换：

```text
p_sim = T_colmap_to_sim * p_colmap

T_colmap_to_sim =
[[ 0.8255063839646276,  0.0422368018366223,  0.1084796324025602, -2.8707523575629557],
 [ 0.1163980138691317, -0.3115620103245967, -0.7644561361724771, -1.8774184431887595],
 [ 0.0018111992993892,  0.7721125140386250, -0.3144066686061038,  1.1178267409565537],
 [ 0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  1.0000000000000000]]
```

结构配准结果：

```text
统一尺度: 0.8336741378
相对于相机初值的选定尺度因子: 0.995
截断 RMSE: 0.04872 m
稀疏点到 LiDAR 中位距离: 0.03428 m
10 cm ICP fitness: 0.72317
10 cm ICP inlier RMSE: 0.04172 m
稀疏点在 0.04 m 内的比例: 0.55870
稀疏点在 0.08 m 内的比例: 0.71289
```

仅依赖相机中心的最小二乘残差没有达到原始目标：

```text
训练集 median: 0.03360 m
训练集 p95: 0.05409 m
留出集 median: 0.04373 m
留出集 p95: 0.07352 m
原定目标: median <= 0.02 m，p95 <= 0.05 m
```

这说明轨迹中仍存在厘米级 SLAM/SfM 漂移。因此最终物理位置以 LiDAR 结构对齐为准，不宣称相机中心达到 2 cm 内的全局精度。

对齐报告：

```text
${REAL_SCENE_WORKSPACE}/artifacts/odin1/colmap_independent_alignment_repaired/alignment_report.json
${REAL_SCENE_WORKSPACE}/artifacts/odin1/colmap_independent_structural_alignment/structural_alignment_report.json
```

## 7. 混合 USD 场景

部署后的视觉资产：

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin1_colmap_independent_repaired_3dgrut_30k.usdz
```

组合 wrapper：

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin1_colmap_independent_repaired_3dgrut_30k_lidar_collision_scene.usda
SHA-256: fbddb47da0bd2cda8cf77b38d6c7fe46bd244ef248d70e36a8c8304fbc8929fb
```

组合结构：

```text
World
|-- GaussianVisual
|   `-- odin1_colmap_independent_repaired_3dgrut_30k.usdz
|-- CollisionShell
|   `-- odin_kf800_lidar_agree_final22000_collision_shell.usdc
`-- CollisionFloor
    `-- odin_kf800_lidar_agree_final22000_collision_floor.usdc
```

验证后的 Stage 属性：

```text
metersPerUnit: 1
upAxis: Z
NuRec Volume 数量: 1
启用碰撞的 prim 数量: 2
invertColorCorrection: false
invertToneMap: false
```

Prim 状态：

```text
/World/GaussianVisual/gauss/gauss
  类型: Volume
  collision API: 无

/World/CollisionShell/mesh
  collision enabled: true
  visibility: invisible

/World/CollisionFloor/mesh
  collision enabled: true
  visibility: invisible
```

## 8. 任务配置

任务 YAML：

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/tasks/common_env_config/real_scene_odin1_colmap_independent_repaired_3dgrut_30k_lidar_collision_football_vla_smoke_sonic.yaml
SHA-256: b768b98d3b6365e6691ded4ef37237fcb697f921998960dee16ab3e6bf5fadea
```

当前任务参数：

```text
任务: Isaac-Move-Football-Single-G129-Dex3-Wholebody
后端: sonic
仿真 dt: 0.005
Decimation: 4
机器人位置: [-1.27, -2.98, 0.789]
机器人四元数 wxyz: [0.0805, 0, 0, -0.9968]
机器人 yaw: 约 -170.8 deg
足球位置: [-2.74, -3.25, 0.11]
机器人补光位置: [-1.75, -3.10, 2.2]
机器人补光强度: 8000
默认地面和球门资产: 禁用
自动管灯: 禁用
```

当前出生点依据训练帧 `frame_001232.png` 选择：

```text
参考训练相机中心: [-1.317, -3.010, 1.234]
参考相机朝向: 约 -170.8 deg
机器人头部相机水平位置差: 小于 0.01 m
机器人头部相机高度差: 约 0.03 m
机器人到足球距离: 约 1.50 m
```

当前机器人位于中央垫区边缘，面向足球和中央开阔活动空间。旧出生点为 `[-2.74, -4.45, 0.789]`、yaw `+90 deg`，虽然位置接近一段训练轨迹，但与最近训练视角的完整方向差约 133 度。

## 9. VLA 模型与运行环境

VLA 模型：

```text
${HUMANOID_ARENA_ROOT}/vla_ckpts/HOI_football/diffusion_sonic_football_0529/pretrained_model
```

SONIC 模型：

```text
${HUMANOID_ARENA_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
${HUMANOID_ARENA_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
```

VLA 服务：

```text
Python: ${LEROBOT_ENV}/bin/python
服务脚本: ${HUMANOID_ARENA_ROOT}/lerobot/scripts/serve_lerobot_vla_http.py
设备: cuda:0
```

## 10. 当前出生点 Smoke 命令

已经验证的当前出生点 20-step 命令：

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

## 11. 实验结果

### 11.1 旧出生点 20-Step Smoke

```text
机器人出生点: [-2.74, -4.45, 0.789]，yaw +90 deg
步数: 20/20
结束原因: fixed_horizon_complete
耗时: 4.13 s
相关日志错误: 无
结果目录:
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_smoke20_20260730
```

### 11.2 旧出生点 500-Step 正式实验

```text
机器人出生点: [-2.74, -4.45, 0.789]，yaw +90 deg
Seed: 0
Repeat: 1
步数: 500/500
结束原因: fixed_horizon_complete
耗时: 52.77 s
最终奖励: -1.0
相关日志错误: 无
结果目录:
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_vla_front_seed0_repeat1_20260730
```

远端视频：

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_vla_front_seed0_repeat1_20260730/videos/success/vla_ckpts__diffusion_sonic_football_0529__seed_0__repeat_0__episode_0__success.mp4
```

本地视频：

```text
${LOCAL_EXPERIMENT_ARCHIVE}/odin1_colmap_independent_repaired_3dgrut_30k_vla_front_500steps.mp4
SHA-256: 778fee597c939c4f41eab92bd060360cac05fe29c0b4b518ca5014442f5080e3
```

机器人在末帧仍保持站立。该实验发生在调整出生点之前，不能作为当前出生点的 500-step 验证结果。

### 11.3 当前训练视角出生点 Smoke

```text
机器人出生点: [-1.27, -2.98, 0.789]，yaw -170.8 deg
步数: 20/20
结束原因: fixed_horizon_complete
相关日志错误: 无
结果目录:
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_smoke20_20260731
```

远端视频：

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/eval_results/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_smoke20_20260731/videos/success/vla_ckpts__diffusion_sonic_football_0529__seed_0__repeat_0__episode_0__success.mp4
```

本地检查文件：

```text
${LOCAL_EXPERIMENT_ARCHIVE}/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_smoke20.mp4
SHA-256: 585617ac321d1fca74de2aa85932724dfadef5d5cd1dd385e012db925561eeca
${LOCAL_EXPERIMENT_ARCHIVE}/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_firstframe.png
${LOCAL_EXPERIMENT_ARCHIVE}/odin1_colmap_independent_repaired_3dgrut_30k_training_supported_pose_finalframe.png
```

日志检查未发现：

```text
NuRec 加载或渲染错误
USD 引用或组合错误
PhysX 碰撞错误
fall_detected
Traceback
RuntimeError
```

当前出生点尚未执行新的 500-step 正式实验。

## 12. 视觉检查结论

当前出生点首帧能够直接看到足球、中央垫区、周围桌椅、柜体和训练器材。与旧出生点相比，机器人头部相机位置更接近真实训练相机，同时面向中央活动空间，因此更适合检查当前 Gaussian 的实际效果。

周围房间结构能够稳定辨认。机器人正下方和近距离前方的地板仍存在明显模糊和拖影。这主要是因为原始采集以房间外围、接近水平的手持视角为主，而机器人前置相机有较大的向下俯视角，对中央地板和近距离地面区域的训练覆盖不足。

出生点调整没有修改 Gaussian 模型，只修改了机器人、足球和补光灯的任务初始位置。

## 13. 失败段对应的采集建议

`frame_000000-frame_000130` 同时包含多个不利于注册的条件：

- 前约 50 帧平移很少。
- 空白白墙占据较大画面比例。
- 窗户严重过曝。
- 从近似静止突然过渡到平移和转向同时发生。
- 初始墙面/窗户视角与后续沙袋/桌柜视角之间缺少持续稳定的共同纹理。

后续采集建议：

1. 录制开始后先完成至少 `0.5-1.0 m` 的平移，不要只在原地旋转。
2. 连续帧中保留桌角、架子边缘、设备、地面接缝等稳定纹理。
3. 在窗边开始采集前锁定或降低曝光。
4. 降低转弯速度，避免突然大角度转向和运动模糊。
5. 每经过一个弱纹理区后重新回到已有稳定路线，形成闭环。
6. 增加中央地板的低视角和向下斜视视角。
7. 增加当前机器人出生点及可能的足球运动路线视角。
8. 统一相机安装方向和图像方向元数据；当前源图像显示时整体存在固定 180 度旋转。

## 14. 结果解释与限制

- 结果中的 `success=true` 仅表示评估器完成固定步数，不表示足球任务奖励目标成功。
- 旧出生点 500-step 实验的最终奖励仍为 `-1.0`。
- 当前训练视角出生点只通过了 20-step smoke，尚未进行 500-step 正式实验。
- 视觉来自 NuRec Gaussian，物理碰撞只来自 LiDAR 碰撞壳和精确地板。
- 相机中心残差没有达到原定 `median 2 cm / p95 5 cm` 目标，最终物理位置以 LiDAR 结构对齐为准。
- 全局结构 ICP 指标不代表每个货架、墙面或障碍物都具有厘米级局部精度。
- 机器人俯视视角会暴露原始外围水平采集未充分覆盖的地板区域。
- Wrapper 使用相对 USD 引用，因此 USDZ、wrapper 和两个碰撞 USDC 应保存在同一资产目录。

## 15. Manifest、回滚和后续工作

机器可读 manifest：

```text
${HUMANOID_ARENA_ROOT}/isaaclab_twist2_g1/assets/objects/real_scene/odin1_colmap_independent_repaired_3dgrut_30k_manifest.json
SHA-256: 61809bf06a285402f804112d243b0f756c9ebeb13c1076911f06ed722c3eb065
```

所有实验资产均使用新增名称。回滚时只需将任务 YAML 中的 `scene.room.spawn.usd_path` 改回之前的 wrapper，并恢复对应机器人和足球出生点；没有基线资产需要还原。

建议的后续工作：

1. 补采中央垫区和当前机器人到足球路径上的俯视、斜视图像。
2. 对原来的窗边、红色沙袋和储物区路线增加具有充分平移视差的往返采集。
3. 重新运行独立 COLMAP，并在启动 3DGRUT 训练前检查相邻相机运动连续性。
4. 使用相同的 422 张测试图像，对扩充数据后的重建与当前 checkpoint 做可比评估。
5. 确认新视觉覆盖后，对当前出生点执行 500-step 固定时长实验。
6. 在任务涉及墙壁、货架或狭窄通道前，增加可见碰撞叠加图和障碍物接触测试。
