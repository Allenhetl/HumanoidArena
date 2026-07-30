# Real2Sim Gaussian Scene Guide

本文档记录从留形平台导出的 Gaussian PLY 接入 HumanoidArena / Isaac Sim 的标准流程、必须检查的规则，以及本次 `ipark_t2_505_20260721` 场景调试中总结出的注意事项。

## 1. 目标

Real2Sim 场景接入的目标是：

1. 将真实场景重建结果作为 Isaac Sim 中的视觉背景加载。
2. 支持通过 YAML 切换真实场景、机器人初始位姿、光源和交互物体。
3. 保证机器人和交互物体在相机视野内可见，并避免因坐标系、SH degree 或光照配置错误导致渲染失败。

当前流程主要面向 **Gaussian / NuRec USDZ + USDA wrapper**。如果后续需要物理导航、落脚或碰撞，必须额外提供与 Gaussian 对齐的 collision mesh。

## 2. 留形平台导出规则

### 2.1 必须选择 Z-up 导出

留形平台导出 PLY 时必须选择：

```text
Z-up
```

原因：Isaac Sim / Isaac Lab 使用 Z-up 世界坐标。当前 real2sim pipeline 默认留形平台导出的 Gaussian PLY 已经是 Z-up，因此 wrapper USDA 不再做额外 X -90° 或 Y-up 转换。

> 历史问题：上一版本导出时误选了 Y-up，导致高斯坐标轴和当前 Z-up pipeline 不一致。后续不要沿用那次导出的坐标假设，也不要在 wrapper 里盲目添加旋转去“修正”新导出的 Z-up 资产。

规则：

- 新资产导出必须记录 `coordinate_system: z_up`。
- 如果发现需要 wrapper 旋转，必须先回查导出设置，确认是否误选了 Y-up。
- 默认禁止给留形平台 Z-up PLY 加 X -90° wrapper rotation。

### 2.2 必须记录导出元信息

每个场景建议保存一份 sidecar metadata，例如：

```yaml
scene_name: ipark_t2_505_20260721
source_platform: liuxing
export_date: 2026-07-21
coordinate_system: z_up
sh_degree: 2
floor_z: -1.124
robot_init:
  pos: [-3.239, -4.425, -0.335]
  rot: [0.7071, 0.0, 0.0, 0.7071]
notes:
  - gaussian is z-up; no X -90 wrapper rotation
  - mesh/gaussian physical alignment not guaranteed unless separately validated
```

最低限度必须记录：

- `coordinate_system`
- `sh_degree`
- `floor_z`
- 推荐机器人初始 `pos/rot`

## 3. PLY 检查与 USDZ 生成规则

### 3.1 自动检测 SH degree

不要硬编码 NuRec `radiance_sph_degree=3`。

处理 PLY 时必须根据 PLY 属性数量确认 SH degree。本次 ipark 场景导出的 PLY 是 **SH degree 2**，如果 USDZ metadata 写成 degree 3，会导致 NuRec 渲染失败或显示异常。

检查/修复规则：

```text
PLY SH degree == NuRec radiance_sph_degree
```

本次修复：

```text
radiance_sph_degree: 3 -> 2
```

### 3.2 固定 color/tone 反转标志

本次 USDZ 中还需要修正：

```text
invertColorCorrection = 0
invertToneMap = 0
```

否则可能出现颜色、亮度或渲染异常。

建议 pipeline 中固定检查这两个字段；除非有明确版本差异，不要随意改成 1。

### 3.3 wrapper USDA 只做引用和必要配置

对 Z-up Gaussian，wrapper USDA 默认不做坐标旋转：

```text
rotation = identity
```

只有在源资产明确不是 Z-up 时，才允许添加坐标变换；并且必须在 metadata 里记录原因。

## 4. Isaac Lab 场景配置规则

### 4.1 通过 YAML 嵌套路径覆盖已有 cfg 字段

正确示例：

```yaml
overrides:
  scene:
    room:
      spawn:
        usd_path: ${PROJECT_ROOT}/assets/objects/real_scene/ipark_t2_505_20260721_gauss_ccm1.usda
    robot:
      init_state:
        pos: [-3.239, -4.425, -0.335]
        rot: [0.7071, 0.0, 0.0, 0.7071]
```

不要给 `InteractiveSceneCfg` 动态添加无关顶层字段，例如：

```yaml
scene:
  room_usd_path: ...  # 错误
```

原因：`InteractiveScene._add_entities_from_cfg()` 会把未知 `cfg.__dict__` 字段当作 asset 处理，最终报：

```text
ValueError: Unknown asset config type for room_usd_path
```

### 4.2 禁用不需要的默认 asset

如果某个 task 默认会加载额外 asset，但当前 real2sim 变体不需要，可以在 YAML 中设为 `null`：

```yaml
scene:
  ground: null
  goal_net: null
  goal_backdrop_1: null
```

Isaac Lab 的 `InteractiveScene` 会跳过 `asset_cfg is None` 的字段。

## 5. 光照规则

Gaussian 场景“看起来亮”不代表它能照亮机器人和普通 USD mesh。Gaussian / NuRec 主要作为视觉背景，不能依赖它给 robot/object 提供有效物理光照。

如果场景中同时存在机器人、足球、箱子等普通 USD mesh，必须加近场 key light，例如：

```yaml
scene:
  robot_key_light:
    init_state:
      pos: [robot_x, robot_y, robot_z + 1.7]
    spawn:
      intensity: 8000.0   # 可按曝光调整到 8000-12000
      radius: 1.0
```

本次 football ball-only 变体中，如果只保留 DomeLight，机器人和足球会是黑色剪影；添加近场 `SphereLight` 后，机器人和足球正常可见。

### 5.1 ipark football 最终采用的自动 keylight 方案

当前 `ipark_t2_505_20260721` + football sonic 任务采用以下配置作为默认任务光照：

- YAML 任务配置：`tasks/common_env_config/real_scene_ipark_football_sonic.yaml`
- Scene cfg：`tasks/g1_tasks/move_football_single_g1_29dof_dex3_wholebody/move_football_single_g1_29dof_dex3_hw_env_cfg.py`
- DomeLight：`intensity=8000.0`
- 单独局部 `robot_key_light`：在 YAML 中设为 `null`
- 自动位置灯：12 个 `SphereLightCfg`，位于自动灯位规划器选出的 overhead positions

每个自动位置灯使用：

```python
sim_utils.SphereLightCfg(
    color=(1.0, 0.96, 0.9),
    intensity=6000.0,
    radius=1.0,
)
```

注意：这些 asset 的字段名仍沿用 `auto_tube_light_00..11`，因为位置来自 tube-light layout planner；但最终发光类型已经不是 `CylinderLightCfg`，而是 `SphereLightCfg`。

已测试过的替代方案及结论：

- DomeLight-only，即使 `intensity=8000`，机器人/手部仍偏暗。
- 单个近场 `robot_key_light`，提升明显，但机器人移动后容易出现空间不均匀/背光。
- 31 根自动 `CylinderLight` tube，亮度改善有限且渲染成本高。
- 12 根 brighter `CylinderLight` tube，亮度变化仍不明显。
- 大半径 soft fill light，整体只轻微变亮，手部仍黑。
- **12 个自动位置 `SphereLight` keylights**，覆盖更均匀、手部可见性更好；`intensity=10000` 偏亮，最终采用 `6000`。

### 5.2 自动灯位计算流程

自动灯位由 `lab-real-scene/analysis/plan_ipark_tube_lights_v2.py` 生成，核心步骤如下：

1. 读取与 Gaussian 对齐的辅助 mesh：
   - `assets/objects/real_scene/ipark_t2_505_20260721_mesh.ply`
2. 估计地面高度：
   - 文档/配置使用 `floor_z=-1.124`
   - mesh 统计中局部估计约为 `floor_z_est≈-1.1336`
3. 从 mesh 中提取 floor-like 点：
   - 使用 z 分位数/法向/局部高度过滤地面候选点。
4. 从机器人起点 `[-3.239, -4.425]` 出发提取连通地面 mask：
   - 避免把完全不连通的区域当成任务照明区域。
5. 对连通地面点做 oriented room frame 估计：
   - 本次估计房间主方向 yaw 约 `106°`。
6. 在 oriented room frame 上生成候选 overhead positions。
7. 用 eroded connected floor mask 做覆盖率筛选：
   - 剔除落在墙外、边缘、非地面区域的候选点。
8. 从 accepted positions 中选择 12 个：
   - 优先保留机器人/足球附近 seed positions。
   - 再用 farthest-point sampling 保持空间覆盖。
9. 最终在 scene cfg 中把这些位置实例化为 `SphereLightCfg`：
   - `z≈1.866`
   - `radius=1.0`
   - `intensity=6000.0`

相关产物：

```text
local:/Users/kangkang/Public/HumanoidArena-ws/lab-real-scene/analysis/plan_ipark_tube_lights_v2.py
local:/Users/kangkang/Public/HumanoidArena-ws/lab-real-scene/analysis/ipark_auto_tube_light_layout_v2.json
local:/Users/kangkang/Public/HumanoidArena-ws/lab-real-scene/analysis/ipark_auto_tube_light_layout_v2_downsample12.json
remote:/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/analysis_outputs/ipark_light_layout_v2/ipark_auto_tube_light_layout_v2.json
remote:/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/analysis_outputs/ipark_light_layout_v2/ipark_auto_tube_light_layout_v2_downsample12.json
```

### 5.3 自动 keylight 位置是否需要继续优化

当前方案已经可作为 ipark football sonic 默认配置。短期内不建议继续为这个场景大幅调整，因为用户确认 `12 x SphereLight, intensity=6000, radius=1.0` 效果可接受。

后续如果要把该流程泛化到新真实场景，建议做以下优化：

1. **把任务活动区域作为硬约束**
   - 当前 planner 主要从连通 floor mask 覆盖整个房间。
   - 对足球/开门/导航等任务，应额外输入 robot start、object positions、expected trajectory radius。
   - 优先在活动区域周围放灯，远离任务区域的灯可以减少或降权。

2. **从“tube layout”抽象为“light position planner”**
   - 当前脚本名和字段名仍是 tube lights，但最终发现 `SphereLight` 更适合 robot mesh 可见性。
   - 建议保留 floor-mask/oriented-room 位置生成逻辑，把 emitter type 作为参数：`sphere`, `cylinder`, `rect`, `dome_fill`。

3. **加入局部高度/遮挡检查**
   - 本次 12 个点中曾发现一个候选位置局部 mesh 统计为 `above_local`。
   - 后续应对每个候选 xy 检查 floor-ceiling vertical interval，保证 `floor_z + clearance < light_z < ceiling_z - margin`。

4. **按机器人相机视角加权**
   - 目标是前置相机中 robot hands/object 可见，因此灯位不只要覆盖 floor，还要覆盖相机视野内的手部/物体区域。
   - 可以优先选择 robot/object 前方 0.5-3m 的候选点。

5. **记录每次自动生成参数**
   - 包括 floor estimate、room yaw、grid spacing、erosion radius、accepted/rejected count、downsample index、emitter type、intensity/radius。
   - 这样后续能复现实验而不是依赖手动修改 scene cfg。

## 6. 标准 Real2Sim 流程

建议后续新真实场景按以下步骤执行：

1. **留形平台导出**
   - 必须选择 Z-up。
   - 导出 Gaussian PLY。
   - 记录导出 metadata。

2. **PLY 检查**
   - 检查 bbox。
   - 检查 SH degree。
   - 检查颜色相关属性是否完整。

3. **生成 USDZ / NuRec asset**
   - 使用项目转换脚本生成 USDZ。
   - 校验 NuRec metadata。

4. **修正 USDZ metadata**
   - `radiance_sph_degree` 必须匹配 PLY。
   - `invertColorCorrection=0`。
   - `invertToneMap=0`。

5. **生成 wrapper USDA**
   - Z-up 资产默认 identity rotation。
   - 不加 X -90°。

6. **接入 Isaac Lab scene cfg**
   - 在 scene cfg 中定义 `room = AssetBaseCfg(...)`。
   - 用 YAML 覆盖 `scene.room.spawn.usd_path`。
   - 用 YAML 覆盖机器人初始位姿。

7. **确定 floor_z 和机器人初始高度**
   - 记录 `floor_z`。
   - G1 pelvis 初始高度通常约 `floor_z + 0.79`。
   - 本次 ipark：`floor_z=-1.124`，机器人 pelvis z 约 `-0.335`。

8. **添加光源**
   - 至少添加 robot/object 附近 key light。
   - 用首帧截图确认机器人和物体不是黑色剪影。

9. **Smoke test**
   - 先跑 10-100 steps 快速检查。
   - 再跑 500 steps 检查稳定性。
   - 保存视频和首帧截图。

10. **记录结果**
    - 保存 YAML。
    - 保存 result JSON。
    - 保存 smoke video。
    - 保存首帧截图。

## 7. 本次 ipark 场景的关键配置

```yaml
scene_name: ipark_t2_505_20260721
coordinate_system: z_up
sh_degree: 2
floor_z: -1.124
robot_init:
  pos: [-3.239, -4.425, -0.335]
  rot: [0.7071, 0.0, 0.0, 0.7071]
assets:
  usdz: assets/objects/real_scene/ipark_t2_505_20260721.usdz
  wrapper: assets/objects/real_scene/ipark_t2_505_20260721_gauss_ccm1.usda
```

已验证：

- Gaussian 是 Z-up，不需要 X -90° wrapper rotation。
- 修正 SH degree 到 2 后 NuRec 可渲染。
- 修正 `invertColorCorrection=0` 和 `invertToneMap=0` 后颜色/亮度正常。
- 加 key light 后普通 USD mesh 机器人和足球可见。

### 7.1 这些字段是如何获得的

当前 ipark 场景中的 `coordinate_system`、`sh_degree`、`floor_z` 来源不同，自动化程度也不同：

#### coordinate_system: z_up

来源：**导出设置 + 渲染验证**。

当前值不是从 PLY 文件里可靠自动推断出来的，而是根据留形平台导出时选择的坐标系确定，并通过 Isaac Sim 中的场景朝向/重力方向/地面水平关系做验证。

规则：

- 留形平台导出时必须选择 `Z-up`。
- 如果导出时选错为 `Y-up`，PLY 本身不一定携带足够可靠的 metadata 让 pipeline 自动纠正。
- 当前 pipeline 默认新导出的留形 PLY 是 Z-up，不做 X -90° wrapper rotation。

推荐记录方式：导出时在 sidecar metadata 中手动/脚本化写入：

```yaml
coordinate_system: z_up
coordinate_system_source: liuxing_export_option
```

#### sh_degree: 2

来源：**PLY 属性自动/脚本化检查**。

SH degree 可以通过解析 PLY header 中的 Gaussian SH/radiance 属性数量推断。本次 ipark 场景检查发现 PLY 是 SH degree 2，但 USDZ/NuRec metadata 一开始写成了 3，因此修正为：

```text
radiance_sph_degree: 2
```

推荐 pipeline：

1. 读取 PLY header。
2. 统计 SH 相关属性数量。
3. 推断 degree。
4. 校验/写入 NuRec `radiance_sph_degree`。

推荐记录方式：

```yaml
sh_degree: 2
sh_degree_source: ply_header_attributes
```

#### floor_z: -1.124

来源：**人工/半自动场景检查**。

当前 `floor_z=-1.124` 是通过 real-scene 点云/渲染场景中的地面位置检查后选定的，并用机器人初始高度 smoke test 验证。它不是从 PLY metadata 中直接自动获得的。

本次使用：

```text
floor_z = -1.124
G1 pelvis init z ≈ floor_z + 0.79 = -0.335
```

推荐 pipeline：

1. 从 PLY 点云或辅助 mesh 中裁剪机器人起始区域附近的地面点。
2. 对地面点做平面拟合或 z 分位数统计。
3. 人工检查渲染和机器人接触高度。
4. 用 10-100 step smoke test 验证机器人初始高度。

推荐记录方式：

```yaml
floor_z: -1.124
floor_z_source: local_ground_point_inspection_and_smoke_test
robot_init:
  pos: [-3.239, -4.425, -0.335]
```

注意：如果 Gaussian 只作为视觉背景，`floor_z` 主要用于放置机器人/物体；如果需要真实物理接触，必须额外验证 collision mesh 或 physics plane 与该 `floor_z` 一致。

## 8. 常见问题排查

### Gaussian 不显示或渲染异常

优先检查：

1. PLY SH degree 与 `radiance_sph_degree` 是否一致。
2. `invertColorCorrection` / `invertToneMap` 是否为 0。
3. USDZ 路径和 wrapper USDA 引用是否正确。
4. 是否误把 Z-up 资产加了额外 X -90° 旋转。

### 机器人或物体是黑色剪影

优先检查：

1. 是否有 `robot_key_light` 或等效近场光。
2. DomeLight intensity 是否太低。
3. light 是否被 YAML 禁用。
4. 相机曝光/tonemap 是否把普通 mesh 压暗。

### 物体掉落或不稳定

如果禁用了人工 `ground`，Gaussian 视觉背景通常不提供可靠物理碰撞，动态物体可能下落或无法真实接触地面。需要物理交互时必须添加 collision mesh 或临时 physics plane，并明确记录这不是 Gaussian 本身的碰撞。
