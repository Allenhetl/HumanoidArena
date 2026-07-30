# Real Scene Collision Repair A/B Validation

日期：2026-07-18

## 结论

本次补跑了严格 A/B：同一 `ccm1` 参考位姿、同一 `SonicActionProvider` static-ref 验证脚本、同样 100 steps，仅切换场景碰撞资产。

结果：原始碰撞与 repaired collision 两侧都能在 100 步内保持站立，没有出现明显倒地。repaired collision 的几何层已经按预期替换为平整地面，但在当前 100 步 static-ref 设置下，尚不能证明控制稳定性显著优于原始碰撞；它更能证明“修复后的 wrapper 可以被正确加载并稳定运行”。

## 产物路径

本地：

```text
artifacts/real_scene_ab_validation_20260718/ab_original_ccm1ref_100.log
artifacts/real_scene_ab_validation_20260718/ab_repaired_collision_plane15_ccm1ref_100.log
artifacts/real_scene_ab_validation_20260718/ab_original_ccm1ref_100.mp4
artifacts/real_scene_ab_validation_20260718/ab_repaired_collision_plane15_ccm1ref_100.mp4
```

远端：

```text
/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/test_logs/ab_original_ccm1ref_100.log
/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/test_logs/ab_repaired_collision_plane15_ccm1ref_100.log
/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/test_videos/ab_original_ccm1ref_100.mp4
/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/test_videos/ab_repaired_collision_plane15_ccm1ref_100.mp4
```

## 可复现运行环境

远端机器实际使用：

```text
ssh 10.20.81.208
cd /home/lab/zikang/HumanoidArena
```

必须设置的环境：

```bash
export PROJECT_ROOT=/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1
export PYTHONPATH=/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1:/home/lab/fanghai/IsaacLab/source/isaaclab:/home/lab/fanghai/IsaacLab/source/isaaclab_assets:/home/lab/fanghai/IsaacLab/source/isaaclab_tasks:/home/lab/fanghai/IsaacLab/source/isaaclab_rl:/home/lab/fanghai/IsaacLab/source/isaaclab_mimic:/home/lab/kelun/kelun/SIMPLE/third_party/unitree_sdk2_python:${PYTHONPATH:-}
PYTHON=/home/lab/miniconda3/envs/unitree_sim_env/bin/python
REF_NPZ=/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/analysis_outputs/real_scene_provider_static_ref_ccm1.npz
SCRIPT=/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/tools/test_real_scene_lab_sonic_provider_joint29_static_ref.py
```

注意：只激活 `unitree_sim_env` 不够；当前环境没有把 `isaaclab` 和 `unitree_sdk2py` 安装进 site-packages，必须显式加入上述 `PYTHONPATH`。

## A/B 命令

A. 原始 ccm1 collision：

```bash
REAL_SCENE_ROOM_USD=/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/assets/objects/real_scene/small_warehouse_digital_twin_office_fixed_gauss_xneg90_ccm1.usda \
${PYTHON} ${SCRIPT} \
  --num_steps 100 \
  --load_reference_npz ${REF_NPZ} \
  --video_output /home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/test_videos/ab_original_ccm1ref_100.mp4 \
  --headless --enable_cameras \
  > /home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/test_logs/ab_original_ccm1ref_100.log 2>&1
```

B. repaired collision：

```bash
REAL_SCENE_ROOM_USD=/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/assets/objects/real_scene/small_warehouse_digital_twin_office_fixed_gauss_xneg90_ccm1_repaired_collision_plane15.usda \
${PYTHON} ${SCRIPT} \
  --num_steps 100 \
  --load_reference_npz ${REF_NPZ} \
  --video_output /home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/test_videos/ab_repaired_collision_plane15_ccm1ref_100.mp4 \
  --headless --enable_cameras \
  > /home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/test_logs/ab_repaired_collision_plane15_ccm1ref_100.log 2>&1
```

## 指标对比

参考位姿一致：

```text
joint range = [-0.1234, 0.0162]
body_pos = [1.5, 3.9996, -0.2812]
quat ~= [1.0, 0.0029, 0.0, -0.0007]
```

| Scene | step 1 root | step 100 root | sample z range | xy drift | step100 quat | 视频帧 |
| --- | --- | --- | ---: | ---: | --- | ---: |
| original ccm1 | `[1.499, 3.999, -0.286]` | `[1.668, 3.833, -0.274]` | ~13 mm | ~0.237 m | `[0.998, 0.028, 0.020, 0.057]` | 100 |
| repaired collision plane15 | `[1.499, 4.000, -0.286]` | `[1.713, 3.717, -0.286]` | ~8 mm | ~0.355 m | `[0.999, 0.029, 0.027, 0.007]` | 100 |

解释：

1. 两侧在正确 `ccm1ref` 下均能站满 100 步。
2. repaired collision 的 z 更接近起始高度，sample z 波动略小。
3. repaired collision 的 xy 漂移更大，可能是地面平整化改变了足底微碰撞/摩擦响应，也可能只是 SONIC static-ref 本身的动作输出造成，不能只归因于碰撞层。
4. 当前日志只每 20 步采样一次 root 和 quat，缺少足底 contact、足底高度、力/摩擦和完整逐步轨迹；下一版 A/B 应写出 `npz/csv`。

## Collision Split 改进计划

当前 `plane15_close35` 的 replacement floor 已覆盖更接近用户描述的“三条纵向走廊由横向走廊连通”的区域。已有几何改善很明确：reachable floor 原始高度相对 fitted plane 的误差 p50 ~62 mm、p95 ~131 mm，repair floor target residual 为 0 mm。

当前 split 删除规则：

```text
remove_original_floor_face =
  face centroid inside dilated reachable mask
  AND abs(face normal z) >= cos(25 deg)
  AND abs(face centroid z - fitted_plane_z) <= 0.16 m
```

这个规则已经能避免天花板、桌面和多数墙体被 XY 投影误删，但它仍是局部几何阈值规则，没有理解 face graph 上“与起点地面连通”的语义。

### 方向 1：基于 face graph 的连通地面删除

做法：

1. 读取原 mesh face adjacency。
2. 标记候选地面 faces：水平法向、靠近 fitted plane、centroid 落在 reachable/dilated mask 内。
3. 从起点所在 face 或起点 grid cell 关联 faces 做 flood fill。
4. 只删除与起点地面连通的候选 faces，保留同高但被障碍断开的孤立片、桌椅下方悬浮片、误分类碎片。

预期效果：减少误删，尤其是在桌腿/椅腿附近、墙边、走廊边界和多层结构重叠处更稳。代价是需要处理 USD mesh 的 face adjacency 和 mask/grid 到 face 的映射。

### 方向 2：边界 buffer 与过渡带

当前 deleted original floor 和 `floor_repair` 可能在边界处产生“双碰撞边”或“小台阶”。

做法：

1. 删除区比 floor_repair 区域略大，例如 `delete_margin=0.10-0.20m`。
2. floor_repair 区域比可达区域略小或保持 reachable mask。
3. 在边界输出 overlay 图：removed faces、repair floor outline、reachable mask 三者叠加。
4. 增加 `edge_gap_min/max` 检查，避免原地面边缘刚好贴着 repair floor 边缘。

预期效果：减少脚底在 repair/original 交界处卡边和跳变。风险是删除过大时会在视觉地面边缘产生不可碰撞洞，需要用障碍/墙体碰撞补足。

### 方向 3：置信分层的 floor repair

当前 `plane15_close35` 覆盖面积大，但 residual 阈值也宽；`plane12_close30` 更保守。

做法：

1. 把 reachable cells 分成 high confidence 和 low confidence：例如 residual `< 0.10-0.12m` 为高置信，`0.12-0.15m` 为低置信。
2. 高置信区直接生成 floor_repair。
3. 低置信区仅可视化或生成可开关的 secondary repair layer。
4. A/B 测试 `plane12_close30`、`plane15_high_conf`、`plane15_full`。

预期效果：在保留右侧走廊连通性的同时，降低把粗糙/边界/障碍附近区域强行平面化的风险。

### 方向 4：障碍碰撞简化

当前 obstacles 保留 420018 triangles，PhysX 负担偏大。

做法：

1. 按 connected components 或空间块分割 `collision_obstacles`。
2. 过滤小面积浮空碎片和远离 G1 可达区的碎片。
3. 对墙、桌、椅等大块生成简化 mesh 或 convex decomposition。
4. 保留视觉 mesh 不变，仅替换 collision proxy。

预期效果：降低仿真负担和奇异碰撞概率。风险是简化后障碍边界变粗，需要检查 G1 不穿墙/不穿桌椅。

### 方向 5：floor z offset 与物理材料扫描

做法：

1. 生成 `floor_repair` z offset：`-5mm / 0 / +5mm`。
2. 扫描摩擦参数和 restitution。
3. 用同一 `ccm1ref` 和同一路径测试 root z、tilt、feet contact jitter。

预期效果：找到足底接触最稳定的位置。`+5mm` 可能减少脚底陷入原始噪声，但可能视觉穿出；`-5mm` 视觉更保守，但可能接触偏低。

## 下一步测试矩阵

优先级建议：

1. `original ccm1` vs `repaired plane15`，输出逐步 CSV/NPZ 指标。
2. `plane12_close30` vs `plane15_close35`，验证保守/完整覆盖的控制差异。
3. `plane15` 的 `z offset -5/0/+5mm`。
4. `face graph connected split` vs 当前 centroid split。
5. `obstacle full mesh` vs `obstacle simplified proxy`。
