# Real Scene Region-Guided Optimization Plan

日期：2026-07-18

## 目标

仅依赖第一人称站立视频无法直观看出整场景 mesh/collision 修缮是否正确。后续应提供俯瞰图，让用户人工划定区域，再由工具根据区域约束调参、生成候选修缮资产和报告。

## 区域类型

`ground_walkable`：用户确认这里应该是地面可通行区域。

用途：如果当前 mask 没覆盖，工具需要诊断原因：无 mesh data、residual 阈值过严、morph close 不足、clearance erosion 过严，或 face/grid 连通性断裂。

`operation_zone`：任务关键区域，例如站立、转身、操作物体、经过桌边/门口。

用途：不一定整块都可通行，但需要更严格验证足底接触、地面高度、障碍边界距离和碰撞稳定性。

## 标注格式

初始模板：

```text
artifacts/real_scene_region_annotation_template.json
```

坐标使用场景 XY 米制坐标，不使用图片像素坐标。

示例字段：

```json
{
  "id": "walkable_corridor_left",
  "type": "ground_walkable",
  "priority": "must_include",
  "polygon_xy": [[0, 0], [1, 0], [1, 1], [0, 1]],
  "constraints": {
    "max_plane_residual_m": 0.18,
    "allow_hole_fill": true,
    "min_clearance_m": 0.25
  }
}
```

## 工具规划

建议新增：

```text
tools/real_scene_region_guided_tune.py
```

输入：

```text
floor_masks.npz
collision_repair_visual_report.json
region_annotations.json
```

输出：

```text
region_coverage_report.json
region_guided_candidates.md
candidate_conservative/
candidate_balanced/
candidate_aggressive/
```

## 参数优化逻辑

对每个 `ground_walkable` 多边形，统计：

```text
valid_floor coverage
connected coverage
reachable coverage
residual p50/p90/p95/max
clearance p10/p50/min
no-data / hole area
distance to current reachable boundary
```

诊断规则：

```text
valid_floor 低 + no-data 高
  -> 重建缺洞或稀疏，优先尝试 morph close / hole fill，而不是简单放宽 plane_dist。

valid_floor 高但 connected 低
  -> 连通断裂，尝试 grid-assisted bridging 或 face-graph small-gap stitching。

connected 高但 reachable 低
  -> clearance erosion 过严或障碍边界太近，尝试降低 clearance 或做更精细障碍分类。

residual 高
  -> 该区域可能真实高度偏离当前 fitted plane；需考虑局部平面/低阶曲面，而不是全局单平面。
```

## 候选资产

建议每次输出三档：

`conservative`：优先避免误包含，低 residual，较大 clearance。

`balanced`：覆盖用户标注可通行区，允许小洞填补和适度 residual。

`aggressive`：最大化覆盖，用于检查走廊完整结构和缺洞修补上限，不直接作为默认物理资产。

## 当前场景上的直接应用

用户指出 `(x≈4, y≈1)` 应该仍属于地面。当前 `plane15_close35` 中：

```text
(4,1) cell: valid_floor=false, connected=false, reachable=false
附近 residual 接近 150mm 阈值上限，mesh data 稀疏
(3.5,1) 有地面但 clearance≈0.10m，被 0.30m clearance erosion 剔除
```

这说明该区域没有进入可通行区不是单一原因，而是：

```text
mesh 缺洞/稀疏 + residual 接近阈值 + 障碍边界 clearance 收缩
```

放宽到 `plane20_close45_clear25` 后 `(4,1)` 可进入 reachable，但 residual≈176mm，风险偏高。因此更合理的方向是区域约束驱动的局部修补/低置信标注，而不是全局继续加大阈值。
