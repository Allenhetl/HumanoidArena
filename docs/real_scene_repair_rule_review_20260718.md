# Real Scene Repair Rule Review

日期：2026-07-18

## 结论

当前地面补全和可通行区域生成仍是 2.5D 地面层规则，没有考虑机器人高度方向的 body/head clearance。因此它会把桌面下方、桌架下方这类“地面平面上有空隙，但机器人身体不可通过”的架空区域误判为 reachable。

用户指出的 `(x≈2.5, y≈5)` 就是典型例子：

```text
plane15:
(2.5, 5.0)
  valid_floor = false
  connect_floor = true
  connected = true
  reachable = true
  residual = NaN
  clearance = 0.667m
```

也就是说该 cell 不是原始检测到的地面，而是被 `morphological closing` 当作地面空洞补进来的。进一步检查原 mesh，发现该区域地面上方存在大量 0.25m-1.3m 范围内的 mesh faces，其中水平面中位高度约 0.86m，符合桌面/桌下结构。

因此当前规则存在两个主要问题：

1. `morph_close` 只看 floor mask 的 2D 形态，会把桌子投影造成的大空洞误补成可行走地面。
2. `reachable` 只用地面平面上的 2D distance transform 做 G1 footprint clearance，没有检查机器人高度范围内的上方净空和桌面/桌架障碍。

## 当前规则 Review

当前 `tools/real_scene_floor_extract.py` 的核心逻辑：

```text
1. 从起点附近最低水平层拟合 floor plane。
2. face 满足 horizontal + abs(distance_to_plane) <= plane_dist，标记为 valid_floor。
3. 对 valid_floor 做 binary_closing，得到 connect_floor。
4. 从起点 flood fill 得到 connected。
5. 对 connected 做 distance_transform，clearance >= robot_clearance_radius。
6. flood fill 得到 reachable。
```

这套逻辑有以下隐含假设：

```text
floor mask 中的洞主要是重建缺洞，可以用 morphology close 补。
机器人可通行性主要由地面平面内的 2D 宽度决定。
地面上方没有低矮障碍物，或者低矮障碍会被 floor boundary 间接表达。
```

在当前办公/仓库场景中这些假设不完全成立，因为存在桌子、椅子、桌架、桌腿和桌面。桌子区域在地面层投影下可能表现为“空洞”，但这不是应该填补的地面缺洞，而是低净空障碍区域。

## 探针结果

在 `(x≈2.5, y≈5)` 附近：

```text
floor_z ~= -1.08m
0.25m-2.2m 高度范围内上方 mesh faces: 494
水平上方面 faces: 135
低障碍 0.05m-1.4m faces: 658
上方水平面 dz p50 ~= 0.86m
```

在周边点：

```text
(2.5, 4.5): 上方 faces 623，水平上方面 p50 ~= 0.879m
(2.5, 5.5): 上方 faces 319，水平上方面 p50 ~= 0.846m
(3.0, 5.0): 上方 faces 517，水平上方面 p50 ~= 0.891m
(2.0, 5.0): 上方 faces 589，水平上方面 p50 ~= 0.834m
```

这些区域不应该直接作为 G1 可通行区域，即使其下方存在地面或 morphology close 能连通。

## 新增先验规则

### 规则 1：机器人高度净空约束

对每个 grid cell，计算地面 fitted plane 上方机器人高度范围内是否存在障碍：

```text
floor_z = fitted_plane(x, y)
obstacle_above = any mesh face centroid within XY radius r_body
                 AND z in [floor_z + z_min, floor_z + robot_height_clearance]
```

初始参数建议：

```text
r_body = 0.30-0.40m
z_min = 0.15-0.25m
robot_height_clearance = 1.2-1.6m
```

如果该范围内存在大量 faces，尤其是水平 faces，则该 cell 应标为 `overhead_blocked`，不能进入 reachable。

### 规则 2：低净空水平面阻断

桌面是强水平面，不能仅靠地面层 residual 过滤。

```text
low_ceiling_or_table = horizontal face
                       AND z - floor_z in [0.4m, 1.4m]
                       AND XY overlap with floor candidate
```

这类区域应该作为 `occupied_projection` 投影到地面平面，参与 reachable erosion。

### 规则 3：morphological closing 不能跨越障碍投影

当前 `binary_closing(valid_floor)` 会把桌子投影造成的空洞补成地面。应改为 constrained closing：

```text
fillable_hole = no floor data
                AND surrounded by valid floor
                AND not overhead_blocked
                AND not occupied_projection
                AND hole area <= threshold
```

也就是说，只有确认不是桌椅/障碍投影后，才能把洞补进地面。

### 规则 4：区域标注优先级

如果用户在俯瞰图上标了：

```text
ground_walkable
  可尝试放宽 plane_dist / close / bridging，但仍必须通过 overhead clearance。

operation_zone
  不一定全纳入 reachable，但必须输出 overhead clearance、foot clearance、collision risk report。
```

用户标注不能直接覆盖机器人高度净空规则；如果标注区域处于桌面下，应报告冲突，而不是强行设为可通行。

## 推荐算法更新

下一版 `real_scene_floor_extract.py` 建议新增：

```text
overhead_blocked_grid
low_table_projection_grid
fillable_hole_grid
blocked_floor_grid
reachable_before_overhead
reachable_after_overhead
```

新的 reachable 流程：

```text
valid_floor = horizontal && near fitted plane
overhead_blocked = low/mid-height obstacle projection
fillable_holes = hole candidates && !overhead_blocked && small_area
connect_floor = valid_floor + fillable_holes
connected = flood(connect_floor, start)
free_floor = connected && !overhead_blocked
clearance = distance_transform(free_floor)
reachable = flood(free_floor && clearance >= robot_clearance_radius, start)
```

这样 `(2.5,5)` 这类桌下区域会被 `overhead_blocked` 阻断，不会因为 close 被纳入 reachable。

## 对扩阈值实验的 Review

`plane18_close40_clear25` 和 `plane20_close45_clear25` 能扩大走廊覆盖，但也会放大桌下误纳入问题。

```text
plane15 reachable area: 29.28m²
plane18_close40_clear25 reachable area: 42.54m²
plane20_close45_clear25 reachable area: 48.66m²
```

问题是：

```text
(2.5,5) 在三组里都不是 valid_floor，却都被 close 纳入 reachable。
plane20 还会把 (4,1) 这类高 residual / 稀疏区域纳入 reachable。
```

因此下一步不建议继续单纯全局放宽阈值。更合理的路线是：

1. 先加入 overhead/low-table projection 阻断。
2. 再结合用户 `ground_walkable` 标注扩充真实走廊。
3. 对扩充区分 high/low confidence。
4. 最后再生成候选 repair floor 和 collision split。
