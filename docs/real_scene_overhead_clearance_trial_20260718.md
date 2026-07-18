# Real Scene Overhead Clearance Trial

日期：2026-07-18

## 目的

根据反馈，当前可通行区域计算只基于地面平面 2.5D mask，没有考虑 G1 身体高度方向的净空，导致桌面下方的架空区域被误纳入 reachable。本次实现并测试 overhead/table projection 规则。

## 实现内容

修改：

```text
isaaclab_twist2_g1/tools/real_scene_floor_extract.py
```

新增输出：

```text
overhead_blocked_projection.png
low_table_projection.png
hole_fill_allowed.png
hole_fill_blocked_or_rejected.png
reachable_before_after_overhead.png
```

新增 `floor_masks.npz` 字段：

```text
raw_closed_floor
allowed_fill
rejected_fill
blocked_fill
overhead_blocked
overhead_seed
overhead_count
low_table_projection
table_seed
free_floor
reachable_before_overhead
clearance_before_overhead
```

## 关键算法

保留原始 morphology closing 作为地面稀疏重建的连通桥接：

```text
connect_floor = raw_closed_floor
```

然后在最终 reachable 前扣除桌面/低净空投影：

```text
free_floor = connected & ~overhead_blocked
reachable = flood(free_floor & clearance >= robot_clearance_radius, start)
```

当前默认 `overhead_mode=table_only`，即只把低矮水平面投影作为阻断层。`all` overhead 规则曾尝试过，但会把墙体、桌腿和噪声大量投影，过于激进。

## 试验结果

### 过激版本

`overhead_mode=all` 或 `body_clearance_radius=0.35` 的 table-only 会把 reachable 压到起点附近，说明自动投影过宽，不适合作为最终资产。

### 当前可视化版本

参数：

```text
plane_dist=0.15
morph_close_radius=0.35
robot_clearance_radius=0.30
overhead_mode=table_only
body_clearance_radius=0.10
table_min_z=0.60
table_max_z=1.10
```

结果：

```text
reachable_before_overhead = 29.28 m²
reachable_after_overhead  = 2.58 m²
```

这说明桌面投影成功阻断了桌下区域，但仍过保守，会切断一些应保留走廊。因此当前输出应作为“风险层/人工审核层”，暂不直接用于生成最终 floor_repair。

## 关键点验证

`(x≈2.5, y≈5)`：

```text
valid_floor = false
connected = true
reachable_before_overhead = true
overhead_blocked = true
low_table_projection = true
reachable_after_overhead = false
```

这符合预期：该区域原本是 morphology close 补出的桌下洞，现在被桌面投影阻断。

`(x≈4, y≈1)`：

```text
valid_floor = false
connected = false
reachable_before_overhead = false
overhead_blocked = false
reachable_after_overhead = false
```

这说明该区域未纳入可通行，主要仍是地面缺洞/稀疏和 residual 阈值问题，而不是桌面净空问题。

## 本地产物

```text
artifacts/real_scene_floor_repair_20260716/start_1p5_4p0_ccm1_plane15_close35_overhead_table_narrow_v1/
artifacts/real_scene_collision_repair_report_20260718/plane15_overhead_table_narrow_v1/
```

重点查看：

```text
reachable_before_after_overhead.png
overhead_blocked_projection.png
low_table_projection.png
```

## 下一步建议

不要直接把 overhead-aware reachable 作为 final repair floor。下一步应改成：

1. 把 `low_table_projection` 作为风险层展示给用户。
2. 用户在俯瞰图上标注桌子/障碍投影和真实走廊。
3. 根据标注对 table projection 做局部增删。
4. 使用 `reachable_before_overhead - confirmed_table_projection` 生成 balanced candidate。
5. 再结合 `(x≈4,y≈1)` 这类 ground_walkable 标注，局部补地面缺洞，而不是全局放宽阈值。
