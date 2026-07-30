# Real Scene Multi-Seed Stitch: Prior 5 Points

日期：2026-07-18

## 输入 seed

横向走廊点：

```text
(2.5, 0.5)
```

纵向走廊点：

```text
(1.5, 4.0)
(-1.6, 4.0)
(4.4, 4.0)
(7.0, 4.0)
```

每个点独立运行当前 overhead-aware `real_scene_floor_extract.py`，参数一致：

```text
plane_dist=0.15
morph_close_radius=0.35
robot_clearance_radius=0.30
overhead_mode=table_only
body_clearance_radius=0.10
table_min_z=0.60
table_max_z=1.10
max_fill_area=0.25
```

然后用：

```text
tools/real_scene_multi_seed_stitch.py
```

进行拼接和投票统计。

## 本地产物

```text
artifacts/real_scene_multi_seed_stitch_20260718/prior5_plane15_overhead_table_narrow_v1/
```

重点查看：

```text
multi_seed_union_vote.png
per_seed_reachable_before_overhead_overlay.png
per_seed_reachable_after_overhead_overlay.png
multi_seed_overhead_blocking_effect.png
multi_seed_stitch_report.json
```

## 结果摘要

| Seed | start | reachable before overhead | reachable after overhead |
| --- | --- | ---: | ---: |
| cross_2p5_0p5 | `[2.5, 0.5]` | 46.83 m² | 9.69 m² |
| lane_1p5_4p0 | `[1.5, 4.0]` | 29.28 m² | 2.58 m² |
| lane_m1p6_4p0 | `[-1.6, 4.0]` | 36.24 m² | 0.77 m² |
| lane_4p4_4p0 | `[4.4, 4.0]` | 45.89 m² | 9.69 m² |
| lane_7p0_4p0 | `[7.0, 4.0]` | 31.33 m² | 0.25 m² |

Union：

```text
union valid floor area: 56.27 m²
union connected area: 71.88 m²
union reachable before overhead/table blocking: 48.52 m²
union reachable after overhead/table blocking: 13.31 m²
area removed by any overhead/table projection from union: 35.21 m²
```

## 结论

多 seed 计算明显比单点 seed 更符合“横向走廊 + 多条纵向走廊”的结构。尤其 `(2.5,0.5)` 与 `(4.4,4.0)` 的局部平面更平，能把 `(4,1)` 这类区域识别为高置信地面。

但当前自动 table/overhead projection 仍过保守：overhead 前 union reachable 为 48.52 m²，overhead 后只有 13.31 m²。这说明 `low_table_projection` 适合作为风险层和人工审核层，但还不应直接作为最终可通行 mask 的唯一阻断依据。

## 对 GUI / 区域标注流程的启发

后续 GUI 中用户画区域后，可以在每个区域内采样多个 seed 点，分别跑局部平面/连通/可达分析，再做：

```text
同区域内 vote/union
跨区域 stitch
overhead/table risk overlay
用户确认后扣除桌下/障碍投影
```

推荐拼接策略：

```text
1. ground_walkable 区域：采样多个 seed，使用 union/reachable_before_overhead 做候选覆盖。
2. operation_zone：采样中心和边界点，统计 residual/clearance/overhead risk。
3. table/obstacle 区域：由用户或自动 table projection 标为 negative mask。
4. 最终 balanced mask = multi_seed_union_reachable_before_overhead - confirmed_negative_projection。
5. 对 overlap 区域使用投票数和 residual 选择局部 floor plane，避免全场单平面。
```
