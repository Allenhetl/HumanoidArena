# Inspire 灵巧手 Retarget 调研对比 (2026-08-06)

> 对比三个来源的 inspire 灵巧手 retarget 实现，供后续接入参考。
> 相关仓库（均在远端 reference/，未 commit）：
> - `reference/xr_teleoperate/`（unitreerobotics 官方，sparse 拉取 teleop + inspire 资产 + dex-retargeting submodule）
> - `reference/AnyDexRetarget/`（qqsq12321，MIT，13 种手）
> - 我们现有链路：`action_provider/action_provider_mimic_lite.py` 的 `hand_keypoints_to_a_hw`（26x7→a_hw_6）

---

## 1. 关节模型差异（最关键）

| 来源 | 关节数 | 驱动模型 | 说明 |
|---|---|---|---|
| **xr_teleoperate** | URDF **12 关节**，但 retarget 只解 **6 个** | DFX/FTP 手 **6 电机** | `target_joint_names` = 4×proximal + thumb_proximal_pitch/yaw；intermediate/distal 靠**机械联动**跟随 proximal，**不独立解算** |
| **AnyDexRetarget** | **12 关节独立** | inspire_hand_right.urdf | thumb 4（yaw/pitch/intermediate/distal）+ index/middle/ring/pinky 各 2（proximal/intermediate）全独立 |
| **我们 HumanoidArena** | **12 关节独立** | inspire USD（同 AnyDexRetarget 模型） | `a_hw_6` = 4×finger_flex + thumb_flex + thumb_rotation；`_INSPIRE_FINGER_IDX` 用 26x7 的 MC/Prox/Inter 三点角，thumb 用 MC/Prox/Tip |

**结论**：xr_teleoperate 与我们关节模型**不同**（6 驱动 vs 12 独立）；AnyDexRetarget 与我们**同模型**（12 独立关节），直接可参考其参数。

## 2. Retarget 算法对比

| 来源 | 算法 | 特点 |
|---|---|---|
| **xr_teleoperate** | **dex-retargeting 库**（silencht/dex-retargeting，OpenAI dex-retargeting 衍生） | DexPilot（指尖向量 + 拇指-其余手指投影距离，huber 损失，NLopt SLSQP）、Vector（关键向量）两种；`scaling=1.20`、`low_pass_alpha=0.2` |
| **AnyDexRetarget** | **AdaptiveOptimizerAnalytical**（自研解析梯度 + NLopt SLSQP） | TipDirVec（指尖位置+方向）+ FullHandVec（全手向量）+ **pinch 感知**（捏合时强化拇指-指尖接触）；`segment_scaling` 每指每段缩放、`pinch_thresholds`、`lp_alpha=0.4` |
| **我们** | `hand_keypoints_to_a_hw` | 几何三点角（无优化器、无 pinch 感知、无手尺寸适配） |

### AnyDexRetarget 针对 inspire 的优化（pico4_inspire_hand.yaml）
```yaml
retarget:
  huber_delta: 2.0
  w_pos: 1.0
  w_dir: 5.0
  scaling: 1.0
  segment_scaling:            # 每指 3 段（PIP/DIP/TIP）长度缩放，适配 inspire 手指比例
    thumb:  [1.106, 1.301, 1.35]
    index:  [1.175, 1.20, 1.327]
    middle: [1.130, 1.18, 1.31]
    ring:   [1.15, 1.22, 1.356]
    pinky:  [1.276, 1.32, 1.459]
  pinch_thresholds:           # 捏合距离阈值（mm），小于 d1 开始 pinch 强化
    index:  {d1: 2.0, d2: 8.0}
    middle: {d1: 2.0, d2: 4.0}
    ring:   {d1: 2.0, d2: 4.0}
    pinky:  {d1: 2.0, d2: 4.0}
  lp_alpha: 0.4
```

### 校准措施对比
| 来源 | 校准手段 |
|---|---|
| xr_teleoperate | `scaling_factor`（手尺寸）、`low_pass_alpha`（平滑）；无专门标定工具 |
| AnyDexRetarget | `mediapipe_rotation`（坐标旋转）、`wrist_offset_cm`/`thumb_offset_cm`（腕/拇指组偏移标定）、`align_four_mcp_to_robot`（MCP 对齐机器人）；**有 `example/test/calibrate_offset.py`**（对比 MCP landmark 与机器人中性位手指根位置，生成 offset） |
| 我们 | 无（仅固定三点角 + 限位） |

### Pico4 输入（AnyDexRetarget 现成）
- `example/input/pico4.py`：relay（127.0.0.1:63902）或 direct（PC 63901 端口，Pico UDP 广播发现）；**26 SDK 关节→21 MediaPipe 关节映射**（`JOINT_21_INDICES` 去掉 Palm/4×metacarpal）
- `pico4_daemon.py`：独立 discovery/relay 守护进程（不必每次重连 VR）
- 与我们链路共用 26 关节原始数据，但 AnyDexRetarget 先映射成 MediaPipe 21 再算

## 3. 对我们链路的启示

1. **同模型**：AnyDexRetarget 的 inspire_hand 12 关节模型与我们 USD 完全一致，其 `segment_scaling`/`pinch_thresholds` 参数可直接迁移到我们的 `hand_keypoints_to_a_hw` 或替换为优化器。
2. **pinch 感知**：我们目前没有——对"捏瓶盖"这类精细操作，AnyDexRetarget 的 pinch 强化（捏合时拇指-指尖更稳）是明显优势。
3. **校准**：AnyDexRetarget 的 `calibrate_offset.py` 思路可借鉴，为 Pico 4 手做腕/拇指偏移标定。
4. **xr_teleoperate 的 6 电机 DFX 模型与我们 12 关节模型不同**，其 DexPilot 算法可直接参考（投影距离抓取稳定），但不能直接套关节映射。

## 4. 场景缩小 0.75x（已完成并验证）

- `real_scene_ipark_drink_inspire_mimic_lite.yaml` + `move_real_scene_drink_inspire_hw_env_cfg.py`：桌/瓶/盖 `scale=0.75`，位置重算（ipark：桌原点 z=-0.749、桌面顶 -0.37625、瓶身 -0.37625、瓶盖 -0.177575）
- `tools/test_drink101_twist_break.py` plain 模式同步 0.75x
- headless 验证 **PASS**：`body_z=0.236≈桌面顶0.24775`（瓶落桌面）、拧 2π、取盖、瓶身可拿起

## 5. 自碰撞现状（左手-右手-本体穿透）

- inspire USD **52 个有效碰撞 prim**（CollisionAPI + 原型含 Mesh），碰撞几何存在且 PhysX 会解析（body 碰撞正常，瓶子能落桌面）
- **左手-右手-本体穿透根因 = `enabled_self_collisions=False`**（`G129_CFG_WITH_INSPIRE_WHOLEBODY`、`G129_CFG_MIMIC_LITE_INSPIRE` 均如此；dex3 也 False，是全仓库统一约定，防止手指/腿自锁）
- 开启方法：改 drink 任务用 cfg 的 `ArticulationRootPropertiesCfg(enabled_self_collisions=True)`——但需注意会增加求解负担，且手部 convexHull 碰撞可能让手指轻微自碰。**建议先确认用户是否真要开**（单手在当前 retarget 下无自穿，开了主要是为了左右手互碰 + 手碰本体）
