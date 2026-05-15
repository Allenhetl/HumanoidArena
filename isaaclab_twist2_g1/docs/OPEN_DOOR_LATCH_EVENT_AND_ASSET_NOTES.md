# OpenDoor Latch Event and Door Asset Notes

本文记录 `Isaac-Move-Open-Door-G129-Dex3-Wholebody` 中门把手解锁事件的实现约定，以及 OpenDoor replay 和集群推理场景使用不同 door USD 资产的排查规则。

## 背景

OpenDoor 任务的 door 仍然使用 `AssetBaseCfg` 加载，不改成 `ArticulationCfg`。原因是集群上 Isaac Sim 4.5 曾经出现过 door 资产内部 articulation 部分位置错位的问题；当前方案只在 `AssetBaseCfg` 的基础上，通过 IsaacLab `EventTermCfg` 和 `dynamic_control` 访问门页/门把手 joint。

锁门事件的目标是：

1. episode 初始和 reset 后先锁住门页 joint。
2. 门把手 joint 下压到阈值后触发解锁。
3. 解锁后恢复门页 joint 的默认 drive / damping / stiffness 设定。

## 事件启用方式

该事件默认不启用。必须在启动脚本前显式设置：

```bash
OPEN_DOOR_LATCH_ENABLE=1
```

常用调试命令：

```bash
PYTHONUNBUFFERED=1 \
OPEN_DOOR_LATCH_ENABLE=1 \
OPEN_DOOR_LATCH_DEBUG=1 \
OPEN_DOOR_LATCH_POLL_LOG_INTERVAL=5 \
OPEN_DOOR_HANDLE_UNLOCK_ANGLE_DEG=-20 \
bash run_sonic_joint29.sh
```

replay 路径同理：

```bash
PYTHONUNBUFFERED=1 \
OPEN_DOOR_LATCH_ENABLE=1 \
OPEN_DOOR_LATCH_DEBUG=1 \
OPEN_DOOR_LATCH_POLL_LOG_INTERVAL=5 \
OPEN_DOOR_HANDLE_UNLOCK_ANGLE_DEG=-20 \
bash run_replay_sonic.sh
```

如果需要强制关闭：

```bash
OPEN_DOOR_LATCH_DISABLE=1 bash run_sonic_joint29.sh
OPEN_DOOR_LATCH_DISABLE=1 bash run_replay_sonic.sh
```

注意：旧录制数据 replay 如果目标是复现原始轨迹，通常不要启用 latch，除非该录制本身就是在同一 latch 逻辑下采集的。latch 会改变门页物理响应，可能破坏 deterministic replay。

## 预期日志

事件正常注册时，日志中应能看到：

```text
open_door_latch_startup
open_door_latch_reset
open_door_latch_poll
```

锁门成功时应出现：

```text
[open_door_latch] phase=lock ...
```

`dynamic_control` 正确识别 door articulation 时应出现类似：

```text
[open_door_latch] phase=dc_catalog env=0 art_path=/World/envs/env_0/Door leaf_index=0 handle_index=1 dofs=['0:RevoluteJoint_door001:/World/envs/env_0/Door/E_leaf_2/RevoluteJoint_door001', '1:RevoluteJoint:/World/envs/env_0/Door/E_handle_4/RevoluteJoint']
```

门把手下压但未达到阈值时：

```text
[open_door_latch] phase=poll env=0 state=locked reason=manual_post_physics angle_source=dynamic_control handle_angle_deg=-14.285 threshold_deg=-20.000
```

达到阈值并解锁时：

```text
[open_door_latch] phase=unlock env=0 reason=manual_post_physics angle_source=dynamic_control handle_angle_deg=-20.878 threshold_deg=-20.000 ...
```

解锁后应继续看到：

```text
[open_door_latch] phase=poll env=0 state=unlocked ...
```

如果 `handle_angle_deg` 一直是 `0.000`，说明没有读到真实门把手 DOF 运动，优先检查是否出现 `angle_source=dynamic_control` 和 `phase=dc_catalog`。

## Replay 与推理资产区别

OpenDoor 现在存在两套重要 door USD：

- 旧 replay 数据需要使用：
  `assets/objects/small_warehouse/small_warehouse_opendoor/interaction_obj/door001/model_door001_vali.usd`
- 集群推理场景以及本次 latch 事件开发/测试使用：
  `assets/objects/small_warehouse/small_warehouse_opendoor/interaction_obj/door001/model_door001_vali_gate_welded.usd`

资产入口在：

```text
tasks/common_scene/base_scene_open_door.py
```

重点检查 `OpenDoorSceneCfg.door.spawn.usd_path` 当前指向哪一个 USD。

## 为什么资产不能混用

OpenDoor replay 对 USD 资产非常敏感。旧 replay 数据是按当时的 door 资产、joint 结构、collision、drive 参数和 PhysX 行为采集的。切到 `model_door001_vali_gate_welded.usd` 后，即使 task 名和 replay 文件相同，也可能出现：

- robot / door 接触时序漂移；
- replay 中门页、门把手状态无法和录制时一致；
- `Replay env err` 或后续接触结果异常；
- latch 调试日志中的 DOF catalog、joint path 或角度方向与预期不一致。

反过来，集群推理和 latch 开发测试使用的是 `model_door001_vali_gate_welded.usd`。如果误切回旧 `model_door001_vali.usd`，可能导致推理场景和已验证的锁门事件不一致。

## 排查顺序

如果 OpenDoor replay 或推理出现异常，先按下面顺序查：

1. 确认当前用途：
   - 旧数据 replay：使用 `model_door001_vali.usd`。
   - 集群推理 / latch 测试：使用 `model_door001_vali_gate_welded.usd`。
2. 检查 `tasks/common_scene/base_scene_open_door.py` 中 `usd_path` 是否和用途一致。
3. 如果启用了 latch，检查日志是否有 `open_door_latch_startup/reset/poll`。
4. 检查 `phase=dc_catalog` 是否识别出：
   - 门页 joint：`E_leaf_2/RevoluteJoint_door001`
   - 门把手 joint：`E_handle_4/RevoluteJoint`
5. 检查 `handle_angle_deg` 是否随门把手下压变化。
6. 如果已经 `phase=unlock` 但门仍然很难推，检查 `phase=dc_restore` 中恢复后的门页 DOF 参数，重点看 runtime `stiffness`、`damping`、`maxEffort/max_force` 是否仍然过大。

## 相关代码

- `tasks/g1_tasks/move_open_door_g1_29dof_dex3_wholebody/move_open_door_g1_29dof_dex3_hw_env_cfg.py`
  - latch event 注册；
  - startup/reset 锁门；
  - post-physics 轮询门把手角度；
  - 达阈值后恢复门页 joint。
- `tools/get_reward.py`
  - SONIC/replay 路径会绕过标准 `env.step()`；
  - `sync_task_events_after_physics_step()` 用于在手动 physics step 后补跑 task-local event hook。
- `tasks/common_scene/base_scene_open_door.py`
  - door USD 资产切换入口。

