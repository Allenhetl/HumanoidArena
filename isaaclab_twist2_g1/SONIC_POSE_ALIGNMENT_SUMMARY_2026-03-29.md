# SONIC POSE 对齐优化总结

日期：2026-03-29

本文只记录今天已经验证为“有效”的优化项，以及它们对应解决或缓解的问题。  
“有效”指的是：修改后实际观测到稳定性、跟随性或可诊断性有明显提升，而不是单纯的猜测性调参。

## 当前总体结论

今天的排查结果表明，Isaac Lab 复现和 SONIC 原始实现之间，主要差异不只是在 `kp/kd`，更关键的是：

1. 参考窗口语义
2. decoder 历史观测语义
3. 本地额外加入的输出裁剪
4. 调试可见性不足

修完这些后，表现已经从“启动就摔、几乎不可用”提升到：

- 启动初期不再立即摔倒
- 基本动作跟随明显成立
- 调整 ankle 增益后，前倾小碎步问题可被进一步压制
- 上半身和平衡问题仍有残余，但已经进入可继续精细排查的阶段

## 已确认有效的优化

### 1. 将 encoder 参考从“最近收到的历史窗口”改为“流式时间轴 + 播放游标 + current->future gather”

文件：
- [action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)

问题：
- 本地之前直接把 ZMQ 最近 5 帧当作 encoder 输入窗口。
- SONIC deploy 实时遥操并不是这样，它会先把流式数据合并成一条参考时间轴，再按 `current_frame + i` 取 future window。

优化：
- 本地新增了最小版 streamed reference timeline
- 增加 playback cursor
- encoder 的 active blocks 改为优先从 merged timeline 上按 current->future 取 10 帧

效果：
- 整体稳定性有非常大的提升
- encoder 输入明显更接近 SONIC 原始实现
- 这是今天最关键、收益最大的修复之一

### 2. 修正 decoder 的 `body_q` 语义为 `q - default_angles`

文件：
- [action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)

问题：
- 本地一度把绝对关节角直接喂进 decoder 历史。
- SONIC deploy 记录的是 `body_q = measured_q - default_angles`。

优化：
- 本地 decoder joint position history 改成 `joint_pos - self._sonic_default_np`

效果：
- 减少了 decoder 输入分布偏差
- 对站立期和后续稳定性有帮助

### 3. 修正 decoder 的 gravity 观测语义

文件：
- [action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)

问题：
- 本地之前直接用了 Isaac Lab 的 `projected_gravity_b`
- SONIC deploy 实际使用的是 `quat_conjugate(base_quat) rotate [0, 0, -1]`

优化：
- 本地改为显式从 `root_state_w[..., 3:7]` 计算 gravity direction

效果：
- decoder 的平衡相关观测更接近原始实现

### 4. 修正 decoder 的 `last_action_hist` 语义为“未裁剪 raw action”

文件：
- [action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)

问题：
- 本地之前把 `raw action` 先裁到 `[-2, 2]` 再写入 `last_action_hist`
- SONIC deploy 记录的是未裁剪的原始 policy 输出

优化：
- 保留 `raw_sonic_unclipped`
- `last_action_hist` 改为写入 unclipped raw action

效果：
- decoder 994 维输入中的 `his_last_actions_*` 更接近 deploy

### 5. 去掉本地额外加的 `raw_sonic` 和 `target_sonic` 裁剪

文件：
- [action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)

问题：
- 本地曾额外加入：
  - `raw_sonic = clip(raw, -2, 2)`
  - `target_sonic = clip(target, -3, 3)`
- SONIC deploy 原始实现没有这两层统一裁剪

优化：
- 去掉这两层 clip

效果：
- 更符合 SONIC 原始输出语义
- 对上半身幅度受限、下蹲支撑幅度不足这类问题有直接帮助

### 6. Pico sender 补齐 `adjusted_transl`

文件：
- [pico_server_pose_only.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/pico_server/pico_server_pose_only.py)

问题：
- 本地 sender 之前没有把 `adjusted_transl` 发送出来
- receiver 会因此退回用仿真 root z 兜底

优化：
- 补齐 `adjusted_transl` 的插值、缓存和发送

效果：
- root z / body 姿态 / 参考时序更加一致
- 降低了参考链路错配

### 7. 增加低频诊断日志，建立了可用的观测分析手段

文件：
- [action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)

新增日志包括：
- `[SONIC][STAND_DIAG]`
- `[SONIC][STAND_DIAG_RANGES]`
- `[SONIC][STAND_DIAG_TOPK]`
- `[SONIC][TRACKING_TOPK]`
- `[SONIC][TRACKING_SUPPORT]`

效果：
- 能直接看出静止参考下到底是 policy 输出异常，还是执行跟踪不上
- 帮助定位了 ankle、knee、hip_pitch 等关键问题
- 帮助确认“前倾小碎步”不是操作员在动，而是系统在做补偿

### 8. 给 `SONIC_PD_KP / SONIC_PD_KD` 补全逐关节注释

文件：
- [action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)

优化：
- 每一个 `kp/kd` 数值旁边都标注了对应 joint 名称

效果：
- 后续调参时可以直接看出该改哪个关节
- 降低人工数索引出错概率

## 今天确认“不是根因”或“不是优先级最高”的方向

### 1. 单纯加大 ankle 刚度

结论：
- 不是第一根因
- 在 observation / output 语义不对时，盲目加硬往往只会更快执行错误目标

### 2. 把问题简单归结为“初始姿态不对”

结论：
- 初始姿态会影响早期表现
- 但无法解释后续站立补偿、小碎步、蹲下困难等核心现象

### 3. 把问题简单归结为“sender 和原版完全不同”

结论：
- 本地 [`pico_server_pose_only.py`](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/pico_server/pico_server_pose_only.py) 在这条 POSE 链的主要发送逻辑上，与原版 [`pico_manager_thread_server.py`](/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic/scripts/pico_manager_thread_server.py) 基本一致
- 不是当前上半身 “W” 形 的主嫌疑

## 当前仍然存在的问题

1. 上半身动作与 MuJoCo SONIC 原始实现仍有差异  
现象：
- 双手平举时，本地复现仍可能呈现 “W” 形
- elbow 仍存在持续弯曲问题

2. 下半身深蹲稳定性仍不如原始 SONIC  
现象：
- MuJoCo 原始实现可以更自然地下蹲、趴下再站起
- Isaac Lab 复现虽然已经改善，但深蹲保持平衡仍偏困难

## 当前排查优先级

### 第一优先级

继续对比 deploy 与本地在以下几项上的语义：

- `q-default`
- `dq`
- `base_ang_vel`
- `gravity_dir`
- decoder 下肢/上肢输出目标

### 第二优先级

对比执行层差异：

- MuJoCo deploy 的 motor command 语义
- Isaac Lab position mode 的 runtime override
- 当前 `kp/kd` 调整对上半身和平衡的影响

### 第三优先级

在 observation 与输出语义基本确认后，再做精细增益调整：

- ankle pitch / ankle roll
- hip pitch / knee
- shoulder roll / shoulder yaw / elbow

## 建议的回归验证动作

每次修改后，至少验证以下动作：

1. 静止站立  
看是否前倾、小碎步

2. 双手完全平举  
看是否仍呈 “W” 形，elbow 是否持续弯曲

3. 深蹲并保持  
看膝、踝、腰是否能稳定支撑

4. 趴下再起身  
看大幅度动作下的输出范围和恢复能力

## 备注

今天的关键进展，不是某一组 `kp/kd`，而是把 Isaac Lab 复现从“明显不等价”逐步拉回到更接近 SONIC 原始实现的语义上。  
后续继续排查时，应优先保持这个原则：先修语义差异，再调增益。
