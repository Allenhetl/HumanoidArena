# Replay Debug Summary

## 范围

本文总结 `isaaclab_twist2_g1` 中 `TWIST2` 与 `SONIC` 两条录制/回放链路的主要问题、定位过程、已尝试的修复和当前结论。

相关模块主要包括：

- `sim_main.py`
- `action_provider/action_provider_wh_twist2.py`
- `action_provider/action_provider_sonic.py`
- `action_provider/reset_control.py`
- `pico_server/twist2_teleop_server.py`
- `pico_server/pico_server_pose_only.py`
- `run_replay.sh`
- `run_replay_sonic.sh`
- `run_twist2.sh`
- `run_sonic.sh`

---

## 一、核心现象

### 1. TWIST2

- `direct replay` 早期经常无法稳定回放，机器人会摔倒。
- 首段录制有时无法 replay，但 `save_and_reset` 之后录的第二段可以 replay。
- 如果 `pico server` 在 Isaac 完全启动前就开始发送 teleop 数据，这次会话里录到的数据往往不能 replay。
- `inference replay` 与 `direct replay` 表现不一致。

### 2. SONIC

- 第一段录制与第二段录制在 replay 可用性上不稳定。
- 曾出现 replay 开头瞬间向左旋转约 90 度的问题。
- 当前仍存在 replay 漂移/偏差问题，需要继续排查。

---

## 二、重要经验

### 1. 录制开始边界非常关键

- 当前录制不是“录制前额外 reset 一次再开始”，而是环境启动完成后的第一段就开始录。
- 如果录制边界和 replay 的起始边界不一致，哪怕动作序列一样，也很容易无法 replay。

### 2. 发送端时序会污染首段数据

- `run_twist2.sh`/`run_sonic.sh` 启动脚本里只在最开始清一次 Redis，不足以保证首帧输入干净。
- 从脚本启动到 `action provider` 真正第一次消费输入之间有较长窗口。
- 如果 Pico 在这段窗口里提前发送数据，那么 provider 首次读取的就可能是“错误时机的 teleop 输入”。

### 3. direct replay 对“状态转移完全一致”要求很高

- 只要 `Frame 0 pre-action` 与 `Frame 0 action` 后的 `Frame 1` 存在微小差异，后续误差就会积累。
- 这种误差不一定来自动作本身，很多时候来自脚本入口、重力、渲染路径、机器人 USD、输入 provider 等系统差异。

---

## 三、已经确认过的关键根因

### 1. Replay 重力与录制不一致

这是一个已经确认的硬问题。

- 录制侧默认使用 IsaacLab 物理默认重力 `-9.81`
- replay 侧一度显式传了 `--gravity_z -9.8`

这个 `0.01 m/s^2` 的差距会在第一个 `decimation` 周期内直接体现在根速度上，量级与实际观测到的 `Frame 1 root_lin_vel_err` 一致。

结论：

- 这是导致 `TWIST2 direct replay` 初始偏差的明确原因之一。

### 2. Replay 与录制的 headless/render 路径不一致

- 录制脚本是 `--headless`
- replay 脚本一度不是 `--headless`

由于控制循环最后一个 substep 会显式走 `sim.step(render=True)`，这不是单纯 UI 差异，而是会走不同的系统路径。

结论：

- replay 必须和录制保持同样的 `headless` 路径。

### 3. Replay 与录制使用的机器人 USD 不一致

如果录制与 replay 没有完全对齐 `ROBOT_USD_OVERRIDE`，同一任务在 wholebody 平衡场景中足以直接导致 replay 漂移。

结论：

- replay 脚本必须与录制脚本使用完全一致的 robot USD 选择逻辑。

### 4. TWIST2 首段录制被启动前 teleop 输入污染

这是 `TWIST2` 首段不可 replay 的主要原因之一。

现象规律：

- 如果 Pico 在 Isaac 完全 ready 之前就开始发数据，首段录制往往不可 replay
- `save_and_reset` 后重新录的第二段往往可以 replay

结论：

- 污染不是单纯来自 Redis 有旧键，而是来自“错误时机的首个有效 teleop 输入”进入了 provider 历史状态。

### 5. SONIC replay 的 90 度转向问题来自 anchor/heading 状态未恢复

`SONIC replay` 早期会在开头瞬间发生一次明显 yaw 偏转，后来确认与 replay 时没有恢复录制阶段的 anchor/heading 对齐状态有关。

结论：

- replay 必须恢复 anchor 相关内部状态，而不是在首帧重新按当前运行时条件计算。

---

## 四、已经尝试并落地的修改

### A. 通用 replay 入口统一

已将 replay 参数统一进 `sim_main.py`：

- `--replay_file`
- `--replay_mode`
- `--replay_loop`

并根据 `gmt_backend` 路由到不同 `action provider`。

### B. TWIST2 provider 命名统一

- `DDSRLActionProvider` 已改名为 `TWIST2ActionProvider`
- `dds_wholebody` 已统一改为 `twist2_wholebody`

### C. TWIST2 replay 关键修复

1. 对齐 replay 与录制的：
- 重力
- `headless`
- `ROBOT_USD_OVERRIDE`
- ONNX provider 选择逻辑

2. 在 replay 中增加状态一致性对比：
- 初始资产状态
- PhysX 对外可见状态
- 关节角 replay 平均误差日志

3. 将 `direct replay` 与 `inference replay` 的输入来源统一整理到 provider 内部。

### D. 输入 ready barrier 机制

新增公共模块：

- `action_provider/reset_control.py`

实现能力：

- 发布 `ready epoch`
- 清理对应 backend 的 Redis 输入键
- 启动后发布一次 barrier
- reset 后再次发布 barrier

发送端改动：

- `twist2_teleop_server.py`
- `pico_server_pose_only.py`

现在发送端会：

- 只在收到当前 episode 的 `ready epoch` 后才开始发送 live 数据
- 一旦检测到新 epoch，会先重置本地缓存/状态，再开始新 episode 的发送

结论：

- 这是为了解决“启动前/重置前 Pico 已经开始发数据导致首段污染”的问题。

### E. SONIC 录制按键与保存语义对齐 TWIST2

已对齐：

- `key1 -> save_and_reset`
- `key2 -> discard_and_reset`
- 保存改为阻塞等待，不再异步保存

### F. SONIC replay anchor 恢复

已在 `action_provider_sonic.py` 中恢复录制文件里的：

- `anchor_heading_initialized`
- `anchor_use_heading_align`
- `anchor_init_base_quat_wxyz`
- `anchor_init_ref_quat_wxyz`
- `anchor_heading_align_quat_wxyz`

用于修复首帧 yaw 偏差问题。

### G. SONIC 录制命令处理顺序调整

曾确认一个具体 bug：

- `start` 命令如果在 `add_frame` 之后处理，会导致新 episode 的第一帧 fresh 数据没有被录进去

因此已将 `SONIC` 的命令处理顺序调整为：

1. 先处理 recording command
2. 再决定是否记录当前帧

这点与 `TWIST2` 保持一致。

### H. run_replay_sonic.sh 对齐录制脚本

已将 `run_replay_sonic.sh` 对齐到 `run_sonic.sh` 的关键逻辑：

- 使用相同的 `ROBOT_USD_OVERRIDE`
- 清理 `isaac_input_ready_sonic_unitree_g1_with_hands`
- 使用 `--headless`
- 使用当前统一的 replay 参数入口

---

## 五、试过但结论是否定或有限的项

### 1. `enable_enhanced_determinism`

已在 football 相关 task 中打开：

- 它可以帮助减少 PhysX 并行与顺序带来的不确定性
- 但它不是 replay 失败的根本解释

结论：

- 有帮助，但不足以单独解决 replay 问题。

### 2. `configure_seed()`

当前版本并没有这个统一方法。

已经确认更关键的是：

- 在 `gym.make(...)` 之前设置 `env_cfg.seed`

而 provider 里后设 `torch.manual_seed()` 对环境初始化随机性帮助有限。

结论：

- `configure_seed()` 缺失不是主因。

### 3. “开环 replay 一定不行”

这个说法过于绝对。

实际排查结论是：

- 只要初始状态、动作、入口脚本、物理配置完全一致，`direct replay` 是可以成功的
- `TWIST2 direct replay` 在修正重力/脚本差异后已验证可行

结论：

- 开环不是根因，真正的问题在于系统路径不一致和录制边界污染。

---

## 六、当前状态

### 1. TWIST2

当前状态相对清晰：

- `direct replay` 已经能工作
- 首段录制的污染问题已通过 ready barrier 与发送端 gating 明显改善
- `inference replay` 也已对齐 ONNX provider 逻辑

当前 `TWIST2` 已不是主要阻塞项。

### 2. SONIC

`SONIC` 仍是当前主要问题点。

已经确认和修过的内容包括：

- 发送端 ready barrier
- 按键保存逻辑对齐
- anchor heading replay 恢复
- replay 脚本入口与录制入口对齐
- recording command 与 `add_frame` 顺序调整

但截至当前，`SONIC` 仍然存在 replay 偏差，需要继续排查。

---

## 七、当前最值得继续查的方向

### 1. SONIC 录制边界是否仍不一致

重点看：

- 第一帧录制是否对应 `env.reset()` 后的第一帧
- `save_and_reset` 之后第二段的起录边界是否与第一段一致
- `start` 命令与第一帧 fresh pose window 是否严格对齐

### 2. SONIC replay 的 joint error 从第几帧开始抬头

当前已在 provider 中加入：

- replay 关节角 `mae/max/running_mae`

建议利用这组日志判断：

- 是 `frame 0/1` 就开始偏
- 还是前几十帧很小、之后逐渐积累

这能帮助区分：

- 初始化/录制边界问题
- 还是动力学累计问题

### 3. 录制端与 replay 端的脚本入口差异是否还有遗漏

尤其继续关注：

- `headless`
- robot USD
- world camera / image server
- 模型 provider
- reset 后是否有额外的 scene/runtime patch

---

## 八、简要结论

### TWIST2

`TWIST2 replay` 的主要问题并不是“无法 determinism”，而是多处系统级差异叠加：

- replay 重力不一致
- replay 与录制的 render/headless 路径不一致
- robot USD 不一致
- 启动前 Pico 提前发数据导致首段 teleop 污染

这些问题修正后，`TWIST2 direct replay` 已恢复可用。

### SONIC

`SONIC replay` 当前更复杂。

已确认并修复了若干具体问题：

- anchor/heading 状态未恢复
- replay/recording 脚本入口未对齐
- `start` 命令处理时序导致首帧 fresh 数据漏录

但 `SONIC` 仍存在 replay 偏差，说明还有至少一个录制边界或运行时状态路径未完全对齐。

---

## 九、建议的后续排查顺序

1. 用当前 joint replay 误差日志，确认 `SONIC` 偏差从第几帧开始显著放大
2. 针对 `frame 0/1` 检查第二段录制的起录边界
3. 继续对比 `run_sonic.sh` 与 `run_replay_sonic.sh` 在所有系统开关上的一致性
4. 如仍存在问题，再抓 `SONIC` 首帧录制窗口和首帧 replay 输入的逐字段差异

