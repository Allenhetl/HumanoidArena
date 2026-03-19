# 环境完全重建方案 (Environment Recreation)

## 概述

这套代码实现了**完全重建环境**的方案，用于解决 reset 后 replay 失败的问题。

## 问题分析

### 为什么 reset 后的数据无法 replay？

1. **PhysX 是伪随机的**
   - PhysX 的内部状态包含随机数生成器
   - 即使 Frame 0 的物理状态相同，后续演化也会不同
   - 只有从头开始录制且保持随机种子一致才能 replay

2. **env.reset() 不够彻底**
   - `env.sim.reset()` + `env.reset()` 只重置物理状态
   - 不会重置 PhysX 的内部随机状态
   - 不会重新初始化 GPU 内存和缓存

3. **录制 timing 问题**
   - Reset 后立即开始录制，Frame 0 可能捕获默认值
   - 需要等待第一个物理步完成后再开始录制

## 解决方案

### 完全重建环境

当用户按下保存+重置按钮时：

1. **保存当前录制**（阻塞等待完成）
2. **完全销毁环境**：
   - 清理 action_provider
   - 停止 controller
   - 关闭 environment
   - 运行垃圾回收
   - **清除 Redis 缓存**（防止旧数据影响新环境）
3. **重新创建环境**：
   - 调用 `gym.make()` 创建全新环境
   - 重新初始化 PhysX
   - 创建新的 controller 和 action_provider
4. **开始新的录制**（在第一次 get_action() 时）

### Redis 缓存清除

为了确保环境重建后的干净状态，系统会在两个时机清除 Redis 缓存：

1. **启动时**（`run_recreate.sh`）：
   - 清除所有 action、state、control 相关的键
   - 防止使用上次运行遗留的数据

2. **环境重建时**（`recreate_environment_completely()`）：
   - 在垃圾回收后、创建新环境前清除
   - 确保新环境不会读取到旧环境的数据
   - 清除的键包括：
     - action_body/hand/neck
     - state_body/hand/neck
     - controller_data
     - human_smplx_data/human_info
     - 等等

这样可以避免：
- 新环境启动时读取到旧的 action 数据
- 遥操作端收到旧的 state 数据
- 录制系统使用旧的 human 数据

### 相机视角设置

系统会自动切换到第一人称视角（front_cam）：
- 使用固定的相机路径，不遍历 USD stage
- 避免阻塞环境创建流程
- 在环境初始化后立即设置

### 窗口激活说明

Isaac Sim 需要手动点击窗口来激活渲染显示：
- **启动后**：点击一次窗口激活渲染
- **环境重建后**：再次点击窗口激活渲染
- **重要**：点击窗口**不影响**录制和 replay
  - 物理仿真在后台持续运行
  - 录制数据完整无缺失
  - 只是渲染显示需要激活

## 文件说明

### 新文件

1. **sim_main_recreate.py**
   - 主程序，实现完全重建环境的逻辑
   - 监听 Redis 的 `recording_control_unitree_g1_with_hands` 命令
   - 处理 `save_and_reset` 和 `discard_and_reset` 命令
   - 调用 `recreate_environment_completely()` 函数完全重建环境

2. **run_recreate.sh**
   - 启动脚本
   - 使用方法：`./run_recreate.sh --seed 42`

3. **ENVIRONMENT_RECREATION_GUIDE.md**（本文件）
   - 使用说明和技术文档

### 修改的文件

无需修改现有文件，这是一套完全独立的代码。

**注意**：`sim_main_recreate.py` 直接使用原版的 `action_provider_wh_twist2.py`，不需要特殊版本。save_and_reset 和 discard_and_reset 的逻辑完全在 sim_main 层面处理。

## 使用方法

### 1. 启动仿真

```bash
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
./run_recreate.sh --seed 42
```

### 2. 启动遥操作

```bash
# 在另一个终端
cd /home/dreams/Users/taowen/HumanoidArena/TWIST2/deploy_real
python3 xrobot_teleop_to_robot_w_hand.py
```

### 3. 操作流程

1. **开始录制**：仿真启动后自动开始录制
2. **保存+重置**：按下左手柄 key_one
   - 保存当前录制
   - 完全重建环境
   - 自动开始新的录制
3. **丢弃+重置**：按下左手柄 key_two
   - 丢弃当前录制
   - 完全重建环境
   - 自动开始新的录制

## 技术细节

### 环境重建流程

```python
def recreate_environment_completely(args_cli, env_cfg, old_env, old_controller, old_action_provider):
    # 1. 清理旧的 action provider
    old_action_provider.cleanup()
    del old_action_provider

    # 2. 停止旧的 controller
    old_controller.stop()
    del old_controller

    # 3. 关闭旧的 environment
    old_env.close()
    del old_env

    # 4. 垃圾回收
    gc.collect()
    torch.cuda.empty_cache()

    # 5. 创建新环境（调用 gym.make）
    new_env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    new_env.sim.reset()
    new_env.reset()

    # 6. 创建新的 controller 和 action provider
    new_action_provider = create_action_provider(new_env, args_cli)
    new_controller = RobotController(new_env, control_config)
    new_controller.set_action_provider(new_action_provider)

    return new_env, new_controller, new_action_provider
```

### 录制 Timing

```python
# 在 action_provider 的 __init__ 中
self._should_start_recording_on_first_call = True

# 在第一次 get_action() 调用时
if self._should_start_recording_on_first_call:
    self.recording_manager.start_recording()
    self._should_start_recording_on_first_call = False
```

这确保：
- Frame 0 是 env.reset() 后第一个完整物理步的状态
- 不是默认值或中间状态

### Redis 通信

```python
# sim_main_recreate.py 检查 recording_control 命令
recording_control_raw = redis_client.get("recording_control_unitree_g1_with_hands")
if recording_control_raw:
    recording_control = json.loads(recording_control_raw)
    command = recording_control.get("command", "none")

    if command == "save_and_reset":
        # 1. 保存录制（阻塞等待完成）
        action_provider.recording_manager.save_recording(completion_callback=None)
        action_provider.recording_manager.save_queue.join()

        # 2. 完全重建环境
        env, controller, action_provider = recreate_environment_completely(...)

        # 3. 启动新的 controller
        controller.start()

        # 4. 开始新的录制
        action_provider.recording_manager.start_recording()

        # 5. 发送完成信号
        redis_client.set("isaac_reset_complete_unitree_g1_with_hands", ...)

        # 6. 清除命令
        redis_client.delete("recording_control_unitree_g1_with_hands")

    elif command == "discard_and_reset":
        # 类似流程，但第1步改为 cancel_recording()
        ...
```

**关键点**：
- 监听 `recording_control_unitree_g1_with_hands` 而不是 `isaac_reset_trigger`
- 命令是 `save_and_reset` 和 `discard_and_reset`，不是 `reset_category="3"`
- 在 sim_main 层面处理，action_provider 使用原版即可

## 预期效果

### 从头开始录制的数据

- Frame 0：真实物理状态（接近 0）
- 可以 replay，误差接近 0

### Reset 后录制的数据

- Frame 0：真实物理状态（接近 0）
- 由于环境完全重建，PhysX 状态完全清空
- **理论上**可以 replay，但需要测试验证

## 注意事项

1. **性能开销**
   - 完全重建环境需要 2-5 秒
   - 比简单 reset 慢很多
   - 但确保了确定性

2. **随机种子**
   - 必须使用相同的 seed 才能 replay
   - 启动时通过 `--seed` 参数指定

3. **测试建议**
   - 先测试从头开始录制的数据能否 replay
   - 再测试 reset 后录制的数据能否 replay
   - 对比两者的误差

## 调试

### 检查 Frame 0 状态

```python
import numpy as np

data = np.load("recording.npz", allow_pickle=True)
print("Frame 0 qpos[0:6]:", data['robot_qpos_before_decimation'][0, 0:6])
print("Frame 0 qvel[0:6]:", data['robot_qvel_before_decimation'][0, 0:6])

# 应该接近 0，而不是默认值 [-0.2, 0, 0, 0.4, -0.2, 0]
```

### 查看日志

```bash
# sim_main_recreate.py 会打印详细的重建过程
grep "RECREATING ENVIRONMENT" sim_output.log
grep "Frame 0 captures real physics state" sim_output.log
```

## 下一步

1. **测试 replay**
   - 录制一段数据
   - 使用相同的 seed replay
   - 检查误差

2. **测试 reset 后的 replay**
   - 录制 → 保存+重置 → 录制
   - Replay 第二段数据
   - 检查误差

3. **优化性能**
   - 如果重建太慢，考虑只重建必要的组件
   - 或者使用更轻量级的清理方法

## 参考

- 原始代码：`sim_main.py`, `action_provider_wh_twist2.py`
- PhysX 文档：https://docs.omniverse.nvidia.com/py/isaacsim/
- Isaac Lab 文档：https://isaac-sim.github.io/IsaacLab/
