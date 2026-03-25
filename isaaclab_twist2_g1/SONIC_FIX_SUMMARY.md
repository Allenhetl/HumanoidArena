# SONIC 动作填充问题修复总结

## 问题描述

运行 `run_sonic.sh` 时，SMPL 历史观测没有被填充：
```
[SONIC][HISTORY] frame=180 smpl_history_fill=0/10 smpl_valid=False action=hold_current_pose
```

## 根本原因

**ZMQ "slow joiner" 问题**：ZMQ PUB-SUB 模式中，订阅者（Isaac Lab）连接到发布者（pico_server）后，需要一些时间建立订阅。在此期间发送的所有消息都会丢失。

## 已实施的修复

### 1. 添加 ZMQ 订阅延迟（action_provider_sonic.py:844-847）

```python
# Wait for ZMQ subscription to establish (fixes "slow joiner" problem)
print("[SonicActionProvider] Waiting for ZMQ subscription to establish...")
time.sleep(1.0)
print("[SonicActionProvider] ZMQ subscription ready")
```

这确保在开始接收数据前，ZMQ 订阅连接已完全建立。

### 2. 增强调试日志

#### `_setup_zmq()` 函数
- 显示 ZMQ 连接尝试
- 显示连接成功/失败状态
- 显示 Poller 对象信息
- 显示订阅建立过程

#### `_fetch_zmq_pose()` 函数
- 检测 Poller 是否为 None
- 显示数据接收状态（每 50 帧）
- 显示数据大小
- 显示解析成功/失败

#### `_apply_pose_data()` 函数
- 显示接收到的数据字段
- 显示 SMPL 数据形状和内容
- 显示数据有效性标志

## 测试验证

### 独立测试工具：test_zmq_receive.py

已创建独立测试脚本，验证 ZMQ 数据流正常：

```bash
python test_zmq_receive.py
```

**测试结果：✅ 成功**
- pico_server 正在发送数据
- 数据格式正确（5帧批量）
- 包含所有必需字段
- 数据内容有效（非全0）

## 下一步操作

### 1. 重新运行测试

按照正确的启动顺序：

```bash
# Terminal 1: 启动 pico_server（必须先启动）
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl

# 等待看到 "Body data available!" 消息

# Terminal 2: 启动 Isaac Lab（在 pico_server 完全启动后）
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
bash run_sonic.sh
```

### 2. 检查关键日志

**启动阶段应该看到：**
```
[SonicActionProvider] Attempting to connect ZMQ: tcp://localhost:5556 topic=pose
[SonicActionProvider] ✓ ZMQ connected successfully tcp://localhost:5556 topic=pose
[SonicActionProvider] Waiting for ZMQ subscription to establish...
[SonicActionProvider] ZMQ subscription ready
```

**运行阶段应该看到（前3帧）：**
```
[ZMQ] Received raw data, size=XXXX bytes (frame=1)
[ZMQ] Successfully parsed data, calling _apply_pose_data (frame=1)
[ZMQ] Received data keys: ['smpl_joints', 'smpl_pose', 'body_quat_w', ...]
[ZMQ] smpl_joints shape: (5, 24, 3), latest frame sum: XX.XXXX
[ZMQ] SMPL data marked as VALID
[ZMQ][HISTORY] smpl_history_fill=1/10 smpl_valid=True
```

**第10帧后应该看到：**
```
[SONIC][HISTORY] frame=10 smpl_history_fill=10/10 smpl_valid=True action=running_gear_sonic
```

### 3. 预期结果

- ✅ `smpl_history_fill` 应该从 0 逐渐增加到 10
- ✅ `smpl_valid` 应该变为 True
- ✅ 第10帧后，机器人应该开始执行 SONIC 推理的动作
- ✅ 机器人应该跟随 VR 头显的动作

## 如果问题仍然存在

### 情况A：看到 "No data available"

**原因：** pico_server 未运行或端口不匹配

**解决：**
```bash
# 检查 pico_server 是否运行
ps aux | grep pico_server_pose_only

# 检查端口
netstat -tuln | grep 5556

# 重启 pico_server
pkill -f pico_server_pose_only
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl
```

### 情况B：看到 "Failed to parse ZMQ data"

**原因：** 数据格式不匹配

**解决：** 检查 pico_server 版本是否与 action_provider_sonic.py 兼容

### 情况C：smpl_history_fill 增加但 smpl_valid=False

**原因：** SMPL 数据全为 0

**解决：** 检查 VR 头显是否正常工作，body tracking 是否启用

## 文件修改清单

1. **action_provider/action_provider_sonic.py**
   - `_setup_zmq()`: 添加 1 秒延迟和详细日志
   - `_fetch_zmq_pose()`: 添加详细调试日志
   - 已有的 `_apply_pose_data()`: 保持现有日志

2. **test_zmq_receive.py** (新文件)
   - 独立 ZMQ 接收测试工具

3. **SONIC_ZMQ_DEBUG.md** (新文件)
   - 详细的诊断报告

## 技术细节

### ZMQ "Slow Joiner" 问题

ZMQ PUB-SUB 模式的特性：
- 发布者不等待订阅者
- 订阅者连接需要时间（通常 < 1 秒）
- 在订阅建立前的消息会丢失

**标准解决方案：**
1. 先启动发布者，后启动订阅者
2. 订阅者连接后等待一段时间再开始接收
3. 使用 REQ-REP 模式进行同步（性能较差）

我们采用了方案 1 + 2 的组合。

### 数据流时序

```
时间轴：
T0: pico_server 启动，开始发送数据
T1: Isaac Lab 启动
T2: ZMQPoller 创建，开始连接
T3: sleep(1.0) - 等待订阅建立
T4: 开始调用 get_data()，应该能收到数据
```

如果没有 T3 的延迟，T2-T4 之间的消息会丢失。由于 pico_server 以 50Hz 发送，1 秒内会丢失约 50 条消息。

## 总结

✅ 已添加 ZMQ 订阅延迟修复
✅ 已添加详细调试日志
✅ 已验证 pico_server 数据发送正常
✅ 已创建独立测试工具

📋 下一步：重新运行 run_sonic.sh 并检查日志输出
