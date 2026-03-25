# SONIC ZMQ 数据流诊断报告

## 问题描述

运行 `run_sonic.sh` 时，日志显示：
```
[SONIC][HISTORY] frame=180 smpl_history_fill=0/10 smpl_valid=False action=hold_current_pose
```

SMPL 历史缓冲区一直没有被填充，导致机器人保持默认姿态不动。

## 数据流架构

```
Pico VR 头显
    ↓
pico_server_pose_only.py (发送端)
    ↓ ZMQ PUB (tcp://*:5556, topic="pose")
    ↓
action_provider_sonic.py (接收端)
    ↓ ZMQ SUB (tcp://localhost:5556, topic="pose")
    ↓
_fetch_zmq_pose() → _parse_zmq_pose() → _apply_pose_data()
    ↓
SMPL 历史缓冲区填充
```

## 诊断结果

### ✅ 发送端正常

使用 `test_zmq_receive.py` 测试，确认：
- pico_server_pose_only.py 正在发送数据
- 数据格式正确，包含所有必需字段
- 数据内容有效（非全0）
- 发送频率约 50 Hz

### ✅ 数据格式正确

ZMQ 消息包含以下字段（批量发送，5帧一批）：
- `smpl_joints`: shape=[5, 24, 3], dtype=f32 ✓
- `smpl_pose`: shape=[5, 21, 3], dtype=f32 ✓
- `body_quat_w`: shape=[5, 4], dtype=f32 ✓
- `vr_position`: shape=[9], dtype=f32 ✓
- `vr_orientation`: shape=[12], dtype=f32 ✓
- `joint_pos`: shape=[5, 29], dtype=f64 ✓
- `joint_vel`: shape=[5, 29], dtype=f64 ✓

### ❓ 接收端问题

可能的原因：

1. **ZMQ Poller 初始化失败**
   - `_HAS_ZMQ_POLLER = False`（gear_sonic 未正确导入）
   - ZMQPoller 构造函数抛出异常

2. **ZMQ 订阅连接未建立**
   - ZMQ PUB-SUB 模式的"slow joiner"问题
   - 订阅者启动时，发布者已经发送了很多消息
   - 需要等待订阅连接完全建立

3. **数据解析失败**
   - `_parse_zmq_pose()` 返回 None
   - 消息格式不匹配

## 已添加的调试日志

### 1. `_setup_zmq()` 函数
```python
[SonicActionProvider] Attempting to connect ZMQ: tcp://localhost:5556 topic=pose
[SonicActionProvider] ✓ ZMQ connected successfully tcp://localhost:5556 topic=pose
[SonicActionProvider] ZMQ Poller object: <ZMQPoller instance>
```

### 2. `_fetch_zmq_pose()` 函数
```python
[ZMQ] ERROR: _zmq_poller is None!  # 如果 poller 未初始化
[ZMQ] No data available (frame=X)  # 如果没有收到数据
[ZMQ] Received raw data, size=XXXX bytes (frame=X)  # 收到数据
[ZMQ] ERROR: Failed to parse ZMQ data (frame=X)  # 解析失败
[ZMQ] Successfully parsed data, calling _apply_pose_data (frame=X)  # 成功
```

### 3. `_apply_pose_data()` 函数
```python
[ZMQ] Received data keys: ['smpl_joints', 'smpl_pose', ...]
[ZMQ] smpl_joints shape: (5, 24, 3), latest frame sum: XX.XXXX
[ZMQ] SMPL data marked as VALID
```

## 下一步操作

### 1. 重新运行 run_sonic.sh

确保按照以下顺序启动：

```bash
# Terminal 1: 启动 pico_server（发布者）
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl

# 等待 5-10 秒，确保 pico_server 完全启动

# Terminal 2: 启动 Isaac Lab 仿真（订阅者）
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
bash run_sonic.sh
```

### 2. 检查日志输出

查找以下关键日志：

**初始化阶段：**
- `[SonicActionProvider] Attempting to connect ZMQ: ...`
- `[SonicActionProvider] ✓ ZMQ connected successfully ...`

**运行阶段（前3帧和每50帧）：**
- `[ZMQ] No data available (frame=X)` - 说明没有收到数据
- `[ZMQ] Received raw data, size=XXXX bytes (frame=X)` - 说明收到数据
- `[ZMQ] Successfully parsed data, calling _apply_pose_data (frame=X)` - 说明解析成功
- `[ZMQ] Received data keys: [...]` - 说明 _apply_pose_data 被调用
- `[ZMQ] SMPL data marked as VALID` - 说明数据有效

### 3. 根据日志判断问题

**情况A：看到 "ZMQ Poller object: None" 或 "ERROR: _zmq_poller is None!"**
- 问题：ZMQ Poller 初始化失败
- 解决：检查 gear_sonic 是否正确安装，路径是否正确

**情况B：一直看到 "No data available"**
- 问题：ZMQ 订阅连接未建立或发布者未发送数据
- 解决：
  1. 确认 pico_server 正在运行（`ps aux | grep pico_server`）
  2. 确认端口正确（默认 5556）
  3. 增加启动延迟，先启动 pico_server，等待 10 秒后再启动 run_sonic.sh

**情况C：看到 "Received raw data" 但没有看到 "Successfully parsed data"**
- 问题：数据解析失败
- 解决：检查 `_parse_zmq_pose()` 函数的错误日志

**情况D：看到 "Successfully parsed data" 但 smpl_history_fill 仍然是 0**
- 问题：`_apply_pose_data()` 逻辑问题
- 解决：检查 `got_pose_frame` 是否被正确设置

## 测试工具

### test_zmq_receive.py

独立测试 ZMQ 数据接收：

```bash
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
python test_zmq_receive.py
```

这个脚本会：
- 连接到 ZMQ 端口 5556
- 订阅 "pose" topic
- 打印收到的所有消息
- 显示消息频率

如果这个脚本能收到数据，说明发送端正常，问题在 action_provider_sonic.py 的接收逻辑。

## 可能的根本原因

基于 ZMQ PUB-SUB 模式的特性，最可能的原因是：

**ZMQ "slow joiner" 问题**

ZMQ SUB socket 在调用 `connect()` 后需要一些时间（通常几毫秒到几百毫秒）来建立订阅。在此期间，发布者发送的所有消息都会丢失。

如果 Isaac Lab 启动很快，在订阅连接完全建立之前就开始调用 `get_data()`，会导致前面的消息全部丢失。

### 解决方案

在 `_setup_zmq()` 后添加延迟：

```python
def _setup_zmq(self):
    # ... 现有代码 ...
    if self._zmq_poller is not None:
        print("[SonicActionProvider] Waiting for ZMQ subscription to establish...")
        time.sleep(1.0)  # 等待 1 秒让订阅连接建立
        print("[SonicActionProvider] ZMQ subscription ready")
```

或者在 `_fetch_zmq_pose()` 中添加重试逻辑，在前几帧没有数据时不报错。

## 总结

1. ✅ pico_server 正在正常发送数据
2. ✅ 数据格式正确
3. ❓ action_provider_sonic.py 可能没有收到数据
4. 🔧 已添加详细的调试日志
5. 📋 下一步：重新运行并检查日志输出
