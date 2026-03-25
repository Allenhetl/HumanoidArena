# SONIC 动作填充问题 - 最终修复方案

## 🎯 问题根源

**GR00T-WholeBodyControl 路径错误导致 ZMQPoller 无法导入**

### 错误的路径计算
```python
_GROOT_ROOT = os.path.join(os.path.dirname(_TWIST2_ROOT), "GR00T-WholeBodyControl")
# 结果: /home/dreams/Users/taowen/HumanoidArena/GR00T-WholeBodyControl (不存在!)
```

### 实际路径
```
/home/dreams/Users/taowen/GR00T-WholeBodyControl
```

## ✅ 已实施的修复

### 1. **修复 GR00T 路径**（action_provider_sonic.py:56）

```python
_GROOT_ROOT = "/home/dreams/Users/taowen/GR00T-WholeBodyControl"
```

### 2. **添加 ZMQ 订阅延迟**（action_provider_sonic.py:844-847）

```python
# Wait for ZMQ subscription to establish (fixes "slow joiner" problem)
print("[SonicActionProvider] Waiting for ZMQ subscription to establish...")
time.sleep(1.0)
print("[SonicActionProvider] ZMQ subscription ready")
```

### 3. **增强调试日志**

- `_setup_zmq()`: 显示连接状态、Poller 对象、订阅建立过程
- `_fetch_zmq_pose()`: 显示数据接收、解析状态（每 50 帧）
- `_apply_pose_data()`: 显示数据内容和有效性

## 📋 测试步骤

### 1. 启动 pico_server（发布者）

```bash
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl
```

等待看到：
```
Body data available!
[Main] FPS: XX.XX, Step: XXXX
```

### 2. 启动 Isaac Lab（订阅者）

```bash
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
bash run_sonic.sh
```

## ✅ 预期日志输出

### 启动阶段（应该看到）

```
[SonicActionProvider] Attempting to connect ZMQ: tcp://localhost:5556 topic=pose
[SonicActionProvider] ✓ ZMQ connected successfully tcp://localhost:5556 topic=pose
[SonicActionProvider] ZMQ Poller object: <gear_sonic.utils.teleop.zmq.zmq_poller.ZMQPoller object at 0x...>
[SonicActionProvider] Waiting for ZMQ subscription to establish...
[SonicActionProvider] ZMQ subscription ready
[SonicActionProvider] loaded model_encoder.onnx providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
[SonicActionProvider] loaded model_decoder.onnx providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
Successful load sonic model
[SonicActionProvider] POSE mode ready  pose_source=zmq  (zmq=localhost:5556  redis=localhost:6379)
```

### 运行阶段（前 3 帧）

```
[SONIC] Real get_action path enabled
[ZMQ] Received raw data, size=XXXX bytes (frame=1)
[ZMQ] Successfully parsed data, calling _apply_pose_data (frame=1)
[ZMQ] Received data keys: ['smpl_joints', 'smpl_pose', 'body_quat_w', 'vr_position', 'vr_orientation', ...]
[ZMQ] smpl_joints shape: (5, 24, 3), latest frame sum: XX.XXXX
[ZMQ] smpl_joints latest frame:
[[...]]
[ZMQ] SMPL data marked as VALID
[ZMQ] smpl_pose shape: (5, 21, 3), latest frame:
[[...]]
[ZMQ] body_quat_w shape: (5, 4), latest: [...]
[ZMQ][ANCHOR_INIT] init_base=[...] init_ref=[...] use_heading_align=True/False
[ZMQ][HISTORY] smpl_history_fill=1/10 smpl_valid=True
zmq pose
[SONIC][HISTORY] frame=1 smpl_history_fill=1/10 smpl_valid=True action=hold_current_pose
```

### 第 10 帧后（历史缓冲区填满）

```
[ZMQ][HISTORY] smpl_history_fill=10/10 smpl_valid=True
[SONIC] _run_gear_sonic called
[SONIC] encoder=True, decoder=True
[SONIC] _smpl_data_valid=True
[SONIC] SMPL joints buffer sum: XX.XXXX
[SONIC] Encoder input shape: (1, 1762), expected: (1, 1762)
[SONIC] ✓ Encoder output latent shape: (1, 64)
[SONIC] ✓ Decoder output shape: (1, 29)
[SONIC] ✓ Final target range (after safety clip): [min, max]
[SONIC][PERF] encoder_ms=X.XX decoder_ms=X.XX total_ms=X.XX
```

此时机器人应该开始执行 SONIC 推理的动作，跟随 VR 头显移动。

## ❌ 之前的错误日志（已修复）

```
[SonicActionProvider] WARNING: ZMQPoller not available (_HAS_ZMQ_POLLER=False)
[ZMQ] ERROR: _zmq_poller is None!
[SONIC][HISTORY] frame=180 smpl_history_fill=0/10 smpl_valid=False action=hold_current_pose
```

## 🔧 故障排除

### 问题 A：仍然看到 "ZMQPoller not available"

**原因：** gear_sonic 模块未安装或路径错误

**解决：**
```bash
# 检查 GR00T 路径
ls -la /home/dreams/Users/taowen/GR00T-WholeBodyControl

# 检查 gear_sonic 模块
python -c "import sys; sys.path.insert(0, '/home/dreams/Users/taowen/GR00T-WholeBodyControl'); from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller; print('OK')"
```

### 问题 B：看到 "No data available"

**原因：** pico_server 未运行或端口不匹配

**解决：**
```bash
# 检查 pico_server 进程
ps aux | grep pico_server_pose_only

# 检查端口
netstat -tuln | grep 5556

# 重启 pico_server
pkill -f pico_server_pose_only
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl
```

### 问题 C：smpl_history_fill 增加但机器人不动

**原因：** SMPL 数据全为 0 或 VR 头显未正常工作

**解决：**
```bash
# 使用测试工具检查数据
python test_zmq_receive.py

# 检查 VR 头显 body tracking
# 确保 Pico 头显、手腕控制器、脚踝 tracker 都正常工作
```

## 📊 性能指标

### 正常运行时的性能

- **ZMQ 接收频率**: ~50 Hz（与 pico_server 发送频率一致）
- **SMPL 历史填充**: 10 帧（约 0.2 秒）
- **Encoder 推理**: ~5-10 ms（CUDA）
- **Decoder 推理**: ~5-10 ms（CUDA）
- **总控制循环**: ~50 Hz（20 ms/帧）

### 预期行为

1. **前 10 帧**: 机器人保持当前姿态（hold_current_pose），同时填充 SMPL 历史缓冲区
2. **第 10 帧后**: 开始执行 SONIC 推理，机器人跟随 VR 头显动作
3. **稳定运行**: 机器人实时跟随 VR 输入，延迟 < 100 ms

## 📝 文件修改清单

### 修改的文件

1. **action_provider/action_provider_sonic.py**
   - 第 56 行: 修复 GR00T 路径（硬编码为正确路径）
   - 第 844-847 行: 添加 ZMQ 订阅延迟（1 秒）
   - 第 831-851 行: 增强 `_setup_zmq()` 调试日志
   - 第 1022-1041 行: 增强 `_fetch_zmq_pose()` 调试日志

### 新增的文件

2. **test_zmq_receive.py**
   - 独立 ZMQ 接收测试工具
   - 用于验证 pico_server 数据发送

3. **SONIC_FIX_SUMMARY.md**
   - 修复总结文档（本文件）

4. **SONIC_ZMQ_DEBUG.md**
   - 详细诊断报告

## 🎓 技术细节

### ZMQ PUB-SUB 模式特性

- **发布者（pico_server）**: 不等待订阅者，持续发送数据
- **订阅者（Isaac Lab）**: 连接后需要时间建立订阅（通常 < 1 秒）
- **消息丢失**: 订阅建立前的消息会丢失（"slow joiner" 问题）

### 解决方案

1. **启动顺序**: 先启动发布者，后启动订阅者
2. **订阅延迟**: 订阅者连接后等待 1 秒再开始接收
3. **调试日志**: 详细记录连接和数据接收状态

### 数据流时序

```
T0: pico_server 启动，开始发送数据（50 Hz）
T1: Isaac Lab 启动
T2: ZMQPoller 创建，开始连接
T3: sleep(1.0) - 等待订阅建立
T4: 开始调用 get_data()，应该能收到数据
T5: 填充 SMPL 历史缓冲区（10 帧，约 0.2 秒）
T6: 开始 SONIC 推理，机器人跟随 VR 输入
```

## ✅ 验证清单

在重新运行前，确认：

- [ ] pico_server_pose_only.py 正在运行
- [ ] 看到 "Body data available!" 消息
- [ ] VR 头显、手腕控制器、脚踝 tracker 都正常工作
- [ ] 端口 5556 正在监听（`netstat -tuln | grep 5556`）
- [ ] GR00T-WholeBodyControl 路径正确（`ls /home/dreams/Users/taowen/GR00T-WholeBodyControl`）

然后运行：
```bash
bash run_sonic.sh
```

## 🎉 预期结果

- ✅ 启动时看到 "ZMQ connected successfully"
- ✅ 启动时看到 "ZMQ subscription ready"
- ✅ 运行时看到 "Received raw data"
- ✅ 运行时看到 "SMPL data marked as VALID"
- ✅ `smpl_history_fill` 从 0 增加到 10
- ✅ 第 10 帧后开始 SONIC 推理
- ✅ 机器人跟随 VR 头显动作

---

**修复完成时间**: 2026-03-21
**修复内容**: GR00T 路径错误 + ZMQ slow joiner 问题
**状态**: ✅ 已修复，待测试验证
