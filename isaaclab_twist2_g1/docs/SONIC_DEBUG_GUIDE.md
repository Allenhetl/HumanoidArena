# SONIC调试完全指南

当IsaacLab中的机器人动作与VR动作不一致时，按照以下步骤系统化排查。

## 快速诊断流程

### 步骤1：验证Pico端数据发送（2分钟）

```bash
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tools
python monitor_sonic_live.py
```

**在VR中做一个明显动作（举右手），观察输出：**

✅ **正常情况**：
```
SMPL Joints 变化幅度: 0.1234  范围: [-0.5, 0.8]
Joint Pos 变化幅度:   0.2345  范围: [-1.2, 1.5]
右肩pitch [12]: +0.523  ← 举手时这个值应该变大
右肘      [22]: +0.234
```

❌ **异常情况A - 数据不变**：
```
SMPL Joints 变化幅度: 0.0001  ← 几乎为0
⚠️  SMPL数据几乎不变 - 请在VR中做动作！
```
**原因**：Pico追踪失败
**解决**：
1. 检查5个追踪点（头+双手腕+双脚踝）是否都亮绿灯
2. 重启`pico_manager_thread_server.py`
3. 确认`roboticsservice`正在运行

❌ **异常情况B - 连接失败**：
```
zmq.error.Again: Resource temporarily unavailable
```
**原因**：pico_server未运行或端口错误
**解决**：
```bash
# 检查pico_server是否运行
ps aux | grep pico_manager

# 如果没有，启动它
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/pico_server
python pico_manager_thread_server.py --manager --port 5556
```

---

### 步骤2：验证完整数据流（5分钟）

```bash
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tools
python diagnose_sonic_dataflow.py
```

这个脚本会测试7个关键环节：

1. ✅ ZMQ数据接收
2. ✅ SMPL数据解析
3. ✅ Encoder输入构建（1762维）
4. ✅ Encoder推理（输出64维latent）
5. ✅ Decoder输入构建（994维）
6. ✅ Decoder推理（输出29维action）
7. ✅ 输出后处理（缩放+默认姿态）

**如果某个环节失败，会明确指出问题所在。**

常见问题：

❌ **Encoder模型不存在**：
```
❌ Encoder模型不存在: /path/to/model_encoder.onnx
```
**解决**：检查模型路径，修改`action_provider_sonic.py:381-382`

❌ **维度不匹配**：
```
❌ Encoder输入: (1, 1500), 期望: (1, 1762)
```
**解决**：SMPL数据格式错误，检查pico_server版本

❌ **Encoder推理失败**：
```
❌ Encoder推理失败: CUDA out of memory
```
**解决**：GPU内存不足，关闭其他程序或使用CPU推理

---

### 步骤3：检查IsaacLab端日志（关键！）

启动IsaacLab时，查看终端输出，重点关注以下日志：

#### 3.1 初始化日志

```
[SonicActionProvider] POSE mode ready  zmq=localhost:5556
encoder=/path/to/model_encoder.onnx  decoder=/path/to/model_decoder.onnx
```
✅ 确认模型路径正确

#### 3.2 数据接收日志（每帧）

```
[ZMQ] Received data keys: ['smpl_joints', 'smpl_pose', 'body_quat_w', 'joint_pos', ...]
[ZMQ] smpl_joints shape: (5, 24, 3), latest frame sum: 12.3456
[ZMQ] SMPL data marked as VALID  ← 必须看到这个！
```

❌ **如果看到**：
```
[ZMQ] smpl_joints latest frame sum: 0.0001
```
说明SMPL数据无效，回到步骤1检查Pico端。

#### 3.3 Anchor对齐日志（初始化时）

```
[ZMQ][ANCHOR_INIT] init_base=[1.0, 0.0, 0.0, 0.0] init_ref=[0.98, 0.0, 0.15, 0.0]
raw_init_angle_deg=17.23 aligned_init_angle_deg=8.45 use_heading_align=True
```

这个日志显示机器人朝向与VR朝向的对齐方式。

**关键参数**：
- `use_heading_align=True`：使用heading对齐（推荐）
- `raw_init_angle_deg`：初始角度差
- 如果角度差>45度，可能需要重新校准VR

#### 3.4 Encoder输入日志（每25帧）

```
[SONIC] Encoder input shape: (1, 1762), expected: (1, 1762)
[SONIC] SMPL joints sum: 45.6789  ← 应该>1.0
[SONIC][SMPL_MODE] encoder_mode_vec=[0, 0, 1, 0]  ← mode 2
```

❌ **如果SMPL joints sum < 0.1**：
说明SMPL数据未正确传递到encoder，检查历史缓冲是否填满。

#### 3.5 Encoder/Decoder推理日志

```
[SONIC] Running encoder inference...
[SONIC] ✓ Encoder output latent shape: (1, 64)
[SONIC] Latent range: [-2.3456, 3.4567]

[SONIC] Running decoder inference...
[SONIC] ✓ Decoder output action shape: (1, 29)
[SONIC] Action range: [-0.8, 0.9]
```

✅ **正常范围**：
- Latent: [-5, +5]
- Action: [-1, +1]

❌ **异常情况 - 输出全0或固定值**：
```
[SONIC] Action range: [0.0, 0.0]  ← 全0
```
**原因**：
1. 历史缓冲未填满（需要等待10帧）
2. Encoder输入全0
3. 模型文件损坏

#### 3.6 最终输出日志

```
[SONIC] sonic_targets range: [-0.5, 0.8]
[SONIC] sonic_targets[0:10]: [-0.312, -0.312, 0.0, ...]
```

**关键检查**：
- 数值范围应该在[-2, +2]之间
- 数值应该**随时间变化**，不是固定值
- 在VR中做动作时，这些数值应该明显变化

❌ **如果输出固定不变**：
```
[SONIC] sonic_targets[0:10]: [-0.312, -0.312, 0.0, ...]  ← 每帧都一样
```
说明decoder输出有问题，检查decoder输入的历史缓冲。

---

### 步骤4：对比VR动作与机器人动作

在VR中做以下标准动作，观察机器人响应：

#### 测试1：举右手
**VR动作**：右手从身体侧面举到头顶
**预期机器人动作**：
- 右肩pitch增大（joint[12]从0.0→+0.8）
- 右肩roll增大（joint[16]从0.0→+0.5）
- 右肘可能弯曲（joint[22]变化）

**检查日志**：
```
[SONIC][WRIST_DIAG] ref_wrist=[...] sim_wrist=[...]
```
ref_wrist（来自ZMQ）应该变化，sim_wrist（机器人状态）应该跟随。

#### 测试2：蹲下
**VR动作**：身体下蹲
**预期机器人动作**：
- 髋关节pitch增大（joint[0,1]）
- 膝关节弯曲（joint[9,10]增大）
- 踝关节调整（joint[13,14]）

#### 测试3：转身
**VR动作**：身体左右转动
**预期机器人动作**：
- 腰部yaw旋转（joint[2]）

**检查日志**：
```
[ZMQ][ANCHOR] rel_angle_deg=XX.XX selected=aligned/raw
```
rel_angle_deg应该随转身变化。

---

## 常见问题诊断树

```
机器人没有动作
├─ 步骤1失败：Pico数据不变
│  ├─ 追踪点未连接 → 检查硬件
│  ├─ roboticsservice未运行 → 启动服务
│  └─ pico_server崩溃 → 查看pico_server日志
│
├─ 步骤2失败：某个环节报错
│  ├─ 模型文件不存在 → 检查路径
│  ├─ 维度不匹配 → 检查pico_server版本
│  └─ 推理失败 → 检查GPU/CUDA
│
├─ 步骤3：IsaacLab日志异常
│  ├─ SMPL data未标记VALID → 回到步骤1
│  ├─ Encoder输入全0 → 检查历史缓冲
│  ├─ Action输出全0 → 检查decoder输入
│  └─ Action输出固定值 → 等待历史填满（10帧）
│
└─ 步骤4：有动作但不对
   ├─ 动作方向相反 → 检查关节映射
   ├─ 动作幅度太小 → 调整action_scale
   ├─ 动作延迟严重 → 减小smooth_steps
   └─ 动作抖动 → 增大smooth_steps
```

---

## 高级调试：添加自定义日志

如果上述步骤都正常，但机器人仍然不动，可以添加更详细的日志：

### 在action_provider_sonic.py中添加：

```python
# 在_run_gear_sonic()函数的decoder推理后（约960行）
print(f"[DEBUG] decoder raw output: {action_raw[0, :10]}")  # 前10维
print(f"[DEBUG] action_scaled: {action_scaled[:10]}")
print(f"[DEBUG] sonic_targets: {sonic_targets[:10]}")
print(f"[DEBUG] current robot joint_pos: {robot.joint_pos[0, self._sonic_idx[:10]].cpu().numpy()}")
```

### 在compute_actions()函数中添加：

```python
# 在返回actions之前（约1050行）
print(f"[DEBUG] final actions sent to sim: {actions[0, :10].cpu().numpy()}")
print(f"[DEBUG] actions range: [{actions.min():.3f}, {actions.max():.3f}]")
```

这样可以看到从decoder输出到最终发送给仿真器的完整数据流。

---

## 性能优化建议

如果一切正常但响应慢：

1. **减少平滑步数**（action_provider_sonic.py:385）：
   ```python
   self._sonic_smooth_steps = 10  # 默认20，减小可降低延迟
   ```

2. **检查decimation**（sim_main.py中的--decimation参数）：
   ```bash
   # 默认4，表示每4个仿真步执行1次action
   # 减小可提高响应速度，但会增加计算负担
   ```

3. **使用CUDA加速**（确认encoder/decoder使用GPU）：
   ```python
   providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
   ```

4. **移除调试打印**（验证后注释掉所有print语句）

---

## 联系支持

如果按照上述步骤仍无法解决，请提供：

1. `monitor_sonic_live.py`的完整输出（至少10秒）
2. `diagnose_sonic_dataflow.py`的完整输出
3. IsaacLab启动后前100行日志
4. 在VR中做动作时的日志（标注具体做了什么动作）
5. GPU型号和CUDA版本

---

**最后提醒**：
- 确保等待至少1秒（10帧）让历史缓冲填满
- 在VR中做**大幅度、持续**的动作（不要只是轻微移动）
- 检查PD控制器参数（stiffness/damping）是否合理
- 如果使用replay模式测试过，确保切换回sonic_wholebody模式
