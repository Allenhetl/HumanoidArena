# SONIC动作不正确的完整排查清单

根据代码分析，数据流是正确的。问题可能出在以下几个方面：

## ✅ 已确认正确的部分

1. **SMPL数据转换** - Pico server正确将全局坐标转为局部坐标
2. **Encoder输入** - 1762维，包含正确的SMPL joints和anchor orientation
3. **Decoder输入** - 994维，使用IsaacLab仿真中的实际机器人状态（819-822行）
4. **历史缓冲更新** - 每帧都在更新（不是只在warmup时）

## ⚠️ 需要检查的关键点

### 1. Anchor朝向对齐（最可能的问题！）

**位置**：`action_provider_sonic.py:658-723`

**问题描述**：
- 机器人的base朝向和VR的body朝向可能不一致
- 如果初始对齐不正确，所有动作都会有方向偏差

**检查方法**：
在IsaacLab日志中查找：
```
[ZMQ][ANCHOR_INIT] raw_init_angle_deg=XX.XX aligned_init_angle_deg=XX.XX use_heading_align=True/False
```

**判断标准**：
- 如果`raw_init_angle_deg > 45度`，说明初始朝向差异很大
- 如果`use_heading_align=False`，可能导致动作方向错误

**解决方法**：
```python
# 在action_provider_sonic.py:676行，强制使用heading对齐
self._anchor_use_heading_align = True  # 原来是根据角度自动判断
```

### 2. 动作缩放系数

**位置**：`action_provider_sonic.py:284-314`

**问题描述**：
- `G1_ACTION_SCALE_ISAACLAB`定义了每个关节的缩放系数
- 如果缩放不当，动作幅度会过大或过小

**检查方法**：
在IsaacLab日志中查找：
```
[SONIC] Action range: [-X.XX, X.XX]
[SONIC] sonic_targets range: [-X.XX, X.XX]
```

**判断标准**：
- Action应该在[-1, +1]范围内
- sonic_targets应该在[-2, +2]范围内
- 如果超出范围，说明缩放有问题

### 3. 动作平滑延迟

**位置**：`action_provider_sonic.py:385`

**问题描述**：
```python
self._sonic_smooth_steps = int(getattr(args_cli, "sonic_smooth_steps", 20))
```
- 默认20步平滑，在50Hz下是0.4秒延迟
- 加上历史缓冲10帧（0.2秒），总延迟0.6秒

**检查方法**：
在VR中做一个快速动作（比如快速举手），观察机器人响应时间

**解决方法**：
```bash
# 在run_sonic.sh中添加参数
--sonic_smooth_steps 5  # 减少到5步，降低延迟
```

### 4. PD控制器参数

**位置**：任务配置文件中的`stiffness`和`damping`

**问题描述**：
- 如果stiffness太小，机器人响应慢
- 如果damping太大，机器人动作被抑制

**检查方法**：
查看任务配置文件（例如`Isaac-Move-Cylinder-G129-Dex3-Wholebody`）

**典型值**：
```python
stiffness = 100.0  # 对于大关节（髋、膝）
stiffness = 50.0   # 对于小关节（手腕）
damping = 5.0      # 一般是stiffness的5-10%
```

### 5. 关节限位

**位置**：URDF文件中的joint limits

**问题描述**：
- 如果目标位置超出关节限位，会被截断
- 导致动作不完整

**检查方法**：
```python
# 在action_provider_sonic.py的compute_actions()中添加
print(f"[DEBUG] targets before clamp: {sonic_targets[:5]}")
print(f"[DEBUG] joint limits: {robot.joint_pos_limits[0, self._sonic_idx[:5]]}")
```

### 6. SMPL数据质量

**位置**：Pico VR追踪

**问题描述**：
- 5个追踪点（头+双手腕+双脚踝）中任何一个失效都会影响SMPL拟合
- 遮挡、光照、反光都可能导致追踪失败

**检查方法**：
运行`monitor_sonic_live.py`，观察：
```
SMPL Joints 变化幅度: X.XXXX
```
- 应该>0.01，否则说明追踪数据无效

**解决方法**：
- 确保5个追踪点都在VR视野内
- 避免遮挡（特别是脚踝tracker）
- 穿紧身衣以保证tracker稳定

### 7. Encoder/Decoder模型版本

**位置**：模型文件路径

**问题描述**：
- 如果使用了错误版本的模型，输出会不正确
- SMPL mode (mode 2)需要特定训练的模型

**检查方法**：
```bash
# 确认模型文件日期和大小
ls -lh /home/dreams/Users/Alyssa/HumanoidArena_V1/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_*.onnx
```

**预期**：
- model_encoder.onnx: ~10-20MB
- model_decoder.onnx: ~5-10MB
- 文件日期应该是最新的release版本

## 🔍 系统化调试流程

### 步骤1：验证数据接收（30秒）
```bash
python tools/monitor_sonic_live.py
# 在VR中举右手，看数值是否变化
```

### 步骤2：检查Anchor对齐（查看日志）
```
grep "ANCHOR_INIT" sonic_debug.log
# 检查raw_init_angle_deg是否合理
```

### 步骤3：检查动作输出范围（查看日志）
```
grep "sonic_targets range" sonic_debug.log
# 应该看到数值随时间变化，不是固定值
```

### 步骤4：添加详细调试日志

在`action_provider_sonic.py:930`（encoder推理后）添加：
```python
print(f"[DEBUG] Latent变化: {np.abs(latent - self._latent_prev).sum() if hasattr(self, '_latent_prev') else 0:.4f}")
self._latent_prev = latent.copy()
```

在`action_provider_sonic.py:960`（decoder推理后）添加：
```python
print(f"[DEBUG] Action变化: {np.abs(action_raw - self._action_prev).sum() if hasattr(self, '_action_prev') else 0:.4f}")
self._action_prev = action_raw.copy()
print(f"[DEBUG] Robot joint_pos前5维: {self._robot_joint_pos_hist[-1, :5]}")
print(f"[DEBUG] Robot joint_vel前5维: {self._robot_joint_vel_hist[-1, :5]}")
```

### 步骤5：对比VR动作和机器人动作

**测试动作**：在VR中举右手到头顶，保持5秒

**预期日志**：
```
[DEBUG] Latent变化: 0.5234  ← 应该>0.1
[DEBUG] Action变化: 0.3456  ← 应该>0.05
[SONIC] sonic_targets range: [-0.8, 1.2]  ← 右肩pitch应该增大
```

**如果Latent变化<0.01**：
- 问题在Encoder输入
- 检查SMPL joints是否变化

**如果Action变化<0.01**：
- 问题在Decoder输入
- 检查robot joint_pos/vel是否正确

**如果Action变化正常但机器人不动**：
- 问题在PD控制器或关节限位
- 检查stiffness/damping参数

## 🎯 最可能的原因排序

根据"有动作但不按照行为执行"的描述，最可能的原因是：

1. **Anchor朝向对齐问题**（70%概率）
   - 症状：机器人有动作，但方向不对
   - 解决：强制启用heading对齐

2. **动作平滑延迟过大**（15%概率）
   - 症状：机器人动作滞后，跟不上VR
   - 解决：减小smooth_steps到5

3. **PD控制器参数不当**（10%概率）
   - 症状：机器人动作幅度小或响应慢
   - 解决：增大stiffness

4. **SMPL追踪质量差**（5%概率）
   - 症状：机器人动作断断续续
   - 解决：改善追踪环境

## 📝 快速修复建议

**立即尝试**：

1. 在`action_provider_sonic.py:676`行，强制启用heading对齐：
```python
self._anchor_use_heading_align = True
```

2. 在`run_sonic.sh`中减小平滑步数：
```bash
--sonic_smooth_steps 5
```

3. 重启IsaacLab，在VR中做一个大幅度、持续的动作（举手5秒）

4. 观察日志中的`[ZMQ][ANCHOR]`和`[SONIC] sonic_targets range`

如果修改后仍然不正确，请提供：
- 完整的IsaacLab启动日志（前200行）
- 做动作时的日志（标注具体动作）
- `monitor_sonic_live.py`的输出
