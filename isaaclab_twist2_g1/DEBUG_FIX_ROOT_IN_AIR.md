# Debug Mode: Fix Root in Air

## 功能说明

为了方便调试PID参数和观察遥控操作时的跟踪效果，添加了一个调试模式，可以将G1机器人的root固定在空中，防止机器人摔倒。

## 使用方法

### 启用/禁用

在 `action_provider/action_provider_wh_twist2.py` 文件中，找到第482行：

```python
self._debug_fix_root_in_air = True  # 设置为 True 启用，False 禁用
```

### 调整固定高度

在第485行可以调整机器人悬浮的高度（默认0.9米）：

```python
self._debug_fixed_root_position = torch.tensor([0.0, 0.0, 0.9], device=self.env.device, dtype=torch.float32)
#                                                 x    y    z (单位：米)
```

### 调整固定姿态

在第487行可以调整机器人的姿态（默认为直立，四元数 [w, x, y, z] = [1, 0, 0, 0]）：

```python
self._debug_fixed_root_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
#                                                    w    x    y    z
```

## 工作原理

1. 在每个physics step之后，强制设置机器人的root状态
2. 固定位置到指定的 [x, y, z] 坐标
3. 固定姿态到指定的四元数
4. 将所有速度（线速度和角速度）清零，防止漂移

## 效果

- ✅ 机器人悬浮在空中，不会接触地面
- ✅ 不会因为失去平衡而摔倒
- ✅ 可以清楚观察关节的跟踪效果
- ✅ 方便调试PID参数，无需反复重启

## 注意事项

⚠️ **这是调试模式，仅用于PID调试和遥控跟踪效果观察**

- 调试完成后，记得将 `self._debug_fix_root_in_air` 设置为 `False`
- 正常训练和测试时应该禁用此功能
- 此模式下机器人不会受到重力和地面接触力的影响

## 启动提示

启用此功能后，启动时会看到以下提示：

```
[DDSActionProvider] 🔧 DEBUG MODE: Root fixed in air at z=0.90m
[DDSActionProvider] 🔧 This prevents falling during PID tuning and teleop debugging
[DDSActionProvider] 🔧 Set self._debug_fix_root_in_air = False to disable
```
