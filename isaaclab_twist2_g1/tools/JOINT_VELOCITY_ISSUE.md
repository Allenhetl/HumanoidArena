## 关节速度异常问题 - 根因与解决方案

### 问题现象

Encoder输入范围异常：`[-31.9, 29.8]`，远超正常范围`[-10, 10]`

### 根本原因

**PD控制器阻尼不足，导致yaw关节震荡**

诊断输出显示：
```
⚠️  WARNING: Abnormal joint velocity detected!
  Max velocity: 32.00 rad/s at joint 6 (left_hip_yaw_joint)
  Top 5 fastest joints:
    left_hip_yaw_joint:  32.00 rad/s
    waist_yaw_joint:     27.58 rad/s
    right_hip_yaw_joint: 24.51 rad/s
```

当前PD参数（`assets/robots/g1-29dof_wholebody_dex1/config.yaml`）：
- **Stiffness: 100.0**
- **Damping: 1.0**
- **比例: 100:1** ← 严重欠阻尼！

### 为什么会震荡？

PD控制器的运动方程：
```
τ = Kp * (q_target - q) - Kd * q_dot
```

当Kd太小时：
1. 关节快速向目标移动（Kp驱动）
2. 超过目标位置（过冲）
3. 反向加速（Kp反向驱动）
4. 再次超过目标（震荡）
5. 循环往复，速度越来越大

### 临时解决方案（已实施）

在`action_provider_sonic.py`中clip速度到encoder的训练范围：

```python
self._robot_joint_vel_hist[-1] = np.clip(joint_vel_sonic, -10.0, 10.0)
```

**优点**：
- 立即生效，无需修改配置
- Encoder能正常工作
- 不影响仿真本身

**缺点**：
- 治标不治本
- 仿真中机器人仍然震荡
- 可能影响跟踪精度

### 长期解决方案（推荐）

**方案1：增加阻尼（推荐）**

修改`assets/robots/g1-29dof_wholebody_dex1/config.yaml`：

```yaml
gains:
  stiffness: 100.0
  damping: 15.0  # 从1.0增加到15.0
```

临界阻尼：`damping = 2 * sqrt(stiffness) = 20`
推荐值：`damping = 15`（略微欠阻尼，保持响应速度）

**方案2：降低刚度**

```yaml
gains:
  stiffness: 50.0   # 从100.0降低到50.0
  damping: 10.0     # 相应调整
```

**方案3：针对yaw关节单独设置**

如果Isaac Lab支持per-joint PD参数，可以只增加yaw关节的阻尼：

```yaml
joint_specific_gains:
  left_hip_yaw_joint:
    stiffness: 100.0
    damping: 20.0
  waist_yaw_joint:
    stiffness: 100.0
    damping: 20.0
  right_hip_yaw_joint:
    stiffness: 100.0
    damping: 20.0
```

### 验证方法

修改PD参数后，重启IsaacLab，检查：

1. **速度警告消失**
   ```
   ⚠️  WARNING: Abnormal joint velocity detected!
   ```
   这个警告应该不再出现

2. **Encoder输入范围正常**
   ```
   [DIAGNOSTIC] Encoder input field ranges:
     motion_joint_vel_step5: [-10.0, 10.0]  # 应该在这个范围内
   ```

3. **机器人运动平滑**
   - 在仿真中观察机器人
   - 不应该有明显的抖动或震荡
   - 跟随VR动作应该更平滑

### 相关文件

- 临时修复：`action_provider/action_provider_sonic.py` (第844-850行)
- PD参数配置：`assets/robots/g1-29dof_wholebody_dex1/config.yaml`
- 诊断输出：第825-838行

### 参考

- PD控制器理论：https://en.wikipedia.org/wiki/PID_controller
- 临界阻尼计算：`ζ = Kd / (2 * sqrt(Kp * m))`，对于单位质量，`Kd_critical = 2 * sqrt(Kp)`
- Isaac Lab文档：关于articulation PD gains的配置
