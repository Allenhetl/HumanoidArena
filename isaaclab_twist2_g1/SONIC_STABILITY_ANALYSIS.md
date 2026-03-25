# SONIC 稳定性问题分析 - Kp/Kd参数的训练vs部署差异

## 🚨 核心问题

用户反馈：使用C++ SONIC部署参数（低Kp）后，Isaac Lab中G1**更容易摔倒**，尽管动作更柔顺。

---

## 🔍 深入分析

### 1. **SONIC的两套参数体系**

#### **训练时（MuJoCo仿真）**
```yaml
# g1_29dof_sonic_model12.yaml
MOTOR_KP: [150, 150, 150, 200, 40, 40, ...]  # 高Kp，硬控制
MOTOR_KD: [2, 2, 2, 4, 2, 2, ...]
SIMULATE_DT: 0.005  # 200 Hz
```

#### **部署时（C++真机/MuJoCo）**
```cpp
// policy_parameters.hpp
stiffness = armature × (10Hz × 2π)²
STIFFNESS_7520_22 = 99.0   // hip/knee
STIFFNESS_7520_14 = 40.2   // hip_yaw/waist_yaw
STIFFNESS_5020 = 14.2      // shoulder/elbow
control_dt = 0.02  // 50 Hz
```

### 2. **Kp差异对比**

| 关节组 | 训练Kp | 部署Kp | 降低幅度 | Action Scale | 实际扭矩比率 |
|--------|--------|--------|----------|--------------|--------------|
| hip | 150.0 | 99.0 | -34% | 0.351 | 23.17% |
| knee | 200.0 | 99.0 | -51% | 0.351 | 17.38% |
| **waist** | **250.0** | **40.2** | **-84%** | 0.547 | **8.80%** |
| **shoulder** | **100.0** | **14.2** | **-86%** | 0.440 | **6.25%** |
| ankle | 40.0 | 28.4 | -29% | 0.220 | 15.62% |

**关键发现**：
- 腰部Kp降低84%（250 → 40.2）
- 肩部Kp降低86%（100 → 14.2）
- Action scale补偿了部分差异，但**实际扭矩输出仍远低于训练时**

---

## 🎯 为什么SONIC C++能用低Kp？

### **假设1: MuJoCo的PD控制器实现更稳定**

MuJoCo可能有：
- 更好的数值积分稳定性
- 隐式的阻尼或摩擦补偿
- 更精确的接触力计算

### **假设2: 真机有额外的稳定机制**

真实G1机器人可能有：
- 硬件级的力矩限制和保护
- 更复杂的摩擦模型
- 机械结构的自然阻尼

### **假设3: Action scale的补偿作用**

```python
# 训练时: 高Kp，小action幅度
torque_train = Kp_train × action_small

# 部署时: 低Kp，大action幅度（通过action_scale）
torque_deploy = Kp_deploy × (action_small × action_scale)
                = Kp_deploy × action_small × (0.25 × effort / Kp_deploy)
                = 0.25 × effort × action_small
```

**理论上扭矩相同**，但**动态响应不同**：
- 低Kp → 更大的位置误差 → 更慢的收敛
- 高Kp → 更小的位置误差 → 更快的收敛

---

## ⚠️ Isaac Lab中的问题

### **问题1: 低Kp导致跟踪不足**

在动态运动中：
```
期望位置: q_des(t)
实际位置: q(t)
误差: e(t) = q_des(t) - q(t)
扭矩: τ = Kp × e(t) - Kd × dq(t)
```

**低Kp的影响**：
- 相同误差 → 更小的恢复扭矩
- 在快速运动中 → 累积误差更大
- 腰部/肩部 → 无法维持姿态稳定

### **问题2: Isaac Lab的物理引擎差异**

Isaac Lab (PhysX) vs MuJoCo:
- 接触力模型不同
- 摩擦系数处理不同
- 数值积分方法不同
- 可能对低Kp更敏感

### **问题3: 频率匹配但响应不同**

虽然都是：
- sim_dt = 0.005s (200 Hz)
- control_dt = 0.02s (50 Hz)
- decimation = 4

但**PD控制器的实际行为可能不同**：
- MuJoCo: 可能有内部的力矩平滑
- Isaac Lab: 可能是纯粹的PD控制

---

## 🔧 可能的解决方案

### **方案1: 使用训练时的Kp（推荐尝试）**

**理由**：
- 策略在高Kp环境下训练
- 学到的动作幅度适配高Kp
- 可能在Isaac Lab中更稳定

**需要调整action_scale**：
```python
# 重新计算action_scale以匹配训练Kp
G1_ACTION_SCALE_ISAACLAB_TRAIN = np.array([
    0.25 * 139.0 / 150.0,  # hip: 0.2317 (was 0.3507)
    0.25 * 139.0 / 200.0,  # knee: 0.1738 (was 0.3507)
    0.25 * 88.0 / 250.0,   # waist: 0.0880 (was 0.5475)
    0.25 * 25.0 / 100.0,   # shoulder: 0.0625 (was 0.4386)
    0.25 * 25.0 / 40.0,    # ankle: 0.1562 (was 0.4386)
])
```

**权衡**：
- ✅ 更稳定的姿态控制
- ✅ 更快的响应速度
- ❌ 动作幅度变小（因为action_scale变小）
- ❌ 可能与训练数据分布不完全匹配

### **方案2: 混合方案 - 关键关节用高Kp**

只对容易失稳的关节提高Kp：
```python
stiffness={
    # 腿部：保持中等Kp（支撑稳定性）
    ".*_hip_pitch_joint": 120.0,  # 介于99和150之间
    ".*_knee_joint": 120.0,

    # 腰部：提高Kp（躯干稳定性）
    "waist_yaw_joint": 150.0,     # 从40.2提高到150
    "waist_roll_joint": 100.0,    # 从28.4提高到100
    "waist_pitch_joint": 100.0,

    # 手臂：保持低Kp（柔顺性）
    ".*_shoulder_.*_joint": 14.2,
    ".*_elbow_joint": 14.2,
}
```

**需要部分调整action_scale**（只调整改变Kp的关节）

### **方案3: 增加Kd阻尼**

保持低Kp，但增加Kd以提高稳定性：
```python
damping={
    ".*_hip_pitch_joint": 5.0,    # 从3.16提高到5.0
    ".*_knee_joint": 5.0,
    "waist_yaw_joint": 3.0,       # 从1.28提高到3.0
    "waist_roll_joint": 2.0,      # 从0.91提高到2.0
    ".*_shoulder_.*_joint": 1.0,  # 从0.45提高到1.0
}
```

**优点**：
- 不改变action_scale
- 增加阻尼减少震荡
- 可能提高稳定性

**缺点**：
- 可能使动作过于僵硬
- 与训练环境偏离

### **方案4: 调整sim_dt和decimation**

尝试更高的控制频率：
```python
self.sim.dt = 0.0025  # 400 Hz (更精细的仿真)
self.decimation = 8   # 控制频率仍为50Hz
```

**理由**：
- 更高的仿真频率 → 更稳定的数值积分
- 可能减少低Kp带来的不稳定性

---

## 📊 测试建议

### **测试1: 使用训练Kp + 调整action_scale**

```bash
# 1. 修改 robots/unitree.py 使用训练Kp
# 2. 修改 action_provider_sonic.py 重新计算action_scale
# 3. 测试稳定性
bash run_sonic.sh
```

### **测试2: 混合方案（腰部高Kp）**

```bash
# 只提高腰部Kp到150，其他保持不变
# 部分调整action_scale
bash run_sonic.sh
```

### **测试3: 增加Kd阻尼**

```bash
# 保持当前Kp，增加Kd到训练值
bash run_sonic.sh
```

---

## 🎯 推荐方案

**优先尝试方案2（混合方案）**：

1. **腰部和髋部用训练Kp**（稳定性关键）
2. **手臂保持低Kp**（柔顺性）
3. **部分调整action_scale**

**理由**：
- 平衡稳定性和柔顺性
- 最小化对action_scale的改动
- 针对性解决摔倒问题（主要是躯干不稳）

---

**文档创建时间**: 2026-03-21
**状态**: ⚠️ 需要测试验证不同方案
**下一步**: 实现混合方案并测试
