# SONIC KDKP 最终修复 - 使用C++部署参数

## 🎯 问题根源

经过深入分析SONIC pipeline，发现**KDKP参数不匹配**是根本原因：

### **SONIC有两套PD参数！**

1. **训练时（MuJoCo仿真）**: `g1_29dof_sonic_model12.yaml`
   - hip_pitch: Kp=150, Kd=2
   - waist_yaw: Kp=250, Kd=5
   - shoulder: Kp=100, Kd=5

2. **部署时（C++/真机）**: `policy_parameters.hpp`（基于armature计算）
   - hip_pitch: Kp=99.0, Kd=3.16
   - waist_yaw: Kp=40.2, Kd=1.28
   - shoulder: Kp=14.2, Kd=0.45

**关键发现**: `action_scale`是基于**部署时的Kp**计算的！

```cpp
// policy_parameters.hpp:27
action_scale = 0.25 × effort_limit / stiffness

// 示例:
hip_pitch_scale = 0.25 × 139.0 / 99.0 = 0.3507  // 基于Kp=99.0
```

如果Isaac Lab使用Kp=150，但action_scale仍用0.3507（基于Kp=99），则：
- 实际动作幅度 = raw_action × 0.3507
- 但PD控制器期望的幅度 = raw_action × (0.25 × 139.0 / 150.0) = raw_action × 0.2317
- **动作幅度放大了1.5倍！**

---

## 🔧 最终修复方案

### **使用C++ SONIC部署时的Kp/Kd**

**文件**: `robots/unitree.py` (Line 853-973)

### 1. **Stiffness计算公式**

```python
# C++ policy_parameters.hpp:49-53
NATURAL_FREQ = 10 Hz × 2π = 62.83 rad/s
stiffness = armature × NATURAL_FREQ²

# 电机类型:
STIFFNESS_7520_22 = 0.025101925 × 62.83² = 99.0
STIFFNESS_7520_14 = 0.010177520 × 62.83² = 40.2
STIFFNESS_5020 = 0.003609725 × 62.83² = 14.2
STIFFNESS_4010 = 0.00425 × 62.83² = 16.7
```

### 2. **Damping计算公式**

```python
# C++ policy_parameters.hpp:55-59
DAMPING_RATIO = 2.0
damping = 2 × DAMPING_RATIO × armature × NATURAL_FREQ

# 电机类型:
DAMPING_7520_22 = 2 × 2.0 × 0.025101925 × 62.83 = 3.16
DAMPING_7520_14 = 2 × 2.0 × 0.010177520 × 62.83 = 1.28
DAMPING_5020 = 2 × 2.0 × 0.003609725 × 62.83 = 0.45
DAMPING_4010 = 2 × 2.0 × 0.00425 × 62.83 = 0.54
```

### 3. **完整参数表**

| 关节组 | 电机类型 | Kp | Kd | Effort Limit |
|--------|---------|----|----|--------------|
| hip_pitch/roll | 7520_22 | 99.0 | 3.16 | 139.0 |
| hip_yaw | 7520_14 | 40.2 | 1.28 | 88.0 |
| knee | 7520_22 | 99.0 | 3.16 | 139.0 |
| ankle_pitch/roll | 5020 × 2 | 28.4 | 0.91 | 25.0 |
| waist_yaw | 7520_14 | 40.2 | 1.28 | 88.0 |
| waist_roll/pitch | 5020 × 2 | 28.4 | 0.91 | 25.0 |
| shoulder_* | 5020 | 14.2 | 0.45 | 25.0 |
| elbow | 5020 | 14.2 | 0.45 | 25.0 |
| wrist_roll | 5020 | 14.2 | 0.45 | 25.0 |
| wrist_pitch/yaw | 4010 | 16.7 | 0.54 | 5.0 |

---

## 📊 修改对比

### **Legs（腿部）**

| 参数 | 之前（MuJoCo YAML） | 现在（C++ 部署） | 变化 |
|------|-------------------|----------------|------|
| hip Kp | 150.0 | **99.0** | -34% |
| hip Kd | 2.0 | **3.16** | +58% |
| hip_yaw Kp | 150.0 | **40.2** | -73% |
| hip_yaw Kd | 2.0 | **1.28** | -36% |
| knee Kp | 200.0 | **99.0** | -51% |
| knee Kd | 4.0 | **3.16** | -21% |

### **Waist（腰部）**

| 参数 | 之前（MuJoCo YAML） | 现在（C++ 部署） | 变化 |
|------|-------------------|----------------|------|
| yaw Kp | 250.0 | **40.2** | -84% |
| yaw Kd | 5.0 | **1.28** | -74% |
| roll/pitch Kp | 250.0 | **28.4** | -89% |
| roll/pitch Kd | 5.0 | **0.91** | -82% |

### **Arms（手臂）**

| 参数 | 之前（MuJoCo YAML） | 现在（C++ 部署） | 变化 |
|------|-------------------|----------------|------|
| shoulder Kp | 100.0 | **14.2** | -86% |
| shoulder Kd | 5.0 | **0.45** | -91% |
| elbow Kp | 40.0 | **14.2** | -64% |
| elbow Kd | 2.0 | **0.45** | -78% |
| wrist Kp | 20.0 | **14.2/16.7** | -29%/+17% |
| wrist Kd | 2.0 | **0.45/0.54** | -78%/-73% |

---

## 🎯 为什么这样修改？

### 1. **Action Scale一致性**

**C++ action_scale公式**（policy_parameters.hpp:107）:
```cpp
action_scale = 0.25 × effort_limit / stiffness
```

**当前Isaac Lab的action_scale**（action_provider_sonic.py:650-680）:
```python
G1_ACTION_SCALE_ISAACLAB = np.array([
    0.3506614566,  # hip_pitch: 0.25 × 139.0 / 99.0
    0.5475464463,  # hip_yaw: 0.25 × 88.0 / 40.2
    0.4385773242,  # ankle: 0.25 × 25.0 / 14.2
    0.0745008737,  # wrist_pitch/yaw: 0.25 × 5.0 / 16.7
])
```

这些值是基于**C++部署时的Kp**计算的！如果Isaac Lab用不同的Kp，动作幅度会不匹配。

### 2. **物理意义**

**更低的Kp（14.2 vs 100）**:
- 更柔软的控制
- 更大的动作幅度（因为action_scale更大）
- 更符合真机的柔顺性

**更低的Kd（0.45 vs 5.0）**:
- 更少的阻尼
- 更自然的动作
- 减少过度抑制

### 3. **训练vs部署的差异**

SONIC策略在MuJoCo中训练时：
- 使用高Kp（150/250/100）→ 硬控制
- 学习到的动作幅度较小

部署到真机时：
- 使用低Kp（99/40/14）→ 软控制
- 通过action_scale补偿，保持相同的实际扭矩

**Isaac Lab必须匹配部署参数，否则动作幅度错误！**

---

## ⚠️ 其他发现

### 1. **NO Action Smoothing**

SONIC C++代码中**没有任何动作平滑**：
- ❌ NO exponential moving average
- ❌ NO low-pass filter
- ❌ NO temporal smoothing

所有平滑来自：
- PID控制器的Kd阻尼
- 策略训练时学到的时间一致性

### 2. **Isaac Lab的额外Clipping**

**C++ SONIC**: 无clipping
**Isaac Lab**: 两次clipping
```python
raw_sonic = np.clip(raw_sonic, -2.0, 2.0)  # Line 1540
target_sonic = np.clip(target_sonic, -3.0, 3.0)  # Line 1558
```

**建议**: 移除这些clipping，与C++保持一致。

---

## 📋 修改总结

| 文件 | 修改位置 | 状态 |
|------|---------|------|
| `robots/unitree.py` | Line 853-973 | ✅ 已修复 |
| `SONIC_PIPELINE_ANALYSIS.md` | 新文档 | ✅ 已创建 |
| `SONIC_KDKP_FINAL_FIX.md` | 新文档 | ✅ 已创建 |

---

## 🧪 测试验证

```bash
# Terminal 1: pico_server
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl

# Terminal 2: Isaac Lab (with C++ deployment KDKP)
bash run_sonic.sh
```

**预期效果**:
1. ✅ 动作幅度与C++ SONIC一致
2. ✅ 更柔顺的控制（低Kp）
3. ✅ 更自然的动作（低Kd）
4. ✅ 与action_scale完美匹配

---

**修复时间**: 2026-03-21
**状态**: ✅ 已完成，使用C++部署参数
**关键**: action_scale必须与Kp匹配！
