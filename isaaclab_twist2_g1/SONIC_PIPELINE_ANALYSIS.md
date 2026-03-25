# SONIC Pipeline 完整分析 - 动作处理流程检查

## 🔍 问题描述

KDKP参数调整后仍有问题，需要检查SONIC pipeline中是否缺少关键的动作处理步骤。

---

## 📊 SONIC C++ 原版动作处理流程

### 1. **动作缩放公式**（policy_parameters.hpp）

```cpp
// Line 27: Action scaling formula
action_scale = 0.25 × effort_limit / stiffness

// Line 109-139: Per-joint action scales (MuJoCo order)
const std::array<double, 29> g1_action_scale = {
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  // hip_pitch: 0.3507
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  // hip_roll: 0.3507
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  // hip_yaw: 0.5475
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  // knee: 0.3507
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        // ankle_pitch: 0.4386
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        // ankle_roll: 0.4386
    // ... (29 joints total)
};
```

**计算示例**:
- **Hip pitch**: `0.25 × 139.0 / 99.0 = 0.3507`
- **Hip yaw**: `0.25 × 88.0 / 40.2 = 0.5475`
- **Wrist pitch/yaw**: `0.25 × 5.0 / 16.7 = 0.0745`

### 2. **动作应用流程**（g1_deploy_onnx_ref.cpp:2803-2834）

```cpp
bool CreatePolicyCommand() {
  // Step 1: Policy inference
  if (!policy_engine_->Infer()) return false;

  // Step 2: Get raw actions from decoder
  auto& action_buffer = policy_engine_->GetActionBuffer();
  float* floatarr = action_buffer.data();

  // Step 3: Transform actions to motor commands
  MotorCommand motor_command_tmp;
  for (int i = 0; i < G1_NUM_MOTOR; i++) {
    // 3a. Remap from IsaacLab order to MuJoCo/hardware order
    int isaaclab_idx = isaaclab_to_mujoco[i];

    // 3b. Scale action
    const double action_value = static_cast<double>(floatarr[isaaclab_idx])
                                * g1_action_scale[i];

    // 3c. Add default offset
    motor_command_tmp.q_target.at(i) = static_cast<float>(
        default_angles[i] + action_value
    );

    // 3d. Set PD gains
    motor_command_tmp.kp.at(i) = kps[i];
    motor_command_tmp.kd.at(i) = kds[i];
    motor_command_tmp.tau_ff.at(i) = 0.0;
    motor_command_tmp.dq_target.at(i) = 0.0;

    // 3e. Store for history (in IsaacLab order)
    last_action[i] = static_cast<double>(floatarr[i]);
  }

  motor_command_buffer_.SetData(motor_command_tmp);
  return true;
}
```

**关键点**:
- ❌ **NO action smoothing/filtering**
- ❌ **NO action clipping** (except implicit joint limits)
- ✅ **Joint order remapping**: IsaacLab → MuJoCo
- ✅ **Per-joint scaling**: `action × action_scale`
- ✅ **Default offset**: `+ default_angles`

---

## 📊 Isaac Lab 当前实现对比

### 1. **动作缩放**（action_provider_sonic.py:650-680）

```python
G1_ACTION_SCALE_ISAACLAB = np.array([
    0.3506614566,  # 0: left_hip_pitch_joint
    0.3506614566,  # 1: right_hip_pitch_joint
    0.5475464463,  # 2: waist_yaw_joint
    0.3506614566,  # 3: left_hip_roll_joint
    # ... (29 joints, IsaacLab order)
], dtype=np.float32)
```

**问题**: 这些值是否与C++版本完全一致？需要验证！

### 2. **动作应用流程**（action_provider_sonic.py:1531-1572）

```python
# Step 1: Decoder inference
action_sonic = self._decoder.run(None, dec_inputs)[0]
raw_sonic = action_sonic.flatten()[:29]

# Step 2: Clip raw action (NOT in C++ version!)
raw_sonic = np.clip(raw_sonic, -2.0, 2.0)

# Step 3: Update action history (raw action, before scaling)
self._last_action_hist = np.roll(self._last_action_hist, -1, axis=0)
self._last_action_hist[-1] = raw_sonic

# Step 4: Scale and add default offset
target_sonic = raw_sonic * G1_ACTION_SCALE_ISAACLAB + self._sonic_default_np

# Step 5: Safety clip (NOT in C++ version!)
target_sonic = np.clip(target_sonic, -3.0, 3.0)

return target_sonic.astype(np.float32)
```

**差异**:
- ⚠️ **额外的clipping**: Isaac Lab有两次clip，C++版本没有
- ⚠️ **Joint order**: Isaac Lab直接使用IsaacLab order，C++需要remapping

---

## 🔍 关键发现

### 1. **NO Action Smoothing in SONIC**

**搜索结果**: 在SONIC的三个核心文件中，**没有找到任何动作平滑机制**：
- ❌ NO exponential moving average (EMA)
- ❌ NO low-pass filter
- ❌ NO temporal smoothing
- ❌ NO action interpolation

**唯一的"平滑"来自**:
- PID控制器的Kd阻尼
- 策略训练时学到的时间一致性
- 参考动作的blend（仅planner模式）

### 2. **Action Scale 计算公式**

**C++ 公式**（policy_parameters.hpp:27）:
```cpp
action_scale = 0.25 × effort_limit / stiffness
```

**物理意义**:
- `effort_limit`: 电机最大扭矩（N·m）
- `stiffness`: PD控制器Kp增益
- `0.25`: 安全系数（限制动作幅度为最大扭矩的25%）

**示例计算**:
```
hip_pitch: 0.25 × 139.0 / 99.0 = 0.3507
hip_yaw:   0.25 × 88.0 / 40.2 = 0.5475
wrist_yaw: 0.25 × 5.0 / 16.7 = 0.0745
```

### 3. **Stiffness 计算公式**

**C++ 公式**（policy_parameters.hpp:49-53）:
```cpp
NATURAL_FREQ = 10 Hz × 2π = 62.83 rad/s
stiffness = armature × NATURAL_FREQ²

// Example:
STIFFNESS_7520_22 = 0.025101925 × 62.83² = 99.0
STIFFNESS_5020 = 0.003609725 × 62.83² = 14.2
STIFFNESS_4010 = 0.00425 × 62.83² = 16.7
```

**但是！Isaac Lab中的Kp与此不同！**

---

## ⚠️ **发现的问题**

### 问题1: **Stiffness不匹配**

**C++ SONIC (policy_parameters.hpp:143-173)**:
```cpp
const std::array<float, 29> kps = {
    STIFFNESS_7520_22,        // hip_pitch: 99.0
    STIFFNESS_7520_22,        // hip_roll: 99.0
    STIFFNESS_7520_14,        // hip_yaw: 40.2
    STIFFNESS_7520_22,        // knee: 99.0
    2.0 * STIFFNESS_5020,     // ankle_pitch: 28.4
    2.0 * STIFFNESS_5020,     // ankle_roll: 28.4
    STIFFNESS_7520_14,        // waist_yaw: 40.2
    2.0 * STIFFNESS_5020,     // waist_roll: 28.4
    2.0 * STIFFNESS_5020,     // waist_pitch: 28.4
    STIFFNESS_5020,           // shoulder_*: 14.2
    STIFFNESS_5020,           // elbow: 14.2
    STIFFNESS_5020,           // wrist_roll: 14.2
    STIFFNESS_4010,           // wrist_pitch/yaw: 16.7
};
```

**Isaac Lab (unitree.py:858-976, 刚更新的KDKP)**:
```python
stiffness={
    ".*_hip_pitch_joint": 150.0,    # ❌ C++: 99.0
    ".*_hip_roll_joint": 150.0,     # ❌ C++: 99.0
    ".*_hip_yaw_joint": 150.0,      # ❌ C++: 40.2
    ".*_knee_joint": 200.0,         # ❌ C++: 99.0
    ".*_ankle_pitch_joint": 40.0,   # ❌ C++: 28.4
    ".*_ankle_roll_joint": 40.0,    # ❌ C++: 28.4
    "waist_yaw_joint": 250.0,       # ❌ C++: 40.2
    "waist_roll_joint": 250.0,      # ❌ C++: 28.4
    "waist_pitch_joint": 250.0,     # ❌ C++: 28.4
    ".*_shoulder_pitch_joint": 100.0,  # ❌ C++: 14.2
    ".*_shoulder_roll_joint": 100.0,   # ❌ C++: 14.2
    ".*_shoulder_yaw_joint": 40.0,     # ❌ C++: 14.2
    ".*_elbow_joint": 40.0,            # ❌ C++: 14.2
    ".*_wrist_roll_joint": 20.0,       # ❌ C++: 14.2
    ".*_wrist_pitch_joint": 20.0,      # ❌ C++: 16.7
    ".*_wrist_yaw_joint": 20.0,        # ❌ C++: 16.7
}
```

**结论**: Isaac Lab的Kp值来自MuJoCo YAML（`g1_29dof_sonic_model12.yaml`的`MOTOR_KP`），但**C++部署代码使用的是完全不同的Kp值**（基于armature计算）！

### 问题2: **Action Scale可能不匹配**

**如果Kp不同，action_scale也应该不同！**

**C++ action_scale公式**:
```
action_scale = 0.25 × effort_limit / stiffness
```

**如果Isaac Lab的Kp是150而不是99**:
```
hip_pitch_scale_cpp = 0.25 × 139.0 / 99.0 = 0.3507
hip_pitch_scale_isaaclab = 0.25 × 139.0 / 150.0 = 0.2317  # ❌ 不同！
```

**当前Isaac Lab使用的action_scale**（Line 651）:
```python
0.3506614566,  # left_hip_pitch_joint
```

这是基于C++ Kp=99.0计算的，但Isaac Lab实际Kp=150.0！

---

## 🎯 **根本原因分析**

### **SONIC有两套Kp/Kd参数！**

1. **训练时（MuJoCo）**: 使用`g1_29dof_sonic_model12.yaml`的`MOTOR_KP/KD`
   - hip_pitch Kp=150, Kd=2
   - waist_yaw Kp=250, Kd=5
   - shoulder Kp=100, Kd=5

2. **部署时（C++/真机）**: 使用`policy_parameters.hpp`的计算值
   - hip_pitch Kp=99.0, Kd=3.16
   - waist_yaw Kp=40.2, Kd=1.28
   - shoulder Kp=14.2, Kd=0.45

**这意味着**:
- 策略在MuJoCo中训练时，机器人的"硬度"是一个值
- 部署到真机/C++仿真时，"硬度"变成了另一个值
- **action_scale是基于部署时的Kp计算的**，用于补偿Kp差异

---

## ✅ **解决方案**

### 方案1: **使用C++ SONIC的Kp/Kd（推荐）**

**理由**: action_scale是基于这些Kp计算的，必须匹配。

**修改Isaac Lab `unitree.py`**:
```python
stiffness={
    ".*_hip_pitch_joint": 99.0,     # STIFFNESS_7520_22
    ".*_hip_roll_joint": 99.0,
    ".*_hip_yaw_joint": 40.2,       # STIFFNESS_7520_14
    ".*_knee_joint": 99.0,
    ".*_ankle_pitch_joint": 28.4,   # 2.0 * STIFFNESS_5020
    ".*_ankle_roll_joint": 28.4,
    "waist_yaw_joint": 40.2,        # STIFFNESS_7520_14
    "waist_roll_joint": 28.4,       # 2.0 * STIFFNESS_5020
    "waist_pitch_joint": 28.4,
    ".*_shoulder_pitch_joint": 14.2,  # STIFFNESS_5020
    ".*_shoulder_roll_joint": 14.2,
    ".*_shoulder_yaw_joint": 14.2,
    ".*_elbow_joint": 14.2,
    ".*_wrist_roll_joint": 14.2,
    ".*_wrist_pitch_joint": 16.7,   # STIFFNESS_4010
    ".*_wrist_yaw_joint": 16.7,
}
damping={
    ".*_hip_pitch_joint": 3.16,     # DAMPING_7520_22
    ".*_hip_roll_joint": 3.16,
    ".*_hip_yaw_joint": 1.28,       # DAMPING_7520_14
    ".*_knee_joint": 3.16,
    ".*_ankle_pitch_joint": 0.91,   # 2.0 * DAMPING_5020
    ".*_ankle_roll_joint": 0.91,
    "waist_yaw_joint": 1.28,
    "waist_roll_joint": 0.91,
    "waist_pitch_joint": 0.91,
    ".*_shoulder_pitch_joint": 0.45,  # DAMPING_5020
    ".*_shoulder_roll_joint": 0.45,
    ".*_shoulder_yaw_joint": 0.45,
    ".*_elbow_joint": 0.45,
    ".*_wrist_roll_joint": 0.45,
    ".*_wrist_pitch_joint": 0.54,   # DAMPING_4010
    ".*_wrist_yaw_joint": 0.54,
}
```

### 方案2: **重新计算action_scale（如果坚持用MuJoCo Kp）**

**如果必须使用MuJoCo的Kp=150/250**，则需要重新计算action_scale:

```python
# 基于Isaac Lab实际Kp重新计算
G1_ACTION_SCALE_ISAACLAB = np.array([
    0.25 * 139.0 / 150.0,  # hip_pitch: 0.2317 (was 0.3507)
    # ... 需要重新计算所有29个关节
])
```

**但这会导致**:
- 动作幅度变小（因为Kp更大）
- 可能与训练时的动作分布不匹配

---

## 📋 **检查清单**

| 项目 | C++ SONIC | Isaac Lab | 状态 |
|------|-----------|-----------|------|
| Action smoothing | ❌ None | ❌ None | ✅ 一致 |
| Action clipping | ❌ None | ⚠️ 有两次clip | ⚠️ 不一致 |
| Joint order | MuJoCo | IsaacLab | ✅ 已remapping |
| Action scale | 基于Kp=99/40/14 | 基于Kp=99/40/14 | ✅ 一致 |
| Kp (stiffness) | 99/40/14/28/17 | 150/200/250/100/40/20 | ❌ **不一致** |
| Kd (damping) | 3.16/1.28/0.45/0.91/0.54 | 2/4/5 | ❌ **不一致** |
| Effort limit | 139/88/25/5 | 88/139/50/25/5 | ⚠️ 部分不一致 |
| Default angles | Line 210-240 | `_sonic_default_np` | ❓ 需验证 |

---

## 🎯 **结论**

**KDKP调整无效的根本原因**:

1. **Kp/Kd不匹配**: Isaac Lab使用MuJoCo训练参数，C++使用部署参数
2. **Action scale基于错误的Kp**: action_scale是为Kp=99设计的，但Isaac Lab用Kp=150
3. **动作幅度不一致**: 相同的raw action，在不同Kp下产生不同的实际扭矩

**推荐修复**:
- ✅ 使用C++ SONIC的Kp/Kd（基于armature计算）
- ✅ 保持当前的action_scale不变
- ✅ 移除Isaac Lab中的额外clipping

---

**文档创建时间**: 2026-03-21
**状态**: ⚠️ 发现关键问题，待修复
