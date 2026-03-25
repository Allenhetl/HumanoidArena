# SONIC Isaac Lab KDKP参数同步

## 🎯 目标

将Isaac Lab中的PD控制器参数（Kp, Kd, effort_limit）调整为与SONIC MuJoCo训练环境完全一致。

## 📊 参数对比

### SONIC MuJoCo 配置源
- **文件**: `/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`
- **关键参数**:
  - `MOTOR_KP` (Line 85-91): Stiffness gains
  - `MOTOR_KD` (Line 93-99): Damping gains
  - `motor_effort_limit_list` (Line 241-249): Torque limits

### 修改的Isaac Lab配置
- **文件**: `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/robots/unitree.py`
- **配置**: `G129_CFG_WITH_DEX3_WHOLEBODY` (Line 799-998)

---

## 🔧 详细参数映射

### 1. Legs（腿部）

| 关节 | SONIC Kp | SONIC Kd | SONIC Effort | Isaac Lab (更新后) |
|------|----------|----------|--------------|-------------------|
| hip_pitch | 150 | 2 | 88 N·m | Kp=150, Kd=2, τ=88 |
| hip_roll | 150 | 2 | 88 N·m | Kp=150, Kd=2, τ=88 |
| hip_yaw | 150 | 2 | 88 N·m | Kp=150, Kd=2, τ=88 |
| knee | 200 | 4 | 139 N·m | Kp=200, Kd=4, τ=139 |

**SONIC原值（MOTOR_KP）**:
```yaml
MOTOR_KP: [
    150, 150, 150, 200, 40, 40,  # left leg
    150, 150, 150, 200, 40, 40,  # right leg
    ...
]
```

### 2. Feet（脚踝）

| 关节 | SONIC Kp | SONIC Kd | SONIC Effort | Isaac Lab (更新后) |
|------|----------|----------|--------------|-------------------|
| ankle_pitch | 40 | 2 | 50 N·m | Kp=40, Kd=2, τ=50 |
| ankle_roll | 40 | 2 | 50 N·m | Kp=40, Kd=2, τ=50 |

### 3. Waist（腰部）

| 关节 | SONIC Kp | SONIC Kd | SONIC Effort | Isaac Lab (更新后) |
|------|----------|----------|--------------|-------------------|
| waist_yaw | 250 | 5 | 88 N·m | Kp=250, Kd=5, τ=88 |
| waist_roll | 250 | 5 | 50 N·m | Kp=250, Kd=5, τ=50 |
| waist_pitch | 250 | 5 | 50 N·m | Kp=250, Kd=5, τ=50 |

**SONIC原值（MOTOR_KP Line 87-88）**:
```yaml
250, 250, 250,  # waist_yaw, waist_roll, waist_pitch
```

### 4. Arms（手臂）

| 关节 | SONIC Kp | SONIC Kd | SONIC Effort | Isaac Lab (更新后) |
|------|----------|----------|--------------|-------------------|
| shoulder_pitch | 100 | 5 | 25 N·m | Kp=100, Kd=5, τ=25 |
| shoulder_roll | 100 | 5 | 25 N·m | Kp=100, Kd=5, τ=25 |
| shoulder_yaw | 40 | 2 | 25 N·m | Kp=40, Kd=2, τ=25 |
| elbow | 40 | 2 | 25 N·m | Kp=40, Kd=2, τ=25 |
| wrist_roll | 20 | 2 | 25 N·m | Kp=20, Kd=2, τ=25 |
| wrist_pitch | 20 | 2 | 5 N·m | Kp=20, Kd=2, τ=5 |
| wrist_yaw | 20 | 2 | 5 N·m | Kp=20, Kd=2, τ=5 |

**SONIC原值（MOTOR_KP Line 89-91）**:
```yaml
100, 100, 40, 40, 20, 20, 20,  # left arm
100, 100, 40, 40, 20, 20, 20   # right arm
```

**SONIC原值（motor_effort_limit_list Line 244-248）**:
```yaml
25.0, 25.0, 25.0, 25.0,   # shoulder + elbow
25.0, 5.0, 5.0,           # wrist (roll=25, pitch/yaw=5)
```

---

## 🔍 关键变化

### 修改前（Isaac Lab原值）
```python
# Legs
stiffness={".*_hip_.*_joint": 50.0, ".*_knee_joint": 50.0}
effort_limit_sim={".*_hip_.*_joint": 100.0, ".*_knee_joint": 150.0}

# Waist
stiffness={yaw: 60.0, roll: 60.0, pitch: 100.0}

# Arms
stiffness={".*_shoulder_.*_joint": 150.0-300.0, ".*_elbow_joint": 150.0, ".*_wrist_.*_joint": 150.0}
effort_limit_sim=200.0 (全部)
```

### 修改后（匹配SONIC）
```python
# Legs
stiffness={".*_hip_.*_joint": 150.0, ".*_knee_joint": 200.0}
effort_limit_sim={".*_hip_.*_joint": 88.0, ".*_knee_joint": 139.0}

# Waist
stiffness=250.0 (全部)
effort_limit_sim={yaw: 88.0, roll: 50.0, pitch: 50.0}

# Arms
stiffness={"shoulder_pitch/roll": 100.0, "shoulder_yaw": 40.0, "elbow": 40.0, "wrist": 20.0}
effort_limit_sim={"shoulder/elbow": 25.0, "wrist_roll": 25.0, "wrist_pitch/yaw": 5.0}
```

---

## 📈 预期效果

### 1. **更高的跟踪精度**
- **Legs**: Kp从50→150/200（提升3-4倍）
- **Waist**: Kp从60-100→250（提升2.5-4倍）
- 更强的位置跟踪能力，减少延迟

### 2. **更真实的力限制**
- **Arms**: effort从200→25/5（降低40倍）
- **Legs**: effort从100/150→88/139（略微降低）
- 与SONIC训练环境的硬件限制一致

### 3. **更稳定的动力学**
- **Waist**: Kd从4→5（提升25%）
- **Arms**: Kd保持2-5（与SONIC一致）
- 更好的阻尼，减少震荡

---

## 🧪 测试验证

### 测试步骤
```bash
# Terminal 1: pico_server
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl

# Terminal 2: Isaac Lab (with updated KDKP)
bash run_sonic.sh
```

### 预期观察
1. ✅ 机器人动作更贴近VR输入（减少延迟）
2. ✅ 腰部和腿部更稳定（更高的Kp）
3. ✅ 手臂动作更柔和（降低的effort_limit）
4. ✅ 无过度震荡或发散（适当的Kd）

### 监控指标
- Joint tracking error（关节跟踪误差）
- Control frequency（控制频率）
- Simulation stability（仿真稳定性）

---

## 📋 SONIC关节顺序（MuJoCo order）

**SONIC MOTOR_KP顺序**（g1_29dof_sonic_model12.yaml Line 85-91）:
```
[0-5]:   left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
[6-11]:  right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
[12-14]: waist_yaw, waist_roll, waist_pitch
[15-21]: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
         left_wrist_roll, left_wrist_pitch, left_wrist_yaw
[22-28]: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow,
         right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

**Isaac Lab关节顺序**（与SONIC不同，但已通过`joint_names_expr`正确映射）

---

## ✅ 修改总结

| 配置文件 | 修改位置 | 状态 |
|---------|---------|------|
| `robots/unitree.py` | Line 858-976 | ✅ 已完成 |
| `G129_CFG_WITH_DEX3_WHOLEBODY` | actuators配置 | ✅ 已更新 |

**修改内容**:
- ✅ Legs: Kp, Kd, effort_limit
- ✅ Feet: Kp, Kd, effort_limit
- ✅ Waist: Kp, Kd, effort_limit
- ✅ Arms: Kp, Kd, effort_limit（包括wrist细分）

---

**修改时间**: 2026-03-21
**参考源**: SONIC `g1_29dof_sonic_model12.yaml`
**状态**: ✅ 已完成，待测试验证
