# SONIC motion_anchor_orientation 修复 - 相对旋转vs绝对旋转

## 🚨 问题发现

### 错误的理解
之前认为 `motion_anchor_orientation` 应该是**世界坐标系下的绝对朝向**，导致直接使用 `body_quat_w` 而不计算相对旋转。

### 实际情况（C++ 原版代码）
通过仔细阅读 `g1_deploy_onnx_ref.cpp:514-594`，发现：

```cpp
// Line 585: 关键计算
auto base_to_ref_quat = quat_mul_d(quat_conjugate_d(base_quat), new_ref_root_rot);
auto rotation_matrix = quat_to_rotation_matrix_d(base_to_ref_quat);
```

**`motion_anchor_orientation` 是相对于机器人base坐标系的旋转，不是世界坐标系的绝对旋转！**

## 🔍 C++ 原版逻辑分析

### 数据流
```cpp
// 1. 读取机器人当前朝向
auto hist = state_logger_->GetLatest(1, sample_dt);
const auto& base_quat = hist[0].base_quat;  // 机器人当前朝向

// 2. 读取参考数据（SMPL）朝向
const auto motion_body_quat = current_motion_->BodyQuaternions(target_frame);
std::array<double, 4> ref_data_root_rot_array = motion_body_quat[0];

// 3. 应用heading对齐（如果需要）
auto new_ref_root_rot = quat_mul_d(apply_delta_heading, ref_data_root_rot_array);

// 4. 计算相对旋转：ref 相对于 base
auto base_to_ref_quat = quat_mul_d(quat_conjugate_d(base_quat), new_ref_root_rot);
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^
                                   base^(-1)                    ref

// 5. 转换为rot6d（旋转矩阵前2列，按行展开）
auto rotation_matrix = quat_to_rotation_matrix_d(base_to_ref_quat);
std::array<double, 6> motion_anchor_ori_b = {
    rotation_matrix[0][0], rotation_matrix[0][1],  // Row 0
    rotation_matrix[1][0], rotation_matrix[1][1],  // Row 1
    rotation_matrix[2][0], rotation_matrix[2][1]   // Row 2
};
```

### 关键点
- **输入坐标系**: 世界坐标系（`base_quat`, `ref_quat`）
- **输出坐标系**: 机器人局部坐标系（`base_to_ref_quat`）
- **物理意义**: "参考朝向在机器人局部坐标系下的表示"

## ✅ 正确的 Python 实现

### 修改前（错误）
```python
# Line 1220-1223（之前的错误实现）
ref_quat_wxyz = quat_normalize_wxyz(bq[-1])
rot6d_latest = quat_to_rotation_6d(ref_quat_wxyz.reshape(1, 4))[0]
# ❌ 直接使用世界坐标系的绝对朝向，不计算相对旋转
```

**问题**: 当 `init_rot` 改变时，机器人朝向改变，但 encoder 输入不变，导致模型认为机器人在错误的朝向。

### 修改后（正确）
```python
# Line 1221-1236（现在的正确实现）
# 获取机器人当前朝向（从Isaac Lab）
robot = self.env.scene["robot"].data
base_quat_wxyz = robot.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)  # [w,x,y,z]

# 获取参考数据（SMPL）的朝向
ref_quat_wxyz = quat_normalize_wxyz(bq[-1])  # [w,x,y,z]

# 计算相对旋转：ref 相对于 base（与C++一致）
# base_to_ref = base^(-1) * ref
rel_quat_wxyz = quat_mul_wxyz(
    quat_conjugate_wxyz(base_quat_wxyz),  # base^(-1)
    ref_quat_wxyz                          # ref
)

# 转换为rot6d
rot6d_latest = quat_to_rotation_6d(rel_quat_wxyz.reshape(1, 4))[0]
```

## 📊 为什么相对旋转是正确的？

### 训练数据格式
训练时，encoder 输入的 `motion_anchor_orientation` 是：
- 参考动作（SMPL）相对于机器人base坐标系的旋转
- 不是世界坐标系下的绝对旋转

### 物理意义
- **相对旋转**: 告诉模型"参考朝向在我的坐标系下是什么方向"
- **绝对旋转**: 告诉模型"参考朝向在世界坐标系下是什么方向"（错误）

### 为什么 `init_rot=(1,0,0,0)` 能"巧合地"工作？
当 `init_rot=(1,0,0,0)` 时：
- 机器人朝向：`base_quat = [1,0,0,0]`（单位四元数，无旋转）
- 相对旋转：`base^(-1) * ref = [1,0,0,0]^(-1) * ref = ref`
- 结果**巧合地等于**直接使用 `ref`

当 `init_rot=(0.7071,0,0,0.7071)` 时：
- 机器人朝向：`base_quat = [0.7071,0,0,0.7071]`（绕Z轴旋转90度）
- 相对旋转：`base^(-1) * ref ≠ ref`
- 如果不计算相对旋转，encoder 会认为机器人朝向错误

## 🎯 修复效果

### 修复前
- `init_rot=(1,0,0,0)`: ✅ 能工作（巧合）
- `init_rot=(0.7071,0,0,0.7071)`: ❌ G1整体旋转，动作不正常

### 修复后
- `init_rot=(1,0,0,0)`: ✅ 应该能工作
- `init_rot=(0.7071,0,0,0.7071)`: ✅ 应该能工作
- 任意 `init_rot`: ✅ 理论上都应该能工作

## 🔧 修改文件

**文件**: `action_provider/action_provider_sonic.py`
**位置**: Line 1212-1252
**状态**: ✅ 已修复

## 📝 技术细节

### 四元数运算
```python
# 四元数共轭（相当于逆旋转）
quat_conjugate([w,x,y,z]) = [w,-x,-y,-z]

# 四元数乘法（旋转组合）
quat_mul(q1, q2) 表示先应用q2旋转，再应用q1旋转

# 相对旋转计算
rel_quat = base^(-1) * ref
```

### Rotation 6D
- 旋转矩阵的前2列（3x2 = 6维）
- 按行展开：`[R00, R01, R10, R11, R20, R21]`
- 与原版C++完全一致

## ✅ 验证方法

重新测试：
```bash
# Terminal 1: pico_server
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl

# Terminal 2: Isaac Lab with default init_rot
bash run_sonic.sh

# 预期: 机器人跟随VR输入，动作正常
```

---

**修复时间**: 2026-03-21
**修复内容**: 恢复相对旋转计算（与C++原版一致）
**状态**: ✅ 已完成，待测试验证
