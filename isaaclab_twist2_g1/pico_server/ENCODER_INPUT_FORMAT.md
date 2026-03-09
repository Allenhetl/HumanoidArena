# SONIC Encoder输入格式分析

## 根据官方文档的正确配置

### SMPL模式 (mode_id: 2) 的Encoder输入

根据 `observation_config.yaml` 和文档，SMPL模式需要以下输入：

```yaml
- name: "smpl"
  mode_id: 2
  required_observations:
    - encoder_mode_4                              # 4维
    - smpl_joints_10frame_step1                   # 720维 (10×24×3)
    - smpl_anchor_orientation_10frame_step1       # 60维 (10×6)
    - motion_joint_positions_wrists_10frame_step1 # 60维 (10×6)
```

**总维度**: 4 + 720 + 60 + 60 = **844维**

### 各输入详解

#### 1. encoder_mode_4 (4维)
- One-hot编码的模式标识
- SMPL模式 (mode_id=2): `[0, 0, 1, 0]`
- 参考C++: `GatherEncoderMode(buf, offset, 3)` - 第3个模式

#### 2. smpl_joints_10frame_step1 (720维)
- 10帧连续的SMPL关节位置
- 每帧24个关节，每个关节3个坐标 (x, y, z)
- 形状: (10, 24, 3) → 展平为 (720,)
- **展平顺序**: 按帧展平，即 `[frame0_joint0_xyz, frame0_joint1_xyz, ..., frame9_joint23_xyz]`

#### 3. smpl_anchor_orientation_10frame_step1 (60维)
- 10帧连续的anchor方向（6D旋转表示）
- 每帧6个值（旋转矩阵前2列）
- 形状: (10, 6) → 展平为 (60,)
- **展平顺序**: `[frame0_rot6d, frame1_rot6d, ..., frame9_rot6d]`

#### 4. motion_joint_positions_wrists_10frame_step1 (60维)
- 10帧连续的手腕关节位置
- 每帧6个关节（左右手腕各3个）
- 形状: (10, 6) → 展平为 (60,)
- **关节顺序** (IsaacLab order):
  ```python
  wrist_joints = [
      "left_shoulder_pitch_joint",   # 0
      "left_shoulder_roll_joint",    # 1
      "left_shoulder_yaw_joint",     # 2
      "right_shoulder_pitch_joint",  # 3
      "right_shoulder_roll_joint",   # 4
      "right_shoulder_yaw_joint",    # 5
  ]
  ```

## C++实现参考

### 数据收集方式

C++中所有观察值都被写入一个连续的1D缓冲区：

```cpp
// 每个观察函数将数据写入buffer的指定offset
bool GatherMotionSmplJointsMultiFrame(std::vector<double>& buf, size_t offset,
                                      int num_frames, int step_size) {
    // 写入 num_frames × 24 × 3 个值到 buf[offset...]
    for (int frame = 0; frame < num_frames; frame++) {
        for (int joint = 0; joint < 24; joint++) {
            buf[offset++] = smpl_joints[frame][joint][0];  // x
            buf[offset++] = smpl_joints[frame][joint][1];  // y
            buf[offset++] = smpl_joints[frame][joint][2];  // z
        }
    }
}
```

### Encoder输入构建

```cpp
// 1. 清空encoder输入缓冲区
std::fill(encoder_obs_buffer_.begin(), encoder_obs_buffer_.end(), 0.0);

// 2. 按顺序调用各个观察函数，每个函数写入自己的offset
size_t offset = 0;
GatherEncoderMode(encoder_obs_buffer_, offset, 2);  // 写入4个值
offset += 4;

GatherMotionSmplJointsMultiFrame(encoder_obs_buffer_, offset, 10, 1);  // 写入720个值
offset += 720;

GatherMotionAnchorOrientationMutiFrame(encoder_obs_buffer_, offset, 10, 1);  // 写入60个值
offset += 60;

GatherMotionJointPositionsMultiFrame(encoder_obs_buffer_, offset, 10, 1, wrist_indices);  // 写入60个值
offset += 60;

// 3. 将encoder_obs_buffer_传给ONNX Runtime
```

## IsaacLab实现建议

### 当前问题

当前实现传递的是多维数组：
```python
enc_inputs = {
    'input_0': smpl_joints_in,  # (1, 10, 24, 3)
    'input_1': anchor_orient,   # (1, 10, 6)
    'input_2': joint_pos_hist,  # (1, 10, 29)  ← 错误：应该是手腕关节
}
```

### 正确实现

应该传递展平的1D数组（或带batch维度的2D数组）：

```python
# 1. encoder_mode (4,)
encoder_mode = np.array([0., 0., 1., 0.], dtype=np.float32)  # SMPL模式

# 2. smpl_joints_10frame_step1 (720,)
smpl_joints_flat = self._smpl_joints_buf.reshape(-1)  # (10, 24, 3) → (720,)

# 3. smpl_anchor_orientation_10frame_step1 (60,)
anchor_orient_flat = self._body_rot6d_buf.reshape(-1)  # (10, 6) → (60,)

# 4. motion_joint_positions_wrists_10frame_step1 (60,)
# 需要从robot joint positions中提取手腕关节
wrist_indices = [...]  # 6个手腕关节的索引
wrist_pos_hist = np.zeros((10, 6), dtype=np.float32)
for i in range(10):
    wrist_pos_hist[i] = self._robot_joint_pos[wrist_indices]  # 需要历史数据
wrist_pos_flat = wrist_pos_hist.reshape(-1)  # (10, 6) → (60,)

# 组合成单个输入（如果encoder只有一个输入）
encoder_input = np.concatenate([
    encoder_mode,           # 4
    smpl_joints_flat,       # 720
    anchor_orient_flat,     # 60
    wrist_pos_flat,         # 60
])[np.newaxis]  # (1, 844)

# 或者分别传递（如果encoder有多个输入）
enc_inputs = {
    'encoder_mode': encoder_mode[np.newaxis],           # (1, 4)
    'smpl_joints': smpl_joints_flat[np.newaxis],        # (1, 720)
    'anchor_orient': anchor_orient_flat[np.newaxis],    # (1, 60)
    'wrist_pos': wrist_pos_flat[np.newaxis],            # (1, 60)
}
```

## 验证步骤

1. **检查encoder输入数量和名称**:
   ```python
   for inp in encoder.get_inputs():
       print(f"{inp.name}: shape={inp.shape}, type={inp.type}")
   ```

2. **检查输入形状**:
   - 如果是单个输入: 应该是 `(1, 844)` 或 `(batch, 844)`
   - 如果是多个输入: 检查每个输入的维度是否匹配

3. **检查数据范围**:
   - SMPL joints: 通常在 [-2, 2] 范围内（米）
   - 6D rotation: 每个值在 [-1, 1] 范围内
   - Joint positions: 通常在 [-π, π] 范围内（弧度）

## 下一步行动

1. 使用 `check_encoder_inputs.py` 检查encoder的实际输入要求
2. 根据实际输入形状调整 `action_provider_sonic.py`
3. 确保手腕关节索引正确
4. 添加encoder_mode输入
5. 测试完整流程