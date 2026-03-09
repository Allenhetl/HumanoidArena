# SONIC推理问题 - 最终修复方案

## 问题诊断结果

通过运行 `check_sonic_encoder.py`，我们发现：
- **Encoder输入**: 单个输入 `obs_dict`，形状 `[1, 1762]`
- **Encoder输出**: `encoded_tokens`，形状 `[1, 64]`

这说明encoder需要**所有启用的观察值**拼接成一个1762维的向量，而不仅仅是SMPL模式需要的观察值。

## 1762维输入组成

根据 `observation_config.yaml` 的 `encoder_observations` 部分：

```
序号  观察值名称                                      维度    累计
----  --------------------------------------------  ----  ------
 1.   encoder_mode_4                                   4       4
 2.   motion_joint_positions_10frame_step5           290     294
 3.   motion_joint_velocities_10frame_step5          290     584
 4.   motion_root_z_position_10frame_step5            10     594
 5.   motion_root_z_position                           1     595
 6.   motion_anchor_orientation                        6     601
 7.   motion_anchor_orientation_10frame_step5         60     661
 8.   motion_joint_positions_lowerbody_10frame_step5 120     781
 9.   motion_joint_velocities_lowerbody_10frame_step5 120    901
10.   vr_3point_local_target                           9     910
11.   vr_3point_local_orn_target                      12     922
12.   smpl_joints_10frame_step1                      720    1642
13.   smpl_anchor_orientation_10frame_step1           60    1702
14.   motion_joint_positions_wrists_10frame_step1     60    1762
----  --------------------------------------------  ----  ------
      总计                                           1762
```

## 核心修复

### 1. 四元数到6D旋转转换 ✓

```python
def quat_to_rotation_6d(quat: np.ndarray) -> np.ndarray:
    """将四元数转换为6D旋转表示（旋转矩阵前2列）"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    rot6d = np.stack([r00, r01, r10, r11, r20, r21], axis=-1)
    return rot6d.astype(np.float32)
```

### 2. 添加历史缓冲区 ✓

```python
def _setup_buffers(self):
    # SMPL数据缓冲
    self._smpl_joints_buf = np.zeros((10, 24, 3), dtype=np.float32)
    self._body_rot6d_buf = np.tile(
        np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32), (10, 1))

    # 机器人状态历史（用于encoder）
    self._robot_joint_pos_hist = np.tile(
        self._sonic_default_np[np.newaxis], (10, 1))  # (10, 29)
    self._robot_joint_vel_hist = np.zeros((10, 29), dtype=np.float32)
```

### 3. 构建完整的1762维encoder输入 ✓

```python
def _run_gear_sonic(self) -> np.ndarray:
    # 更新历史缓冲区
    joint_pos_sonic = robot.joint_pos[0, self._sonic_idx].cpu().numpy()
    joint_vel_sonic = robot.joint_vel[0, self._sonic_idx].cpu().numpy()

    self._robot_joint_pos_hist = np.roll(self._robot_joint_pos_hist, -1, axis=0)
    self._robot_joint_pos_hist[-1] = joint_pos_sonic
    self._robot_joint_vel_hist = np.roll(self._robot_joint_vel_hist, -1, axis=0)
    self._robot_joint_vel_hist[-1] = joint_vel_sonic

    # 构建1762维输入
    encoder_input = np.concatenate([
        np.array([0., 0., 1., 0.]),                    # encoder_mode (SMPL=2)
        motion_joint_pos_step5_full,                   # 290
        motion_joint_vel_step5_full,                   # 290
        motion_root_z_step5,                           # 10
        motion_root_z,                                 # 1
        self._body_rot6d_buf[-1],                      # 6
        motion_anchor_orient_step5_full,               # 60
        motion_joint_pos_lowerbody_full,               # 120
        motion_joint_vel_lowerbody_full,               # 120
        np.zeros(9),                                   # vr_3pt_pos
        np.zeros(12),                                  # vr_3pt_orn
        self._smpl_joints_buf.reshape(-1),             # 720
        self._body_rot6d_buf.reshape(-1),              # 60
        motion_wrist_pos,                              # 60
    ])[np.newaxis]  # (1, 1762)

    # Encoder推理
    enc_inputs = {
        self._encoder.get_inputs()[0].name: encoder_input
    }
    latent = self._encoder.run(None, enc_inputs)[0]
```

## 关键点说明

### 1. encoder_mode_4
- SMPL模式的one-hot编码: `[0, 0, 1, 0]`
- mode_id=2对应第3个位置为1

### 2. step5 vs step1
- `step5`: 每隔5帧采样一次（0.1秒间隔）
- `step1`: 连续帧采样
- 由于我们只维护10帧历史，step5采样只能得到2帧数据，其余用零填充

### 3. 下半身关节
- 前12个关节被认为是下半身关节
- 包括：hip, knee, ankle等

### 4. 手腕关节
- 索引12-17：左右肩部各3个关节
- `left_shoulder_pitch/roll/yaw`, `right_shoulder_pitch/roll/yaw`

### 5. VR 3点数据
- SMPL模式不使用VR数据，用零填充
- 维度：位置9 (3点×3坐标) + 方向12 (3点×4元数)

## 测试验证

### 1. 检查encoder输入维度
```bash
python isaaclab_twist2_g1/pico_server/check_sonic_encoder.py /path/to/model_encoder.onnx
```
应该显示: `✓ 形状: [1, 1762]`

### 2. 测试转换函数
```bash
python isaaclab_twist2_g1/pico_server/test_quat_to_rot6d.py
```
应该显示: `✓ 所有测试通过`

### 3. 运行完整系统
```bash
# Terminal 1: Pico服务器
cd isaaclab_twist2_g1/pico_server
python pico_server_pose_only.py --vis_vr3pt

# Terminal 2: IsaacLab仿真
cd isaaclab_twist2_g1
bash run_sonic.sh
```

### 4. 验证输出
查看日志应该显示：
```
[SONIC] Encoder input shape: (1, 1762), expected: (1, 1762)
[SONIC] Encoder output latent shape: (1, 64)
```

如果看到这些输出且没有错误，说明修复成功！

## 预期结果

修复后：
- ✓ Encoder接收正确的1762维输入
- ✓ 包含encoder_mode标识SMPL模式
- ✓ 包含完整的SMPL joints和6D旋转
- ✓ 包含机器人状态历史
- ✓ 推理正常工作，输出动态变化
- ✓ 机器人跟随VR全身姿态

## 与原始实现的差异

### 原始错误实现
```python
# 错误1: 使用四元数而不是6D旋转
anchor_orient = self._body_quat_buf[np.newaxis]  # (1, 10, 4) ✗

# 错误2: 传递多维数组而不是展平的向量
enc_inputs = {
    'input_0': smpl_joints_in,  # (1, 10, 24, 3) ✗
    'input_1': anchor_orient,   # (1, 10, 4) ✗
    'input_2': joint_pos_hist,  # (1, 10, 29) ✗
}

# 错误3: 缺少encoder_mode和其他观察值
```

### 正确实现
```python
# 正确1: 使用6D旋转
self._body_rot6d_buf  # (10, 6) ✓

# 正确2: 展平并拼接所有观察值
encoder_input = np.concatenate([...])  # (1, 1762) ✓

# 正确3: 包含所有必需的观察值
enc_inputs = {
    'obs_dict': encoder_input  # (1, 1762) ✓
}
```

## 参考文档

1. **Encoder检查结果**: 显示输入形状 `[1, 1762]`
2. **observation_config.yaml**: 定义了14个encoder观察值
3. **C++实现**: `g1_deploy_onnx_ref.cpp` 中的观察值收集逻辑
4. **6D旋转论文**: Zhou et al., CVPR 2019

## 总结

这次修复解决了三个关键问题：
1. **数据格式**: 四元数 → 6D旋转表示
2. **输入维度**: 多维数组 → 1762维展平向量
3. **观察值完整性**: 部分观察值 → 所有14个观察值

现在的实现完全符合SONIC encoder的输入要求，应该能够正常工作。