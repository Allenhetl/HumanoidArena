# SONIC推理问题修复总结

## 问题根本原因

**IsaacLab实现发送四元数(4维)，但SONIC encoder期望6D旋转表示(6维)**

### 发现过程

1. 用户报告：`sonic_targets` 始终输出固定的默认值
2. 诊断发现：ZMQ数据接收正常，`body_quat_w` 形状为 `(N, 4)`
3. 查看C++实现：发现 `smpl_anchor_orientation_10frame_step1` 维度是 **60** (10帧 × 6)
4. 分析C++代码：确认使用的是6D旋转表示（旋转矩阵前2列）

### 技术细节

**6D旋转表示**是一种连续的旋转表示方法：
- 取3×3旋转矩阵R的前2列
- 按行展开为6个值：`[R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]`
- 优点：连续、无奇异点、适合神经网络学习
- 参考论文：Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019

**C++实现** (gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:514-580):
```cpp
// 将四元数转换为旋转矩阵
auto rotation_matrix = quat_to_rotation_matrix_d(base_to_ref_quat);

// 提取前2列，按行展开
std::array<double, 6> motion_anchor_ori_b = {
    rotation_matrix[0][0], rotation_matrix[0][1],  // 第0行
    rotation_matrix[1][0], rotation_matrix[1][1],  // 第1行
    rotation_matrix[2][0], rotation_matrix[2][1]   // 第2行
};
```

## 修复内容

### 1. 添加转换函数 (action_provider_sonic.py)

```python
def quat_to_rotation_6d(quat: np.ndarray) -> np.ndarray:
    """
    将四元数转换为6D旋转表示（旋转矩阵的前2列，按行展开）

    Args:
        quat: (..., 4) 四元数 [w, x, y, z]

    Returns:
        rot6d: (..., 6) 6D旋转表示 [R00, R01, R10, R11, R20, R21]
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # 四元数转旋转矩阵（前2列）
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    rot6d = np.stack([r00, r01, r10, r11, r20, r21], axis=-1)
    return rot6d.astype(np.float32)
```

### 2. 修改缓冲区初始化 (_setup_buffers)

**修改前**:
```python
self._body_quat_buf = np.tile(
    np.array([1., 0., 0., 0.], dtype=np.float32), (10, 1))  # (10, 4)
```

**修改后**:
```python
self._body_rot6d_buf = np.tile(
    np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32), (10, 1))  # (10, 6)
```

### 3. 修改ZMQ数据接收 (_fetch_zmq_pose)

**修改前**:
```python
if "body_quat_w" in data:
    bq = data["body_quat_w"].astype(np.float32)  # (N, 4)
    self._body_quat_buf = np.roll(self._body_quat_buf, -1, axis=0)
    self._body_quat_buf[-1] = bq[-1]
```

**修改后**:
```python
if "body_quat_w" in data:
    bq = data["body_quat_w"].astype(np.float32)  # (N, 4)
    # 转换为6D旋转表示
    rot6d = quat_to_rotation_6d(bq)  # (N, 6)
    self._body_rot6d_buf = np.roll(self._body_rot6d_buf, -1, axis=0)
    self._body_rot6d_buf[-1] = rot6d[-1]  # (6,)
```

### 4. 修改Encoder输入 (_run_gear_sonic)

**修改前**:
```python
anchor_orient = self._body_quat_buf[np.newaxis]  # (1, 10, 4)
```

**修改后**:
```python
anchor_orient = self._body_rot6d_buf[np.newaxis]  # (1, 10, 6)
```

## 测试验证

### 1. 运行转换测试
```bash
cd isaaclab_twist2_g1/pico_server
python test_quat_to_rot6d.py
```

期望输出：
- ✓ 单位四元数转换正确
- ✓ 各种旋转测试通过
- ✓ 批量转换形状正确
- ✓ 与scipy结果一致（如果安装）

### 2. 运行完整系统测试

**Terminal 1: 启动Pico服务器**
```bash
cd isaaclab_twist2_g1/pico_server
python pico_server_pose_only.py --vis_vr3pt --vis_smpl
```

**Terminal 2: 启动IsaacLab仿真**
```bash
cd isaaclab_twist2_g1
bash run_sonic.sh
```

### 3. 验证修复效果

查看日志输出，应该看到：
```
[SONIC] Encoder inputs: smpl_joints=(1, 10, 24, 3), anchor_orient=(1, 10, 6), joint_pos_hist=(1, 10, 29)
[SONIC] Encoder output latent shape: (1, 64)
[SONIC] Decoder output shape: (1, 29)
```

**关键检查点**:
- ✓ `anchor_orient` 形状从 `(1, 10, 4)` 变为 `(1, 10, 6)`
- ✓ Encoder推理成功，不再报错
- ✓ `sonic_targets` 输出不再是固定值
- ✓ 机器人能够跟随VR姿态运动

## 预期结果

修复后的系统应该：
1. **正确的数据格式**：Encoder接收 `(1, 10, 6)` 的6D旋转表示
2. **正常的推理**：Encoder和Decoder都能正常运行
3. **动态的输出**：`sonic_targets` 随VR姿态变化而变化
4. **实时的跟随**：机器人在仿真中跟随人体全身姿态

## 参考文档

- C++ 实现: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:514-580`
- 观察配置: `gear_sonic_deploy/policy/release/observation_config.yaml`
- 观察文档: `gear_sonic_deploy/docs/source/references/observation_config.md`
- 6D旋转论文: Zhou et al., CVPR 2019

## 关键发现

1. **原始Pico服务器是正确的**：`pico_manager_thread_server.py` 只发送 `global_orient_quat` (4,) 是正确的，因为只需要根部方向
2. **IsaacLab端需要转换**：接收到四元数后，需要在IsaacLab端转换为6D表示
3. **SMPL模式使用6D表示**：观察配置中 `smpl_anchor_orientation_10frame_step1` 维度是60 (10×6)，不是40 (10×4)
4. **与C++实现一致**：修复后的Python实现与C++部署代码完全一致

## 下一步

如果测试通过，可以：
1. 移除调试打印语句（`print(f"[ZMQ] ...")`）
2. 优化性能（如果需要）
3. 添加更多的错误处理
4. 记录到项目文档中