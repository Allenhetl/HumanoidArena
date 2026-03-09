# SONIC 推理问题 - 根本原因与解决方案

## 问题根本原因

**IsaacLab实现发送的是四元数(4维)，但SONIC encoder期望的是6D旋转表示(6维)**

### 证据

1. **C++ 实现 (gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp)**:
   - Line 1618: `smpl_anchor_orientation_10frame_step1` 维度是 **60** (10帧 × 6)
   - Line 1583: `motion_anchor_orientation` 维度是 **6** (1帧 × 6)
   - 实现代码 (Line 514-580):
     ```cpp
     // 将四元数转换为旋转矩阵
     auto rotation_matrix = quat_to_rotation_matrix_d(base_to_ref_quat);

     // 提取旋转矩阵的前2列，按行展开为6个元素
     std::array<double, 6> motion_anchor_ori_b = {
         rotation_matrix[0][0], rotation_matrix[0][1],  // 第0行，前2列
         rotation_matrix[1][0], rotation_matrix[1][1],  // 第1行，前2列
         rotation_matrix[2][0], rotation_matrix[2][1]   // 第2行，前2列
     };
     ```

2. **IsaacLab 实现 (action_provider_sonic.py)**:
   - Line 301-302: 初始化为 `(10, 4)` - 四元数格式
   - Line 365-369: 接收 `(N, 4)` - 四元数格式
   - Line 407: 传给encoder `(1, 10, 4)` - **错误！应该是 (1, 10, 6)**

3. **Pico服务器 (pico_manager_thread_server.py)**:
   - Line 1334-1336: 只发送 `global_orient_quat` (4,)
   - 这是正确的，因为只需要根部方向四元数
   - **但IsaacLab端需要将其转换为6D表示**

## 6D旋转表示说明

6D旋转表示是一种连续的旋转表示方法，比四元数更适合神经网络：
- 取旋转矩阵R (3×3) 的前两列
- 按行展开：`[R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]`
- 优点：连续、无奇异点、易于学习

参考论文: "On the Continuity of Rotation Representations in Neural Networks" (Zhou et al., CVPR 2019)

## 解决方案

### 方案1: 修改 action_provider_sonic.py (推荐)

在IsaacLab端将四元数转换为6D旋转表示：

```python
import numpy as np

def quat_to_rotation_6d(quat):
    """
    将四元数转换为6D旋转表示

    Args:
        quat: (4,) 或 (..., 4) 四元数 [w, x, y, z]

    Returns:
        rot6d: (..., 6) 6D旋转表示
    """
    # 四元数转旋转矩阵
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # 旋转矩阵的9个元素
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)

    r10 = 2*(x*y + w*z)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - w*x)

    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)
    r22 = 1 - 2*(x*x + y*y)

    # 提取前两列，按行展开
    rot6d = np.stack([r00, r01, r10, r11, r20, r21], axis=-1)

    return rot6d

# 修改 action_provider_sonic.py

class SonicActionProvider:
    def _setup_buffers(self):
        # SMPL 历史帧缓冲（encoder 需要 10 帧）
        self._smpl_joints_buf = np.zeros(
            (10, _N_SMPL_JOINTS, 3), dtype=np.float32)   # (10, 24, 3)
        self._smpl_pose_buf   = np.zeros(
            (10, _N_SMPL_POSES,  3), dtype=np.float32)   # (10, 21, 3)

        # 修改：使用6D旋转表示而不是四元数
        self._body_rot6d_buf = np.tile(
            np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32),  # 单位矩阵的前2列
            (10, 1))  # (10, 6)

        # ... 其他代码保持不变

    def _fetch_zmq_pose(self):
        """从 ZMQ 读取最新 POSE 消息，更新 SMPL 历史缓冲。"""
        if self._zmq_poller is None:
            return

        data = self._zmq_poller.poll_latest()
        if data is None:
            return

        # ... 处理 smpl_joints 等数据

        # 接收四元数并转换为6D表示
        if "body_quat_w" in data:
            bq = data["body_quat_w"].astype(np.float32)  # (N, 4)

            # 转换为6D旋转表示
            rot6d = quat_to_rotation_6d(bq)  # (N, 6)

            # 更新缓冲区
            self._body_rot6d_buf = np.roll(self._body_rot6d_buf, -1, axis=0)
            self._body_rot6d_buf[-1] = rot6d[-1]  # (6,)

        # ... 其他代码

    def _run_gear_sonic(self) -> np.ndarray:
        """GEAR-SONIC 推理"""
        if self._encoder is None or self._decoder is None:
            return self._sonic_default_np.copy()

        if not self._smpl_data_valid:
            return self._sonic_default_np.copy()

        # 准备 Encoder 输入
        smpl_joints_in = self._smpl_joints_buf[np.newaxis]  # (1, 10, 24, 3)
        anchor_orient  = self._body_rot6d_buf[np.newaxis]   # (1, 10, 6) ← 修改
        joint_pos_hist = np.tile(
            self._robot_joint_pos[np.newaxis, np.newaxis],
            (1, 10, 1)).astype(np.float32)  # (1, 10, 29)

        # Encoder 推理
        enc_inputs = {
            self._encoder.get_inputs()[0].name: smpl_joints_in,
            self._encoder.get_inputs()[1].name: anchor_orient,
            self._encoder.get_inputs()[2].name: joint_pos_hist,
        }
        latent = self._encoder.run(None, enc_inputs)[0]
        self._latent = latent

        # ... Decoder 推理保持不变
```

### 方案2: 验证修复

创建测试脚本验证转换正确性：

```python
#!/usr/bin/env python
"""测试四元数到6D旋转表示的转换"""

import numpy as np
from scipy.spatial.transform import Rotation

def quat_to_rotation_6d(quat):
    """四元数转6D旋转表示"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - w*z)
    r10 = 2*(x*y + w*z)
    r11 = 1 - 2*(x*x + z*z)
    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)

    rot6d = np.stack([r00, r01, r10, r11, r20, r21], axis=-1)
    return rot6d

# 测试
test_quat = np.array([1., 0., 0., 0.])  # 单位四元数
rot6d = quat_to_rotation_6d(test_quat)
print(f"单位四元数: {test_quat}")
print(f"6D表示: {rot6d}")
print(f"期望: [1, 0, 0, 1, 0, 0]")

# 使用scipy验证
r = Rotation.from_quat([0., 0., 0., 1.])  # scipy使用 [x,y,z,w] 顺序
mat = r.as_matrix()
rot6d_scipy = mat[:, :2].flatten()
print(f"Scipy验证: {rot6d_scipy}")

# 测试随机四元数
random_quat = np.array([0.7071, 0.7071, 0., 0.])  # 绕X轴旋转90度
rot6d_random = quat_to_rotation_6d(random_quat)
print(f"\n随机四元数: {random_quat}")
print(f"6D表示: {rot6d_random}")
```

## 实施步骤

1. **备份当前代码**:
   ```bash
   cp action_provider_sonic.py action_provider_sonic.py.backup
   ```

2. **应用修改**:
   - 修改 `_setup_buffers()`: 将 `_body_quat_buf` 改为 `_body_rot6d_buf`，形状从 `(10, 4)` 改为 `(10, 6)`
   - 修改 `_fetch_zmq_pose()`: 添加 `quat_to_rotation_6d()` 转换
   - 修改 `_run_gear_sonic()`: 使用 `_body_rot6d_buf` 而不是 `_body_quat_buf`

3. **测试**:
   ```bash
   # Terminal 1: 启动Pico服务器
   cd isaaclab_twist2_g1/pico_server
   python pico_server_pose_only.py --vis_vr3pt

   # Terminal 2: 启动IsaacLab
   cd isaaclab_twist2_g1
   bash run_sonic.sh
   ```

4. **验证**:
   - 检查 `sonic_targets` 是否不再是固定值
   - 观察机器人是否跟随VR姿态运动
   - 查看encoder输入形状是否正确 `(1, 10, 6)`

## 预期结果

修复后：
- ✓ Encoder输入形状正确: `anchor_orient` 从 `(1, 10, 4)` 变为 `(1, 10, 6)`
- ✓ 推理正常工作，输出不再是固定的默认姿态
- ✓ 机器人能够跟随VR全身姿态运动

## 参考

- C++ 实现: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:514-580`
- 观察配置: `gear_sonic_deploy/policy/release/observation_config.yaml:53`
- 6D旋转论文: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019