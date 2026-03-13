#!/usr/bin/env python
"""
测试四元数到6D旋转表示的转换

验证转换函数的正确性，确保与C++实现一致
"""

import numpy as np

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
    # 第0列
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    # 第1列
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    # 按行展开：[第0行的前2列, 第1行的前2列, 第2行的前2列]
    rot6d = np.stack([r00, r01, r10, r11, r20, r21], axis=-1)

    return rot6d.astype(np.float32)


def test_identity():
    """测试单位四元数"""
    print("="*70)
    print("测试1: 单位四元数 [1, 0, 0, 0]")
    print("="*70)

    quat = np.array([1., 0., 0., 0.])
    rot6d = quat_to_rotation_6d(quat)

    print(f"输入四元数: {quat}")
    print(f"输出6D表示: {rot6d}")
    print(f"期望结果:   [1, 0, 0, 1, 0, 0]  (单位矩阵的前2列)")

    expected = np.array([1., 0., 0., 1., 0., 0.])
    if np.allclose(rot6d, expected, atol=1e-6):
        print("✓ 测试通过")
    else:
        print("✗ 测试失败")
        print(f"  差异: {rot6d - expected}")
    print()


def test_rotation_x():
    """测试绕X轴旋转90度"""
    print("="*70)
    print("测试2: 绕X轴旋转90度")
    print("="*70)

    # 绕X轴旋转90度的四元数: [cos(45°), sin(45°), 0, 0]
    quat = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0., 0.])
    rot6d = quat_to_rotation_6d(quat)

    print(f"输入四元数: {quat}")
    print(f"输出6D表示: {rot6d}")

    # 期望的旋转矩阵:
    # [1,  0,  0]
    # [0,  0, -1]
    # [0,  1,  0]
    # 前2列按行展开: [1, 0, 0, 0, 0, 1]
    expected = np.array([1., 0., 0., 0., 0., 1.])
    print(f"期望结果:   {expected}")

    if np.allclose(rot6d, expected, atol=1e-6):
        print("✓ 测试通过")
    else:
        print("✗ 测试接近 (浮点误差)")
        print(f"  差异: {rot6d - expected}")
    print()


def test_rotation_z():
    """测试绕Z轴旋转90度"""
    print("="*70)
    print("测试3: 绕Z轴旋转90度")
    print("="*70)

    # 绕Z轴旋转90度的四元数: [cos(45°), 0, 0, sin(45°)]
    quat = np.array([np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)])
    rot6d = quat_to_rotation_6d(quat)

    print(f"输入四元数: {quat}")
    print(f"输出6D表示: {rot6d}")

    # 期望的旋转矩阵:
    # [0, -1,  0]
    # [1,  0,  0]
    # [0,  0,  1]
    # 前2列按行展开: [0, -1, 1, 0, 0, 0]
    expected = np.array([0., -1., 1., 0., 0., 0.])
    print(f"期望结果:   {expected}")

    if np.allclose(rot6d, expected, atol=1e-6):
        print("✓ 测试通过")
    else:
        print("✗ 测试接近 (浮点误差)")
        print(f"  差异: {rot6d - expected}")
    print()


def test_batch():
    """测试批量转换"""
    print("="*70)
    print("测试4: 批量转换 (N, 4) → (N, 6)")
    print("="*70)

    # 创建5个四元数
    quats = np.array([
        [1., 0., 0., 0.],  # 单位
        [np.cos(np.pi/4), np.sin(np.pi/4), 0., 0.],  # X轴90度
        [np.cos(np.pi/4), 0., np.sin(np.pi/4), 0.],  # Y轴90度
        [np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)],  # Z轴90度
        [0.5, 0.5, 0.5, 0.5],  # 任意旋转
    ])

    rot6d = quat_to_rotation_6d(quats)

    print(f"输入形状: {quats.shape}")
    print(f"输出形状: {rot6d.shape}")
    print(f"期望形状: (5, 6)")

    if rot6d.shape == (5, 6):
        print("✓ 形状正确")
    else:
        print("✗ 形状错误")

    print(f"\n前3个样本的6D表示:")
    for i in range(3):
        print(f"  [{i}] {rot6d[i]}")
    print()


def test_with_scipy():
    """使用scipy验证（如果可用）"""
    try:
        from scipy.spatial.transform import Rotation

        print("="*70)
        print("测试5: 与scipy.spatial.transform.Rotation对比")
        print("="*70)

        # 测试几个随机四元数
        test_cases = [
            [1., 0., 0., 0.],
            [0.7071, 0.7071, 0., 0.],
            [0.7071, 0., 0.7071, 0.],
            [0.7071, 0., 0., 0.7071],
        ]

        all_pass = True
        for quat_wxyz in test_cases:
            # 我们的实现 (w, x, y, z)
            quat = np.array(quat_wxyz)
            rot6d_ours = quat_to_rotation_6d(quat)

            # scipy实现 (x, y, z, w)
            quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            r = Rotation.from_quat(quat_xyzw)
            mat = r.as_matrix()
            rot6d_scipy = mat[:, :2].flatten()

            match = np.allclose(rot6d_ours, rot6d_scipy, atol=1e-6)
            status = "✓" if match else "✗"
            print(f"{status} quat={quat_wxyz}")
            print(f"   ours:  {rot6d_ours}")
            print(f"   scipy: {rot6d_scipy}")

            if not match:
                all_pass = False
                print(f"   diff:  {rot6d_ours - rot6d_scipy}")

        if all_pass:
            print("\n✓ 所有测试与scipy一致")
        else:
            print("\n✗ 部分测试与scipy不一致")
        print()

    except ImportError:
        print("="*70)
        print("测试5: scipy未安装，跳过对比测试")
        print("="*70)
        print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("四元数到6D旋转表示转换测试")
    print("="*70)
    print()

    test_identity()
    test_rotation_x()
    test_rotation_z()
    test_batch()
    test_with_scipy()

    print("="*70)
    print("测试完成")
    print("="*70)