#!/usr/bin/env python3
"""
测试不同的quaternion格式转换
生成多个版本，让用户运行replay来判断哪个是正确的站立姿态
"""

import numpy as np

# 录制的quaternion
quat_recorded = np.array([0.70253754, -0.01670154, 0.01430531, 0.71130675])

print("=" * 80)
print("Quaternion格式测试")
print("=" * 80)
print(f"\n录制的quaternion: {quat_recorded}")
print()

# 测试不同的转换方式
test_cases = {
    "原始不变 (假设已经是正确格式)": quat_recorded,

    "wxyz -> xyzw": np.array([
        quat_recorded[1],  # x
        quat_recorded[2],  # y
        quat_recorded[3],  # z
        quat_recorded[0]   # w
    ]),

    "反转顺序 wzyx": np.array([
        quat_recorded[3],
        quat_recorded[2],
        quat_recorded[1],
        quat_recorded[0]
    ]),

    "xyzw -> wxyz (假设录制时搞反了)": np.array([
        quat_recorded[3],  # w
        quat_recorded[0],  # x
        quat_recorded[1],  # y
        quat_recorded[2]   # z
    ]),

    "取负的w分量 (有些库用不同的符号约定)": np.array([
        -quat_recorded[0],
        quat_recorded[1],
        quat_recorded[2],
        quat_recorded[3]
    ]),

    "共轭quaternion (反向旋转)": np.array([
        quat_recorded[0],
        -quat_recorded[1],
        -quat_recorded[2],
        -quat_recorded[3]
    ]),
}

print("\n测试用例:")
print("=" * 80)
for i, (name, quat) in enumerate(test_cases.items(), 1):
    print(f"\n{i}. {name}")
    print(f"   Quaternion: {quat}")

    # 计算Euler角 (假设是xyzw格式)
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # 尝试wxyz解释
    w_wxyz, x_wxyz, y_wxyz, z_wxyz = quat[0], quat[1], quat[2], quat[3]
    t0 = 2.0 * (w_wxyz * x_wxyz + y_wxyz * z_wxyz)
    t1 = 1.0 - 2.0 * (x_wxyz * x_wxyz + y_wxyz * y_wxyz)
    roll_wxyz = np.arctan2(t0, t1)
    t2 = 2.0 * (w_wxyz * y_wxyz - z_wxyz * x_wxyz)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_wxyz = np.arcsin(t2)
    t3 = 2.0 * (w_wxyz * z_wxyz + x_wxyz * y_wxyz)
    t4 = 1.0 - 2.0 * (y_wxyz * y_wxyz + z_wxyz * z_wxyz)
    yaw_wxyz = np.arctan2(t3, t4)

    print(f"   如果是wxyz格式: Roll={np.degrees(roll_wxyz):.2f}°, Pitch={np.degrees(pitch_wxyz):.2f}°, Yaw={np.degrees(yaw_wxyz):.2f}°")

    # 尝试xyzw解释
    x_xyzw, y_xyzw, z_xyzw, w_xyzw = quat[0], quat[1], quat[2], quat[3]
    t0 = 2.0 * (w_xyzw * x_xyzw + y_xyzw * z_xyzw)
    t1 = 1.0 - 2.0 * (x_xyzw * x_xyzw + y_xyzw * y_xyzw)
    roll_xyzw = np.arctan2(t0, t1)
    t2 = 2.0 * (w_xyzw * y_xyzw - z_xyzw * x_xyzw)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_xyzw = np.arcsin(t2)
    t3 = 2.0 * (w_xyzw * z_xyzw + x_xyzw * y_xyzw)
    t4 = 1.0 - 2.0 * (y_xyzw * y_xyzw + z_xyzw * z_xyzw)
    yaw_xyzw = np.arctan2(t3, t4)

    print(f"   如果是xyzw格式: Roll={np.degrees(roll_xyzw):.2f}°, Pitch={np.degrees(pitch_xyzw):.2f}°, Yaw={np.degrees(yaw_xyzw):.2f}°")

print("\n" + "=" * 80)
print("使用方法:")
print("=" * 80)
print("""
在action_provider_wh_twist2_replay.py中，修改第241-242行的转换代码：

# 测试用例1: 原始不变
root_quat_xyzw = root_quat_wxyz  # 不做任何转换

# 测试用例2: wxyz -> xyzw
root_quat_xyzw = np.array([root_quat_wxyz[1], root_quat_wxyz[2],
                           root_quat_wxyz[3], root_quat_wxyz[0]])

# 测试用例3: 反转顺序
root_quat_xyzw = np.array([root_quat_wxyz[3], root_quat_wxyz[2],
                           root_quat_wxyz[1], root_quat_wxyz[0]])

# 测试用例4: xyzw -> wxyz
root_quat_xyzw = np.array([root_quat_wxyz[3], root_quat_wxyz[0],
                           root_quat_wxyz[1], root_quat_wxyz[2]])

# 测试用例5: 取负的w分量
root_quat_xyzw = np.array([-root_quat_wxyz[0], root_quat_wxyz[1],
                           root_quat_wxyz[2], root_quat_wxyz[3]])

# 测试用例6: 共轭quaternion
root_quat_xyzw = np.array([root_quat_wxyz[0], -root_quat_wxyz[1],
                           -root_quat_wxyz[2], -root_quat_wxyz[3]])

每次修改后运行replay，看机器人是否站立。
找到正确的转换后告诉我是哪个测试用例。
""")
