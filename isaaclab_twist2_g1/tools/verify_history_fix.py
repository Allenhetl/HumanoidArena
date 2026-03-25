#!/usr/bin/env python3
"""
验证历史缓冲区大小修复

测试gather_temporal_window是否能正确从46帧历史中提取10帧（step5采样）
"""

import numpy as np

_STEP5_FRAMES = 10
_STEP5_STRIDE = 5
_STEP1_FRAMES = 10
_STEP5_HISTORY_LEN = (_STEP5_FRAMES - 1) * _STEP5_STRIDE + 1  # 46

def gather_temporal_window(hist: np.ndarray, num_frames: int, stride: int) -> np.ndarray:
    """Take the latest temporal window using `stride` over a history buffer."""
    hist = np.asarray(hist, dtype=np.float32)
    required = (num_frames - 1) * stride + 1
    if hist.shape[0] < required:
        raise ValueError(
            f"History too short for num_frames={num_frames}, stride={stride}: "
            f"len={hist.shape[0]}, required={required}"
        )
    start = hist.shape[0] - required
    window = hist[start::stride]
    if window.shape[0] != num_frames:
        raise ValueError(
            f"Temporal window shape mismatch: got {window.shape[0]}, expected {num_frames}"
        )
    return window.astype(np.float32)

print("=" * 80)
print("验证历史缓冲区大小修复")
print("=" * 80)

# 测试1：step5采样（需要46帧）
print("\n测试1：step5采样（10帧，stride=5）")
print(f"  需要历史长度：{_STEP5_HISTORY_LEN} 帧")

robot_joint_pos_hist = np.random.randn(_STEP5_HISTORY_LEN, 29).astype(np.float32)
try:
    result = gather_temporal_window(robot_joint_pos_hist, _STEP5_FRAMES, _STEP5_STRIDE)
    print(f"  ✅ 成功！输出shape: {result.shape}，预期: ({_STEP5_FRAMES}, 29)")
    assert result.shape == (_STEP5_FRAMES, 29), f"Shape mismatch: {result.shape}"
except Exception as e:
    print(f"  ❌ 失败：{e}")

# 测试2：step1采样（需要10帧）
print("\n测试2：step1采样（10帧，stride=1）")
print(f"  需要历史长度：{_STEP1_FRAMES} 帧")

try:
    result = gather_temporal_window(robot_joint_pos_hist, _STEP1_FRAMES, 1)
    print(f"  ✅ 成功！输出shape: {result.shape}，预期: ({_STEP1_FRAMES}, 29)")
    assert result.shape == (_STEP1_FRAMES, 29), f"Shape mismatch: {result.shape}"
except Exception as e:
    print(f"  ❌ 失败：{e}")

# 测试3：验证采样正确性
print("\n测试3：验证step5采样正确性")
test_hist = np.arange(46).reshape(46, 1).astype(np.float32)  # [0, 1, 2, ..., 45]
result = gather_temporal_window(test_hist, 10, 5)
expected_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
expected = test_hist[expected_indices]
print(f"  输入：[0, 1, 2, ..., 45]")
print(f"  输出：{result.flatten()}")
print(f"  预期：{expected.flatten()}")
if np.allclose(result, expected):
    print(f"  ✅ 采样正确！")
else:
    print(f"  ❌ 采样错误！")

print("\n" + "=" * 80)
print("修复总结")
print("=" * 80)
print("\n问题：")
print("  _robot_joint_pos_hist 只有10帧，但step5采样需要46帧")
print("\n修复：")
print("  将 _robot_joint_pos_hist 大小从10帧改为46帧")
print("  在decoder输入构建时，用gather_temporal_window提取最近10帧")
print("\n现在可以重新启动IsaacLab测试！")
