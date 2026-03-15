#!/usr/bin/env python3
"""
验证seed设置和物理参数
在replay时运行此脚本来检查所有关键参数
"""

import torch
import numpy as np

print("=" * 80)
print("Seed和物理参数验证")
print("=" * 80)

# 1. 检查随机数生成器状态
print("\n【随机数生成器状态】")
print("-" * 80)
print(f"PyTorch seed: {torch.initial_seed()}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA seed: {torch.cuda.initial_seed()}")
print(f"NumPy random state (first value): {np.random.get_state()[1][0]}")

# 2. 检查PyTorch确定性设置
print("\n【PyTorch确定性设置】")
print("-" * 80)
print(f"cudnn.deterministic: {torch.backends.cudnn.deterministic}")
print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")

# 3. 生成测试随机数
print("\n【测试随机数生成】")
print("-" * 80)
print("如果seed设置正确，这些值应该每次运行都相同：")
print(f"torch.rand(3): {torch.rand(3)}")
print(f"np.random.rand(3): {np.random.rand(3)}")

print("\n" + "=" * 80)
print("请在录制和replay时都运行此脚本，对比输出是否一致")
print("=" * 80)
