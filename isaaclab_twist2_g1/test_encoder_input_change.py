"""
分析sonic_debug.log中encoder输入是否真的在变化
"""
import re
import numpy as np

# 读取日志
with open('sonic_debug.log', 'r') as f:
    lines = f.readlines()

# 提取encoder输入的SMPL部分
smpl_inputs = []
for line in lines:
    if 'DEBUG_ENC_DETAIL] Encoder输入SMPL部分(922-932):' in line:
        # 提取数组
        match = re.search(r'\[(.*?)\]', line)
        if match:
            values = [float(x) for x in match.group(1).split()]
            smpl_inputs.append(values)

if len(smpl_inputs) < 2:
    print("日志中没有足够的encoder输入数据")
else:
    print(f"找到 {len(smpl_inputs)} 帧encoder输入")
    
    # 计算相邻帧的差异
    diffs = []
    for i in range(1, len(smpl_inputs)):
        diff = np.abs(np.array(smpl_inputs[i]) - np.array(smpl_inputs[i-1])).max()
        diffs.append(diff)
    
    print(f"\n相邻帧SMPL输入的最大变化:")
    print(f"  平均: {np.mean(diffs):.6f}")
    print(f"  最大: {np.max(diffs):.6f}")
    print(f"  最小: {np.min(diffs):.6f}")
    
    if np.max(diffs) < 0.001:
        print("\n⚠️  警告：SMPL输入几乎不变！")
    else:
        print("\n✅ SMPL输入正常变化")
    
    # 打印前5帧
    print(f"\n前5帧SMPL输入（前3维）:")
    for i in range(min(5, len(smpl_inputs))):
        print(f"  帧{i}: {smpl_inputs[i][:3]}")
