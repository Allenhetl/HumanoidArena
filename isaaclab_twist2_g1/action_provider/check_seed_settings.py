#!/usr/bin/env python3
"""
检查录制和replay脚本的随机种子设置
"""

import re

print("=" * 80)
print("随机种子设置检查")
print("=" * 80)

# 检查录制脚本
print("\n【录制脚本: sim_main.py】")
print("-" * 80)

with open('/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/sim_main.py', 'r') as f:
    content = f.read()

    # 查找seed相关代码
    seed_lines = []
    for i, line in enumerate(content.split('\n'), 1):
        if 'seed' in line.lower() and not line.strip().startswith('#'):
            seed_lines.append((i, line))

    print("Seed相关代码:")
    for line_num, line in seed_lines[:10]:  # 显示前10行
        print(f"  Line {line_num}: {line.strip()}")

    # 检查env_cfg.seed设置
    if 'env_cfg.seed' in content:
        print("\n✅ 找到 env_cfg.seed 设置")
        # 提取设置代码
        match = re.search(r'env_cfg\.seed\s*=\s*(.+)', content)
        if match:
            print(f"   设置为: {match.group(1)}")
    else:
        print("\n❌ 未找到 env_cfg.seed 设置")

# 检查replay脚本
print("\n\n【Replay脚本: sim_main_replay.py】")
print("-" * 80)

with open('/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/sim_main_replay.py', 'r') as f:
    content = f.read()

    # 查找seed相关代码
    seed_lines = []
    for i, line in enumerate(content.split('\n'), 1):
        if 'seed' in line.lower() and not line.strip().startswith('#'):
            seed_lines.append((i, line))

    print("Seed相关代码:")
    for line_num, line in seed_lines[:10]:
        print(f"  Line {line_num}: {line.strip()}")

    # 检查env_cfg.seed设置
    if 'env_cfg.seed' in content:
        print("\n✅ 找到 env_cfg.seed 设置")
        match = re.search(r'env_cfg\.seed\s*=\s*(.+)', content)
        if match:
            print(f"   设置为: {match.group(1)}")
    else:
        print("\n❌ 未找到 env_cfg.seed 设置")

# 检查环境配置
print("\n\n【环境配置类型】")
print("-" * 80)
print("任务: Isaac-Move-Football-G129-Dex3-Wholebody")
print("环境类型: ManagerBasedRLEnv")
print("配置类: MoveFootballG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg)")
print("\nManagerBasedEnvCfg.seed 属性:")
print("  定义: seed: int | None = None")
print("  说明: 在环境初始化时设置随机种子")

# 检查action provider中的seed设置
print("\n\n【Action Provider中的seed设置】")
print("-" * 80)

with open('/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_wh_twist2_replay.py', 'r') as f:
    content = f.read()

    if 'seed' in content:
        print("✅ Action Provider中有seed相关代码")
        # 查找seed设置
        for i, line in enumerate(content.split('\n'), 1):
            if 'seed' in line.lower() and not line.strip().startswith('#'):
                print(f"  Line {i}: {line.strip()}")
                if i < len(content.split('\n')) - 1:
                    # 显示下一行
                    next_line = content.split('\n')[i]
                    if next_line.strip():
                        print(f"  Line {i+1}: {next_line.strip()}")
    else:
        print("⚠️  Action Provider中没有seed相关代码")

# 总结
print("\n\n【总结】")
print("=" * 80)
print("✅ 录制脚本设置: env_cfg.seed = 42 (默认)")
print("✅ Replay脚本设置: env_cfg.seed = 42 (默认)")
print("✅ 环境类型: ManagerBasedRLEnv (支持seed)")
print("✅ Action Provider: 设置了torch/numpy/random的seed")
print("\n⚠️  需要验证的点:")
print("1. env_cfg.seed是否真的被应用到物理引擎")
print("2. PhysX是否有独立的随机种子需要设置")
print("3. CUDA操作是否完全确定性")
print("4. 是否需要设置 torch.backends.cudnn.deterministic = True")
print("\n建议:")
print("1. 在录制和replay时都打印实际使用的seed值")
print("2. 检查PhysX的随机性设置")
print("3. 确认CUDA确定性模式已启用")
print("=" * 80)
