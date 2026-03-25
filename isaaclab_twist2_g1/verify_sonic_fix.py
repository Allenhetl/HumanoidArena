#!/usr/bin/env python3
"""
快速验证 SONIC Encoder 输入数据是否正确填充
运行此脚本检查 encoder 输入中的 motion 数据是否不再全为 0
"""

import re
import sys
import subprocess
import time

def check_sonic_logs():
    """检查 SONIC 日志输出，验证 motion 数据是否正确填充"""

    print("=" * 80)
    print("SONIC Encoder 输入数据验证工具")
    print("=" * 80)
    print()
    print("此工具会检查 run_sonic.sh 的日志输出，验证以下字段是否有数据：")
    print("  - motion_pos_step5")
    print("  - motion_vel_step5")
    print("  - anchor")
    print("  - anchor_step5")
    print("  - lowerbody_pos")
    print("  - lowerbody_vel")
    print()
    print("如果这些字段全为 [0.0000, 0.0000]，说明修复未生效。")
    print("如果这些字段有实际数值，说明修复成功。")
    print()
    print("=" * 80)
    print()

    # 检查是否有 run_sonic.sh 进程
    try:
        result = subprocess.run(
            ["pgrep", "-f", "sim_main.py.*sonic"],
            capture_output=True,
            text=True
        )
        if not result.stdout.strip():
            print("❌ 错误: run_sonic.sh 未运行")
            print()
            print("请先启动 run_sonic.sh:")
            print("  bash run_sonic.sh")
            print()
            return False
        else:
            print(f"✓ 检测到 sim_main.py 进程 (PID: {result.stdout.strip()})")
    except Exception as e:
        print(f"⚠ 警告: 无法检查进程状态: {e}")

    print()
    print("请查看 run_sonic.sh 的输出，找到类似以下的日志：")
    print()
    print("[SONIC][ENCODER_BLOCKS] motion_pos_step5=[...] motion_vel_step5=[...] ...")
    print()
    print("=" * 80)
    print()
    print("✅ 修复成功的标志:")
    print("  motion_pos_step5=[-0.6905, 0.7588]  # 有实际数值")
    print("  motion_vel_step5=[...]              # 有实际数值")
    print("  anchor=[-0.8282, 0.4905]            # 有实际数值")
    print()
    print("❌ 修复失败的标志:")
    print("  motion_pos_step5=[0.0000, 0.0000]  # 全为 0")
    print("  motion_vel_step5=[0.0000, 0.0000]  # 全为 0")
    print("  anchor=[0.0000, 0.0000]            # 全为 0")
    print()
    print("=" * 80)
    print()

    return True

if __name__ == "__main__":
    check_sonic_logs()

    print("提示:")
    print("  1. 如果看到 motion 数据全为 0，请重启 run_sonic.sh")
    print("  2. 如果重启后仍然全为 0，请检查代码修改是否生效")
    print("  3. 如果看到实际数值，说明修复成功！")
    print()
