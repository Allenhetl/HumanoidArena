#!/usr/bin/env python3
"""分析 replay 错误日志，找出错误累积的模式"""

import re
import numpy as np
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    """解析日志文件，提取每一帧的错误数据"""
    with open(log_path, 'r') as f:
        content = f.read()

    # 按帧分割
    frames = re.split(r'={80,}\nFrame (\d+)\n={80,}', content)[1:]  # Skip header

    data = {
        'frame': [],
        'root_pos_error': [],
        'root_quat_error': [],
        'root_lin_vel_error': [],
        'root_ang_vel_error': [],
        'joint_pos_error': [],
        'joint_vel_error': [],
    }

    for i in range(0, len(frames), 2):
        frame_num = int(frames[i])
        frame_content = frames[i+1]

        data['frame'].append(frame_num)

        # Extract errors
        root_pos_match = re.search(r'Root Position:.*?Error \(L2\): ([\d.]+) m', frame_content, re.DOTALL)
        root_quat_match = re.search(r'Root Quaternion.*?Error \(L2\): ([\d.]+)', frame_content, re.DOTALL)
        root_lin_vel_match = re.search(r'Root Linear Velocity:.*?Error \(L2\): ([\d.]+) m/s', frame_content, re.DOTALL)
        root_ang_vel_match = re.search(r'Root Angular Velocity:.*?Error \(L2\): ([\d.]+) rad/s', frame_content, re.DOTALL)
        joint_pos_match = re.search(r'Joint Positions.*?Error \(L2\): ([\d.]+) rad', frame_content, re.DOTALL)
        joint_vel_match = re.search(r'Joint Velocities.*?Error \(L2\): ([\d.]+) rad/s', frame_content, re.DOTALL)

        data['root_pos_error'].append(float(root_pos_match.group(1)) if root_pos_match else 0)
        data['root_quat_error'].append(float(root_quat_match.group(1)) if root_quat_match else 0)
        data['root_lin_vel_error'].append(float(root_lin_vel_match.group(1)) if root_lin_vel_match else 0)
        data['root_ang_vel_error'].append(float(root_ang_vel_match.group(1)) if root_ang_vel_match else 0)
        data['joint_pos_error'].append(float(joint_pos_match.group(1)) if joint_pos_match else 0)
        data['joint_vel_error'].append(float(joint_vel_match.group(1)) if joint_vel_match else 0)

    return data

def analyze_error_growth(data):
    """分析错误增长率"""
    frames = np.array(data['frame'])

    print("="*80)
    print("错误增长率分析")
    print("="*80)

    for key in ['root_pos_error', 'root_quat_error', 'root_lin_vel_error',
                'root_ang_vel_error', 'joint_pos_error', 'joint_vel_error']:
        errors = np.array(data[key])

        # 计算增长率 (每帧的平均增长)
        if len(errors) > 1:
            growth_rates = np.diff(errors)
            avg_growth = np.mean(growth_rates)
            max_growth = np.max(growth_rates)
            max_growth_frame = frames[np.argmax(growth_rates) + 1]

            # 计算指数增长率
            if errors[0] > 0:
                relative_growth = (errors[-1] / errors[0]) ** (1 / len(errors))
            else:
                relative_growth = 0

            print(f"\n{key}:")
            print(f"  初始值: {errors[0]:.6f}")
            print(f"  最终值: {errors[-1]:.6f}")
            print(f"  平均每帧增长: {avg_growth:.6f}")
            print(f"  最大单帧增长: {max_growth:.6f} (Frame {max_growth_frame})")
            print(f"  相对增长率: {relative_growth:.6f}")

            # 找出增长最快的区间
            window_size = 10
            if len(errors) > window_size:
                window_growth = []
                for i in range(len(errors) - window_size):
                    growth = errors[i+window_size] - errors[i]
                    window_growth.append(growth)
                max_window_idx = np.argmax(window_growth)
                max_window_growth = window_growth[max_window_idx]
                print(f"  最快增长区间: Frame {frames[max_window_idx]} - {frames[max_window_idx+window_size]}, 增长 {max_window_growth:.6f}")

def find_divergence_point(data, threshold_multiplier=10):
    """找出错误开始显著增长的帧"""
    print("\n" + "="*80)
    print("错误发散点分析")
    print("="*80)

    frames = np.array(data['frame'])

    for key in ['root_pos_error', 'root_quat_error', 'root_lin_vel_error',
                'root_ang_vel_error', 'joint_pos_error', 'joint_vel_error']:
        errors = np.array(data[key])

        # 计算前10帧的平均误差作为基准
        baseline = np.mean(errors[:min(10, len(errors))])
        threshold = baseline * threshold_multiplier

        # 找出第一个超过阈值的帧
        divergence_frames = np.where(errors > threshold)[0]
        if len(divergence_frames) > 0:
            divergence_frame = frames[divergence_frames[0]]
            divergence_error = errors[divergence_frames[0]]
            print(f"\n{key}:")
            print(f"  基准误差 (前10帧平均): {baseline:.6f}")
            print(f"  阈值 ({threshold_multiplier}x基准): {threshold:.6f}")
            print(f"  发散帧: Frame {divergence_frame}, 误差 {divergence_error:.6f}")

def main():
    log_path = '/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/replay_debug_logs/Isaac-Move-Football-G129-Dex3-Wholebody_1773326197574686.txt'

    print("解析日志文件...")
    data = parse_log_file(log_path)
    print(f"解析完成，共 {len(data['frame'])} 帧\n")

    analyze_error_growth(data)
    find_divergence_point(data)

    print("\n" + "="*80)
    print("关键发现:")
    print("="*80)
    print("1. 检查错误增长率最高的状态变量")
    print("2. 检查错误发散点，看是否有特定事件触发")
    print("3. 如果所有误差都以相似速率增长，可能是系统性参数不匹配")
    print("4. 如果某个误差增长特别快，可能是该状态的设置或控制有问题")

if __name__ == "__main__":
    main()
