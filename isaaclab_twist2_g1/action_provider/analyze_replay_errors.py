#!/usr/bin/env python3
"""
Replay误差分析工具 - 可视化录制数据与仿真状态的差异

使用方法:
    python analyze_replay_errors.py ./replay_debug_logs/recording_20250313_120000.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_debug_data(json_file):
    """加载debug JSON数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def plot_errors(data, output_dir):
    """绘制误差曲线"""
    frames = [f['frame'] for f in data['frames']]

    # 提取各项误差
    errors = {
        'root_pos': [f['errors']['root_pos_l2'] for f in data['frames']],
        'root_quat': [f['errors']['root_quat_l2'] for f in data['frames']],
        'root_lin_vel': [f['errors']['root_lin_vel_l2'] for f in data['frames']],
        'root_ang_vel': [f['errors']['root_ang_vel_l2'] for f in data['frames']],
        'joint_pos': [f['errors']['joint_pos_l2'] for f in data['frames']],
        'joint_vel': [f['errors']['joint_vel_l2'] for f in data['frames']],
    }

    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Replay Error Analysis', fontsize=16)

    titles = [
        ('root_pos', 'Root Position Error (m)'),
        ('root_quat', 'Root Quaternion Error'),
        ('root_lin_vel', 'Root Linear Velocity Error (m/s)'),
        ('root_ang_vel', 'Root Angular Velocity Error (rad/s)'),
        ('joint_pos', 'Joint Position Error (rad)'),
        ('joint_vel', 'Joint Velocity Error (rad/s)'),
    ]

    for idx, (key, title) in enumerate(titles):
        ax = axes[idx // 2, idx % 2]
        ax.plot(frames, errors[key], linewidth=1.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('L2 Error')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # 标记最大误差点
        max_idx = np.argmax(errors[key])
        max_frame = frames[max_idx]
        max_error = errors[key][max_idx]
        ax.plot(max_frame, max_error, 'ro', markersize=8)
        ax.annotate(f'Max: {max_error:.4f}\n@Frame {max_frame}',
                   xy=(max_frame, max_error),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # 保存图表
    output_file = output_dir / 'error_analysis.png'
    plt.savefig(output_file, dpi=150)
    print(f"✅ 误差曲线已保存: {output_file}")

    return errors


def analyze_error_patterns(data, errors):
    """分析误差模式"""
    print("\n" + "="*80)
    print("误差模式分析")
    print("="*80)

    frames = [f['frame'] for f in data['frames']]
    total_frames = len(frames)

    # 1. 误差增长趋势
    print("\n1. 误差增长趋势:")
    for key in ['root_pos', 'joint_pos']:
        err = np.array(errors[key])
        # 计算前10%和后10%的平均误差
        early_avg = np.mean(err[:total_frames//10])
        late_avg = np.mean(err[-total_frames//10:])
        growth_rate = (late_avg - early_avg) / early_avg * 100 if early_avg > 0 else 0

        print(f"  {key:20s}: 初期={early_avg:.6f}, 后期={late_avg:.6f}, "
              f"增长率={growth_rate:+.1f}%")

    # 2. 误差突变检测
    print("\n2. 误差突变检测 (误差增长 > 50%):")
    for key in ['root_pos', 'root_lin_vel', 'joint_pos']:
        err = np.array(errors[key])
        # 计算相邻帧的误差变化率
        err_diff = np.diff(err)
        err_change_rate = np.abs(err_diff / (err[:-1] + 1e-8))

        # 找出变化率 > 0.5 的帧
        spike_indices = np.where(err_change_rate > 0.5)[0]
        if len(spike_indices) > 0:
            print(f"  {key:20s}: 检测到 {len(spike_indices)} 个突变点")
            for idx in spike_indices[:5]:  # 只显示前5个
                frame = frames[idx]
                change = err_change_rate[idx] * 100
                print(f"    Frame {frame}: 误差变化 {change:.1f}%")
        else:
            print(f"  {key:20s}: 无明显突变")

    # 3. 周期性分析
    print("\n3. 误差周期性分析:")
    for key in ['root_pos', 'joint_pos']:
        err = np.array(errors[key])
        # 简单的自相关分析
        if len(err) > 20:
            autocorr = np.correlate(err - np.mean(err), err - np.mean(err), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]

            # 找第一个局部最大值（排除lag=0）
            peaks = []
            for i in range(2, min(50, len(autocorr)-1)):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.3:  # 相关性阈值
                        peaks.append((i, autocorr[i]))

            if peaks:
                period, corr = peaks[0]
                print(f"  {key:20s}: 检测到周期性 (周期≈{period}帧, 相关性={corr:.2f})")
            else:
                print(f"  {key:20s}: 无明显周期性")


def find_problem_frames(data, errors, thresholds):
    """找出问题帧"""
    print("\n" + "="*80)
    print("问题帧识别")
    print("="*80)

    frames = [f['frame'] for f in data['frames']]

    for key, threshold in thresholds.items():
        problem_frames = [f for f, e in zip(frames, errors[key]) if e > threshold]
        if problem_frames:
            print(f"\n{key} 误差 > {threshold}:")
            print(f"  问题帧: {problem_frames[:10]}")  # 只显示前10个
            if len(problem_frames) > 10:
                print(f"  ... 共 {len(problem_frames)} 帧")
        else:
            print(f"\n{key}: ✅ 所有帧误差 < {threshold}")


def generate_report(data, errors, output_dir):
    """生成分析报告"""
    report_file = output_dir / 'error_analysis_report.txt'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Replay误差分析报告\n")
        f.write("="*80 + "\n\n")

        # 基本信息
        f.write(f"总帧数: {data['metadata']['total_frames']}\n")
        f.write(f"分析时间: {data['metadata']['timestamp']}\n\n")

        # 最大误差
        f.write("最大误差统计:\n")
        f.write("-"*80 + "\n")
        for key, max_error in data['max_errors'].items():
            frame = data['max_error_frames'][key]
            f.write(f"  {key:20s}: {max_error:.6f}  (Frame {frame})\n")

        # 平均误差
        f.write("\n平均误差统计:\n")
        f.write("-"*80 + "\n")
        for key in errors.keys():
            avg_error = np.mean(errors[key])
            std_error = np.std(errors[key])
            f.write(f"  {key:20s}: 平均={avg_error:.6f}, 标准差={std_error:.6f}\n")

        # 误差分布
        f.write("\n误差分布 (百分位数):\n")
        f.write("-"*80 + "\n")
        for key in ['root_pos', 'joint_pos']:
            err = np.array(errors[key])
            p50 = np.percentile(err, 50)
            p90 = np.percentile(err, 90)
            p95 = np.percentile(err, 95)
            p99 = np.percentile(err, 99)
            f.write(f"  {key:20s}: P50={p50:.6f}, P90={p90:.6f}, "
                   f"P95={p95:.6f}, P99={p99:.6f}\n")

    print(f"\n✅ 分析报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='分析replay误差数据')
    parser.add_argument('json_file', type=str, help='Debug JSON文件路径')
    parser.add_argument('--thresholds', type=str, default='0.05,0.1',
                       help='问题帧阈值 (root_pos,joint_pos)，单位: m,rad')
    args = parser.parse_args()

    # 加载数据
    print(f"加载数据: {args.json_file}")
    data = load_debug_data(args.json_file)

    output_dir = Path(args.json_file).parent

    # 绘制误差曲线
    print("\n绘制误差曲线...")
    errors = plot_errors(data, output_dir)

    # 分析误差模式
    analyze_error_patterns(data, errors)

    # 找出问题帧
    thresholds_str = args.thresholds.split(',')
    thresholds = {
        'root_pos': float(thresholds_str[0]),
        'joint_pos': float(thresholds_str[1]) if len(thresholds_str) > 1 else 0.1,
    }
    find_problem_frames(data, errors, thresholds)

    # 生成报告
    generate_report(data, errors, output_dir)

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
