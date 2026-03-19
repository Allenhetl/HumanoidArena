#!/usr/bin/env python3
"""
带关节位置可视化的仿真脚本
用于PD参数调试
"""

import os
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import torch
import gymnasium as gym
from pathlib import Path
import threading

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Joint Position Visualization for PD Tuning")
parser.add_argument("--task", type=str, default="Isaac-Move-Football-G129-Dex3-Wholebody",
                    help="task name")
parser.add_argument("--num_envs", type=int, default=1, help="number of environments")
parser.add_argument("--visualize_joints", action="store_true", default=True,
                    help="Enable joint position visualization")
parser.add_argument("--window_size", type=int, default=200,
                    help="Sliding window size for visualization")
parser.add_argument("--env_id", type=int, default=0,
                    help="Which environment to visualize (if num_envs > 1)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入其他模块
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tools.joint_position_visualizer import JointPositionVisualizer


def main():
    # 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print(f"\n{'='*80}")
    print(f"Task: {args_cli.task}")
    print(f"Number of environments: {args_cli.num_envs}")
    print(f"Number of actions: {env.action_space.shape}")
    print(f"Number of observations: {env.observation_space.shape}")
    print(f"{'='*80}\n")

    # 获取关节数量
    robot = env.scene["robot"]
    num_joints = robot.num_joints

    print(f"Robot has {num_joints} joints")
    print(f"Joint names: {robot.data.joint_names}")

    # 创建可视化器
    visualizer = None
    if args_cli.visualize_joints:
        print("\nInitializing joint position visualizer...")
        visualizer = JointPositionVisualizer(
            num_joints=num_joints,
            window_size=args_cli.window_size,
            update_interval=50  # 20Hz更新
        )

        # 在单独的线程中启动可视化
        viz_thread = threading.Thread(target=visualizer.start, daemon=True)
        viz_thread.start()

        print("Visualizer started. Use Left/Right arrow keys to navigate joints.")
        print("Press Ctrl+C in terminal to stop.\n")

    # 重置环境
    obs, _ = env.reset()

    # 主循环
    step_count = 0
    try:
        while simulation_app.is_running():
            # 生成随机动作（或使用你的控制器）
            # 这里使用零动作，让机器人保持初始姿态
            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

            # 也可以使用随机动作测试
            # actions = torch.randn_like(actions) * 0.1

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(actions)

            # 更新可视化
            if visualizer is not None and step_count % 2 == 0:  # 每2步更新一次
                # 获取目标位置和当前位置
                target_pos = robot.data.joint_pos_target[args_cli.env_id].cpu().numpy()
                current_pos = robot.data.joint_pos[args_cli.env_id].cpu().numpy()

                visualizer.update_data(target_pos, current_pos, timestamp=step_count)

            step_count += 1

            # 每1000步打印统计
            if visualizer is not None and step_count % 1000 == 0:
                visualizer.print_statistics()

    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
        if visualizer is not None:
            print("\nFinal statistics:")
            visualizer.print_statistics()

    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
