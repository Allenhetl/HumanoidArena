# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Replay Debug Logger - 用于对比录制数据和replay仿真状态的差异

使用方法:
    logger = ReplayDebugLogger(log_dir="./replay_debug_logs")
    logger.log_frame(frame_idx, recorded_data, simulated_data)
    logger.close()
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional


class ReplayDebugLogger:
    """记录replay过程中录制数据与仿真状态的差异"""

    def __init__(self, log_dir: str = "./replay_debug_logs", log_name: Optional[str] = None):
        """
        初始化日志记录器

        Args:
            log_dir: 日志保存目录
            log_name: 日志文件名（不含扩展名），如果为None则使用时间戳
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 生成日志文件名
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"replay_debug_{timestamp}"

        self.log_file = os.path.join(log_dir, f"{log_name}.txt")
        self.json_file = os.path.join(log_dir, f"{log_name}.json")
        self.summary_file = os.path.join(log_dir, f"{log_name}_summary.txt")

        # 打开日志文件
        self.log_fp = open(self.log_file, 'w', encoding='utf-8')

        # 存储所有帧的差异数据（用于JSON输出）
        self.frame_data = []

        # 统计信息
        self.max_errors = {
            'root_pos': 0.0,
            'root_quat': 0.0,
            'root_lin_vel': 0.0,
            'root_ang_vel': 0.0,
            'joint_pos': 0.0,
            'joint_vel': 0.0,
            'applied_torque': 0.0,
        }
        self.max_error_frames = {
            'root_pos': -1,
            'root_quat': -1,
            'root_lin_vel': -1,
            'root_ang_vel': -1,
            'joint_pos': -1,
            'joint_vel': -1,
            'applied_torque': -1,
        }

        # 写入文件头
        self._write_header()

        print(f"[ReplayDebugLogger] 日志文件: {self.log_file}")
        print(f"[ReplayDebugLogger] JSON文件: {self.json_file}")
        print(f"[ReplayDebugLogger] 摘要文件: {self.summary_file}")

    def _write_header(self):
        """写入日志文件头"""
        self.log_fp.write("=" * 100 + "\n")
        self.log_fp.write("Replay Debug Log - 录制数据 vs 仿真状态对比\n")
        self.log_fp.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_fp.write("=" * 100 + "\n\n")

    def log_frame(self, frame_idx: int, recorded: Dict[str, Any], simulated: Dict[str, Any]):
        """
        记录一帧的对比数据

        Args:
            frame_idx: 帧索引
            recorded: 录制数据字典，包含:
                - root_pos: [3] 根节点位置
                - root_quat: [4] 根节点四元数 (w,x,y,z)
                - root_lin_vel: [3] 根节点线速度
                - root_ang_vel: [3] 根节点角速度
                - joint_pos: [29] 关节位置
                - joint_vel: [29] 关节速度
                - applied_torque: [29] 施加的力矩 (可选)
            simulated: 仿真数据字典（格式同上）
        """
        # 转换为numpy数组
        rec = self._to_numpy(recorded)
        sim = self._to_numpy(simulated)

        # 计算差异
        errors = self._compute_errors(rec, sim)

        # 更新最大误差
        self._update_max_errors(frame_idx, errors)

        # 写入文本日志
        self._write_frame_log(frame_idx, rec, sim, errors)

        # 存储JSON数据
        self._store_frame_data(frame_idx, rec, sim, errors)

    def _to_numpy(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """将数据转换为numpy数组"""
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                result[key] = value
            else:
                result[key] = np.array(value)
        return result

    def _compute_errors(self, rec: Dict, sim: Dict) -> Dict[str, float]:
        """计算各项误差"""
        errors = {}

        # L2范数误差
        for key in ['root_pos', 'root_quat', 'root_lin_vel', 'root_ang_vel', 'joint_pos', 'joint_vel', 'applied_torque']:
            if key in rec and key in sim:
                diff = rec[key] - sim[key]
                errors[f'{key}_l2'] = float(np.linalg.norm(diff))
                errors[f'{key}_max'] = float(np.max(np.abs(diff)))
                errors[f'{key}_mean'] = float(np.mean(np.abs(diff)))

        return errors

    def _update_max_errors(self, frame_idx: int, errors: Dict[str, float]):
        """更新最大误差记录"""
        for key in self.max_errors.keys():
            error_key = f'{key}_l2'
            if error_key in errors:
                if errors[error_key] > self.max_errors[key]:
                    self.max_errors[key] = errors[error_key]
                    self.max_error_frames[key] = frame_idx

    def _write_frame_log(self, frame_idx: int, rec: Dict, sim: Dict, errors: Dict):
        """写入单帧日志"""
        self.log_fp.write(f"\n{'='*100}\n")
        self.log_fp.write(f"Frame {frame_idx}\n")
        self.log_fp.write(f"{'='*100}\n\n")

        # Root位置
        self.log_fp.write("Root Position:\n")
        self.log_fp.write(f"  Recorded:  {self._format_array(rec['root_pos'])}\n")
        self.log_fp.write(f"  Simulated: {self._format_array(sim['root_pos'])}\n")
        self.log_fp.write(f"  Error (L2): {errors['root_pos_l2']:.6f} m\n")
        self.log_fp.write(f"  Error (Max): {errors['root_pos_max']:.6f} m\n\n")

        # Root姿态
        self.log_fp.write("Root Quaternion (w,x,y,z):\n")
        self.log_fp.write(f"  Recorded:  {self._format_array(rec['root_quat'])}\n")
        self.log_fp.write(f"  Simulated: {self._format_array(sim['root_quat'])}\n")
        self.log_fp.write(f"  Error (L2): {errors['root_quat_l2']:.6f}\n")
        self.log_fp.write(f"  Error (Max): {errors['root_quat_max']:.6f}\n\n")

        # Root线速度
        self.log_fp.write("Root Linear Velocity:\n")
        self.log_fp.write(f"  Recorded:  {self._format_array(rec['root_lin_vel'])}\n")
        self.log_fp.write(f"  Simulated: {self._format_array(sim['root_lin_vel'])}\n")
        self.log_fp.write(f"  Error (L2): {errors['root_lin_vel_l2']:.6f} m/s\n")
        self.log_fp.write(f"  Error (Max): {errors['root_lin_vel_max']:.6f} m/s\n\n")

        # Root角速度
        self.log_fp.write("Root Angular Velocity:\n")
        self.log_fp.write(f"  Recorded:  {self._format_array(rec['root_ang_vel'])}\n")
        self.log_fp.write(f"  Simulated: {self._format_array(sim['root_ang_vel'])}\n")
        self.log_fp.write(f"  Error (L2): {errors['root_ang_vel_l2']:.6f} rad/s\n")
        self.log_fp.write(f"  Error (Max): {errors['root_ang_vel_max']:.6f} rad/s\n\n")

        # 关节位置（只显示前5个和最大误差的关节）
        joint_pos_diff = np.abs(rec['joint_pos'] - sim['joint_pos'])
        max_joint_idx = np.argmax(joint_pos_diff)
        self.log_fp.write("Joint Positions (29 DOFs):\n")
        self.log_fp.write(f"  Error (L2): {errors['joint_pos_l2']:.6f} rad\n")
        self.log_fp.write(f"  Error (Max): {errors['joint_pos_max']:.6f} rad (joint {max_joint_idx})\n")
        self.log_fp.write(f"  Error (Mean): {errors['joint_pos_mean']:.6f} rad\n")
        self.log_fp.write(f"  First 5 joints recorded:  {self._format_array(rec['joint_pos'][:5])}\n")
        self.log_fp.write(f"  First 5 joints simulated: {self._format_array(sim['joint_pos'][:5])}\n")
        self.log_fp.write(f"  Max error joint [{max_joint_idx}]: rec={rec['joint_pos'][max_joint_idx]:.6f}, "
                         f"sim={sim['joint_pos'][max_joint_idx]:.6f}\n\n")

        # 关节速度
        joint_vel_diff = np.abs(rec['joint_vel'] - sim['joint_vel'])
        max_vel_idx = np.argmax(joint_vel_diff)
        self.log_fp.write("Joint Velocities (29 DOFs):\n")
        self.log_fp.write(f"  Error (L2): {errors['joint_vel_l2']:.6f} rad/s\n")
        self.log_fp.write(f"  Error (Max): {errors['joint_vel_max']:.6f} rad/s (joint {max_vel_idx})\n")
        self.log_fp.write(f"  Error (Mean): {errors['joint_vel_mean']:.6f} rad/s\n\n")

        # 施加的力矩（如果有）
        if 'applied_torque' in rec and 'applied_torque' in sim:
            torque_diff = np.abs(rec['applied_torque'] - sim['applied_torque'])
            max_torque_idx = np.argmax(torque_diff)
            self.log_fp.write("Applied Torques (29 DOFs):\n")
            self.log_fp.write(f"  Error (L2): {errors['applied_torque_l2']:.6f} Nm\n")
            self.log_fp.write(f"  Error (Max): {errors['applied_torque_max']:.6f} Nm (joint {max_torque_idx})\n")
            self.log_fp.write(f"  Error (Mean): {errors['applied_torque_mean']:.6f} Nm\n")
            self.log_fp.write(f"  First 5 joints recorded:  {self._format_array(rec['applied_torque'][:5])}\n")
            self.log_fp.write(f"  First 5 joints simulated: {self._format_array(sim['applied_torque'][:5])}\n")
            self.log_fp.write(f"  Max error joint [{max_torque_idx}]: rec={rec['applied_torque'][max_torque_idx]:.6f}, "
                             f"sim={sim['applied_torque'][max_torque_idx]:.6f}\n\n")

        # 刷新缓冲区
        self.log_fp.flush()

    def _format_array(self, arr: np.ndarray, precision: int = 6) -> str:
        """格式化数组输出"""
        return '[' + ', '.join([f'{x:.{precision}f}' for x in arr.flatten()]) + ']'

    def _store_frame_data(self, frame_idx: int, rec: Dict, sim: Dict, errors: Dict):
        """存储帧数据到JSON"""
        frame_info = {
            'frame': frame_idx,
            'recorded': {k: v.tolist() for k, v in rec.items()},
            'simulated': {k: v.tolist() for k, v in sim.items()},
            'errors': errors
        }
        self.frame_data.append(frame_info)

    def write_summary(self):
        """写入摘要信息"""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("Replay Debug Summary - 最大误差统计\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"总帧数: {len(self.frame_data)}\n\n")

            f.write("最大误差 (L2范数):\n")
            f.write("-" * 100 + "\n")
            for key, max_error in self.max_errors.items():
                frame = self.max_error_frames[key]
                f.write(f"  {key:20s}: {max_error:.6f}  (Frame {frame})\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write("建议:\n")
            f.write("-" * 100 + "\n")

            # 根据误差大小给出建议
            if self.max_errors['root_pos'] > 0.1:
                f.write(f"⚠️  Root位置误差较大 ({self.max_errors['root_pos']:.3f}m)，"
                       f"在Frame {self.max_error_frames['root_pos']}达到峰值\n")
                f.write("   建议检查: 1) 初始状态设置 2) 物理参数 3) 接触力模型\n\n")

            if self.max_errors['root_lin_vel'] > 0.5:
                f.write(f"⚠️  Root线速度误差较大 ({self.max_errors['root_lin_vel']:.3f}m/s)，"
                       f"在Frame {self.max_error_frames['root_lin_vel']}达到峰值\n")
                f.write("   建议检查: 1) 摩擦系数 2) 地面接触 3) 质量分布\n\n")

            if self.max_errors['joint_pos'] > 0.1:
                f.write(f"⚠️  关节位置误差较大 ({self.max_errors['joint_pos']:.3f}rad)，"
                       f"在Frame {self.max_error_frames['joint_pos']}达到峰值\n")
                f.write("   建议检查: 1) PD控制器参数 2) 关节限位 3) 动作目标设置\n\n")

            if 'applied_torque' in self.max_errors and self.max_errors['applied_torque'] > 5.0:
                f.write(f"⚠️  施加力矩误差较大 ({self.max_errors['applied_torque']:.3f}Nm)，"
                       f"在Frame {self.max_error_frames['applied_torque']}达到峰值\n")
                f.write("   建议检查: 1) PD控制器参数一致性 2) 关节位置/速度误差 3) 力矩限制设置\n\n")

        print(f"[ReplayDebugLogger] 摘要已保存: {self.summary_file}")

    def close(self):
        """关闭日志记录器"""
        # 防止重复关闭
        if not hasattr(self, 'log_fp') or self.log_fp.closed:
            print(f"[ReplayDebugLogger] Logger already closed")
            return

        # 写入JSON文件
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_frames': len(self.frame_data),
                    'timestamp': datetime.now().isoformat(),
                },
                'max_errors': self.max_errors,
                'max_error_frames': self.max_error_frames,
                'frames': self.frame_data
            }, f, indent=2)

        # 写入摘要
        self.write_summary()

        # 关闭文本日志
        self.log_fp.close()

        print(f"[ReplayDebugLogger] 日志已关闭，共记录 {len(self.frame_data)} 帧")
        print(f"[ReplayDebugLogger] 查看详细日志: {self.log_file}")
        print(f"[ReplayDebugLogger] 查看JSON数据: {self.json_file}")
        print(f"[ReplayDebugLogger] 查看摘要: {self.summary_file}")
