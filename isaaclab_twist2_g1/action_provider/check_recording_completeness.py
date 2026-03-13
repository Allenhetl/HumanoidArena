#!/usr/bin/env python3
"""
验证录制数据的完整性
用法: python check_recording_completeness.py <npz_file_path>
"""

import sys
import numpy as np

def check_recording_completeness(npz_file):
    """检查录制文件是否包含所有必需的数据"""

    print("=" * 80)
    print("录制数据完整性检查")
    print("=" * 80)
    print(f"\n文件: {npz_file}")

    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"\n❌ 无法加载文件: {e}")
        return False

    # 定义必需的字段
    required_fields = {
        # 基础状态
        'robot_qpos_before_decimation': ('Joint positions', True),
        'robot_qvel_before_decimation': ('Joint velocities', True),
        'robot_root_position': ('Root position', True),
        'robot_root_orientation': ('Root orientation', True),

        # 速度数据（旧格式）
        'robot_root_lin_vel_local': ('Root linear velocity (local)', True),

        # 速度数据（新格式 - 推荐）
        'robot_root_lin_vel_world': ('Root linear velocity (world)', False),
        'robot_root_ang_vel_local': ('Root angular velocity (local)', False),
        'robot_root_ang_vel_world': ('Root angular velocity (world)', False),

        # 其他重要数据
        'robot_twist2_inference_qpos': ('TWIST2 inference output', True),
        'robot_obs_buf': ('Observation buffer', True),
    }

    print(f"\n总帧数: {data.get('num_frames', 'N/A')}")

    print(f"\n【数据完整性检查】")
    print("-" * 80)

    missing_required = []
    missing_optional = []
    present = []

    for field, (desc, is_required) in required_fields.items():
        if field in data:
            shape = data[field].shape if hasattr(data[field], 'shape') else 'N/A'
            present.append((field, desc, shape))
            print(f"✓ {desc:40s} {field:35s} {shape}")
        else:
            if is_required:
                missing_required.append((field, desc))
                print(f"✗ {desc:40s} {field:35s} [必需]")
            else:
                missing_optional.append((field, desc))
                print(f"⚠ {desc:40s} {field:35s} [可选]")

    # 检查数据格式版本
    print(f"\n【数据格式版本】")
    print("-" * 80)

    has_world_vel = 'robot_root_lin_vel_world' in data
    has_ang_vel = 'robot_root_ang_vel_world' in data or 'robot_root_ang_vel_local' in data

    if has_world_vel and has_ang_vel:
        print("✅ 新格式 (完整版本) - 包含世界坐标系速度和角速度")
        version = "v2_complete"
    elif has_world_vel or has_ang_vel:
        print("⚠️  新格式 (部分) - 部分新字段存在")
        version = "v2_partial"
    else:
        print("⚠️  旧格式 - 仅包含局部坐标系线速度")
        version = "v1_legacy"

    # 显示Frame 0的数据示例
    if present:
        print(f"\n【Frame 0 数据示例】")
        print("-" * 80)

        if 'robot_root_position' in data:
            print(f"Root position: {data['robot_root_position'][0]}")

        if 'robot_root_orientation' in data:
            print(f"Root orientation: {data['robot_root_orientation'][0]}")

        if 'robot_root_lin_vel_world' in data:
            print(f"Root lin vel (world): {data['robot_root_lin_vel_world'][0]}")
        elif 'robot_root_lin_vel_local' in data:
            print(f"Root lin vel (local): {data['robot_root_lin_vel_local'][0]}")

        if 'robot_root_ang_vel_world' in data:
            print(f"Root ang vel (world): {data['robot_root_ang_vel_world'][0]}")
        elif 'robot_root_ang_vel_local' in data:
            print(f"Root ang vel (local): {data['robot_root_ang_vel_local'][0]}")

        if 'robot_qpos_before_decimation' in data:
            qpos = data['robot_qpos_before_decimation'][0]
            print(f"Joint pos (前5个): {qpos[:5]}")

    # 总结
    print(f"\n【总结】")
    print("=" * 80)

    if missing_required:
        print(f"\n❌ 缺少必需字段 ({len(missing_required)}):")
        for field, desc in missing_required:
            print(f"   - {desc} ({field})")
        print(f"\n⚠️  此文件不完整，无法用于完整replay")
        return False

    if missing_optional:
        print(f"\n⚠️  缺少可选字段 ({len(missing_optional)}):")
        for field, desc in missing_optional:
            print(f"   - {desc} ({field})")
        print(f"\n✓ 此文件可用于replay，但使用旧格式（{version}）")
        print(f"   建议：使用更新后的录制脚本重新录制以获得完整数据")
    else:
        print(f"\n✅ 所有字段完整！数据格式: {version}")
        print(f"   此文件包含完整的状态信息，可用于高质量replay")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_recording_completeness.py <npz_file_path>")
        print("\n或者检查最新的录制文件:")
        print("python check_recording_completeness.py latest")
        sys.exit(1)

    npz_file = sys.argv[1]

    if npz_file == "latest":
        import os
        import glob

        recording_dir = "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data"
        files = glob.glob(os.path.join(recording_dir, "*.npz"))

        if not files:
            print(f"❌ 在 {recording_dir} 中没有找到录制文件")
            sys.exit(1)

        # 按修改时间排序，获取最新的
        npz_file = max(files, key=os.path.getmtime)
        print(f"检查最新文件: {os.path.basename(npz_file)}\n")

    check_recording_completeness(npz_file)
