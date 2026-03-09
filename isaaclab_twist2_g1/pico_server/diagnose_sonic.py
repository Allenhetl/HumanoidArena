#!/usr/bin/env python
"""
诊断 SONIC 推理问题
检查 ZMQ 数据接收和 SMPL 数据有效性
"""

import numpy as np
import zmq
import time

def parse_zmq_pose(raw_bytes):
    """解析ZMQ pose消息 - 使用自定义格式

    消息格式: [topic_bytes][1280-byte JSON header][packed binary payload]
    """
    try:
        import json
        import struct

        # 1. 找到topic结束位置（查找第一个{字符，表示JSON header开始）
        header_start = raw_bytes.find(b'{')
        if header_start == -1:
            print("Cannot find JSON header start")
            return None

        topic = raw_bytes[:header_start].decode('utf-8')
        print(f"   Topic: {topic}")

        # 2. 读取1280字节的JSON header
        HEADER_SIZE = 1280
        header_bytes = raw_bytes[header_start:header_start + HEADER_SIZE]
        # 去除padding的null字节
        header_json = header_bytes.rstrip(b'\x00').decode('utf-8')
        header = json.loads(header_json)

        print(f"   Header version: {header.get('v')}")
        print(f"   Header fields: {[f['name'] for f in header.get('fields', [])]}")

        # 3. 解析binary payload
        payload_start = header_start + HEADER_SIZE
        payload = raw_bytes[payload_start:]

        data = {}
        offset = 0

        for field in header['fields']:
            name = field['name']
            dtype = field['dtype']
            shape = field['shape']

            # 计算元素数量
            num_elements = 1
            for dim in shape:
                num_elements *= dim

            # 根据dtype确定字节大小
            if dtype == 'f32':
                element_size = 4
                np_dtype = np.float32
            elif dtype == 'f64':
                element_size = 8
                np_dtype = np.float64
            elif dtype == 'i32':
                element_size = 4
                np_dtype = np.int32
            elif dtype == 'i64':
                element_size = 8
                np_dtype = np.int64
            elif dtype == 'bool' or dtype == 'u8':
                element_size = 1
                np_dtype = np.uint8
            else:
                print(f"Unknown dtype: {dtype}")
                continue

            # 读取数据
            num_bytes = num_elements * element_size
            field_data = payload[offset:offset + num_bytes]
            offset += num_bytes

            # 转换为numpy数组
            arr = np.frombuffer(field_data, dtype=np_dtype)
            if len(shape) > 1:
                arr = arr.reshape(shape)

            data[name] = arr

        return data

    except Exception as e:
        print(f"Failed to parse ZMQ message: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*70)
    print("SONIC 推理诊断工具")
    print("="*70)

    # 连接ZMQ
    print("\n1. 连接ZMQ服务器...")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
    print("   ✓ 已连接到 tcp://localhost:5556")

    # 接收数据
    print("\n2. 等待接收数据...")
    try:
        topic_bytes = socket.recv()
        print(f"   ✓ 收到消息，大小: {len(topic_bytes)} bytes")

        # 解析数据
        print("\n3. 解析数据...")
        data = parse_zmq_pose(topic_bytes)

        if data is None:
            print("   ✗ 解析失败")
            return

        print(f"   ✓ 解析成功")
        print(f"   数据字段: {list(data.keys())}")

        # 检查关键字段
        print("\n4. 检查关键字段...")

        # smpl_joints
        if "smpl_joints" in data:
            sj = np.array(data["smpl_joints"], dtype=np.float32)
            print(f"   ✓ smpl_joints: shape={sj.shape}")
            print(f"     最新帧: {sj[-1]}")
            print(f"     绝对值和: {np.abs(sj[-1]).sum():.4f}")

            if np.abs(sj[-1]).sum() > 0.01:
                print(f"     ✓ SMPL数据有效（非全0）")
            else:
                print(f"     ✗ SMPL数据无效（全0或接近0）")
        else:
            print(f"   ✗ 缺少 smpl_joints 字段")

        # body_quat_w
        if "body_quat_w" in data:
            bq = np.array(data["body_quat_w"], dtype=np.float32)
            print(f"   ✓ body_quat_w: shape={bq.shape}")
            print(f"     最新帧: {bq[-1]}")
        else:
            print(f"   ✗ 缺少 body_quat_w 字段")

        # joint_pos
        if "joint_pos" in data:
            jp = np.array(data["joint_pos"], dtype=np.float32)
            print(f"   ✓ joint_pos: shape={jp.shape}")
            print(f"     最新帧: {jp[-1]}")
        else:
            print(f"   ✗ 缺少 joint_pos 字段")

        # 检查数据流
        print("\n5. 检查数据流...")
        frame_count = 0
        start_time = time.time()

        for i in range(10):
            try:
                topic_bytes = socket.recv()
                frame_count += 1
                print(f"   帧 {frame_count}: {len(topic_bytes)} bytes", end='\r')
            except zmq.Again:
                print(f"\n   ⚠ 超时，只收到 {frame_count} 帧")
                break

        elapsed = time.time() - start_time
        if frame_count > 0:
            fps = frame_count / elapsed
            print(f"\n   ✓ 收到 {frame_count} 帧，FPS: {fps:.2f}")

        print("\n" + "="*70)
        print("诊断完成")
        print("="*70)

        # 给出建议
        print("\n建议:")
        if "smpl_joints" not in data:
            print("  ✗ 缺少 smpl_joints 数据")
            print("    → 检查 pico_server_pose_only.py 是否正确发送数据")
        elif np.abs(sj[-1]).sum() < 0.01:
            print("  ✗ SMPL数据全0或无效")
            print("    → 检查 Pico 设备是否正常工作")
            print("    → 检查身体追踪是否启动")
        else:
            print("  ✓ ZMQ数据接收正常")
            print("    → 问题可能在 SONIC 模型推理部分")
            print("    → 检查 encoder/decoder 模型是否正确加载")
            print("    → 检查输入数据的形状和类型")

    except zmq.Again:
        print("   ✗ 超时，未收到数据")
        print("\n建议:")
        print("  1. 检查 pico_server_pose_only.py 是否正在运行")
        print("  2. 检查端口是否正确（默认5556）")
        print("  3. 检查防火墙设置")
    except Exception as e:
        print(f"   ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()
