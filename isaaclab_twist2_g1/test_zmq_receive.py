#!/usr/bin/env python3
"""Test script to verify ZMQ pose data reception from pico_server_pose_only.py"""

import sys
import os
import time
import json
import numpy as np

# Add GR00T path for ZMQPoller
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GROOT_ROOT = "/home/dreams/Users/taowen/GR00T-WholeBodyControl"
if _GROOT_ROOT not in sys.path:
    sys.path.insert(0, _GROOT_ROOT)

try:
    from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
except ImportError as e:
    print(f"ERROR: Cannot import ZMQPoller: {e}")
    print(f"Make sure GR00T-WholeBodyControl is available at: {_GROOT_ROOT}")
    sys.exit(1)

_HEADER_SIZE = 1280

def parse_zmq_pose(raw: bytes):
    """Parse ZMQ pose message"""
    try:
        topic_end = raw.index(b"{")
        header_bytes = raw[topic_end: topic_end + _HEADER_SIZE]
        payload = raw[topic_end + _HEADER_SIZE:]
        header = json.loads(header_bytes.rstrip(b"\x00").decode("utf-8"))

        print(f"\n[HEADER] Fields in message:")
        for f in header.get("fields", []):
            print(f"  - {f['name']}: shape={f['shape']}, dtype={f['dtype']}")

        _DTYPE_MAP = {
            "f32": (np.float32, 4), "f64": (np.float64, 8),
            "i32": (np.int32, 4),   "i64": (np.int64, 8),
            "u8":  (np.uint8,  1),  "bool": (np.bool_,  1),
        }
        result, offset = {}, 0
        for f in header.get("fields", []):
            np_dtype, itemsize = _DTYPE_MAP.get(f["dtype"], (np.float32, 4))
            n = int(np.prod(f["shape"]))
            arr = np.frombuffer(payload[offset: offset + n * itemsize],
                                dtype=np_dtype).reshape(f["shape"])
            result[f["name"]] = arr
            offset += n * itemsize
        return result
    except Exception as e:
        print(f"[ERROR] Parse error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    host = "localhost"
    port = 5556

    print("=" * 70)
    print("ZMQ Pose Data Reception Test")
    print("=" * 70)
    print(f"Connecting to: tcp://{host}:{port}")
    print(f"Topic: pose")
    print("=" * 70)

    try:
        poller = ZMQPoller(host=host, port=port, topic="pose")
        print(f"✓ ZMQ Poller created successfully")
    except Exception as e:
        print(f"✗ Failed to create ZMQ Poller: {e}")
        return

    print("\nWaiting for pose messages... (Press Ctrl+C to stop)")
    print("-" * 70)

    msg_count = 0
    last_report = time.time()

    try:
        while True:
            raw = poller.get_data()
            if raw is None:
                time.sleep(0.001)
                continue

            msg_count += 1
            data = parse_zmq_pose(raw)

            if data is not None:
                print(f"\n[MESSAGE #{msg_count}] Received at {time.time():.3f}")
                print(f"  Keys in data: {list(data.keys())}")

                # Check critical fields
                if "smpl_joints" in data:
                    sj = data["smpl_joints"]
                    print(f"  ✓ smpl_joints: shape={sj.shape}, sum={np.abs(sj).sum():.4f}")
                else:
                    print(f"  ✗ smpl_joints: MISSING")

                if "smpl_pose" in data:
                    sp = data["smpl_pose"]
                    print(f"  ✓ smpl_pose: shape={sp.shape}, sum={np.abs(sp).sum():.4f}")
                else:
                    print(f"  ✗ smpl_pose: MISSING")

                if "body_quat_w" in data:
                    bq = data["body_quat_w"]
                    print(f"  ✓ body_quat_w: shape={bq.shape}, value={bq}")
                else:
                    print(f"  ✗ body_quat_w: MISSING")

                if "vr_position" in data:
                    vp = data["vr_position"]
                    print(f"  ✓ vr_position: shape={vp.shape}, range=[{vp.min():.4f}, {vp.max():.4f}]")
                else:
                    print(f"  ✗ vr_position: MISSING")

                if "vr_orientation" in data:
                    vo = data["vr_orientation"]
                    print(f"  ✓ vr_orientation: shape={vo.shape}, range=[{vo.min():.4f}, {vo.max():.4f}]")
                else:
                    print(f"  ✗ vr_orientation: MISSING")

            # Report rate every 5 seconds
            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - last_report
                rate = msg_count / elapsed if elapsed > 0 else 0
                print(f"\n[STATS] Message rate: {rate:.2f} Hz")
                msg_count = 0
                last_report = now

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        try:
            poller.close()
        except:
            pass

if __name__ == "__main__":
    main()
