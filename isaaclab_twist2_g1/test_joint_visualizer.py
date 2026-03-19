#!/usr/bin/env python3
"""
测试关节可视化工具（独立测试，不依赖Isaac Sim）
"""

import os
import sys

# Fix Qt plugin path conflict
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

print("="*80)
print("Testing Joint Position Visualizer")
print("="*80)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("\n1. Importing visualizer...")
try:
    from tools.joint_position_visualizer import JointPositionVisualizer
    print("   ✓ Import successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Creating visualizer...")
try:
    viz = JointPositionVisualizer(num_joints=29, window_size=200)
    print("   ✓ Visualizer created")
except Exception as e:
    print(f"   ✗ Creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Starting non-blocking mode...")
try:
    viz.start_non_blocking()
    print("   ✓ Visualizer started")
    print("   You should see a matplotlib window with 6 subplots")
except Exception as e:
    print(f"   ✗ Start failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Simulating data updates...")
print("   Generating fake joint data for 5 seconds...")
print("   Watch the plots update in real-time")
print("   Press Ctrl+C to stop early\n")

import numpy as np
import time

try:
    for t in range(250):  # 5 seconds at 50Hz
        # Generate fake data
        target = np.sin(t * 0.1 + np.arange(29) * 0.2)
        current = target + np.random.randn(29) * 0.05

        viz.update_data(target, current, timestamp=t)

        if t % 50 == 0:
            print(f"   Step {t}/250...")

        time.sleep(0.02)  # 50Hz

except KeyboardInterrupt:
    print("\n   Interrupted by user")

print("\n5. Printing statistics...")
viz.print_statistics()

print("\n" + "="*80)
print("✓ Test completed successfully!")
print("="*80)
print("\nThe matplotlib window should still be open.")
print("Close it manually or press Ctrl+C to exit.")

try:
    import matplotlib.pyplot as plt
    plt.show(block=True)  # Keep window open
except KeyboardInterrupt:
    print("\nExiting...")
