#!/bin/bash
# 测试关节位置可视化功能

echo "Testing Joint Position Visualizer Integration"
echo "=============================================="
echo ""
echo "This will launch the simulation with joint visualization enabled."
echo "You should see:"
echo "  1. Isaac Sim window with the robot"
echo "  2. A matplotlib window showing joint positions"
echo ""
echo "Controls:"
echo "  - Left/Right Arrow: Navigate between joint groups"
echo "  - Close matplotlib window or Ctrl+C to stop"
echo ""
echo "Press Enter to continue..."
read

# Run with a simple task
python sim_main.py \
    --task Isaac-Move-Football-G129-Dex3-Wholebody \
    --action_source dds_wholebody \
    --enable_wholebody_dds \
    --headless

echo ""
echo "Test completed!"
