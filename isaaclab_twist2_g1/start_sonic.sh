#!/bin/bash
# 快速启动脚本 - SONIC 全身遥操作（Isaac Lab）

echo "=========================================="
echo "SONIC 全身遥操作启动脚本"
echo "=========================================="
echo ""

# 检查 pico_server 是否运行
if pgrep -f "pico_server_pose_only.py" > /dev/null; then
    echo "✓ pico_server 正在运行"
else
    echo "✗ pico_server 未运行！"
    echo ""
    echo "请先在另一个终端启动 pico_server："
    echo "  cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1"
    echo "  python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl"
    echo ""
    read -p "按 Enter 继续（如果已在其他终端启动），或 Ctrl+C 退出..."
fi

# 检查端口
if netstat -tuln 2>/dev/null | grep -q ":5556"; then
    echo "✓ ZMQ 端口 5556 正在监听"
else
    echo "⚠ 警告: 端口 5556 未监听，pico_server 可能未完全启动"
fi

# 检查 GR00T 路径
if [ -d "/home/dreams/Users/taowen/GR00T-WholeBodyControl" ]; then
    echo "✓ GR00T-WholeBodyControl 路径存在"
else
    echo "✗ 错误: GR00T-WholeBodyControl 路径不存在！"
    exit 1
fi

echo ""
echo "=========================================="
echo "启动 Isaac Lab SONIC 仿真..."
echo "=========================================="
echo ""

# 运行 run_sonic.sh
bash run_sonic.sh
