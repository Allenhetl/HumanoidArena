# Pico 视频串流修复说明

## 问题描述

用户报告 Pico 无法接收到 Isaac Lab 的视频串流。

## 根本原因分析

通过对比 `sim_main.py`（工作正常）和 `sim_main_recreate.py`（视频串流失败），发现了以下关键差异：

### 1. 缺少 multi_image_writer 清理

**问题**：`sim_main_recreate.py` 的 `cleanup_all()` 函数中缺少对全局 `multi_image_writer` 的清理。

**影响**：
- 共享内存没有正确释放
- ImageServer 无法正确读取相机数据
- 导致视频串流失败

**位置**：
- `sim_main.py:193-197` - 有正确的清理代码
- `sim_main_recreate.py:129-141` - 缺少清理代码

### 2. 重启逻辑不正确

**问题**：原来的实现使用 `sys.exit(0)` 退出程序，期望外部脚本重启。

**影响**：
- 需要额外的 shell 脚本包装
- 增加了复杂性
- 不符合用户要求（重启应该在 sim_main 内部完成）

## 修复方案

### 修复 1：添加 multi_image_writer 清理

在 `sim_main_recreate.py` 的 `cleanup_all()` 函数中添加：

```python
# Clean up global shared memory writer from camera_state.py
try:
    from tasks.common_observations.camera_state import multi_image_writer
    print("[sim_main_recreate] Cleaning up global camera multi_image_writer...")
    multi_image_writer.cleanup()
except Exception as e:
    print(f"[sim_main_recreate] Failed to cleanup camera shared memory: {e}")
```

**位置**：`sim_main_recreate.py:135-141`

### 修复 2：使用 subprocess.Popen() 实现进程重启

将 `save_and_reset` 和 `discard_and_reset` 的退出逻辑改为：

```python
# 4. 启动新进程并退出当前进程
print("[RESET] 🔄 Starting new process and exiting current one...")

# 启动新进程（detached，不等待）
import subprocess
subprocess.Popen([sys.executable] + sys.argv,
               start_new_session=True,
               stdin=subprocess.DEVNULL,
               stdout=None,
               stderr=None)

# 清理并退出当前进程
cleanup_all()
print("[RESET] ✅ New process started, exiting current process...")
sys.exit(0)
```

**位置**：
- `sim_main_recreate.py:506-520` (save_and_reset)
- `sim_main_recreate.py:547-561` (discard_and_reset)

**工作原理**：
- `subprocess.Popen()` 启动一个完全独立的新进程
- `start_new_session=True` 创建新的会话，避免继承父进程状态
- 当前进程清理资源后退出
- 新进程从头开始，确保 PhysX 状态完全清空
- 保持相同的命令行参数

## 参数验证

对比 `run.sh` 和 `run_recreate.sh`，确认所有参数一致：

```bash
# 两者都使用相同的参数
--device cuda
--enable_cameras
--task "${TASK_NAME}"
--robot_type g129
--enable_dex3_dds
--image_transport xrobot
--image_xrobot_host 10.42.0.35
--image_xrobot_port 12345
--image_xrobot_width 480
--image_xrobot_height 320
--image_xrobot_bitrate 4194304
--image_fps 30
--image_xrobot_ffmpeg /usr/bin/ffmpeg
--recording_save_dir /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/0315/zk
--seed "${SEED}"
```

## 测试建议

1. **启动仿真**：
   ```bash
   cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
   ./run_recreate.sh
   ```

2. **验证视频串流**：
   - 检查 Pico 是否能接收到视频
   - 确认视频流畅度和延迟

3. **测试重启功能**：
   - 按下手柄的 save_and_reset 按钮
   - 观察程序是否自动重启（不退出）
   - 确认重启后视频串流仍然正常

4. **检查日志**：
   - 查找 `[sim_main_recreate] Cleaning up global camera multi_image_writer...`
   - 查找 `[RESET] 🔄 Restarting Python process...`
   - 确认没有共享内存相关的错误

## 预期效果

- ✅ Pico 能够正常接收视频串流
- ✅ 按下 reset 按钮后程序自动重启（不需要外部脚本）
- ✅ 重启后视频串流继续正常工作
- ✅ 录制数据可以正确 replay（因为使用完全重启，PhysX 状态完全清空）

## 技术细节

### 为什么需要清理 multi_image_writer？

`camera_state.py` 在模块加载时创建了一个全局的 `MultiImageWriter` 对象：

```python
# camera_state.py:31
multi_image_writer = MultiImageWriter()
```

这个对象使用共享内存（shared memory）来传递图像数据给 `ImageServer`。如果不正确清理：
- 共享内存会保持锁定状态
- 新的进程无法访问或重新创建共享内存
- ImageServer 读取到的是旧数据或空数据

### 为什么使用 os.execv()？

`os.execv()` 的优势：
1. **完全重启**：替换整个进程内存空间，等同于重新运行程序
2. **清空 PhysX 状态**：所有 C++ 扩展模块都会重新加载
3. **保持确定性**：与从头启动程序完全相同，确保 replay 可用
4. **无需外部脚本**：在 Python 代码内部完成，简化部署

替代方案的问题：
- `sys.exit(0)` + shell 循环：需要额外脚本，增加复杂性
- `env.reset()`：不够彻底，PhysX 内部状态不会清空，导致 replay 失败
- `gym.make()`：同样不够彻底，随机数生成器状态不会重置

## 相关文件

- `sim_main_recreate.py` - 主程序（已修复）
- `run_recreate.sh` - 启动脚本（无需修改）
- `camera_state.py` - 相机状态和共享内存管理
- `image_server.py` - 图像服务器（读取共享内存并发送到 Pico）

## 修复日期

2026-03-16
