#!/usr/bin/env python3
"""
sim_main_recreate.py

完全重建环境的版本 - 用于确保 replay 的确定性

主要特性：
1. 保存+重置：完全销毁并重新创建环境（包括 gym.make）
2. 确保 PhysX 状态完全清空
3. 录制从第一个 get_action() 调用开始，确保 Frame 0 是真实物理状态
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import time
import signal
import torch
import gymnasium as gym
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Unitree Simulation with Environment Recreation")
parser.add_argument("--task", type=str, default="Isaac-Move-Football-G129-Dex3-Wholebody", help="task name")
parser.add_argument("--action_source", type=str, default="dds_wholebody",
                   choices=["dds", "file", "trajectory", "policy", "replay", "dds_wholebody", "sonic_wholebody"],
                   help="Action source")
parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dex3 DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire DDS")
parser.add_argument("--enable_wholebody_dds", action="store_true", help="enable wholebody DDS")
parser.add_argument("--model_path", type=str, default="/home/dreams/Users/taowen/HumanoidArena/TWIST2/assets/ckpts/twist2_1017_20k.onnx", help="path to policy model")
parser.add_argument("--control_frequency", type=int, default=50, help="control frequency")
parser.add_argument("--step_hz", type=int, default=50, help="control step frequency")
parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
parser.add_argument("--recording_save_dir", type=str,
                   default="/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data",
                   help="directory to save recordings")
parser.add_argument("--auto_start_recording", action="store_true", help="auto start recording")

# Image streaming parameters (for Pico)
parser.add_argument("--image_transport", type=str, default="zmq", choices=["zmq", "redis", "dds", "xrobot"],
                   help="image transport for streaming")
parser.add_argument("--image_fps", type=int, default=30, help="image streaming fps cap")
parser.add_argument("--image_zmq_port", type=int, default=5555, help="ZMQ port for image streaming")
parser.add_argument("--image_redis_host", type=str, default="localhost", help="Redis host for image streaming")
parser.add_argument("--image_redis_port", type=int, default=6379, help="Redis port for image streaming")
parser.add_argument("--image_redis_db", type=int, default=0, help="Redis db for image streaming")
parser.add_argument("--image_redis_key_prefix", type=str, default="isaac_image", help="Redis key prefix")
parser.add_argument("--image_redis_channel", type=str, default="", help="Redis pubsub channel")
parser.add_argument("--image_dds_topic", type=str, default="rt/isaac_image", help="DDS topic for image streaming")
parser.add_argument("--image_xrobot_host", type=str, default="172.20.10.2", help="XRobot/Pico IP")
parser.add_argument("--image_xrobot_port", type=int, default=12345, help="XRobot/Pico port")
parser.add_argument("--image_xrobot_bitrate", type=int, default=4000000, help="XRobot/Pico H264 bitrate")
parser.add_argument("--image_xrobot_ffmpeg", type=str, default="", help="ffmpeg path for XRobot streaming")

# 添加 Isaac Lab 的参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
simulation_app = AppLauncher(args_cli).app

# 导入其他模块（必须在 AppLauncher 之后）
from layeredcontrol.robot_control_system import RobotController, ControlConfig
from action_provider.create_action_provider import create_action_provider
from dds.dds_create import create_dds_objects
from image_server.image_server import ImageServer
from tools.grass_ground_material import apply_grass_pbr_to_ground
from tools.football_physics_material import apply_football_physics_material
from tools.pitch_lines import create_simple_debug_lines
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# 全局变量
env = None
controller = None
action_provider = None
dds_manager = None
reset_pose_dds = None
sim_state_dds = None
image_server = None


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    print("\n\nReceived interrupt signal, cleaning up...")
    cleanup_all()
    sys.exit(0)


def cleanup_all():
    """清理所有资源"""
    global env, controller, action_provider, dds_manager, image_server, simulation_app

    print("\nCleaning up resources...")

    try:
        if action_provider and hasattr(action_provider, 'cleanup'):
            action_provider.cleanup()
    except Exception as e:
        print(f"Error cleaning up action provider: {e}")

    try:
        if controller:
            controller.stop()
    except Exception as e:
        print(f"Error stopping controller: {e}")

    try:
        if env:
            env.close()
    except Exception as e:
        print(f"Error closing environment: {e}")

    try:
        if dds_manager:
            dds_manager.stop_all_communication()
    except Exception as e:
        print(f"Error stopping DDS: {e}")

    try:
        if image_server:
            image_server._close()
    except Exception as e:
        print(f"Error stopping image server: {e}")

    # Clean up global shared memory writer from camera_state.py
    try:
        from tasks.common_observations.camera_state import multi_image_writer
        print("[sim_main_recreate] Cleaning up global camera multi_image_writer...")
        multi_image_writer.cleanup()
    except Exception as e:
        print(f"[sim_main_recreate] Failed to cleanup camera shared memory: {e}")

    try:
        if simulation_app:
            print("[sim_main_recreate] Closing simulation app...")
            simulation_app.close()
            print("[sim_main_recreate] ✅ Simulation app closed")
    except Exception as e:
        print(f"[sim_main_recreate] ❌ Error closing simulation app: {e}")
        import traceback
        traceback.print_exc()

    print("Cleanup complete")

    # Force exit to ensure all background threads are terminated
    # This is necessary because Isaac Sim may have background threads that prevent normal exit
    print("[sim_main_recreate] Forcing process exit...")
    try:
        os._exit(0)
    except Exception:
        raise SystemExit(0)


def create_environment(args_cli, env_cfg):
    """创建新环境"""
    print("\n" + "="*80)
    print("🌍 CREATING ENVIRONMENT")
    print("="*80)

    try:
        new_env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print(f"✅ Environment created: {args_cli.task}")
        print(f"  Robot init pos: {new_env.cfg.scene.robot.init_state.pos}")
        print(f"  Robot USD: {new_env.cfg.scene.robot.spawn.usd_path}")

        # 初始化环境

        print("✅ Environment initialized")

        # 足球任务：套用草坪 PBR 材质到地面（需在 reset 前，确保 stage 已完整载入）
        if "football" in args_cli.task.lower():
            print("\n🌱 Applying grass material to ground...")
            grass_ok_pre = apply_grass_pbr_to_ground(prim_path="/World/GroundPlane", uv_scale=(150.0, 150.0))
            print(f"[grass_ground_material] before reset apply result: {grass_ok_pre}")
            try:
                apply_football_physics_material(restitution=0.75)
                print("✅ Football physics material applied")
            except Exception as e:
                print(f"⚠️ Football physics material skipped: {e}")
            try:
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                create_simple_debug_lines(stage, line_color=(32.0 / 255.0, 32.0 / 255.0, 32.0 / 255.0))
                print("✅ Pitch lines created")
            except Exception as e:
                print(f"⚠️ Pitch lines skipped: {e}")

        new_env.sim.reset()
        new_env.reset()
        if "football" in args_cli.task.lower():
            grass_ok_post = apply_grass_pbr_to_ground(prim_path="/World/GroundPlane", uv_scale=(150.0, 150.0))
            print(f"[grass_ground_material] after reset apply result: {grass_ok_post}")

        # 设置第一人称视角（仅在非 headless 模式下）
        if not args_cli.headless:
            try:
                import omni.kit.viewport.utility as vp_utils

                # 直接使用已知的相机路径，避免遍历 stage
                cam_path = "/World/envs/env_0/Robot/d435_link/front_cam"

                vp = vp_utils.get_active_viewport()
                if vp is not None:
                    print(f"\n📷 Setting viewport camera to: {cam_path}")
                    vp.set_active_camera(cam_path)

                    # 验证是否设置成功
                    new_cam = vp.get_active_camera()
                    if new_cam == cam_path:
                        print(f"  ✅ Successfully switched to first-person view")
                    else:
                        print(f"  ⚠️ Camera path may not exist yet: {new_cam}")
                else:
                    print("⚠️ No active viewport found")
            except Exception as e:
                print(f"⚠️ Failed to set camera: {e}")
        else:
            print("ℹ️  Running in headless mode, skipping viewport camera setup")

        return new_env

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        raise


def create_controller_and_action_provider(new_env, args_cli):
    """创建控制器和 action provider"""
    print("\n" + "="*80)
    print("🎮 CREATING CONTROLLER AND ACTION PROVIDER")
    print("="*80)

    try:
        # 自动检测并设置 wholebody 模式（与 sim_main.py 保持一致）
        if "Wholebody" in args_cli.task:
            print(f"ℹ️  Task contains 'Wholebody', auto-enabling wholebody mode")
            args_cli.action_source = "dds_wholebody"
            args_cli.enable_wholebody_dds = True

        # 创建 action provider
        print(f"Creating action provider: {args_cli.action_source}...")
        new_action_provider = create_action_provider(new_env, args_cli)
        if new_action_provider is None:
            raise RuntimeError("Failed to create action provider")
        print("✅ Action provider created")

        # 创建控制器
        # 计算 step_hz（与原始 sim_main.py 保持一致）
        use_wholebody = args_cli.action_source in ["dds_wholebody", "sonic_wholebody"]
        if use_wholebody:
            physics_dt = new_env.physics_dt
            decimation = new_env.cfg.decimation
            step_hz = int(1.0 / (physics_dt * decimation))
        else:
            step_hz = args_cli.control_frequency

        control_config = ControlConfig(
            step_hz=step_hz,
            replay_mode=False,
            use_rl_action_mode=use_wholebody,
        )
        new_controller = RobotController(new_env, control_config)
        new_controller.set_action_provider(new_action_provider)
        print("✅ Controller created")

        # 设置 action provider 到 env（用于 camera 访问）
        new_env.action_provider = new_action_provider

        return new_controller, new_action_provider

    except Exception as e:
        print(f"❌ Failed to create controller/action provider: {e}")
        raise


def recreate_environment_completely(args_cli, env_cfg, old_env, old_controller, old_action_provider):
    """
    完全销毁并重新创建环境

    这个函数会：
    1. 清理旧的 action provider
    2. 停止旧的 controller
    3. 关闭旧的 environment
    4. 运行垃圾回收
    5. 清除 Redis 缓存
    6. 创建全新的 environment（调用 gym.make）
    7. 创建全新的 controller 和 action provider

    注意：不关闭 simulation_app，避免需要重新初始化整个 Isaac Sim

    Returns:
        tuple: (new_env, new_controller, new_action_provider)
    """
    print("\n" + "="*80)
    print("🔄 COMPLETE ENVIRONMENT RECREATION")
    print("="*80)
    print("⚠️  Note: You may need to click the window again after recreation")
    print("="*80)

    # Step 1: 清理旧的 action provider
    try:
        if old_action_provider is not None:
            print("[1/6] Cleaning up old action provider...")
            if hasattr(old_action_provider, 'cleanup'):
                old_action_provider.cleanup()
            del old_action_provider
            print("  ✅ Done")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")

    # Step 2: 停止旧的 controller
    try:
        if old_controller is not None:
            print("[2/6] Stopping old controller...")
            if hasattr(old_controller, 'stop'):
                old_controller.stop()
            del old_controller
            print("  ✅ Done")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")

    # Step 3: 关闭旧的 environment
    try:
        if old_env is not None:
            print("[3/6] Closing old environment...")
            old_env.close()
            del old_env
            print("  ✅ Done")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")

    # Step 4: 垃圾回收
    try:
        print("[4/6] Running garbage collection...")
        import gc
        import time
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 等待一下，确保资源完全释放
        time.sleep(0.5)
        print("  ✅ Done")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")

    # Step 4.5: 清除 Redis 缓存（防止旧数据影响新环境）
    try:
        print("[4.5/6] Clearing Redis cache...")
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # 清除 action 相关的键
        keys_to_delete = [
            "action_body_unitree_g1_with_hands",
            "action_hand_left_unitree_g1_with_hands",
            "action_hand_right_unitree_g1_with_hands",
            "action_neck_unitree_g1_with_hands",
            "controller_data",
            "t_action",
            "isaac_reset_trigger",
            # 也清除 state 相关的键，确保完全干净
            "state_body_unitree_g1_with_hands",
            "state_hand_left_unitree_g1_with_hands",
            "state_hand_right_unitree_g1_with_hands",
            "state_neck_unitree_g1_with_hands",
            "t_state",
        ]

        deleted_count = redis_client.delete(*keys_to_delete)
        print(f"  ✅ Cleared {deleted_count} Redis keys")
    except redis.ConnectionError:
        print(f"  ⚠️ Redis not available, skipping cache clear")
    except Exception as e:
        print(f"  ⚠️ Error clearing Redis cache: {e}")

    # Step 5: 创建新环境
    print("[5/6] Creating new environment...")
    new_env = create_environment(args_cli, env_cfg)

    # Step 6: 创建新的 controller 和 action provider
    print("[6/6] Creating new controller and action provider...")
    new_controller, new_action_provider = create_controller_and_action_provider(new_env, args_cli)

    print("="*80)
    print("✅ ENVIRONMENT RECREATION COMPLETE")
    print("="*80 + "\n")

    return new_env, new_controller, new_action_provider


def main():
    """主函数"""
    global env, controller, action_provider, dds_manager, reset_pose_dds, sim_state_dds, image_server

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("="*80)
    print("🚀 ROBOT CONTROL SYSTEM (RECREATE VERSION)")
    print(f"Task: {args_cli.task}")
    print(f"Action source: {args_cli.action_source}")
    print(f"Seed: {args_cli.seed}")
    print("="*80)

    # 解析环境配置
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task

        # 设置随机种子
        seed_value = args_cli.seed if args_cli.seed is not None else 42
        env_cfg.seed = seed_value
        print(f"[CONFIG] Environment seed set to: {seed_value}")

    except Exception as e:
        print(f"❌ Failed to parse environment configuration: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建初始环境
    env = create_environment(args_cli, env_cfg)

    # 创建 ImageServer（用于 Pico 串流）
    try:
        print("\n" + "="*80)
        print("📹 CREATING IMAGE SERVER")
        print("="*80)
        image_server = ImageServer(
            fps=args_cli.image_fps,
            port=args_cli.image_zmq_port,
            Unit_Test=False,
            transport=args_cli.image_transport,
            redis_host=args_cli.image_redis_host,
            redis_port=args_cli.image_redis_port,
            redis_db=args_cli.image_redis_db,
            redis_key_prefix=args_cli.image_redis_key_prefix,
            redis_channel=args_cli.image_redis_channel,
            dds_topic=args_cli.image_dds_topic,
            xrobot_host=args_cli.image_xrobot_host,
            xrobot_port=args_cli.image_xrobot_port,
            xrobot_bitrate=args_cli.image_xrobot_bitrate,
            xrobot_width=None,
            xrobot_height=None,
            xrobot_ffmpeg=args_cli.image_xrobot_ffmpeg or None,
        )
        print(f"✅ Image server created (transport={args_cli.image_transport}, port={args_cli.image_zmq_port})")
    except Exception as e:
        print(f"❌ Failed to create image server: {e}")
        import traceback
        traceback.print_exc()
        image_server = None

    # 创建 DDS
    try:
        print("\n" + "="*80)
        print("📡 CREATING DDS")
        print("="*80)
        reset_pose_dds, sim_state_dds, dds_manager = create_dds_objects(args_cli, env)
        print("✅ DDS created")
    except Exception as e:
        print(f"❌ Failed to create DDS: {e}")
        return

    # 创建 controller 和 action provider
    controller, action_provider = create_controller_and_action_provider(env, args_cli)

    # 立即启动录制（在第一次 get_action() 之前）
    # 这确保从 env.reset() 后就开始捕获所有状态，避免随机序列不同步
    if hasattr(action_provider, '_should_start_recording_on_first_call'):
        if action_provider._should_start_recording_on_first_call:
            print("\n" + "="*80)
            print("🔴 STARTING RECORDING IMMEDIATELY AFTER ENV.RESET()")
            print("="*80)
            action_provider.recording_manager.start_recording()
            action_provider._should_start_recording_on_first_call = False
            print("✅ Recording started to capture all random state from the beginning")
            print("="*80 + "\n")

    # 启动 controller
    print("\n" + "="*80)
    print("▶️  STARTING CONTROLLER")
    print("="*80)
    controller.start()
    print("✅ Controller started")

    # 主循环
    loop_count = 0
    last_stats_time = time.time()

    print("\n" + "="*80)
    print("🔄 ENTERING MAIN LOOP")
    print("="*80)
    print("ℹ️  Note: Please click the Isaac Sim window if rendering appears frozen")
    print("="*80 + "\n")

    try:
        with torch.inference_mode():
            while simulation_app.is_running() and controller.is_running:
                loop_count += 1

                # 检查 Redis 中的 reset 命令
                try:
                    import redis
                    import json

                    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

                    # 检查 isaac_reset_trigger 命令（action_provider发送的）
                    reset_trigger_raw = redis_client.get("isaac_reset_trigger")

                    # 调试：每50次循环打印一次检查状态
                    if loop_count % 50 == 0:
                        if reset_trigger_raw:
                            print(f"[DEBUG] Loop {loop_count}: Found reset_trigger: {reset_trigger_raw}")
                        else:
                            print(f"[DEBUG] Loop {loop_count}: No reset_trigger found")

                    if reset_trigger_raw:
                        reset_trigger = json.loads(reset_trigger_raw)
                        reset_category = reset_trigger.get("reset_category", "0")

                        print(f"\n[RESET] Received reset command (category {reset_category})")

                        # 清除 reset trigger
                        redis_client.delete("isaac_reset_trigger")

                        if reset_category == "1":
                            # Reset object only (参考 sim_main.py)
                            print("🔄 Resetting object...")
                            env_cfg.event_manager.trigger("reset_object_self", env)
                            print("✅ Object reset complete")

                        elif reset_category == "2":
                            # Reset all (robot + objects) (参考 sim_main.py)
                            print("🔄 Resetting all (robot + objects)...")
                            env_cfg.event_manager.trigger("reset_all_self", env)
                            print("✅ All reset complete")

                        elif reset_category == "3":
                            # Complete reset: notify parent process to restart
                            print("🔄 Complete reset: notifying parent process...")

                            # 清除 reset trigger
                            redis_client.delete("isaac_reset_trigger")

                            # 检查是否由父进程管理
                            is_managed = os.environ.get('ISAAC_SIM_MANAGED') == '1'

                            if is_managed:
                                # 由父进程管理：通知父进程并退出
                                print("[RESET] Running under parent process manager")

                                # 1. Clean up action provider (saves pending recordings)
                                try:
                                    print("[RESET] Saving recordings and cleaning up...")
                                    if action_provider and hasattr(action_provider, 'cleanup'):
                                        action_provider.cleanup()
                                    print("[RESET] ✅ Action provider cleaned up")
                                except Exception as e:
                                    print(f"[RESET] Warning: {e}")

                                # 2. 通知父进程执行 reset
                                try:
                                    reset_manager_signal = {
                                        "reset_category": "3",
                                        "timestamp": int(time.time() * 1000)
                                    }
                                    redis_client.set("isaac_reset_trigger_manager",
                                                   json.dumps(reset_manager_signal))
                                    redis_client.expire("isaac_reset_trigger_manager", 10)
                                    print("[RESET] ✅ Notified parent process to restart")
                                except Exception as e:
                                    print(f"[RESET] ⚠️ Failed to notify parent: {e}")

                                # 3. 优雅退出，让父进程 kill 我们
                                print("[RESET] Exiting gracefully...")
                                sys.stdout.flush()
                                sys.stderr.flush()
                                sys.exit(0)

                            else:
                                # 独立运行：使用 os.execv 重启
                                print("[RESET] Running independently, using execv to restart")

                                # 快速清理关键资源
                                try:
                                    if action_provider and hasattr(action_provider, 'cleanup'):
                                        action_provider.cleanup()
                                except Exception as e:
                                    print(f"[RESET] Warning: {e}")

                                try:
                                    if controller:
                                        controller.stop()
                                except Exception as e:
                                    print(f"[RESET] Warning: {e}")

                                try:
                                    if image_server:
                                        image_server._close()
                                except Exception as e:
                                    print(f"[RESET] Warning: {e}")

                                try:
                                    from tasks.common_observations.camera_state import multi_image_writer
                                    multi_image_writer.cleanup()
                                except Exception as e:
                                    print(f"[RESET] Warning: {e}")

                                # 发送重置完成信号
                                try:
                                    reset_complete_signal = {
                                        "status": "complete",
                                        "timestamp": int(time.time() * 1000)
                                    }
                                    redis_client.set("isaac_reset_complete_unitree_g1_with_hands",
                                                   json.dumps(reset_complete_signal))
                                    redis_client.expire("isaac_reset_complete_unitree_g1_with_hands", 5)
                                    print("[RESET] ✅ Reset complete signal sent via Redis")
                                except Exception as e:
                                    print(f"[RESET] ⚠️ Failed to send reset complete signal: {e}")

                                # 使用 execv 替换进程
                                print("[RESET] Replacing current process...")
                                sys.stdout.flush()
                                sys.stderr.flush()
                                os.execv(sys.executable, [sys.executable] + sys.argv)

                                # 如果 execv 失败
                                print("[RESET] ❌ execv failed, forcing exit...")
                                os._exit(1)

                        # 发送重置完成信号（仅对 category 1 和 2）
                        if reset_category in ["1", "2"]:
                            try:
                                reset_complete_signal = {
                                    "status": "complete",
                                    "timestamp": int(time.time() * 1000)
                                }
                                redis_client.set("isaac_reset_complete_unitree_g1_with_hands",
                                               json.dumps(reset_complete_signal))
                                redis_client.expire("isaac_reset_complete_unitree_g1_with_hands", 5)
                                print("[RESET] ✅ Reset complete signal sent via Redis")
                            except Exception as e:
                                print(f"[RESET] ⚠️ Failed to send reset complete signal: {e}")

                except redis.ConnectionError:
                    pass  # Redis 不可用，忽略
                except Exception as e:
                    if loop_count % 100 == 0:
                        print(f"[WARN] Error checking reset trigger: {e}")

                # 执行控制步骤（驱动物理仿真）
                controller.step()

                # 定期打印状态
                if loop_count % 500 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_stats_time
                    fps = 500 / elapsed if elapsed > 0 else 0
                    print(f"[STATS] Loop {loop_count}, FPS: {fps:.1f}")
                    last_stats_time = current_time

                # 检查环境状态
                if env.sim.is_stopped():
                    print("\n[INFO] Environment stopped")
                    break

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Received keyboard interrupt")
    except Exception as e:
        print(f"\n[ERROR] Exception in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_all()


if __name__ == "__main__":
    main()
