import sys
import coverage, coverage.types
print("PY:", sys.executable)
print("coverage:", coverage.__version__, coverage.__file__)
print("has Tracer:", hasattr(coverage.types, "Tracer"))
print("sys.path head:", sys.path[:8])
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
#!/usr/bin/env python3
# sim_main_replay.py - Replay recorded data
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import time
import sys
import signal
import torch
import gymnasium as gym
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from image_server.image_server import ImageServer

# add command line arguments
parser = argparse.ArgumentParser(description="Unitree Simulation Replay")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")

# Replay-specific arguments
parser.add_argument("--replay_file", type=str, required=True, help="Path to replay .npz file")
parser.add_argument("--replay_mode", type=str, default="inference",
                   choices=["inference", "direct"],
                   help="Replay mode: inference (ONNX) or direct (recorded qpos)")
parser.add_argument("--replay_loop", action="store_true", default=False,
                   help="Loop replay when reaching end")

parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

# ONNX model path (for inference mode)
parser.add_argument("--model_path", type=str,
                   default="/home/dreams/Users/taowen/HumanoidArena/TWIST2/assets/ckpts/twist2_1017_20k.onnx",
                   help="ONNX model path for inference mode")

# image streaming parameters
parser.add_argument(
    "--image_transport",
    type=str,
    default="zmq",
    choices=["zmq", "redis", "dds", "xrobot"],
    help="image transport for streaming (zmq/redis/dds/xrobot)",
)
parser.add_argument("--image_fps", type=int, default=30, help="image streaming fps cap")
parser.add_argument("--image_zmq_port", type=int, default=5555, help="ZMQ port for image streaming")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=50, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=False, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")

parser.add_argument("--gravity_z", type=float, default=-9.8, help="override gravity z (e.g., -9.8)")

# random seed for reproducibility
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility (default: None)")

# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from layeredcontrol.robot_control_system import (
    RobotController,
    ControlConfig,
)

import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from action_provider.action_provider_wh_twist2_replay import ReplayActionProvider


def setup_signal_handlers(controller, image_server=None, simulation_app=None):
    """set signal handlers

    Args:
        controller: robot controller instance
        image_server: ImageServer instance
        simulation_app: simulation app instance
    """
    _handling = {"in_progress": False}

    def signal_handler(signum, frame):
        print(f"\nreceived signal {signum}, stopping controller...")
        # Prevent running cleanup multiple times
        if _handling["in_progress"]:
            print("signal handler already running; forcing exit")
            os._exit(0)
        _handling["in_progress"] = True
        try:
            controller.stop()
        except Exception as e:
            print(f"Failed to stop controller: {e}")

        # Close image server
        try:
            if image_server is not None:
                image_server._close()
                print(f"[sim_main_replay] Image server closed successfully")
        except Exception as e:
            print(f"Failed to stop image server: {e}")

        # Clean up global shared memory writer from camera_state.py
        try:
            from tasks.common_observations.camera_state import multi_image_writer
            print("[sim_main_replay] Cleaning up global camera multi_image_writer...")
            multi_image_writer.cleanup()
        except Exception as e:
            print(f"[sim_main_replay] Failed to cleanup camera shared memory: {e}")

        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation app: {e}")

        # Force exit
        try:
            import os as _os
            _os._exit(0)
        except Exception:
            raise SystemExit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """main function"""
    print("=" * 60)
    print("robot control system started (REPLAY MODE)")
    print(f"Task: {args_cli.task}")
    print(f"Replay file: {args_cli.replay_file}")
    print(f"Replay mode: {args_cli.replay_mode}")
    print(f"Replay loop: {args_cli.replay_loop}")
    print("=" * 60)

    # parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
        # Set seed: command line argument takes priority, otherwise use default 42
        seed_value = args_cli.seed if args_cli.seed is not None else 42
        env_cfg.seed = seed_value
        print(f"[CONFIG] Setting environment seed: {seed_value}")
    except Exception as e:
        print(f"Failed to parse environment configuration: {e}")
        return

    # create environment
    print("\ncreate environment...")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print(f"\ncreate environment success ...")
        print("robot cfg init pos:", env.cfg.scene.robot.init_state.pos)
        print("robot usd:", env.cfg.scene.robot.spawn.usd_path)

        # Set rendering mode via viewport API
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport:
                viewport.set_hd_engine("rtx", "RaytracedLighting")
                print("[RENDER] ✅ Switched to Real-Time (RaytracedLighting) mode")
            else:
                print("[RENDER] ⚠️ No active viewport found")
        except Exception as e:
            print(f"[RENDER] ⚠️ Failed to set render mode: {e}")

        # Override gravity
        if args_cli.gravity_z is not None:
            g = float(args_cli.gravity_z)
            env.sim.get_physics_context().set_gravity(g)

    except Exception as e:
        print(f"\nFailed to create environment: {e}")
        return

    print("\n")
    print("***  Please left-click on the Sim window to activate rendering. ***")
    print("\n")

    # reset environment
    env.sim.reset()
    env.reset()

    # --- set default viewport camera to first-person view (GUI only) ---
    try:
        # Viewport camera switching only makes sense when rendering is enabled
        import omni.kit.viewport.utility as vp_utils
        from pxr import Usd, UsdGeom

        # Debug: List all cameras in the scene
        stage = env.sim.stage
        print("\n🔍 [DEBUG] Available cameras in USD stage:")
        camera_paths = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                cam_path_str = str(prim.GetPath())
                camera_paths.append(cam_path_str)
                print(f"  - {cam_path_str}")

        if not camera_paths:
            print("  ⚠️ No cameras found in stage!")

        # Try to find front_cam
        front_cam_candidates = [p for p in camera_paths if "front_cam" in p.lower()]

        if front_cam_candidates:
            cam_path = front_cam_candidates[0]
            print(f"\n✅ Found front_cam at: {cam_path}")
        else:
            # Fallback to the expected path
            cam_path = "/World/envs/env_0/Robot/d435_link/front_cam"
            print(f"\n⚠️ No front_cam found, trying expected path: {cam_path}")

        vp = vp_utils.get_active_viewport()
        if vp is not None:
            # Check current camera before switching
            current_cam = vp.get_active_camera()
            print(f"  Current viewport camera: {current_cam}")

            # Try to set the camera
            vp.set_active_camera(cam_path)

            # Verify the switch
            new_cam = vp.get_active_camera()
            if new_cam == cam_path:
                print(f"  ✅ Successfully switched viewport to first-person view: {cam_path}")
            else:
                print(f"  ❌ Failed to switch! Viewport still at: {new_cam}")
        else:
            print("⚠️ No active viewport found; skip setting default camera")
    except Exception as e:
        # Likely running headless / no viewport
        import traceback
        print(f"⚠️ Failed to set viewport default camera: {e}")
        print(traceback.format_exc())

    # create simplified control configuration
    try:
        # Match step_hz with physics frequency for TWIST2
        physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
        policy_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
        step_hz = policy_hz
        print(f"✅ TWIST2 Replay: step_hz set to {step_hz}Hz (physics_dt={physics_dt}s, decimation={env_cfg.decimation})")

        control_config = ControlConfig(
            step_hz=step_hz,
            replay_mode=False,
            use_rl_action_mode=True,
        )
    except Exception as e:
        print(f"Failed to create control configuration: {e}")
        return

    # create image server
    print("========= create image server =========")
    try:
        image_server = ImageServer(
            fps=args_cli.image_fps,
            port=args_cli.image_zmq_port,
            Unit_Test=False,
            transport=args_cli.image_transport,
        )
        print(f"[sim_main_replay] ImageServer started on port {args_cli.image_zmq_port}")
    except Exception as e:
        print(f"Failed to create image server: {e}")
        return
    print("========= create image server success =========")

    # create replay action provider
    print(f"\ncreate replay action provider...")
    try:
        action_provider = ReplayActionProvider(env=env, args_cli=args_cli)
        print(f"✅ Replay action provider created successfully")
    except Exception as e:
        print(f"Failed to create replay action provider: {e}")
        return

    # NOTE: Initial state restoration is now handled by ReplayActionProvider
    # in its get_action() method when current_frame == 0.
    # This ensures proper timing and avoids conflicts with the control loop.
    print(f"\n========= Initial state will be restored by ReplayActionProvider =========")
    print(f"Initial state restoration happens in the first get_action() call")
    print("=" * 60)

    # create controller
    print("========= create controller =========")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)

    # Also set action_provider on env for camera_state.py to access
    env.action_provider = action_provider
    print(f"[sim_main_replay] Set action_provider on env: {type(action_provider)}")

    print("========= create controller success =========")

    # configure performance analysis
    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        print(f"performance analysis enabled, report every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        print("performance analysis disabled")

    # set signal handlers
    setup_signal_handlers(controller, image_server, simulation_app)

    try:
        # start controller
        print("========= start controller =========")
        controller.start()
        print("========= start controller success =========")

        # main loop
        last_stats_time = time.time()
        loop_start_time = time.time()
        loop_count = 0
        last_loop_time = time.time()
        recent_loop_times = []

        # use torch.inference_mode() and handle KeyboardInterrupt
        try:
            with torch.inference_mode():
                while simulation_app.is_running() and controller.is_running:
                    current_time = time.time()
                    loop_count += 1

                    # calculate instantaneous loop time
                    loop_dt = current_time - last_loop_time
                    last_loop_time = current_time
                    recent_loop_times.append(loop_dt)

                    # keep recent 100 loop times
                    if len(recent_loop_times) > 100:
                        recent_loop_times.pop(0)

                    # execute control step
                    controller.step()

                    # print statistics periodically
                    if current_time - last_stats_time >= args_cli.stats_interval:
                        elapsed_time = current_time - loop_start_time
                        loop_frequency = loop_count / elapsed_time if elapsed_time > 0 else 0

                        if recent_loop_times:
                            avg_loop_time = sum(recent_loop_times) / len(recent_loop_times)
                            moving_avg_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                            min_loop_time = min(recent_loop_times)
                            max_loop_time = max(recent_loop_times)
                            max_freq = 1.0 / min_loop_time if min_loop_time > 0 else 0
                            min_freq = 1.0 / max_loop_time if max_loop_time > 0 else 0
                        else:
                            moving_avg_frequency = 0
                            min_freq = max_freq = 0

                        print(f"\n=== While loop execution frequency statistics ===")
                        print(f"loop execution count: {loop_count}")
                        print(f"running time: {elapsed_time:.2f} seconds")
                        print(f"overall average frequency: {loop_frequency:.2f} Hz")
                        print(
                            f"moving average frequency: {moving_avg_frequency:.2f} Hz (last {len(recent_loop_times)} times)"
                        )
                        print(f"frequency range: {min_freq:.2f} - {max_freq:.2f} Hz")
                        print(f"average loop time: {(elapsed_time/loop_count*1000):.2f} ms")
                        if recent_loop_times:
                            print(f"recent loop time: {(avg_loop_time*1000):.2f} ms")
                        print(f"=============================")

                        last_stats_time = current_time

                    # check environment state
                    if env.sim.is_stopped():
                        print("\nenvironment stopped")
                        break

        except KeyboardInterrupt:
            print("\nuser interrupted program")

    except Exception as e:
        print(f"\nprogram exception: {e}")

    finally:
        # clean up resources
        print("\nclean up resources...")
        controller.cleanup()

        env.close()
        print("cleanup completed")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("Performing final cleanup...")

        import os
        import subprocess
        import signal
        import time

        current_pid = os.getpid()
        print(f"Current main process PID: {current_pid}")

        try:
            # Find all related Python processes
            result = subprocess.run(['pgrep', '-f', 'sim_main_replay.py'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                print(f"Found related processes: {pids}")

                for pid in pids:
                    if pid and pid != str(current_pid):
                        try:
                            print(f"Terminating child process: {pid}")
                            os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            print(f"Process {pid} does not exist")
                        except Exception as e:
                            print(f"Failed to terminate process {pid}: {e}")

                # Wait for processes to exit
                time.sleep(2)

                # Check if there are any remaining processes, force kill them
                result2 = subprocess.run(['pgrep', '-f', 'sim_main_replay.py'],
                                       capture_output=True, text=True)
                if result2.returncode == 0:
                    remaining_pids = result2.stdout.strip().split('\n')
                    for pid in remaining_pids:
                        if pid and pid != str(current_pid):
                            try:
                                print(f"Force killing process: {pid}")
                                os.kill(int(pid), signal.SIGKILL)
                            except Exception as e:
                                print(f"Failed to force kill process {pid}: {e}")

        except Exception as e:
            print(f"Error during process cleanup: {e}")

        try:
            simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation application: {e}")

        print("Program exit completed")

        # Force exit
        os._exit(0)

