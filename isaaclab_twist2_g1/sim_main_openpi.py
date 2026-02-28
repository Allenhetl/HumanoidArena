#!/usr/bin/env python3
"""
OpenPI-integrated simulation main script for IsaacLab TWIST2 G1.

This script integrates OpenPI0.5 model with TWIST2 motion tracking for
verification in Isaac Lab simulation.
"""

import sys
import os

# Setup project root
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

# Import utilities
from image_server.image_server import ImageServer
from dds.dds_create import create_dds_objects

# Command line arguments
parser = argparse.ArgumentParser(description="OpenPI + TWIST2 + IsaacLab Simulation")

# Task parameters
parser.add_argument("--task", type=str, default="Isaac-Wholebody-G129-Dex3",
                   help="Isaac Lab task name")
parser.add_argument("--action_source", type=str, default="openpi",
                   help="Action source (set to 'openpi' for this script)")

# OpenPI specific parameters
parser.add_argument("--openpi_checkpoint", type=str, required=True,
                   help="Path to OpenPI checkpoint directory")
parser.add_argument("--language_instruction", type=str, required=True,
                   help="Language instruction for OpenPI (e.g., 'walk forward slowly')")
parser.add_argument("--smplx_model_path", type=str,
                   default="/home/hcl4070-1/Desktop/taowen/projects/smplx_models",
                   help="Path to SMPL-X model files")
parser.add_argument("--human_height", type=float, default=1.75,
                   help="Human height in meters for GMR scaling")

# TWIST2 parameters
parser.add_argument("--twist2_model_path", type=str,
                   default="/home/hcl4070-1/Desktop/taowen/projects/TWIST2/assets/ckpts/twist2_1017_20k.onnx",
                   help="Path to TWIST2 policy model (ONNX or PT)")

# Video recording parameters
parser.add_argument("--video_save_dir", type=str, default="./videos/openpi",
                   help="Directory to save output videos")
parser.add_argument("--video_fps", type=int, default=30,
                   help="Video frame rate")
parser.add_argument("--enable_smpl_vis", action="store_true", default=True,
                   help="Enable SMPL skeleton visualization in video")

# Robot and hand parameters
parser.add_argument("--robot_type", type=str, default="g129",
                   help="Robot type")
parser.add_argument("--enable_dex3_dds", action="store_true",
                   help="Enable dexterous hand DDS")
parser.add_argument("--enable_dex1_dds", action="store_true",
                   help="Enable gripper DDS")
parser.add_argument("--enable_inspire_dds", action="store_true",
                   help="Enable inspire hand DDS")

# Image streaming parameters
parser.add_argument("--image_transport", type=str, default="zmq",
                   choices=["zmq", "redis", "dds", "xrobot"],
                   help="Image transport method")
parser.add_argument("--image_fps", type=int, default=30,
                   help="Image streaming FPS cap")
parser.add_argument("--image_zmq_port", type=int, default=5555,
                   help="ZMQ port for image streaming")

# Camera parameters
parser.add_argument("--enable_world_camera", action="store_true", default=False,
                   help="Enable world camera (third-person view)")
parser.add_argument("--world_camera_port", type=int, default=5556,
                   help="ZMQ port for world camera")

# Control parameters
parser.add_argument("--step_hz", type=int, default=500,
                   help="Control frequency in Hz")
parser.add_argument("--gravity_z", type=float, default=-9.8,
                   help="Gravity override (Z-axis)")

# Performance parameters
parser.add_argument("--enable_profiling", action="store_true", default=False,
                   help="Enable performance profiling")
parser.add_argument("--profile_interval", type=int, default=500,
                   help="Profiling report interval (steps)")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate arguments
if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: Only one hand type can be enabled at a time")
    sys.exit(1)

# Launch Isaac Lab
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after app launch
from layeredcontrol.robot_control_system import RobotController, ControlConfig
from dds.reset_pose_dds import *
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider


def setup_signal_handlers(controller, dds_manager=None, image_servers=None, simulation_app=None):
    """Setup signal handlers for graceful shutdown."""
    _handling = {"in_progress": False}

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, stopping controller...")

        if _handling["in_progress"]:
            print("Signal handler already in progress, ignoring...")
            return

        _handling["in_progress"] = True

        try:
            # Stop controller
            if controller:
                print("Stopping controller...")
                controller.stop()

            # Stop image servers
            if image_servers:
                if not isinstance(image_servers, list):
                    image_servers = [image_servers]
                for server in image_servers:
                    if server:
                        print(f"Stopping image server {server.name}...")
                        server.stop()

            # Close DDS
            if dds_manager:
                print("Closing DDS manager...")
                dds_manager.close_all()

            # Close simulation
            if simulation_app:
                print("Closing simulation...")
                simulation_app.close()

            print("Graceful shutdown complete")
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main function."""
    print("=" * 80)
    print("OpenPI + TWIST2 + IsaacLab Simulation")
    print(f"Task: {args_cli.task}")
    print(f"Language: {args_cli.language_instruction}")
    print(f"OpenPI checkpoint: {args_cli.openpi_checkpoint}")
    print("=" * 80)

    # Parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
    except Exception as e:
        print(f"Failed to parse environment configuration: {e}")
        return

    # Create environment
    print("\nCreating environment...")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print(f"Environment created successfully")
        print(f"Robot init pos: {env.cfg.scene.robot.init_state.pos}")
        print(f"Robot USD: {env.cfg.scene.robot.spawn.usd_path}")

        # Override gravity if specified
        if args_cli.gravity_z is not None:
            g = float(args_cli.gravity_z)
            env.sim.get_physics_context().set_gravity(g)
            print(f"Gravity set to {g} m/s^2")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nPlease left-click on the Sim window to activate rendering.\n")

    # Create DDS objects (for reset and state publishing)
    reset_pose_dds, sim_state_dds, dds_manager = create_dds_objects(args_cli, env)

    # Create image servers (first-person and optional world camera)
    image_servers = []
    try:
        # Front camera (first-person)
        camera_names = list(env.scene.sensors.keys())
        if len(camera_names) > 0:
            front_camera = env.scene.sensors[camera_names[0]]
            front_server = ImageServer(
                name="front_camera",
                camera=front_camera,
                transport=args_cli.image_transport,
                fps=args_cli.image_fps,
                zmq_port=args_cli.image_zmq_port,
            )
            front_server.start()
            image_servers.append(front_server)
            print(f"Front camera image server started: {args_cli.image_transport}")

        # World camera (third-person, optional)
        if args_cli.enable_world_camera and len(camera_names) > 1:
            world_camera = env.scene.sensors[camera_names[1]]
            world_server = ImageServer(
                name="world_camera",
                camera=world_camera,
                transport=args_cli.image_transport,
                fps=args_cli.image_fps,
                zmq_port=args_cli.world_camera_port,
            )
            world_server.start()
            image_servers.append(world_server)
            print(f"World camera image server started")
    except Exception as e:
        print(f"Warning: Failed to create image servers: {e}")

    # Create action provider (OpenPI + TWIST2)
    print("\nCreating OpenPI action provider...")
    try:
        action_provider = create_action_provider(env, args_cli)
        print(f"Action provider created: {action_provider.name}")
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create robot controller
    step_hz = args_cli.step_hz
    use_wholebody = True  # OpenPI always uses wholebody control
    control_config = ControlConfig(
        step_hz=step_hz,
        use_rl_action_mode=use_wholebody
    )

    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)

    # Setup signal handlers
    setup_signal_handlers(
        controller=controller,
        dds_manager=dds_manager,
        image_servers=image_servers,
        simulation_app=simulation_app
    )

    # Reset environment
    print("\nResetting environment...")
    env.reset()

    # Start controller
    print("\nStarting controller...")
    controller.start()

    # Main loop
    print("\nEntering main loop...")
    print("Press Ctrl+C to stop")

    step_count = 0
    start_time = time.time()

    try:
        while simulation_app.is_running() and controller.is_running:
            # Execute control step
            controller.step()

            # Publish simulation state (for DDS monitoring)
            if sim_state_dds:
                env_state = env.scene.get_state()
                env_state_json = sim_state_to_json(env_state)
                sim_state_dds.write_sim_state_data(env_state_json)

            # Check for reset command
            if reset_pose_dds:
                reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                if reset_pose_cmd:
                    print(f"Received reset command: {reset_pose_cmd}")
                    env_cfg.event_manager.trigger("reset_object_self", env)

            step_count += 1

            # Print progress
            if step_count % 1000 == 0:
                elapsed = time.time() - start_time
                hz = step_count / elapsed if elapsed > 0 else 0
                print(f"Step {step_count}, Avg FPS: {hz:.1f}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    except Exception as e:
        print(f"\nError in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")

        # Stop controller
        if controller:
            controller.stop()

        # Cleanup action provider (saves video)
        if action_provider and hasattr(action_provider, 'cleanup'):
            action_provider.cleanup()

        # Stop image servers
        for server in image_servers:
            server.stop()

        # Close DDS
        if dds_manager:
            dds_manager.close_all()

        # Close environment
        env.close()

        # Close simulation
        simulation_app.close()

        elapsed = time.time() - start_time
        print(f"\nSimulation complete:")
        print(f"  Total steps: {step_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average FPS: {step_count/elapsed:.1f}")
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
