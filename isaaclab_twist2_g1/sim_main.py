import sys
import coverage, coverage.types
print("PY:", sys.executable)
print("coverage:", coverage.__version__, coverage.__file__)
print("has Tracer:", hasattr(coverage.types, "Tracer"))
print("sys.path head:", sys.path[:8])
#exit()
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
#!/usr/bin/env python3
# main.py
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
from dds.dds_create import create_dds_objects,create_dds_objects_replay
# add command line arguments
parser = argparse.ArgumentParser(description="Unitree Simulation")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")
parser.add_argument("--action_source", type=str, default="dds", 
                   choices=["dds", "file", "trajectory", "policy", "replay","dds_wholebody"], 
                   help="Action source")


parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dexterous hand DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire hand DDS")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

parser.add_argument("--file_path", type=str, default="/home/unitree/newDisk/sim-data/Placewoodenblock", help="file path (when action_source=file)")
parser.add_argument("--generate_data_dir", type=str, default="./data", help="save data dir")
parser.add_argument("--generate_data", action="store_true", default=False, help="generate data")
parser.add_argument("--rerun_log", action="store_true", default=False, help="rerun log")
parser.add_argument("--replay_data",  action="store_true", default=False, help="replay data")

parser.add_argument("--modify_light",  action="store_true", default=False, help="modify light")
parser.add_argument("--modify_camera",  action="store_true", default=False,    help="modify camera")

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
parser.add_argument("--image_redis_host", type=str, default="localhost", help="Redis host for image streaming")
parser.add_argument("--image_redis_port", type=int, default=6379, help="Redis port for image streaming")
parser.add_argument("--image_redis_db", type=int, default=0, help="Redis db for image streaming")
parser.add_argument(
    "--image_redis_key_prefix",
    type=str,
    default="isaac_image",
    help="Redis key prefix for image streaming",
)
parser.add_argument(
    "--image_redis_channel",
    type=str,
    default="",
    help="Redis pubsub channel (optional) for image streaming",
)
parser.add_argument("--image_dds_topic", type=str, default="rt/isaac_image", help="DDS topic for image streaming")
parser.add_argument("--image_xrobot_host", type=str, default="172.20.10.2", help="XRobot/Pico IP for image streaming")
parser.add_argument("--image_xrobot_port", type=int, default=12345, help="XRobot/Pico port for image streaming")
parser.add_argument("--image_xrobot_bitrate", type=int, default=4000000, help="XRobot/Pico H264 bitrate (bps)")
parser.add_argument("--image_xrobot_width", type=int, default=256, help="XRobot/Pico output width (0=use source)")
parser.add_argument("--image_xrobot_height", type=int, default=256, help="XRobot/Pico output height (0=use source)")
parser.add_argument("--image_xrobot_ffmpeg", type=str, default="", help="ffmpeg path for XRobot streaming")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=500, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=True, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")

parser.add_argument("--model_path", type=str, default="/home/hcl4070-1/Desktop/taowen/projects/TWIST2/assets/ckpts/twist2_1017_20k.onnx", help="model path")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="enable wh dds")
parser.add_argument("--setpgrp", action="store_true", default=False, help="detach to a new process group")


parser.add_argument("--gravity_z", type=float, default=-9.8, help="override gravity z (e.g., -9.8)")

# world camera parameters
parser.add_argument("--enable_world_camera", action="store_true", default=False, help="enable world camera (third-person view)")
parser.add_argument("--world_camera_port", type=int, default=5556, help="ZMQ port for world camera streaming")

# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: enable_dex3_dds and enable_dex1_dds and enable_inspire_dds cannot be enabled at the same time")
    print("Please select one of the options")
    sys.exit(1)


import pinocchio 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from layeredcontrol.robot_control_system import (
    RobotController, 
    ControlConfig,
)

from dds.reset_pose_dds import *
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tools.augmentation_utils import (
    update_light,
    batch_augment_cameras_by_name,
)
from tools.grass_ground_material import apply_grass_pbr_to_ground
from tools.pitch_lines import create_simple_debug_lines
from tools.football_physics_material import apply_football_physics_material

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider
from tools.get_stiffness import get_robot_stiffness_from_env
from tools.get_reward import get_step_reward_value,get_current_rewards

def setup_signal_handlers(controller, dds_manager=None, image_servers=None, simulation_app=None):
    """set signal handlers

    Args:
        controller: robot controller instance
        dds_manager: DDS manager instance
        image_servers: list of ImageServer instances or single ImageServer
        simulation_app: simulation app instance
    """
    _handling = {"in_progress": False}
    def signal_handler(signum, frame):
        print(f"\nreceived signal {signum}, stopping controller...")
        # Prevent running cleanup multiple times (Ctrl-C can be pressed repeatedly)
        if _handling["in_progress"]:
            print("signal handler already running; forcing exit")
            os._exit(0)
        _handling["in_progress"] = True
        try:
            controller.stop()
        except Exception as e:
            print(f"Failed to stop controller: {e}")

        # Close image servers (support single or multiple servers)
        try:
            if image_servers is not None:
                # Handle both single server and list of servers
                servers_list = image_servers if isinstance(image_servers, list) else [image_servers]
                for idx, server in enumerate(servers_list):
                    if server is not None:
                        try:
                            server._close()
                            print(f"[sim_main] Image server {idx} closed successfully")
                        except Exception as e:
                            print(f"[sim_main] Failed to close image server {idx}: {e}")
        except Exception as e:
            print(f"Failed to stop image servers: {e}")

        # Clean up global shared memory writer from camera_state.py
        try:
            from tasks.common_observations.camera_state import multi_image_writer
            print("[sim_main] Cleaning up global camera multi_image_writer...")
            multi_image_writer.cleanup()
        except Exception as e:
            print(f"[sim_main] Failed to cleanup camera shared memory: {e}")

        try:
            if dds_manager is not None:
                dds_manager.stop_all_communication()
        except Exception as e:
            print(f"Failed to stop DDS: {e}")
        try:
            if simulation_app is not None:
                simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation app: {e}")
        # If any background threads (DDS / OmniKit) keep the process alive, force exit.
        try:
            import os as _os
            _os._exit(0)
        except Exception:
            raise SystemExit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)



def main():
    """main function"""
    import os
    import atexit
    try:
        if args_cli.setpgrp:
            os.setpgrp()
            current_pgid = os.getpgrp()
            print(f"Setting process group: {current_pgid}")
        
            def cleanup_process_group():
                try:
                    print(f"Cleaning up process group: {current_pgid}")
                    import signal
                    os.killpg(current_pgid, signal.SIGTERM)
                except Exception as e:
                    print(f"Failed to clean up process group: {e}")
        
            atexit.register(cleanup_process_group)
        
    except Exception as e:
        print(f"Failed to set process group: {e}")
    print("=" * 60)
    print("robot control system started")
    print(f"Task: {args_cli.task}")
    print(f"Action source: {args_cli.action_source}")
    print("=" * 60)

    # parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
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

        # Optional: override gravity (IsaacLab/IsaacSim APIs differ across versions)
        if args_cli.gravity_z is not None:
            # import pdb; pdb.set_trace()
            g = float(args_cli.gravity_z)
            # gravity = (0.0, 0.0, g)
            env.sim.get_physics_context().set_gravity(g)
            # # Prefer public APIs when available
            # if hasattr(env, "sim") and hasattr(env.sim, "set_gravity"):
            #     env.sim.set_gravity(gravity)
            #     print(f"[sim] gravity set via env.sim.set_gravity: {gravity}")
            # # Some versions expose the physics context explicitly
            # elif hasattr(env, "sim") and hasattr(env.sim, "get_physics_context"):
            #     ctx = env.sim.get_physics_context()
            #     if hasattr(ctx, "set_gravity"):
            #         try:
            #             ctx.set_gravity(gravity)  # 尝试 vec3
            #             print(f"[sim] gravity set via physics context (vec3): {gravity}")
            #         except TypeError:
            #             ctx.set_gravity(g)  # 回退 scalar
            #             print(f"[sim] gravity set via physics context (scalar): {g}")
            #         print(f"[sim] gravity set via physics context: {gravity}")
            #     else:
            #         print("[sim] physics context has no set_gravity; gravity not changed")
            # # Fallback: set on the config (may require env reset/recreate to take effect)
            # elif hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "gravity"):
            #     env_cfg.sim.gravity = gravity
            #     print(f"[sim] gravity written to env_cfg.sim.gravity (may require reset): {gravity}")
            # else:
            #     print("[sim] could not find a supported gravity API on this IsaacLab version")
    except Exception as e:
        print(f"\nFailed to create environment: {e}")
        return
    
    # get robot stiffness and damping parameters from runtime environment
    print("\n" + "="*60)
    print("🔍 Getting robot stiffness and damping parameters from runtime environment")
    print("="*60)
    
    try:
        stiffness_data = get_robot_stiffness_from_env(env)
        if stiffness_data:
            print("✅ Successfully got robot parameters!")
        else:
            print("⚠️ Failed to get robot parameters, will try again after environment reset")
    except Exception as e:
        print(f"⚠️ Error getting robot parameters: {e}")
    
    print("="*60)
    
    print("\n")
    print("***  Please left-click on the Sim window to activate rendering. ***")
    print("\n")
    # reset environment
    if args_cli.modify_light:
        update_light(
            prim_path="/World/light",
            color=(0.75, 0.75, 0.75),
            intensity=500.0,
            # position=(1.0, 2.0, 3.0),
            radius=0.1,
            enabled=True,
            cast_shadows=True
        )
    if args_cli.modify_camera:
        batch_augment_cameras_by_name(
            names=["front_cam"],
            focal_length=3.0,
            horizontal_aperture=22.0,
            vertical_aperture=16.0,
            exposure=0.8,
            focus_distance=1.2
        )
    env.sim.reset()
    env.reset()
    # 足球任務：套用草坪 PBR 材質到地面（需在 reset 後，確保 stage 已完整載入）
    # uv_scale=(150,150) 維持草地 UV 密度，渲染區域由地面尺寸（10×10m）控制
    if "football" in args_cli.task.lower():
        apply_grass_pbr_to_ground(prim_path="/World/GroundPlane", uv_scale=(150.0, 150.0))
        try:
            apply_football_physics_material(restitution=0.75)
        except Exception as e:
            print(f"[football_physics] 跳過: {e}")
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            create_simple_debug_lines(stage)
        except Exception as e:
            print(f"[pitch_lines] 標線未建立或跳過: {e}")

    # ================= Debug: print Box & Cube physics properties =================
    print("\n" + "=" * 60)
    print("[DEBUG] Inspecting physics properties using IsaacLab API")

    # Get Box properties
    if "box" in env.scene.keys():
        box_obj = env.scene["box"]
        print(f"\n[BOX] Prim path: {box_obj.cfg.prim_path}")

        # Get mass (before modification)
        masses_before = box_obj.root_physx_view.get_masses()
        print(f"  Mass (original from USD): {masses_before[0] if len(masses_before) > 0 else 'N/A'} kg")
        print(f"  Mass (from config): {box_obj.cfg.spawn.mass_props.mass if hasattr(box_obj.cfg.spawn, 'mass_props') and box_obj.cfg.spawn.mass_props else 'Not set in config'} kg")

        # Try to set mass using correct format
        print(f"\n  Attempting to set mass to 0.3 kg...")
        try:
            import torch
            # PhysX requires CPU tensors (device -1 or 'cpu')
            new_mass = torch.tensor([[0.3]], dtype=torch.float32, device='cpu')
            box_obj.root_physx_view.set_masses(new_mass, indices=torch.tensor([0], device='cpu'))

            # Verify the change
            masses_after = box_obj.root_physx_view.get_masses()
            print(f"  Mass (after setting): {masses_after[0]} kg")
            print(f"  ✅ Successfully set mass to 0.3 kg!")
        except Exception as e:
            print(f"  ❌ Failed to set mass: {e}")
            print(f"  Note: mass_props in config is ignored for USD files")

        # Get friction and restitution from PhysX runtime
        print(f"\n  Gravity disabled: {box_obj.cfg.spawn.rigid_props.disable_gravity}")

        # Try to get material properties from PhysX view
        try:
            # Get material properties using PhysX view API
            materials = box_obj.root_physx_view.get_material_properties()
            if materials is not None and len(materials) > 0:
                mat = materials[0]
                print(f"  Static friction (from PhysX): {mat[0].item()}")
                print(f"  Dynamic friction (from PhysX): {mat[1].item()}")
                print(f"  Restitution (from PhysX): {mat[2].item()}")
            else:
                print(f"  Material properties: Unable to retrieve via PhysX view API")
        except Exception as e:
            print(f"  Material properties: Unable to retrieve ({e})")

        print(f"\n  Note: Box uses USD file - friction/restitution from USD defaults")
        print(f"        USD file does NOT contain explicit mass/friction/restitution properties")
        print(f"        Physics engine auto-calculates mass from geometry volume + default density")

    # Get Cube properties
    if "cube" in env.scene.keys():
        cube_obj = env.scene["cube"]
        print(f"\n[CUBE] Prim path: {cube_obj.cfg.prim_path}")

        # Get mass
        masses = cube_obj.root_physx_view.get_masses()
        print(f"  Mass (from runtime): {masses[0] if len(masses) > 0 else 'N/A'} kg")
        print(f"  Mass (from config): {cube_obj.cfg.spawn.mass_props.mass if hasattr(cube_obj.cfg.spawn, 'mass_props') else 'Not set'} kg")

        # Get friction from config
        print(f"\n  Gravity disabled: {cube_obj.cfg.spawn.rigid_props.disable_gravity}")
        if hasattr(cube_obj.cfg.spawn, 'physics_material') and cube_obj.cfg.spawn.physics_material:
            phys_mat = cube_obj.cfg.spawn.physics_material
            print(f"  Static friction (from config): {phys_mat.static_friction}")
            print(f"  Dynamic friction (from config): {phys_mat.dynamic_friction}")
            print(f"  Restitution (from config): {phys_mat.restitution}")

        print(f"\n  Note: Cube uses procedural geometry (CuboidCfg)")
        print(f"        Config mass_props and physics_material are applied correctly")

    print("\n[DEBUG] Finished inspecting physics properties")
    print("=" * 60 + "\n")
    # ==================================================================

    # --- set default viewport camera (GUI only) ---
    try:
        # Viewport camera switching only makes sense when rendering is enabled
        import omni.kit.viewport.utility as vp_utils

        # Set to front camera (first-person view)
        cam_path = "/World/envs/env_0/Robot/d435_link/front_cam"
        vp = vp_utils.get_active_viewport()
        if vp is not None:
            vp.set_active_camera(cam_path)
            print(f"✅ Viewport active camera set to: {cam_path}")
        else:
            print("⚠️ No active viewport found; skip setting default camera")
    except Exception as e:
        # Likely running headless / no viewport
        print(f"⚠️ Failed to set viewport default camera (maybe headless): {e}")


    # create simplified control configuration
    try:
        use_wholebody = "Wholebody" in args_cli.task or args_cli.enable_wholebody_dds
        if use_wholebody:
            args_cli.action_source = "dds_wholebody"
            args_cli.enable_wholebody_dds = True
            physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
            policy_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
            if args_cli.step_hz != policy_hz:
                print(f"⚠️  Overriding step_hz {args_cli.step_hz} -> {policy_hz} to match TWIST2 policy rate")
            step_hz = policy_hz
        else:
            step_hz = args_cli.step_hz

        control_config = ControlConfig(
            step_hz=step_hz,
            replay_mode=args_cli.replay_data,
            use_rl_action_mode=use_wholebody,
        )
    except Exception as e:
        print(f"Failed to create control configuration: {e}")
        return
    
    # create controller

    if not args_cli.replay_data:
        print("========= create image server(s) =========")
        image_servers = []  # List to hold all image servers

        # Create front camera image server (always enabled)
        try:
            front_server = ImageServer(
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
                xrobot_width=args_cli.image_xrobot_width or None,
                xrobot_height=args_cli.image_xrobot_height or None,
                xrobot_ffmpeg=args_cli.image_xrobot_ffmpeg or None,
            )
            image_servers.append(front_server)
            print(f"[sim_main] Front camera ImageServer started on port {args_cli.image_zmq_port}")
        except Exception as e:
            print(f"Failed to create front camera image server: {e}")
            return

        # Create world camera image server (optional, based on --enable_world_camera)
        if args_cli.enable_world_camera:
            try:
                world_server = ImageServer(
                    fps=args_cli.image_fps,
                    port=args_cli.world_camera_port,
                    Unit_Test=False,
                    transport=args_cli.image_transport,
                    redis_host=args_cli.image_redis_host,
                    redis_port=args_cli.image_redis_port,
                    redis_db=args_cli.image_redis_db,
                    redis_key_prefix=args_cli.image_redis_key_prefix + "_world",
                    redis_channel=args_cli.image_redis_channel + "_world" if args_cli.image_redis_channel else "",
                    dds_topic=args_cli.image_dds_topic + "_world",
                    xrobot_host=args_cli.image_xrobot_host,
                    xrobot_port=args_cli.image_xrobot_port + 1,  # Use different port for world camera
                    xrobot_bitrate=args_cli.image_xrobot_bitrate,
                    xrobot_width=args_cli.image_xrobot_width or None,
                    xrobot_height=args_cli.image_xrobot_height or None,
                    xrobot_ffmpeg=args_cli.image_xrobot_ffmpeg or None,
                    camera_name="world_camera",  # Specify which camera to use
                )
                image_servers.append(world_server)
                print(f"[sim_main] World camera ImageServer started on port {args_cli.world_camera_port}")
            except Exception as e:
                print(f"Warning: Failed to create world camera image server: {e}")
                print("[sim_main] Continuing without world camera...")
        else:
            print("[sim_main] World camera disabled (use --enable_world_camera to enable)")

        print(f"========= created {len(image_servers)} image server(s) success =========")
        print("========= create dds =========")
        try:
            reset_pose_dds,sim_state_dds,dds_manager = create_dds_objects(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
    else:
        print("========= create dds =========")
        try:
            create_dds_objects_replay(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
        from tools.data_json_load import get_data_json_list
        print("========= get data json list =========")
        data_idx=0
        data_json_list = get_data_json_list(args_cli.file_path)
        if args_cli.action_source != "replay":
            args_cli.action_source = "replay"
        print("========= get data json list success =========")
    # create action provider

    print(f"\ncreate action provider: {args_cli.action_source}...")
    try:
        print(f"args_cli.task: {args_cli.task}")
        # import pdb; pdb.set_trace()
        action_provider = create_action_provider(env, args_cli)
        if action_provider is None:
            print("action provider creation failed, exiting")
            return
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        return
    
    # set action provider
    print("========= create controller =========")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)
    print("========= create controller success =========")
    
    # configure performance analysis
    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        print(f"performance analysis enabled, report every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        print("performance analysis disabled")


    # set signal handlers
    if not args_cli.replay_data:
        setup_signal_handlers(controller, dds_manager, image_servers, simulation_app)
    else:
        setup_signal_handlers(controller, None, None, simulation_app)
    print("Note: The DDS in Sim transmits messages on channel 1. Please ensure that other DDS instances use the same channel for message exchange by setting: ChannelFactoryInitialize(1).")
    try:
        # start controller - start asynchronous components
        print("========= start controller =========")
        controller.start()
        print("========= start controller success =========")
        
        # main loop - execute in main thread to support rendering
        last_stats_time = time.time()
        loop_start_time = time.time()
        loop_count = 0
        last_loop_time = time.time()
        recent_loop_times = []  # for calculating moving average frequency

        # use torch.inference_mode() and handle KeyboardInterrupt
        try:
            with torch.inference_mode():
                while simulation_app.is_running() and controller.is_running:
                    current_time = time.time()
                    loop_count += 1
                    if not args_cli.replay_data:
                        try:
                            env_state = env.scene.get_state()
                            env_state_json = sim_state_to_json(env_state)
                            sim_state = {"init_state": env_state_json, "task_name": args_cli.task}
                        except Exception as e:
                            print(f"Failed to get env state: {e}")
                            raise e
                        try:
                            # sim_state = json.dumps(sim_state)
                            sim_state_dds.write_sim_state_data(sim_state)
                        except Exception as e:
                            print(f"Failed to write sim state: {e}")
                            raise e
                        # print(f"reset_pose_dds: {reset_pose_dds}")
                        try:
                            reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                        except Exception as e:
                            print(f"Failed to get reset pose command: {e}")
                            raise e
                        # # print(f"reset_pose_cmd: {reset_pose_cmd}")
                        # Compute current reward values manually if needed for debugging
                        try:
                            current_reward = get_step_reward_value(env)
                            print(f"reward: {current_reward}")
                        except Exception as e:
                            print(f"奖励计算失败: {e}")
                            pass

                        if reset_pose_cmd is not None:
                            try:
                                reset_category = reset_pose_cmd.get("reset_category")
                                # print(f"reset_category: {reset_category}")
                                if (args_cli.enable_wholebody_dds and (reset_category == '1' or reset_category == '2')) or (
                                    not args_cli.enable_wholebody_dds and reset_category == '1'
                                ):
                                    print("reset object")
                                    env_cfg.event_manager.trigger("reset_object_self", env)
                                    reset_pose_dds.write_reset_pose_command(-1)
                                elif reset_category == '2' and not args_cli.enable_wholebody_dds:
                                    print("reset all")
                                    env_cfg.event_manager.trigger("reset_all_self", env)
                                    reset_pose_dds.write_reset_pose_command(-1)
                            except Exception as e:
                                print(f"Failed to write reset pose command: {e}")
                                raise e
                    else:
                        if action_provider.get_start_loop() and data_idx < len(data_json_list):
                            print(f"data_idx: {data_idx}")
                            try:
                                sim_state, task_name = action_provider.load_data(data_json_list[data_idx])
                                if task_name != args_cli.task:
                                    raise ValueError(
                                        f" The {task_name} in the dataset is different from the {args_cli.task} being executed ."
                                    )
                            except Exception as e:
                                print(f"Failed to load data: {e}")
                                raise e
                            try:
                                env.reset_to(sim_state, torch.tensor([0], device=env.device), is_relative=True)
                                env.sim.reset()
                                time.sleep(1)
                                action_provider.start_replay()
                                data_idx += 1
                            except Exception as e:
                                print(f"Failed to start replay: {e}")
                                raise e
                    # print(f"env_state: {env_state}")
                    # calculate instantaneous loop time
                    loop_dt = current_time - last_loop_time
                    last_loop_time = current_time
                    recent_loop_times.append(loop_dt)

                    # keep recent 100 loop times
                    if len(recent_loop_times) > 100:
                        recent_loop_times.pop(0)

                    # execute control step (in main thread, support rendering)
                    controller.step()

                    # print statistics and loop frequency periodically
                    if current_time - last_stats_time >= args_cli.stats_interval:
                        # calculate while loop execution frequency
                        elapsed_time = current_time - loop_start_time
                        loop_frequency = loop_count / elapsed_time if elapsed_time > 0 else 0

                        # calculate moving average frequency (based on recent loop times)
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

                        # print_stats(controller)
                        last_stats_time = current_time

                    # check environment state
                    if env.sim.is_stopped():
                        print("\nenvironment stopped")
                        break
                    # rate_limiter.sleep(env)
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
        
        # Get current process information
        import os
        import subprocess
        import signal
        import time
        
        current_pid = os.getpid()
        print(f"Current main process PID: {current_pid}")
        
        try:
            # Find all related Python processes
            result = subprocess.run(['pgrep', '-f', 'sim_main.py'], 
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
                result2 = subprocess.run(['pgrep', '-f', 'sim_main.py'], 
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

# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex1-Joint    --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129

# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-RedBlock-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex3-Joint     --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Inspire-Joint     --enable_inspire_dds --robot_type g129




# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody  --robot_type g129 --enable_dex1_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex3-Wholebody  --robot_type g129 --enable_dex3_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Inspire-Wholebody  --robot_type g129 --enable_inspire_dds 
