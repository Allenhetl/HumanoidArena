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
                   choices=["dds", "file", "trajectory", "policy", "replay",
                            "dds_wholebody", "sonic_wholebody"],
                   help="Action source")

# SONIC-specific arguments (used when action_source=sonic_wholebody)
parser.add_argument("--sonic_zmq_host", type=str, default="localhost",
                    help="ZMQ host for SONIC pose topic (pico_manager_thread_server)")
parser.add_argument("--sonic_zmq_port", type=int, default=5556,
                    help="ZMQ port for SONIC pose topic")
parser.add_argument("--sonic_encoder_path", type=str, default="",
                    help="Path to GEAR-SONIC encoder ONNX model")
parser.add_argument("--sonic_decoder_path", type=str, default="",
                    help="Path to GEAR-SONIC decoder ONNX model")


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

parser.add_argument("--model_path", type=str, default="/home/dreams/Users/taowen/HumanoidArena/TWIST2/assets/ckpts/twist2_1017_20k.onnx", help="model path")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="enable wh dds")
parser.add_argument("--setpgrp", action="store_true", default=False, help="detach to a new process group")

# recording parameters
parser.add_argument("--recording_save_dir", type=str, default="./recording_data", help="directory to save recording data")
parser.add_argument("--auto_start_recording", action="store_true", default=False, help="automatically start recording on startup (for testing from-reset reproducibility)")

# random seed for reproducibility
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility (default: None)")

parser.add_argument("--gravity_z", type=float, default=-9.8, help="override gravity z (e.g., -9.8)")

# world camera parameters
parser.add_argument("--enable_world_camera", action="store_true", default=False, help="enable world camera (third-person view)")
parser.add_argument("--world_camera_port", type=int, default=5556, help="ZMQ port for world camera streaming")

# camera configuration parameters
parser.add_argument("--camera_enable_depth", action="store_true", default=False, help="enable depth data (distance_to_image_plane) for cameras")
parser.add_argument("--camera_width", type=int, default=640, help="camera image width")
parser.add_argument("--camera_height", type=int, default=480, help="camera image height")

# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: enable_dex3_dds and enable_dex1_dds and enable_inspire_dds cannot be enabled at the same time")
    print("Please select one of the options")
    sys.exit(1)


# import pinocchio  # 注释掉：与 NumPy 2.x 不兼容，且当前未使用
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
import tools.pitch_lines as pitch_lines_mod
from tools.pitch_lines import create_simple_debug_lines
from tools.football_physics_material import apply_football_physics_material

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider
from tools.get_stiffness import get_robot_stiffness_from_env
from tools.get_reward import get_step_reward_value,get_current_rewards
# Use text-based tracker instead of GUI visualizer to avoid matplotlib issues
from tools.joint_position_tracker import JointPositionTracker

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
        # Apply camera configuration from command line arguments
        if hasattr(env_cfg.scene, 'front_camera'):
            # Set data types based on depth flag
            data_types = ["rgb"]
            if args_cli.camera_enable_depth:
                data_types.append("distance_to_image_plane")

            env_cfg.scene.front_camera.data_types = data_types
            env_cfg.scene.front_camera.width = args_cli.camera_width
            env_cfg.scene.front_camera.height = args_cli.camera_height

            print(f"[CONFIG] Camera settings:")
            print(f"  - Resolution: {args_cli.camera_width}x{args_cli.camera_height}")
            print(f"  - Data types: {data_types}")

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print(f"\ncreate environment success ...")
        print("robot cfg init pos:", env.cfg.scene.robot.init_state.pos)
        print("robot usd:", env.cfg.scene.robot.spawn.usd_path)

        # Set rendering mode via viewport API
        # try:
        #     from omni.kit.viewport.utility import get_active_viewport
        #     viewport = get_active_viewport()
        #     if viewport:
        #         # Switch to Real-Time mode (RaytracedLighting) - fastest RTX mode
        #         viewport.set_hd_engine("rtx", "RaytracedLighting")
        #         print("[RENDER] ✅ Switched to Real-Time (RaytracedLighting) mode via viewport API")
        #         # === Other render modes (commented out for comparison) ===
        #         # viewport.set_hd_engine("rtx", "PathTracing")        # Path Tracing (slower, higher quality)
        #         # viewport.set_hd_engine("iray", "iray")              # iray (not available in Isaac Sim)
        #
        #         # Verify viewport settings
        #         print(f"[RENDER] Viewport hydra_engine: {viewport.hydra_engine}")
        #         print(f"[RENDER] Viewport render_mode: {viewport.render_mode}")
        #
        #         # Print current render settings
        #         import carb
        #         settings = carb.settings.get_settings()
        #         print("\n" + "="*60)
        #         print(" REAL-TIME RENDER PARAMETERS")
        #         print("="*60)
        #         print(f"  /rtx/rendermode: {settings.get('/rtx/rendermode')}")
        #         print(f"  Antialiasing: DLAA (configured in env_cfg)")
        #         print("="*60 + "\n")
        #
        #         # === Path Tracing code (commented out for comparison) ===
        #         # viewport.set_hd_engine("rtx", "PathTracing")
        #         # print("[RENDER] ✅ Switched to Path Tracing mode via viewport API")
        #         # settings.set("/rtx/pathtracing/spp", 1)  # Samples per pixel per frame = 1
        #         # settings.set("/rtx/pathtracing/totalSpp", 1)  # Total samples per pixel
        #         # settings.set("/rtx/pathtracing/maxBounces", 4)  # Max light bounces
        #         # pt_settings = [
        #         #     "/rtx/rendermode",
        #         #     "/rtx/pathtracing/spp",
        #         #     "/rtx/pathtracing/totalSpp",
        #         #     "/rtx/pathtracing/maxBounces",
        #         #     "/rtx/pathtracing/maxSpecularAndTransmissionBounces",
        #         #     "/rtx/pathtracing/maxVolumeBounces",
        #         #     "/rtx/pathtracing/clampSpp",
        #         #     "/rtx/pathtracing/optixDenoiser/enabled",
        #         #     "/rtx/pathtracing/cached/enabled",
        #         #     "/rtx/pathtracing/aa/op",
        #         # ]
        #         # for setting in pt_settings:
        #         #     value = settings.get(setting)
        #         #     print(f"  {setting}: {value}")
        #         # === End Path Tracing code ===
        #     else:
        #         print("[RENDER] ⚠️ No active viewport found, cannot set render mode")
        # except Exception as e:
        #     print(f"[RENDER] ⚠️ Failed to set render mode: {e}")

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
    scene_deactivate_keywords = tuple(getattr(env.cfg, "scene_deactivate_keywords", ()))
    scene_deactivate_exclude_keywords = tuple(getattr(env.cfg, "scene_deactivate_exclude_keywords", ()))
    if len(scene_deactivate_keywords) > 0:
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            deactivated_paths = []
            keywords = tuple(k.lower() for k in scene_deactivate_keywords)
            exclude_keywords = tuple(k.lower() for k in scene_deactivate_exclude_keywords)
            for prim in stage.Traverse():
                if not prim.IsActive():
                    continue
                prim_path = prim.GetPath().pathString
                prim_name = prim.GetName()
                path_lower = prim_path.lower()
                name_lower = prim_name.lower()
                if any((k in path_lower) or (k in name_lower) for k in exclude_keywords):
                    continue
                if any((k in path_lower) or (k in name_lower) for k in keywords):
                    prim.SetActive(False)
                    deactivated_paths.append(prim_path)
            print(
                f"[scene_filter] deactivate keywords={scene_deactivate_keywords}, "
                f"exclude={scene_deactivate_exclude_keywords}, count={len(deactivated_paths)}"
            )
            for prim_path in deactivated_paths[:20]:
                print(f"[scene_filter] deactivated: {prim_path}")
        except Exception as e:
            print(f"[scene_filter] deactivate failed: {e}")
    scene_reposition_rules = tuple(getattr(env.cfg, "scene_reposition_rules", ()))
    if len(scene_reposition_rules) > 0:
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
            stage = omni.usd.get_context().get_stage()
            moved_items = []
            all_active_prims = [p for p in stage.Traverse() if p.IsActive()]
            for rule in scene_reposition_rules:
                keywords = tuple(str(k).lower() for k in rule.get("keywords", ()))
                if len(keywords) == 0:
                    continue
                offset = tuple(float(v) for v in rule.get("offset", (0.0, 0.0, 0.0)))
                if abs(offset[0]) + abs(offset[1]) + abs(offset[2]) < 1e-9:
                    continue
                candidates = []
                for prim in all_active_prims:
                    prim_path = prim.GetPath().pathString
                    prim_name = prim.GetName()
                    path_lower = prim_path.lower()
                    name_lower = prim_name.lower()
                    if "joint" in path_lower or "/materials" in path_lower:
                        continue
                    if not any((k in name_lower) or (k in path_lower) for k in keywords):
                        continue
                    depth = prim_path.count("/")
                    candidates.append((depth, len(prim_path), prim_path))
                if len(candidates) == 0:
                    continue
                candidates.sort(key=lambda x: (x[0], x[1]))
                target_path = candidates[0][2]
                target_prim = stage.GetPrimAtPath(target_path)
                if not target_prim.IsValid():
                    continue
                xformable = UsdGeom.Xformable(target_prim)
                translate_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                if translate_op is None:
                    translate_op = xformable.AddTranslateOp()
                current_t = translate_op.Get()
                if current_t is None:
                    current_t = Gf.Vec3d(0.0, 0.0, 0.0)
                new_t = Gf.Vec3d(
                    float(current_t[0]) + offset[0],
                    float(current_t[1]) + offset[1],
                    float(current_t[2]) + offset[2],
                )
                translate_op.Set(new_t)
                moved_items.append((target_path, rule.get("name", "rule"), offset))
            print(f"[scene_filter] reposition rules={len(scene_reposition_rules)}, moved={len(moved_items)}")
            for prim_path, rule_name, offset in moved_items[:20]:
                print(f"[scene_filter] moved({rule_name}): {prim_path}, offset={offset}")
        except Exception as e:
            print(f"[scene_filter] reposition failed: {e}")
    if "livingroom" in args_cli.task.lower() and os.environ.get("LIVINGROOM_RUNTIME_PATCH", "0") == "1":
        try:
            import omni.usd
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            xform_cache = UsdGeom.XformCache()
            target_paths = (
                "/World/envs/env_0/Room/model_teatable",
                "/World/envs/env_0/Room/model_teatable/E_body_179",
                "/World/envs/env_0/Room/model_table_1",
                "/World/envs/env_0/Room/model_table_1/E_body_1",
            )
            print("[scene_probe] begin teatable/table_1 world pose dump")
            for prim_path in target_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid() or (not prim.IsActive()):
                    continue
                world_m = xform_cache.GetLocalToWorldTransform(prim)
                world_t = world_m.ExtractTranslation()
                print(
                    f"[scene_probe] {prim_path} "
                    f"world_pos=({float(world_t[0]):.4f}, {float(world_t[1]):.4f}, {float(world_t[2]):.4f})"
                )
            print("[scene_probe] end teatable/table_1 world pose dump")
        except Exception as e:
            print(f"[scene_probe] failed: {e}")
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
        
    # 足球任務：套用草坪 PBR 材質到地面（需在 reset 前，確保 stage 已完整載入）
    # uv_scale=(150,150) 維持草地 UV 密度，渲染區域由地面尺寸（10×10m）控制
    if "football" in args_cli.task.lower():
        grass_ok_pre = apply_grass_pbr_to_ground(prim_path="/World/GroundPlane", uv_scale=(100.0, 100.0))
        print(f"[grass_ground_material] before reset apply result: {grass_ok_pre}")
        try:
            apply_football_physics_material(restitution=0.75)
        except Exception as e:
            print(f"[football_physics] 跳過: {e}")
        try:
            import omni.usd
            from tasks.g1_tasks.move_football_g1_29dof_dex3_wholebody.move_football_g1_29dof_dex3_hw_env_cfg import (
                GOAL_REFERENCE_LINE_ABSOLUTE_CENTERS,
                GOAL_REFERENCE_LINE_COLOR,
                GOAL_REFERENCE_LINE_LENGTH,
                GOAL_REFERENCE_LINE_RELATIVE_OFFSETS,
                GOAL_REFERENCE_LINE_WIDTH_RATIO,
            )
            from tools.pitch_lines import DEFAULT_LINE_WIDTH
            stage = omni.usd.get_context().get_stage()
            print(f"[pitch_lines] module file: {pitch_lines_mod.__file__}")
            create_simple_debug_lines(
                stage,
                line_color=(32.0 / 255.0, 32.0 / 255.0, 32.0 / 255.0),
                draw_goal_reference_lines=True,
                goal_centers=GOAL_REFERENCE_LINE_ABSOLUTE_CENTERS,
                goal_relative_offsets=GOAL_REFERENCE_LINE_RELATIVE_OFFSETS,
                goal_line_length=GOAL_REFERENCE_LINE_LENGTH,
                goal_line_width=DEFAULT_LINE_WIDTH * GOAL_REFERENCE_LINE_WIDTH_RATIO,
                goal_line_color=GOAL_REFERENCE_LINE_COLOR,
            )
            print(
                f"[pitch_lines] goal reference lines color={GOAL_REFERENCE_LINE_COLOR}, "
                f"centers={GOAL_REFERENCE_LINE_ABSOLUTE_CENTERS}, offsets={GOAL_REFERENCE_LINE_RELATIVE_OFFSETS}"
            )
        except Exception as e:
            print(f"[pitch_lines] 標線未建立或跳過: {e}")

    
    if "livingroom" in args_cli.task.lower():
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()

            obstacle_paths = [f"/World/envs/env_0/FloorObstacleDrink{i}" for i in range(1, 11)]
            target_paths = ["/World/envs/env_0/TableDrink", "/World/envs/env_0/Room/model_officechair_3"] + obstacle_paths

            def _enable_dynamic_collision(target_path: str):
                target_prim = stage.GetPrimAtPath(target_path)
                if not target_prim.IsValid():
                    return (0, 0, False)
                rigid_count = 0
                collider_count = 0
                has_existing_rigid = False
                for sub_prim in Usd.PrimRange(target_prim):
                    sub_rigid = UsdPhysics.RigidBodyAPI.Get(stage, sub_prim.GetPath())
                    if sub_rigid:
                        has_existing_rigid = True
                        sub_rigid.GetRigidBodyEnabledAttr().Set(True)
                        sub_rigid.GetKinematicEnabledAttr().Set(False)
                        rigid_count += 1
                    if sub_prim.IsA(UsdGeom.Mesh):
                        collision_api = UsdPhysics.CollisionAPI.Apply(sub_prim)
                        collision_api.GetCollisionEnabledAttr().Set(True)
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(sub_prim)
                        mesh_collision_api.GetApproximationAttr().Set("convexHull")
                        collider_count += 1
                if not has_existing_rigid:
                    root_rigid = UsdPhysics.RigidBodyAPI.Apply(target_prim)
                    root_rigid.GetRigidBodyEnabledAttr().Set(True)
                    root_rigid.GetKinematicEnabledAttr().Set(False)
                    rigid_count += 1
                return (rigid_count, collider_count, True)

            for target_path in target_paths:
                rigid_count, collider_count, ok = _enable_dynamic_collision(target_path)
                print(
                    f"[livingroom_collision] target={target_path} ok={ok} "
                    f"rigid_bodies={rigid_count} colliders={collider_count}"
                )
            desired_total_mass = {
                "/World/envs/env_0/TableDrink": 0.35,
                "/World/envs/env_0/Room/model_officechair_3": 8.0,
                "/World/envs/env_0/FloorObstacleDrink1": 1.2,
                "/World/envs/env_0/FloorObstacleDrink2": 0.8,
                "/World/envs/env_0/FloorObstacleDrink3": 1.8,
                "/World/envs/env_0/FloorObstacleDrink4": 1.5,
                "/World/envs/env_0/FloorObstacleDrink5": 2.8,
                "/World/envs/env_0/FloorObstacleDrink6": 1.4,
                "/World/envs/env_0/FloorObstacleDrink7": 1.3,
                "/World/envs/env_0/FloorObstacleDrink8": 1.0,
                "/World/envs/env_0/FloorObstacleDrink9": 2.4,
                "/World/envs/env_0/FloorObstacleDrink10": 1.4,
            }

            def _retune_mass(target_path: str):
                target_prim = stage.GetPrimAtPath(target_path)
                if not target_prim.IsValid():
                    return (False, 0, 0.0, 0.0)
                rigid_prims = []
                for sub_prim in Usd.PrimRange(target_prim):
                    sub_rigid = UsdPhysics.RigidBodyAPI.Get(stage, sub_prim.GetPath())
                    if sub_rigid:
                        rigid_prims.append(sub_prim)
                if len(rigid_prims) == 0:
                    return (False, 0, 0.0, 0.0)
                target_total = float(desired_total_mass.get(target_path, 1.0))
                per_mass = max(0.02, target_total / float(len(rigid_prims)))
                for sub_prim in rigid_prims:
                    mass_api = UsdPhysics.MassAPI.Apply(sub_prim)
                    mass_api.GetMassAttr().Set(per_mass)
                return (True, len(rigid_prims), target_total, per_mass)

            for target_path in target_paths:
                ok_tune, rigid_cnt_tune, target_total, per_mass = _retune_mass(target_path)
                print(
                    f"[livingroom_mass_tune] target={target_path} ok={ok_tune} "
                    f"rigid_bodies={rigid_cnt_tune} target_total_mass={target_total} per_rigid_mass={per_mass}"
                )
            def _inspect_usd_mass(target_path: str):
                target_prim = stage.GetPrimAtPath(target_path)
                if not target_prim.IsValid():
                    return (False, 0, 0, 0.0)
                rigid_prims = []
                explicit_mass_values = []
                for sub_prim in Usd.PrimRange(target_prim):
                    sub_rigid = UsdPhysics.RigidBodyAPI.Get(stage, sub_prim.GetPath())
                    if sub_rigid:
                        rigid_prims.append(sub_prim)
                        mass_api = UsdPhysics.MassAPI.Get(stage, sub_prim.GetPath())
                        if mass_api:
                            mass_attr = mass_api.GetMassAttr()
                            if mass_attr:
                                v = mass_attr.Get()
                                if v is not None:
                                    explicit_mass_values.append(float(v))
                return (True, len(rigid_prims), len(explicit_mass_values), float(sum(explicit_mass_values)))

            scene_key_map = {"TableDrink": "table_drink"}
            for i in range(1, 11):
                scene_key_map[f"FloorObstacleDrink{i}"] = f"floor_obstacle_drink_{i}"
            for target_path in target_paths:
                ok, rigid_count, explicit_count, explicit_sum = _inspect_usd_mass(target_path)
                basename = target_path.split("/")[-1]
                scene_key = scene_key_map.get(basename)
                runtime_mass = None
                if scene_key is not None and scene_key in env.scene.keys():
                    obj = env.scene[scene_key]
                    if hasattr(obj, "root_physx_view") and obj.root_physx_view is not None:
                        masses = obj.root_physx_view.get_masses()
                        if masses is not None and len(masses) > 0:
                            runtime_mass = float(masses[0])
                print(
                    f"[livingroom_mass] target={target_path} ok={ok} rigid_bodies={rigid_count} "
                    f"explicit_mass_count={explicit_count} explicit_mass_sum={explicit_sum} "
                    f"runtime_root_mass={runtime_mass}"
                )
        except Exception as e:
            print(f"[livingroom_collision] setup failed: {e}")
    env.sim.reset()
    env.reset()
    if "football" in args_cli.task.lower():
        grass_ok_post = apply_grass_pbr_to_ground(prim_path="/World/GroundPlane", uv_scale=(15.0, 15.0))
        print(f"[grass_ground_material] after reset apply result: {grass_ok_post}")

    # ================= Debug: print Box & Cube physics properties =================
    print("\n" + "=" * 60)
    print("[DEBUG] Inspecting physics properties using IsaacLab API")

    inspect_targets = ["object_l", "object", "box", "hurdle"]
    for target_name in inspect_targets:
        if target_name not in env.scene.keys():
            continue
        target_obj = env.scene[target_name]
        print(f"\n[{target_name.upper()}] Prim path: {target_obj.cfg.prim_path}")
        print(f"  Scene object type: {type(target_obj).__name__}")
        has_physx_view = hasattr(target_obj, "root_physx_view") and target_obj.root_physx_view is not None
        if has_physx_view:
            masses = target_obj.root_physx_view.get_masses()
            print(f"  Mass (runtime): {masses[0] if len(masses) > 0 else 'N/A'} kg")
            print(f"  Mass (from config): {target_obj.cfg.spawn.mass_props.mass if hasattr(target_obj.cfg.spawn, 'mass_props') and target_obj.cfg.spawn.mass_props else 'Not set in config'} kg")
            try:
                materials = target_obj.root_physx_view.get_material_properties()
                if materials is not None and len(materials) > 0:
                    mat = materials[0]
                    print(f"  Static friction (from PhysX): {mat[0].item()}")
                    print(f"  Dynamic friction (from PhysX): {mat[1].item()}")
                    print(f"  Restitution (from PhysX): {mat[2].item()}")
                else:
                    print(f"  Material properties: Unable to retrieve via PhysX view API")
            except Exception as e:
                print(f"  Material properties: Unable to retrieve ({e})")
        else:
            print("  Runtime mass/material inspection skipped (no root_physx_view)")
        if hasattr(target_obj.cfg, "spawn") and hasattr(target_obj.cfg.spawn, "rigid_props") and target_obj.cfg.spawn.rigid_props:
            print(f"  Gravity disabled: {target_obj.cfg.spawn.rigid_props.disable_gravity}")
        break

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

        # Try to find world camera first (third-person view), then fallback to front_cam
        world_cam_candidates = [p for p in camera_paths if "PerspectiveCamera" in p or "world_camera" in p.lower()]
        front_cam_candidates = [p for p in camera_paths if "front_cam" in p.lower()]

        if world_cam_candidates:
            cam_path = world_cam_candidates[0]
            print(f"\n✅ Found world_camera at: {cam_path}")
        elif front_cam_candidates:
            cam_path = front_cam_candidates[0]
            print(f"\n✅ Found front_cam at: {cam_path}")
        else:
            # Fallback to the expected path
            cam_path = "/World/envs/env_0/Robot/d435_link/front_cam"
            print(f"\n⚠️ No camera found, trying expected path: {cam_path}")

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
                print(f"  ✅ Successfully switched viewport to: {cam_path}")
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
        # sonic_wholebody 优先：不被 task 名中的 "Wholebody" 覆盖
        if args_cli.action_source == "sonic_wholebody":
            use_wholebody = True
            physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
            policy_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
            if args_cli.step_hz != policy_hz:
                print(f"⚠️  Overriding step_hz {args_cli.step_hz} -> {policy_hz} to match TWIST2 policy rate")
            step_hz = policy_hz
        elif "Wholebody" in args_cli.task or args_cli.enable_wholebody_dds:
            use_wholebody = True
            args_cli.action_source = "dds_wholebody"
            args_cli.enable_wholebody_dds = True
            # Match step_hz with physics frequency for TWIST2
            physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
            policy_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
            step_hz = policy_hz  # Use physics frequency instead of args_cli.step_hz
            print(f"✅ TWIST2 Wholebody: step_hz set to {step_hz}Hz (physics_dt={physics_dt}s, decimation={env_cfg.decimation})")
        else:
            use_wholebody = False
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

    # Also set action_provider on env for camera_state.py to access
    env.action_provider = action_provider
    print(f"[sim_main] Set action_provider on env: {type(action_provider)}")

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

    # Initialize joint position tracker (text-based, no GUI)
    joint_tracker = None
    try:
        print("\n" + "="*80)
        print("Initializing Joint Position Tracker (Text-based)")
        print("="*80)
        robot = env.scene["robot"]
        joint_tracker = JointPositionTracker(
            num_joints=robot.num_joints,
            window_size=200
        )
        # Update joint names
        if hasattr(robot.data, 'joint_names'):
            joint_tracker.joint_names = robot.data.joint_names
        print(f"Tracking {robot.num_joints} joints")
        print("Statistics will be printed every 10 seconds")
        print("="*80 + "\n")
    except Exception as e:
        print(f"Warning: Failed to initialize joint tracker: {e}")
        print("Continuing without tracking...")

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
                        # Only update state and reward every 10 loops to improve performance
                        if loop_count % 10 == 0:
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
                        # Check for reset command from Redis (from Pico controller)
                        try:
                            import redis
                            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                            reset_trigger = redis_client.get("isaac_reset_trigger")

                            # Debug: print every 50 loops to show we're checking
                            if loop_count % 50 == 0:
                                print(f"[DEBUG] Checking Redis reset trigger... (value: {reset_trigger})")

                            if reset_trigger:
                                import json
                                reset_cmd_redis = json.loads(reset_trigger)
                                reset_category = reset_cmd_redis.get("reset_category", "2")
                                print(f"[DEBUG] ✅ Received reset from Redis: category={reset_category}")

                                if reset_category == "1":
                                    print("🔄 Resetting object...")
                                    env_cfg.event_manager.trigger("reset_object_self", env)
                                elif reset_category == "2":
                                    print("🔄 Resetting all (robot + objects)...")
                                    env_cfg.event_manager.trigger("reset_all_self", env)
                                elif reset_category == "3":
                                    print("🔄 Complete reset (PhysX + all entities)...")
                                    # Complete reset: clear PhysX internal state
                                    env.sim.reset()  # Clear PhysX internal state
                                    env.reset()      # Reset environment
                                    env_cfg.event_manager.trigger("reset_all_self", env)
                                    print("✅ Complete reset finished")

                                    # Send reset complete signal via Redis
                                    try:
                                        reset_complete_signal = {
                                            "status": "complete",
                                            "timestamp": int(time.time() * 1000)
                                        }
                                        redis_client.set("isaac_reset_complete_unitree_g1_with_hands",
                                                       json.dumps(reset_complete_signal))
                                        redis_client.expire("isaac_reset_complete_unitree_g1_with_hands", 5)
                                        print("✅ Reset complete signal sent via Redis")
                                    except Exception as e:
                                        print(f"❌ Failed to send reset complete signal: {e}")

                                # Clear the trigger
                                redis_client.delete("isaac_reset_trigger")
                                print("[DEBUG] Reset trigger cleared from Redis")
                        except redis.ConnectionError as e:
                            if loop_count % 100 == 0:  # Print occasionally
                                print(f"[WARN] Redis not available for reset trigger: {e}")
                        except Exception as e:
                            print(f"[ERROR] Failed to check Redis reset trigger: {e}")

                        # Check for reset command from DDS (original method)
                        # print(f"reset_pose_dds: {reset_pose_dds}")
                        try:
                            reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                            # Debug: print when command is received
                            if reset_pose_cmd is not None:
                                print(f"[DEBUG] Received reset command: {reset_pose_cmd}")
                        except Exception as e:
                            print(f"Failed to get reset pose command: {e}")
                            raise e
                        # # print(f"reset_pose_cmd: {reset_pose_cmd}")
                        # Compute current reward values manually if needed for debugging
                        if loop_count % 10 == 0:  # Only compute reward every 10 loops
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

                                if reset_category == '1':
                                    # Reset object only
                                    print("🔄 Resetting object...")
                                    env_cfg.event_manager.trigger("reset_object_self", env)
                                    reset_pose_dds.write_reset_pose_command(-1)
                                elif reset_category == '2':
                                    # Reset all (robot + objects)
                                    print("🔄 Resetting all (robot + objects)...")
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

                    # Update joint tracker
                    if joint_tracker and loop_count % 2 == 0:  # Update every 2 steps
                        try:
                            target_pos = env.scene["robot"].data.joint_pos_target[0].cpu().numpy()
                            current_pos = env.scene["robot"].data.joint_pos[0].cpu().numpy()
                            joint_tracker.update_data(target_pos, current_pos, timestamp=loop_count)

                            # Print compact summary every 100 steps
                            if loop_count % 100 == 0:
                                joint_tracker.print_compact_summary()
                        except Exception as e:
                            if loop_count % 500 == 0:  # Print error occasionally
                                print(f"Warning: Joint tracker update failed: {e}")

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

                        # Print joint tracking statistics
                        if joint_tracker:
                            try:
                                joint_tracker.print_statistics(top_n=joint_tracker.num_joints)  # Print all joints
                            except Exception as e:
                                print(f"Warning: Failed to print joint statistics: {e}")

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
