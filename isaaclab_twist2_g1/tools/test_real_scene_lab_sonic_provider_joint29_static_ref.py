#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = "/home/lab/zikang/HumanoidArena/isaaclab_twist2_g1"
HUMANOID_ROOT = "/home/lab/zikang/HumanoidArena"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"
ENCODER = f"{HUMANOID_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
DECODER = f"{HUMANOID_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"


def flush(msg):
    print(msg, flush=True)


def make_provider_args(cli):
    return SimpleNamespace(
        task=TASK_NAME,
        action_source="sonic_wholebody",
        input_source="pico_twist2",
        gmt_backend="sonic_joint29",
        sonic_pose_source="redis",
        sonic_redis_host="localhost",
        sonic_redis_port=6379,
        sonic_encoder_path=ENCODER,
        sonic_decoder_path=DECODER,
        sonic_input_timeout_s=2.0,
        robot_type="g129",
        enable_dex1_dds=False,
        enable_dex3_dds=False,
        enable_inspire_dds=False,
        replay_file="",
        replay_mode="inference_replay",
        replay_loop=False,
        record_during_replay=False,
        exit_when_replay_complete=False,
        recording_save_dir=f"{PROJECT_ROOT}/recording_data/real_scene_lab_provider_static_ref",
        recording_save_workers=1,
        recording_save_queue_size=2,
        enable_world_camera=False,
        enable_perspective_camera=False,
        lerobot_gripper_threshold=0.5,
        lerobot_server_url="",
        lerobot_server_timeout=5.0,
        lerobot_server_verify_ssl=False,
        sonic_debug=cli.sonic_debug,
        sonic_log_every=cli.sonic_log_every,
        enable_rtf_monitor=False,
        sonic_effort_control=cli.sonic_effort_control,
        sonic_smooth_steps=20,
        sonic_output_delay_steps=0,
        disable_front_camera=True,
    )


def inject_static_joint29(provider, robot, joint_ref, body_pos_ref, body_quat_ref, frame_idx):
    data = {
        "body_quat_w": body_quat_ref.reshape(1, 4).astype(np.float32),
        "adjusted_transl": body_pos_ref.reshape(1, 3).astype(np.float32),
        "joint_pos": joint_ref.reshape(1, 29).astype(np.float32),
        "joint_vel": np.zeros((1, 29), dtype=np.float32),
        "frame_index": np.array([frame_idx], dtype=np.int64),
        "timestamp_realtime": np.array([time.time()], dtype=np.float64),
        "timestamp_monotonic": np.array([time.monotonic()], dtype=np.float64),
        "heading_increment": np.array([0.0], dtype=np.float32),
    }
    provider._apply_pose_data(data, "lerobot_vla_joint29_v3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--video_output", type=str, default=f"{PROJECT_ROOT}/test_videos/real_scene_lab_sonic_provider_joint29_static_ref_100.mp4")
    parser.add_argument("--sonic_debug", action="store_true")
    parser.add_argument("--sonic_log_every", type=int, default=20)
    parser.add_argument("--sonic_effort_control", action="store_true")
    parser.add_argument("--save_png_dir", type=str, default="")
    parser.add_argument("--save_png_step", type=int, default=10)
    parser.add_argument("--capture_camera", type=str, default="front_camera")
    parser.add_argument("--save_reference_npz", type=str, default="")
    parser.add_argument("--load_reference_npz", type=str, default="")
    parser.add_argument("--env_config_yaml", type=str, default="",
                        help="Path to env config YAML (e.g. real_scene_ipark_sonic.yaml)")
    parser.add_argument("--settle_steps", type=int, default=0,
                        help="Number of zero-action physics steps to run before inference, to let robot settle on mesh")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    device = getattr(args, "device", "cuda:0") or "cuda:0"
    args.headless = True
    args.enable_cameras = True
    args.multi_gpu = False
    kit = (getattr(args, "kit_args", "") or "").strip()
    for part in ["--/renderer/multiGpu/enabled=False", f"--/renderer/activeGpu={device}"]:
        if part not in kit:
            kit = (kit + " " + part).strip()
    args.kit_args = kit

    flush(f"[provider_static] device={device} steps={args.num_steps} effort={args.sonic_effort_control}")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.common_runtime import apply_optional_runtime_augments
    from action_provider.action_provider_sonic import SonicActionProvider
    SonicActionProvider._resolve_ort_device_id = lambda self: None

    env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
    env_cfg.env_name = TASK_NAME
    if args.env_config_yaml:
        from tasks.common_env_config.loader import apply_env_config_yaml
        apply_env_config_yaml(
            env_cfg,
            args.env_config_yaml,
            task_name=TASK_NAME,
            route_name="sonic",
        )
    env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
    init_fn = getattr(env_cfg, "initialize_task_scene", None)
    if callable(init_fn):
        init_fn(env, args)
    else:
        apply_optional_runtime_augments(args)
    env.reset()
    robot = env.scene["robot"]
    flush("[provider_static] reset OK")
    # Debug: print initial robot state right after reset
    _root_state = robot.data.root_state_w[0].detach().cpu().numpy()
    _joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
    _joint_vel = robot.data.joint_vel[0].detach().cpu().numpy()
    _root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
    flush(
        "[provider_static] post-reset: root_pos={:.4f},{:.4f},{:.4f} root_quat={:.4f},{:.4f},{:.4f},{:.4f}"
        " root_lin_vel={:.4f},{:.4f},{:.4f} root_ang_vel_b={:.4f},{:.4f},{:.4f}"
        " joint_vel_range=[{:.4f},{:.4f}]".format(
            _root_state[0], _root_state[1], _root_state[2],
            _root_state[3], _root_state[4], _root_state[5], _root_state[6],
            _root_state[7], _root_state[8], _root_state[9],
            _root_ang_vel_b[0], _root_ang_vel_b[1], _root_ang_vel_b[2],
            float(_joint_vel.min()), float(_joint_vel.max()),
        )
    )
    # Settle: run physics steps with zero action to let robot stabilize on the mesh
    _settle_steps = getattr(args, "settle_steps", 0)
    if _settle_steps > 0:
        flush(f"[provider_static] settling for {_settle_steps} steps...")
        import torch as _torch
        _act_dim = env.action_manager.total_action_dim if hasattr(env.action_manager, 'total_action_dim') else 29
        _zero_action = _torch.zeros((1, _act_dim), device=env.device)
        for _si in range(_settle_steps):
            env.step(_zero_action)
        _root_state2 = robot.data.root_state_w[0].detach().cpu().numpy()
        _joint_vel2 = robot.data.joint_vel[0].detach().cpu().numpy()
        _root_ang_vel_b2 = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
        flush(
            "[provider_static] post-settle: root_pos={:.4f},{:.4f},{:.4f} root_quat={:.4f},{:.4f},{:.4f},{:.4f}"
            " root_ang_vel_b={:.4f},{:.4f},{:.4f} joint_vel_range=[{:.4f},{:.4f}]".format(
                _root_state2[0], _root_state2[1], _root_state2[2],
                _root_state2[3], _root_state2[4], _root_state2[5], _root_state2[6],
                _root_ang_vel_b2[0], _root_ang_vel_b2[1], _root_ang_vel_b2[2],
                float(_joint_vel2.min()), float(_joint_vel2.max()),
            )
        )

    provider = SonicActionProvider(env, make_provider_args(args))
    provider.on_env_reset()
    # Debug: print sonic_idx and joint_pos at sonic_idx
    _sonic_idx_np = provider._sonic_idx.cpu().numpy()
    _joint_pos_sonic = robot.data.joint_pos[0, provider._sonic_idx].detach().cpu().numpy()
    _joint_pos_full = robot.data.joint_pos[0].detach().cpu().numpy()
    _motion_hist_sum = float(np.abs(provider._motion_joint_pos_hist).sum())
    flush(
        "[provider_static] sonic_idx={} joint_pos_sonic_range=[{:.4f},{:.4f}] joint_pos_full_range=[{:.4f},{:.4f}]"
        " motion_hist_sum={:.4f} motion_hist_range=[{:.4f},{:.4f}]".format(
            _sonic_idx_np.tolist()[:5],
            float(_joint_pos_sonic.min()), float(_joint_pos_sonic.max()),
            float(_joint_pos_full.min()), float(_joint_pos_full.max()),
            _motion_hist_sum,
            float(provider._motion_joint_pos_hist.min()), float(provider._motion_joint_pos_hist.max()),
        )
    )
    if args.load_reference_npz:
        ref = np.load(args.load_reference_npz)
        joint_ref = ref["joint_ref"].astype(np.float32)
        body_pos_ref = ref["body_pos_ref"].astype(np.float32)
        body_quat_ref = ref["body_quat_ref"].astype(np.float32)
        flush(f"[provider_static] loaded_reference_npz={args.load_reference_npz}")
    else:
        joint_ref = robot.data.joint_pos[0, provider._sonic_idx].detach().cpu().numpy().astype(np.float32)
        body_pos_ref = robot.data.root_state_w[0, :3].detach().cpu().numpy().astype(np.float32)
        body_quat_ref = robot.data.root_state_w[0, 3:7].detach().cpu().numpy().astype(np.float32)
    if args.save_reference_npz:
        Path(args.save_reference_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_reference_npz,
            joint_ref=joint_ref,
            body_pos_ref=body_pos_ref,
            body_quat_ref=body_quat_ref,
        )
        flush(f"[provider_static] saved_reference_npz={args.save_reference_npz}")
    flush(
        "[provider_static] ref joint range=[{:.4f},{:.4f}] body_pos={} quat={}".format(
            float(joint_ref.min()), float(joint_ref.max()),
            np.array2string(body_pos_ref, precision=4, separator=","),
            np.array2string(body_quat_ref, precision=4, separator=","),
        )
    )
    fetch_counter = {"frame": 0}

    def _inject_fetch():
        frame_idx = fetch_counter["frame"]
        fetch_counter["frame"] = frame_idx + 1
        inject_static_joint29(provider, robot.data, joint_ref, body_pos_ref, body_quat_ref, frame_idx)

    provider._fetch_redis_pose = _inject_fetch

    # Pre-inject one frame so that _apply_pose_data initializes heading alignment
    # using the actual reference quaternion (which matches the robot init rotation)
    # instead of the default identity quaternion that _prime_default_reference_heading_align
    # would otherwise use before the first _fetch_redis_pose call.
    inject_static_joint29(provider, robot.data, joint_ref, body_pos_ref, body_quat_ref, 0)
    fetch_counter["frame"] = 1

    video_path = Path(args.video_output)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir = Path(args.save_png_dir) if args.save_png_dir else None
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    start = time.time()
    for step in range(args.num_steps):
        provider.get_action(env)
        if args.capture_camera in env.scene.keys():
            rgb = env.scene[args.capture_camera].data.output.get("rgb")
            if rgb is not None:
                frame = rgb[0].detach().cpu().numpy()
                if frame.ndim == 3:
                    if frame.shape[-1] == 4:
                        frame = frame[..., :3]
                    if png_dir is not None and step == args.save_png_step:
                        arr = frame
                        flush(
                            "[provider_static][rgb_stats] step={} dtype={} shape={} min={:.6g} max={:.6g} mean={:.6g} pct>=250={:.3f} pct_white245={:.3f}".format(
                                step,
                                arr.dtype,
                                arr.shape,
                                float(np.nanmin(arr)),
                                float(np.nanmax(arr)),
                                float(np.nanmean(arr)),
                                float(np.mean(np.nanmax(arr, axis=-1) >= 250.0) * 100.0),
                                float(np.mean(np.all(arr[..., :3] >= 245.0, axis=-1)) * 100.0),
                            )
                        )
                    if frame.dtype != np.uint8:
                        frame = frame.clip(0, 255).astype(np.uint8)
                    if png_dir is not None and step == args.save_png_step:
                        cv2.imwrite(str(png_dir / f"{args.capture_camera}_raw_clip_step_{step:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(video_path), fourcc, 50, (w, h))
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frames += 1
        if step == 0 or (step + 1) % 20 == 0:
            root = robot.data.root_state_w[0, :3].detach().cpu().numpy()
            quat = robot.data.root_state_w[0, 3:7].detach().cpu().numpy()
            raw = getattr(provider, "_latest_decoder_raw_action", np.zeros(29, dtype=np.float32))
            target = getattr(provider, "_latest_decoder_target", np.zeros(29, dtype=np.float32))
            flush(
                "  step {:3d}: root=[{:.3f},{:.3f},{:.3f}] quat=[{:.3f},{:.3f},{:.3f},{:.3f}] raw=[{:.3f},{:.3f}] target=[{:.3f},{:.3f}] frames={}".format(
                    step + 1, root[0], root[1], root[2], quat[0], quat[1], quat[2], quat[3],
                    float(raw.min()), float(raw.max()), float(target.min()), float(target.max()), frames
                )
            )
    if writer:
        writer.release()
    flush(f"[provider_static] {args.num_steps} steps in {time.time() - start:.1f}s, video={video_path} frames={frames}")
    env.close()
    simulation_app.close()
    flush("[provider_static] DONE")


if __name__ == "__main__":
    main()
