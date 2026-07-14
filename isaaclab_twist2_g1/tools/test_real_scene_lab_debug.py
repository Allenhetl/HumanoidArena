#!/usr/bin/env python3
"""Debug version of real_scene_lab headless test - finds exact hang point."""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"

def flush_print(msg):
    print(msg, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--video_output", type=str, default=str(Path(PROJECT_ROOT) / "test_videos" / "real_scene_lab_front_cam.mp4"))
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _append_kit_arg(args, kit_arg: str) -> None:
    existing = (getattr(args, "kit_args", "") or "").strip()
    parts = existing.split() if existing else []
    if kit_arg not in parts:
        parts.append(kit_arg)
    args.kit_args = " ".join(parts)


def main() -> int:
    parser = _build_parser()
    args_cli = parser.parse_args()

    args_cli.headless = True
    args_cli.enable_cameras = True
    args_cli.multi_gpu = False
    _append_kit_arg(args_cli, "--/renderer/multiGpu/enabled=False")
    device = getattr(args_cli, "device", "cuda:0") or "cuda:0"
    _append_kit_arg(args_cli, f"--/renderer/activeGpu={device}")

    # ===== STEP 1: AppLauncher =====
    flush_print("[debug] Step 1: Launching AppLauncher ...")
    t0 = time.time()
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    flush_print(f"[debug] Step 1: AppLauncher OK ({time.time()-t0:.1f}s)")

    # ===== STEP 2: Import gym =====
    flush_print("[debug] Step 2: import gymnasium ...")
    t1 = time.time()
    import gymnasium as gym
    flush_print(f"[debug] Step 2: gymnasium imported ({time.time()-t1:.1f}s)")

    # ===== STEP 3: Import tasks =====
    flush_print("[debug] Step 3: import tasks ...")
    t2 = time.time()
    import tasks  # noqa: F401
    flush_print(f"[debug] Step 3: tasks imported ({time.time()-t2:.1f}s)")

    # ===== STEP 4: parse_env_cfg =====
    flush_print(f"[debug] Step 4: parse_env_cfg for {TASK_NAME} ...")
    t3 = time.time()
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
    env_cfg.env_name = TASK_NAME
    flush_print(f"[debug] Step 4: env_cfg parsed ({time.time()-t3:.1f}s)")

    # ===== STEP 5: gym.make =====
    flush_print(f"[debug] Step 5: gym.make({TASK_NAME}) ...")
    t4 = time.time()
    env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
    flush_print(f"[debug] Step 5: gym.make OK ({time.time()-t4:.1f}s)")

    # ===== STEP 6: initialize_task_scene =====
    flush_print("[debug] Step 6: initialize_task_scene ...")
    t5 = time.time()
    from tasks.common_runtime import apply_optional_runtime_augments
    try:
        init_fn = getattr(env_cfg, "initialize_task_scene", None)
        if callable(init_fn):
            init_fn(env, args_cli)
        else:
            apply_optional_runtime_augments(args_cli)
        flush_print(f"[debug] Step 6: scene init OK ({time.time()-t5:.1f}s)")
    except Exception as exc:
        flush_print(f"[debug] Step 6: WARNING - scene init skipped: {exc}")

    # ===== STEP 7: env.reset =====
    flush_print("[debug] Step 7: env.reset ...")
    t6 = time.time()
    obs, info = env.reset()
    flush_print(f"[debug] Step 7: env.reset OK ({time.time()-t6:.1f}s)")

    # ===== STEP 8: Check camera =====
    flush_print(f"[debug] Step 8: Checking camera ... scene keys: {list(env.scene.keys())}")
    has_front_cam = "front_camera" in env.scene.keys()
    flush_print(f"[debug] Step 8: front_camera in scene: {has_front_cam}")

    # ===== STEP 9: Record video =====
    flush_print("[debug] Step 9: Recording video ...")
    import cv2
    video_path = args_cli.video_output
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    frame_count = 0

    t7 = time.time()
    # SONIC standing pose for 29 body joints
    sonic_body_default = np.array([
        -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2,
        0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ], dtype=np.float64)
    # Compute default joint positions from current state (SONIC default offset)
    init_joints = env.scene["robot"].data.joint_pos[0, :29].cpu().numpy()
    body_offset = sonic_body_default - init_joints
    flush_print(f"[debug] init_joints[0]={init_joints[0]:.4f}, sonic_default[0]={sonic_body_default[0]:.4f}, offset[0]={body_offset[0]:.4f}")

    for step in range(args_cli.num_steps):
        action = torch.zeros(env.action_space.shape, dtype=torch.float64, device=env.device)
        if action.dim() == 1:
            action[:29] = torch.from_numpy(body_offset).to(device=env.device, dtype=torch.float64)
        else:
            action[0, :29] = torch.from_numpy(body_offset).to(device=env.device, dtype=torch.float64)
        obs, reward, terminated, truncated, info = env.step(action)

        if has_front_cam:
            try:
                camera = env.scene["front_camera"]
                rgb = camera.data.output.get("rgb")
                if rgb is not None:
                    frame = rgb[0].detach().cpu().numpy()
                    if frame.ndim == 3:
                        if frame.shape[-1] == 4:
                            frame = frame[..., :3]
                        if frame.dtype != np.uint8:
                            frame = frame.clip(0, 255).astype(np.uint8)
                        if writer is None:
                            h, w = frame.shape[:2]
                            writer = cv2.VideoWriter(str(video_path), fourcc, 50, (w, h))
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        frame_count += 1
            except Exception as e:
                if step == 0:
                    flush_print(f"[debug] Step 9: camera error: {e}")

        if (step + 1) % 5 == 0 or step == 0:
            root_pos = env.scene["robot"].data.root_state_w[0, :3].cpu().numpy()
            flush_print(f"  step {step+1}: root_pos=[{root_pos[0]:.3f},{root_pos[1]:.3f},{root_pos[2]:.3f}] frames={frame_count}")

    if writer:
        writer.release()
    elapsed = time.time() - t7
    flush_print(f"[debug] Step 9: {args_cli.num_steps} steps in {elapsed:.1f}s ({args_cli.num_steps/elapsed:.1f} steps/s)")
    flush_print(f"[debug] Video saved: {video_path} ({frame_count} frames)")

    env.close()
    simulation_app.close()
    flush_print("[debug] SUCCESS")
    return 0


if __name__ == "__main__":
    import traceback
    try:
        exit_code = main()
    except Exception:
        flush_print("[debug] FATAL: " + traceback.format_exc())
        exit_code = 1
    finally:
        try:
            from isaaclab.app import AppLauncher
        except:
            pass
    sys.exit(exit_code)
