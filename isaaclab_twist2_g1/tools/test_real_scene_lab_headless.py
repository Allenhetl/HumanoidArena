#!/usr/bin/env python3
"""Headless smoke test + front-camera video recording for real-scene-lab."""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"
STEPS_PER_SECOND = 50
SIM_SECONDS = 10
NUM_STEPS = STEPS_PER_SECOND * SIM_SECONDS
VIDEO_OUTPUT = str(Path(PROJECT_ROOT) / "test_videos" / "real_scene_lab_front_cam.mp4")


def _capture_front_camera_rgb(env):
    try:
        if "front_camera" not in env.scene.keys():
            return None
        camera = env.scene["front_camera"]
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        frame = rgb[0].detach().cpu().numpy()
        if frame.ndim != 3:
            return None
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)
        return frame
    except Exception:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless test + video recording for real-scene-lab")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help=f"Simulation steps (default={NUM_STEPS} for {SIM_SECONDS}s)")
    parser.add_argument("--video_output", type=str, default=VIDEO_OUTPUT)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _append_kit_arg(args, kit_arg: str) -> None:
    existing = (getattr(args, "kit_args", "") or "").strip()
    parts = existing.split() if existing else []
    if kit_arg not in parts:
        parts.append(kit_arg)
    args.kit_args = " ".join(parts)


class SimpleVideoRecorder:
    """Minimal single-stream MP4 recorder that writes frames incrementally."""
    def __init__(self, save_path: str, fps: int = 30):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer = None
        self.frame_count = 0
        self._frame_size = None

    def _ensure_writer(self, img: np.ndarray):
        h, w = img.shape[:2]
        size = (w, h)
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.save_path), fourcc, self.fps, size)
            self._frame_size = size
        elif self._frame_size != size:
            img = cv2.resize(img, self._frame_size)
        return img

    def add_frame(self, img: np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        img = self._ensure_writer(img)
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def main() -> int:
    parser = _build_parser()
    args_cli = parser.parse_args()

    args_cli.headless = True
    args_cli.enable_cameras = True
    args_cli.multi_gpu = False
    _append_kit_arg(args_cli, "--/renderer/multiGpu/enabled=False")
    device = getattr(args_cli, "device", "cuda:0") or "cuda:0"
    _append_kit_arg(args_cli, f"--/renderer/activeGpu={device}")

    print("[test] Launching Isaac Sim in headless mode ...")
    t0 = time.time()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print(f"[test] AppLauncher OK ({time.time() - t0:.1f}s)")

    print("[test] Importing task registrations ...")
    import gymnasium as gym
    import tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.common_runtime import apply_optional_runtime_augments
    print("[test] Tasks imported OK")

    print(f"[test] Creating environment: {TASK_NAME} ...")
    t1 = time.time()
    env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
    env_cfg.env_name = TASK_NAME
    env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
    print(f"[test] gym.make OK ({time.time() - t1:.1f}s)")

    try:
        init_fn = getattr(env_cfg, "initialize_task_scene", None)
        if callable(init_fn):
            init_fn(env, args_cli)
        else:
            apply_optional_runtime_augments(args_cli)
        print("[test] initialize_task_scene OK")
    except Exception as exc:
        print(f"[test] WARNING: scene init skipped: {exc}")

    print("[test] Resetting environment ...")
    t2 = time.time()
    obs, info = env.reset()
    print(f"[test] env.reset OK ({time.time() - t2:.1f}s)")

    # Create video recorder
    video_path = args_cli.video_output
    recorder = SimpleVideoRecorder(save_path=video_path, fps=STEPS_PER_SECOND)
    print(f"[test] Recording video to: {video_path}")
    print(f"[test] Running {args_cli.num_steps} steps (~{args_cli.num_steps/STEPS_PER_SECOND:.0f}s sim) ...")

    t3 = time.time()
    fall_step = -1
    for step in range(args_cli.num_steps):
        action = torch.from_numpy(env.action_space.sample()).to(env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = _capture_front_camera_rgb(env)
        if frame is not None:
            recorder.add_frame(frame)
        if (step + 1) % 100 == 0:
            root_pos = env.scene["robot"].data.root_state_w[0, :3].cpu().numpy()
            print(f"  step {step+1:4d}: root_pos=[{root_pos[0]:.3f},{root_pos[1]:.3f},{root_pos[2]:.3f}] frames_recorded={recorder.frame_count}")
        if (terminated or truncated) and fall_step < 0:
            fall_step = step
    elapsed = time.time() - t3
    fps = args_cli.num_steps / elapsed if elapsed > 0 else 0

    recorder.close()
    print(f"[test] Simulation done: {args_cli.num_steps} steps in {elapsed:.1f}s ({fps:.0f} steps/s)")
    print(f"[test] Video saved: {video_path} ({recorder.frame_count} frames, {recorder.frame_count/STEPS_PER_SECOND:.1f}s)")
    if fall_step >= 0:
        print(f"[test] Robot fell at step {fall_step}")

    env.close()
    simulation_app.close()
    print("[test] SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
