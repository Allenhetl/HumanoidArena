#!/usr/bin/env python3
"""SONIC-default-pose headless test for real_scene_lab + front-camera video."""
import argparse, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"

SONIC_BODY_POS = np.array([
    -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2,
    0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float64)

def flush_print(msg):
    print(msg, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--video_output", type=str,
                    default=str(Path(PROJECT_ROOT) / "test_videos" / "real_scene_lab_front_cam.mp4"))
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
args_cli.enable_cameras = True
args_cli.multi_gpu = False
kit_args = (getattr(args_cli, "kit_args", "") or "").strip()
if "--/renderer/multiGpu/enabled=False" not in kit_args:
    args_cli.kit_args = (kit_args + " --/renderer/multiGpu/enabled=False").strip()
device = getattr(args_cli, "device", "cuda:0") or "cuda:0"
kit_args = (getattr(args_cli, "kit_args", "") or "").strip()
if f"--/renderer/activeGpu={device}" not in kit_args:
    args_cli.kit_args = (kit_args + f" --/renderer/activeGpu={device}").strip()

flush_print(f"[test] device={device} steps={args_cli.num_steps}")

t0 = time.time()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
flush_print(f"[test] AppLauncher OK ({time.time()-t0:.1f}s)")

import gymnasium as gym
import tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_runtime import apply_optional_runtime_augments
flush_print("[test] imports OK")

env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
env_cfg.env_name = TASK_NAME
env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
flush_print("[test] env OK")

init_fn = getattr(env_cfg, "initialize_task_scene", None)
if callable(init_fn):
    init_fn(env, args_cli)
else:
    apply_optional_runtime_augments(args_cli)

obs, info = env.reset()
flush_print("[test] reset OK")

robot = env.scene["robot"]
has_cam = "front_camera" in env.scene.keys()
flush_print(f"[test] front_camera={has_cam}")

# Compute SONIC body joint offset from current state
body_now = robot.data.joint_pos[0, :29].cpu().numpy().astype(np.float64)
body_offset = SONIC_BODY_POS - body_now
flush_print(f"[test] body_offset[0:5]={body_offset[:5].round(4)}, range=[{body_offset.min():.4f},{body_offset.max():.4f}]")

video_path = args_cli.video_output
Path(video_path).parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = None
frame_count = 0

t_loop = time.time()
for step in range(args_cli.num_steps):
    action = torch.zeros(env.action_space.shape, dtype=torch.float64, device=env.device)
    if action.dim() == 1:
        action[:29] = torch.from_numpy(body_offset).to(device=env.device, dtype=torch.float64)
    else:
        action[0, :29] = torch.from_numpy(body_offset).to(device=env.device, dtype=torch.float64)
    obs, reward, terminated, truncated, info = env.step(action)

    if has_cam:
        try:
            rgb = env.scene["front_camera"].data.output.get("rgb")
            if rgb is not None:
                f = rgb[0].detach().cpu().numpy()
                if f.ndim == 3:
                    if f.shape[-1] == 4: f = f[..., :3]
                    if f.dtype != np.uint8: f = f.clip(0, 255).astype(np.uint8)
                    if vid_writer is None:
                        vid_writer = cv2.VideoWriter(str(video_path), fourcc, 50, f.shape[:2][::-1])
                    vid_writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    frame_count += 1
        except Exception:
            pass

    if (step + 1) % 20 == 0 or step == 0:
        rp = robot.data.root_state_w[0, :3].cpu().numpy()
        flush_print(f"  step {step+1:3d}: root=[{rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f}] frames={frame_count}")

if vid_writer: vid_writer.release()
flush_print(f"[test] {args_cli.num_steps} steps in {time.time()-t_loop:.1f}s, video: {video_path} ({frame_count}f)")
env.close()
simulation_app.close()
flush_print("[test] DONE")
