#!/usr/bin/env python3
"""Record front camera video with all render enhancements enabled."""
import os, sys, time, argparse, cv2, numpy as np, torch
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=100)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
device = getattr(args, "device", "cuda:0") or "cuda:0"

args.headless = True; args.enable_cameras = True; args.multi_gpu = False
kit = (getattr(args, "kit_args", "") or "").strip()
if "--/renderer/multiGpu/enabled=False" not in kit:
    kit += " --/renderer/multiGpu/enabled=False"
if f"--/renderer/activeGpu={device}" not in kit:
    kit += f" --/renderer/activeGpu={device}"
args.kit_args = kit.strip()

print(f"[enhanced] device={device} steps={args.num_steps}", flush=True)

t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[enhanced] AppLauncher OK ({time.time()-t0:.1f}s)", flush=True)

# Enable all render extensions
import carb, omni.kit.app
settings = carb.settings.get_settings()
settings.set("/rtx/post/tonemap/op", 2)
ext_mgr = omni.kit.app.get_app().get_extension_manager()
for ext in ["omni.nurec", "omni.nurec.core", "omni.nurec.renderer",
             "omni.gaussian", "omni.gaussian_splatting", "omni.particle_field"]:
    try:
        ext_mgr.set_extension_enabled(ext, True)
        print(f"[enhanced] {ext} enabled", flush=True)
    except:
        pass
print(f"[enhanced] tonemap/op={settings.get('/rtx/post/tonemap/op')}", flush=True)

import gymnasium as gym
import tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_runtime import apply_optional_runtime_augments

env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
env_cfg.env_name = TASK_NAME
env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
print("[enhanced] env OK", flush=True)

init_fn = getattr(env_cfg, "initialize_task_scene", None)
if callable(init_fn):
    init_fn(env, args)
else:
    apply_optional_runtime_augments(args)
obs, info = env.reset()
print("[enhanced] reset OK", flush=True)

robot = env.scene["robot"]
has_cam = "front_camera" in env.scene.keys()
print(f"[enhanced] front_camera={has_cam}", flush=True)

video_path = str(Path(PROJECT_ROOT) / "test_videos" / "real_scene_lab_enhanced.mp4")
Path(video_path).parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = None
frame_count = 0

t_loop = time.time()
for step in range(args.num_steps):
    action = torch.zeros(env.action_space.shape, dtype=torch.float64, device=env.device)
    obs, _, _, _, _ = env.step(action)
    if has_cam:
        try:
            rgb = env.scene["front_camera"].data.output.get("rgb")
            if rgb is not None:
                f = rgb[0].detach().cpu().numpy()
                if f.ndim == 3:
                    if f.shape[-1] == 4: f = f[..., :3]
                    if f.dtype != np.uint8: f = f.clip(0, 255).astype(np.uint8)
                    if vid_writer is None:
                        h, w = f.shape[:2]
                        vid_writer = cv2.VideoWriter(str(video_path), fourcc, 50, (w, h))
                    vid_writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    frame_count += 1
        except:
            pass
    if (step+1) % 20 == 0 or step == 0:
        rp = robot.data.root_state_w[0,:3].cpu().numpy()
        print(f"  step {step+1:3d}: root=[{rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f}] frames={frame_count}", flush=True)

if vid_writer: vid_writer.release()
print(f"[enhanced] {args.num_steps} steps in {time.time()-t_loop:.1f}s, video: {video_path} ({frame_count}f)", flush=True)
env.close()
simulation_app.close()
print("[enhanced] DONE", flush=True)
