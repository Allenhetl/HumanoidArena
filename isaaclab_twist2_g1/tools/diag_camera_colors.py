#!/usr/bin/env python3
"""Phase A: Front camera RGB pixel diagnostic for NuRec rendering gray issue."""
import os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"

def flush(msg):
    print(msg, flush=True)

# --- Parse args properly ---
parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=10)
parser.add_argument("--device_override", type=str, default="cuda:2")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
device = getattr(args, "device_override", args.device)

args.headless = True
args.enable_cameras = True
args.multi_gpu = False
kit = (getattr(args, "kit_args", "") or "").strip()
if "--/renderer/multiGpu/enabled=False" not in kit:
    kit += " --/renderer/multiGpu/enabled=False"
if f"--/renderer/activeGpu={device}" not in kit:
    kit += f" --/renderer/activeGpu={device}"
args.kit_args = kit.strip()

flush(f"[diag] device={device} steps={args.num_steps}")

# --- Launch ---
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
flush(f"[diag] AppLauncher OK ({time.time()-t0:.1f}s)")

import gymnasium as gym
import tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_runtime import apply_optional_runtime_augments

env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
env_cfg.env_name = TASK_NAME
env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
flush("[diag] env OK")

init_fn = getattr(env_cfg, "initialize_task_scene", None)
if callable(init_fn):
    init_fn(env, args)
else:
    apply_optional_runtime_augments(args)

obs, info = env.reset()
flush("[diag] reset OK")
robot = env.scene["robot"]

# --- Check all cameras ---
for cam_name in ["front_camera", "world_camera"]:
    has = cam_name in env.scene.keys()
    flush(f"[diag] {cam_name} present: {has}")

# --- Run steps and collect stats ---
all_stats = {"front_camera": [], "world_camera": []}

for step in range(args.num_steps):
    action = torch.zeros(env.action_space.shape, dtype=torch.float64, device=env.device)
    obs, _, _, _, _ = env.step(action)

    for cam_name in ["front_camera", "world_camera"]:
        if cam_name not in env.scene.keys():
            continue
        rgb = env.scene[cam_name].data.output.get("rgb")
        if rgb is None:
            if step == 0:
                flush(f"[diag] WARNING: {cam_name} rgb is None!")
            continue
        f = rgb[0].detach().cpu().numpy()
        stats = {
            "step": step,
            "shape": f.shape,
            "dtype": str(f.dtype),
            "min": float(f.min()), "max": float(f.max()),
            "mean": float(f.mean()), "std": float(f.std()),
        }
        if f.ndim == 3 and f.shape[-1] >= 3:
            for ci, cn in enumerate(["R","G","B"]):
                ch = f[..., ci]
                stats[f"{cn}_min"] = float(ch.min())
                stats[f"{cn}_max"] = float(ch.max())
                stats[f"{cn}_mean"] = float(ch.mean())
                stats[f"{cn}_std"] = float(ch.std())
        all_stats[cam_name].append(stats)

# --- Print results ---
for cam_name in ["front_camera", "world_camera"]:
    if not all_stats[cam_name]:
        continue
    s = all_stats[cam_name][-1]  # last step
    flush(f"\n{'='*60}")
    flush(f"[diag] === {cam_name} RGB STATS (step {s['step']}) ===")
    flush(f"  shape={s['shape']} dtype={s['dtype']}")
    flush(f"  overall: min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.4f}")
    for cn in ["R","G","B"]:
        flush(f"  {cn}: min={s[f'{cn}_min']:.4f} max={s[f'{cn}_max']:.4f} mean={s[f'{cn}_mean']:.4f} std={s[f'{cn}_std']:.4f}")

    # Save raw frame for detailed analysis
    rgb = env.scene[cam_name].data.output.get("rgb")
    if rgb is not None:
        frame = rgb[0].detach().cpu().numpy()
        np.save(f"/tmp/{cam_name}_raw_frame.npy", frame)
        flush(f"  raw frame saved: /tmp/{cam_name}_raw_frame.npy ({frame.shape})")

# --- Conclusion ---
fc = all_stats.get("front_camera", [])
if fc:
    s = fc[-1]
    ch_std = max(s.get("R_std",0), s.get("G_std",0), s.get("B_std",0))
    ch_range = max(s.get("R_max",0)-s.get("R_min",0), s.get("G_max",0)-s.get("G_min",0), s.get("B_max",0)-s.get("B_min",0))
    flush(f"\n[diag] === DIAGNOSIS ===")
    if ch_std < 0.001:
        flush("[diag] -> ALL PIXELS SAME VALUE. NuRec SH evaluation completely broken (no output).")
    elif s['std'] < 0.02 and ch_range < 0.1:
        flush(f"[diag] -> Nearly uniform (std={s['std']:.4f}). DC-only rendering, view-dependent SH lost.")
    elif ch_std < 0.05:
        flush(f"[diag] -> Low variation (ch_std={ch_std:.4f}). Partial SH or tone-mapping compression.")
    else:
        flush(f"[diag] -> GOOD color variation (ch_std={ch_std:.4f}). Rendering works, issue elsewhere.")
    flush(f"[diag] R-mean={s['R_mean']:.4f} G-mean={s['G_mean']:.4f} B-mean={s['B_mean']:.4f}")

env.close()
simulation_app.close()
flush("[diag] DONE")
