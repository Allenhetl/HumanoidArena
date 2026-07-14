#!/usr/bin/env python3
"""Check front camera RGB pixel values - renders 5 steps and prints color stats."""
import os, sys, time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"

# Build minimal args
import argparse
parser = argparse.ArgumentParser()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
cli = parser.parse_args([])
cli.headless = True
cli.enable_cameras = True
cli.multi_gpu = False
cli.device = "cuda:1"
cli.kit_args = "--/renderer/multiGpu/enabled=False --/renderer/activeGpu=cuda:1"

t0 = time.time()
app_launcher = AppLauncher(cli)
simulation_app = app_launcher.app
print(f"[color-check] AppLauncher OK ({time.time()-t0:.1f}s)")

import gymnasium as gym
import tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_runtime import apply_optional_runtime_augments

env_cfg = parse_env_cfg(TASK_NAME, device=cli.device, num_envs=1)
env_cfg.env_name = TASK_NAME
env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped

init_fn = getattr(env_cfg, "initialize_task_scene", None)
if callable(init_fn):
    init_fn(env, cli)
else:
    apply_optional_runtime_augments(cli)

obs, info = env.reset()
print("[color-check] env.reset OK")

robot = env.scene["robot"]
has_cam = "front_camera" in env.scene.keys()
print(f"[color-check] front_camera present: {has_cam}")

if has_cam:
    for step in range(3):
        action = torch.zeros(env.action_space.shape, dtype=torch.float64, device=env.device)
        obs, _, _, _, _ = env.step(action)
        rgb = env.scene["front_camera"].data.output.get("rgb")
        if rgb is not None:
            f = rgb[0].detach().cpu().numpy()
            if f.dtype != np.uint8:
                f_u8 = np.clip(f, 0, 255).astype(np.uint8) if f.max() > 1 else np.clip(f*255, 0, 255).astype(np.uint8)
            else:
                f_u8 = f
            print(f"\nStep {step}: shape={f.shape}, dtype={f.dtype}, range=[{f.min():.3f},{f.max():.3f}]")
            for ci, cn in enumerate(["R","G","B"]):
                ch = f[..., ci]
                print(f"  {cn}: min={ch.min():.3f} max={ch.max():.3f} mean={ch.mean():.3f} std={ch.std():.3f}")
            unique = len(np.unique(f_u8.reshape(-1, 3), axis=0))
            print(f"  unique uint8 RGB values: {unique}")
            cy, cx = f.shape[0]//2, f.shape[1]//2
            print(f"  center pixel: {f[cy, cx]}")
            print(f"  corner(0,0): {f[0, 0]}")
            print(f"  corner(-1,-1): {f[-1, -1]}")

env.close()
simulation_app.close()
print("[color-check] DONE")
