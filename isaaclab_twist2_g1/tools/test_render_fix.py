#!/usr/bin/env python3
"""Quick test: set tonemap/op, check camera RGB change."""
import os, sys, time, argparse, numpy as np, torch
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str, default="tonemap",
                    choices=["tonemap", "nurec", "gaussian"])
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

print(f"[test] test={args.test} device={device}", flush=True)

t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[test] AppLauncher OK ({time.time()-t0:.1f}s)", flush=True)

# --- Apply test settings ---
import carb
settings = carb.settings.get_settings()

if args.test == "tonemap":
    before = settings.get("/rtx/post/tonemap/op")
    settings.set("/rtx/post/tonemap/op", 2)
    after = settings.get("/rtx/post/tonemap/op")
    print(f"[test] tonemap/op: {before} -> {after}", flush=True)

elif args.test == "nurec":
    import omni.kit.app
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    # Try to find and enable nurec
    for ext_name in ["omni.nurec", "omni.nurec.core", "omni.nurec.renderer"]:
        try:
            ext_mgr.set_extension_enabled(ext_name, True)
            print(f"[test] Enabled extension: {ext_name}", flush=True)
        except Exception as e:
            print(f"[test] Extension {ext_name}: {e}", flush=True)

elif args.test == "gaussian":
    import omni.kit.app
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    for ext_name in ["omni.gaussian", "omni.gaussian_splatting", "omni.particle_field"]:
        try:
            ext_mgr.set_extension_enabled(ext_name, True)
            print(f"[test] Enabled extension: {ext_name}", flush=True)
        except Exception as e:
            print(f"[test] Extension {ext_name}: {e}", flush=True)

# --- Create env and check camera ---
import gymnasium as gym
import tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_runtime import apply_optional_runtime_augments

env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
env_cfg.env_name = TASK_NAME
env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
print("[test] env OK", flush=True)

init_fn = getattr(env_cfg, "initialize_task_scene", None)
if callable(init_fn):
    init_fn(env, args)
else:
    apply_optional_runtime_augments(args)
obs, info = env.reset()
print("[test] reset OK", flush=True)

has_cam = "front_camera" in env.scene.keys()
print(f"[test] front_camera={has_cam}", flush=True)

# Run 3 steps and print stats
for step in range(3):
    action = torch.zeros(env.action_space.shape, dtype=torch.float64, device=env.device)
    obs, _, _, _, _ = env.step(action)

if has_cam:
    rgb = env.scene["front_camera"].data.output.get("rgb")
    if rgb is not None:
        f = rgb[0].detach().cpu().numpy()
        if f.dtype != np.uint8:
            f = f.clip(0, 255).astype(np.uint8) if f.max() > 1 else (f*255).clip(0,255).astype(np.uint8)
        print(f"\n[test] === front_camera RGB (step 2) ===")
        print(f"  shape={f.shape} dtype={f.dtype}")
        print(f"  overall: min={f.min():.1f} max={f.max():.1f} mean={f.mean():.1f} std={f.std():.1f}")
        for ci, cn in enumerate(["R","G","B"]):
            ch = f[..., ci]
            print(f"  {cn}: min={ch.min():.1f} max={ch.max():.1f} mean={ch.mean():.1f} std={ch.std():.1f}")
        cy, cx = f.shape[0]//2, f.shape[1]//2
        print(f"  center pixel: {f[cy, cx]}")

env.close()
simulation_app.close()
print("[test] DONE", flush=True)
