#!/usr/bin/env python3
"""SONIC decoder closed-loop control test for real_scene_lab with front-camera video."""
import os, sys, time, argparse
from pathlib import Path
import cv2, numpy as np, torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"

DECODER_PATH = str(Path(PROJECT_ROOT).parent / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")

SONIC_DEFAULT = np.array([
    -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2,
    0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

G1_ACTION_SCALE = np.array([
    0.3506614566, 0.3506614566, 0.5475464463, 0.3506614566, 0.3506614566,
    0.4385773242, 0.5475464463, 0.5475464463, 0.4385773242, 0.3506614566,
    0.3506614566, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
    0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
    0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242,
    0.0745008737, 0.0745008737, 0.0745008737, 0.0745008737,
], dtype=np.float32)

HIST_LEN = 10

def flush(msg):
    print(msg, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--video_output", type=str,
    default=str(Path(PROJECT_ROOT) / "test_videos" / "real_scene_lab_sonic.mp4"))
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
device = getattr(args, "device", "cuda:0") or "cuda:0"

args.headless = True
args.enable_cameras = True
args.multi_gpu = False
kit = (getattr(args, "kit_args", "") or "").strip()
if "--/renderer/multiGpu/enabled=False" not in kit:
    kit += " --/renderer/multiGpu/enabled=False"
if f"--/renderer/activeGpu={device}" not in kit:
    kit += f" --/renderer/activeGpu={device}"
args.kit_args = kit.strip()

flush(f"[sonic] device={device} steps={args.num_steps}")

t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
flush(f"[sonic] AppLauncher OK ({time.time()-t0:.1f}s)")

import gymnasium as gym
import tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_runtime import apply_optional_runtime_augments

env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
env_cfg.env_name = TASK_NAME
env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
flush("[sonic] env OK")

init_fn = getattr(env_cfg, "initialize_task_scene", None)
if callable(init_fn):
    init_fn(env, args)
else:
    apply_optional_runtime_augments(args)

obs, info = env.reset()
flush("[sonic] reset OK")
robot = env.scene["robot"]
has_cam = "front_camera" in env.scene.keys()
flush(f"[sonic] front_camera={has_cam}")

# Load decoder
import onnxruntime as ort
decoder = ort.InferenceSession(DECODER_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
dec_in_name = decoder.get_inputs()[0].name
flush("[sonic] decoder loaded")

# Map joint indices
all_joint_names = [j.strip() for j in robot.data.joint_names]
SONIC_ORDER = """left_hip_pitch_joint,right_hip_pitch_joint,waist_yaw_joint,left_hip_roll_joint,right_hip_roll_joint,waist_roll_joint,left_hip_yaw_joint,right_hip_yaw_joint,waist_pitch_joint,left_knee_joint,right_knee_joint,left_shoulder_pitch_joint,right_shoulder_pitch_joint,left_ankle_pitch_joint,right_ankle_pitch_joint,left_shoulder_roll_joint,right_shoulder_roll_joint,left_ankle_roll_joint,right_ankle_roll_joint,left_shoulder_yaw_joint,right_shoulder_yaw_joint,left_elbow_joint,right_elbow_joint,left_wrist_roll_joint,right_wrist_roll_joint,left_wrist_pitch_joint,right_wrist_pitch_joint,left_wrist_yaw_joint,right_wrist_yaw_joint""".split(",")
name_to_idx = {n.strip(): i for i, n in enumerate(all_joint_names)}
sonic_indices = [name_to_idx.get(n.strip(), 0) for n in SONIC_ORDER]
flush(f"[sonic] mapped {len(sonic_indices)} sonic joints")

def get_body_jp(robot):
    return robot.data.joint_pos[0].cpu().numpy()[sonic_indices].astype(np.float32)

def get_body_jv(robot):
    return robot.data.joint_vel[0].cpu().numpy()[sonic_indices].astype(np.float32)

def gravity_dir(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    g_world = np.array([0.0, 0.0, -1.0])
    t0 = qw*g_world[0] + qy*g_world[2] - qz*g_world[1]
    t1 = qw*g_world[1] + qz*g_world[0] - qx*g_world[2]
    t2 = qw*g_world[2] + qx*g_world[1] - qy*g_world[0]
    return np.array([
        qw*t0 + t1*qz - t2*qy - g_world[0],
        qw*t1 + t2*qx - t0*qz - g_world[1],
        qw*t2 + t0*qy - t1*qx - g_world[2],
    ], dtype=np.float32)

# Init history buffers
init_jp = get_body_jp(robot)
init_jv = get_body_jv(robot)
root_ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy().astype(np.float32)
base_quat = robot.data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
gdir = gravity_dir(base_quat)
jp_delta = init_jp - SONIC_DEFAULT

ang_vel_hist = np.tile(root_ang_vel, (HIST_LEN, 1))
jp_hist = np.tile(jp_delta, (HIST_LEN, 1))
jv_hist = np.tile(init_jv, (HIST_LEN, 1))
act_hist = np.zeros((HIST_LEN, 29), dtype=np.float32)
grav_hist = np.tile(gdir, (HIST_LEN, 1))

# Video recorder
video_path = args.video_output
Path(video_path).parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = None
frame_count = 0

t_loop = time.time()
for step in range(args.num_steps):
    # Build decoder input
    latent = np.zeros(64, dtype=np.float32)
    dec_obs = np.concatenate([
        latent, ang_vel_hist.ravel(), jp_hist.ravel(),
        jv_hist.ravel(), act_hist.ravel(), grav_hist.ravel(),
    ])[np.newaxis].astype(np.float32)

    # Run decoder
    raw = decoder.run(None, {dec_in_name: dec_obs})[0].flatten()[:29].astype(np.float32)

    # Target = raw * scale + default
    target_body = raw * G1_ACTION_SCALE + SONIC_DEFAULT

    # Build action (43 dim: 29 body + 14 hand)
    action_np = np.concatenate([target_body.astype(np.float64), np.zeros(14, dtype=np.float64)])
    action = torch.from_numpy(action_np).to(env.device)
    if action.dim() > 1:
        action = action.squeeze(0)

    obs, _, _, _, _ = env.step(action)

    # Update histories
    new_jp = get_body_jp(robot)
    new_jv = get_body_jv(robot)
    new_av = robot.data.root_ang_vel_b[0].cpu().numpy().astype(np.float32)
    new_quat = robot.data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
    new_gdir = gravity_dir(new_quat)
    new_delta = new_jp - SONIC_DEFAULT

    ang_vel_hist = np.roll(ang_vel_hist, -1, axis=0); ang_vel_hist[-1] = new_av
    jp_hist = np.roll(jp_hist, -1, axis=0); jp_hist[-1] = new_delta
    jv_hist = np.roll(jv_hist, -1, axis=0); jv_hist[-1] = new_jv
    act_hist = np.roll(act_hist, -1, axis=0); act_hist[-1] = raw
    grav_hist = np.roll(grav_hist, -1, axis=0); grav_hist[-1] = new_gdir

    # Record frame
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
        except Exception:
            pass

    if (step + 1) % 20 == 0 or step == 0:
        rp = robot.data.root_state_w[0, :3].cpu().numpy()
        flush(f"  step {step+1:3d}: root=[{rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f}] frames={frame_count} decoder_raw=[{raw.min():.3f},{raw.max():.3f}]")

if vid_writer: vid_writer.release()
flush(f"[sonic] {args.num_steps} steps in {time.time()-t_loop:.1f}s, video: {video_path} ({frame_count}f)")
env.close()
simulation_app.close()
flush("[sonic] DONE")
