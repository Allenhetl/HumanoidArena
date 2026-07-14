#!/usr/bin/env python3
"""SONIC closed-loop control: feed robot state as encoder input, decoder produces actions."""
import os, sys, time, argparse, struct, cv2, numpy as np, torch
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK_NAME = "Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody"

ENC_PATH = str(Path(PROJECT_ROOT).parent / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
DEC_PATH = str(Path(PROJECT_ROOT).parent / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")

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

STEP5_FRAMES = 10
HIST_LEN = 10

def flush(msg): print(msg, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument(
    "--run_decoder_without_input",
    action="store_true",
    help="Run the synthetic fixed-reference encoder/decoder experiment even though no live input is present.",
)
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

flush(f"[sonic3] device={device}")

t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
flush(f"[sonic3] AppLauncher OK ({time.time()-t0:.1f}s)")

import gymnasium as gym
import tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_runtime import apply_optional_runtime_augments

env_cfg = parse_env_cfg(TASK_NAME, device=device, num_envs=1)
env_cfg.env_name = TASK_NAME
env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
flush("[sonic3] env OK")

init_fn = getattr(env_cfg, "initialize_task_scene", None)
if callable(init_fn): init_fn(env, args)
else: apply_optional_runtime_augments(args)
obs, info = env.reset()
flush("[sonic3] reset OK")

robot = env.scene["robot"]
has_cam = "front_camera" in env.scene.keys()
flush(f"[sonic3] front_camera={has_cam}")

# Load models
import onnxruntime as ort
encoder = ort.InferenceSession(ENC_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
decoder = ort.InferenceSession(DEC_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
enc_in_name = encoder.get_inputs()[0].name
dec_in_name = decoder.get_inputs()[0].name
flush("[sonic3] encoder+decoder loaded")

# Map joints
all_names = [j.strip() for j in robot.data.joint_names]
SONIC_ORDER = "left_hip_pitch_joint,right_hip_pitch_joint,waist_yaw_joint,left_hip_roll_joint,right_hip_roll_joint,waist_roll_joint,left_hip_yaw_joint,right_hip_yaw_joint,waist_pitch_joint,left_knee_joint,right_knee_joint,left_shoulder_pitch_joint,right_shoulder_pitch_joint,left_ankle_pitch_joint,right_ankle_pitch_joint,left_shoulder_roll_joint,right_shoulder_roll_joint,left_ankle_roll_joint,right_ankle_roll_joint,left_shoulder_yaw_joint,right_shoulder_yaw_joint,left_elbow_joint,right_elbow_joint,left_wrist_roll_joint,right_wrist_roll_joint,left_wrist_pitch_joint,right_wrist_pitch_joint,left_wrist_yaw_joint,right_wrist_yaw_joint".split(",")
name_to_idx = {n.strip(): i for i, n in enumerate(all_names)}
missing_sonic_joints = [name for name in SONIC_ORDER if name not in name_to_idx]
if missing_sonic_joints:
    raise RuntimeError(f"SONIC joints missing from robot articulation: {missing_sonic_joints}")
sonic_idx = [name_to_idx[name] for name in SONIC_ORDER]

# `JointPositionActionCfg(use_default_offset=True)` interprets env.step actions
# as deltas from Isaac Lab's default pose.  SONIC emits absolute targets in its
# own standing frame, so passing those targets through directly double-applies
# an offset and destroys the no-input standing behavior.
env_default_sonic = robot.data.default_joint_pos[0].cpu().numpy()[sonic_idx].astype(np.float32)

def sonic_target_to_env_action(target_sonic):
    action = np.zeros(43, dtype=np.float32)
    action[:29] = np.asarray(target_sonic, dtype=np.float32) - env_default_sonic
    return torch.from_numpy(action).to(env.device).unsqueeze(0)

def get_sonic_jp(robot):
    return robot.data.joint_pos[0].cpu().numpy()[sonic_idx].astype(np.float32)

def get_sonic_jv(robot):
    return robot.data.joint_vel[0].cpu().numpy()[sonic_idx].astype(np.float32)

def gravity_dir(qwxyz):
    qw, qx, qy, qz = qwxyz
    g = np.array([0.0, 0.0, -1.0])
    t0 = qw*g[0]+qy*g[2]-qz*g[1]; t1 = qw*g[1]+qz*g[0]-qx*g[2]; t2 = qw*g[2]+qx*g[1]-qy*g[0]
    return np.array([qw*t0+t1*qz-t2*qy-g[0], qw*t1+t2*qx-t0*qz-g[1], qw*t2+t0*qy-t1*qx-g[2]], dtype=np.float32)

# Build encoder input (joint29 mode): feed robot's own state as the "reference"
def build_encoder_input(robot, reference_jp_delta, reference_jv_delta):
    """Build 1762-dim encoder input using FIXED standing reference (not current state)."""
    jp = get_sonic_jp(robot)
    jv = get_sonic_jv(robot)
    jp_delta = jp - SONIC_DEFAULT  # current state delta

    # Use FIXED standing reference as target (all zeros = default pose)
    motion_jp_window = np.tile(reference_jp_delta, (STEP5_FRAMES, 1))  # (10, 29)
    motion_jv_window = np.tile(reference_jv_delta, (STEP5_FRAMES, 1))  # (10, 29)

    # Anchor orientation (root orientation relative to world)
    base_quat = robot.data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)  # wxyz
    # Convert to rotation 6D representation: use first two columns of rotation matrix
    qw, qx, qy, qz = base_quat
    R00 = 1-2*(qy*qy+qz*qz); R01 = 2*(qx*qy-qz*qw); R02 = 2*(qx*qz+qy*qw)
    R10 = 2*(qx*qy+qz*qw); R11 = 1-2*(qx*qx+qz*qz); R12 = 2*(qy*qz-qx*qw)
    R20 = 2*(qx*qz-qy*qw); R21 = 2*(qy*qz+qx*qw); R22 = 1-2*(qx*qx+qy*qy)
    anchor_6d = np.array([R00, R01, R02, R10, R11, R12], dtype=np.float32)

    root_z = robot.data.root_state_w[0, 2].cpu().numpy().astype(np.float32)

    # Build encoder input (joint29 mode: encoder_mode=[0,0,0,0])
    enc_mode = np.zeros(4, dtype=np.float32)

    enc_input = np.concatenate([
        enc_mode,                              # 4
        motion_jp_window.reshape(-1),          # 290
        motion_jv_window.reshape(-1),          # 290
        np.tile(root_z, (STEP5_FRAMES,)).reshape(-1),  # 10
        np.array([root_z], dtype=np.float32),           # 1
        anchor_6d,                                      # 6
        np.tile(anchor_6d, (STEP5_FRAMES,)).reshape(-1), # 60
        np.zeros(120, dtype=np.float32),        # lowerbody pos (zeros=no VR)
        np.zeros(120, dtype=np.float32),        # lowerbody vel (zeros=no VR)
        np.zeros(9, dtype=np.float32),          # vr 3pt pos
        np.zeros(12, dtype=np.float32),         # vr 3pt orn
        np.zeros(720, dtype=np.float32),        # SMPL joints (zeros in joint29 mode)
        np.zeros(60, dtype=np.float32),         # SMPL anchor (zeros in joint29 mode)
        np.zeros(60, dtype=np.float32),         # wrist vel (zeros in joint29 mode)
    ])[np.newaxis].astype(np.float32)  # (1, 1762)

    return enc_input

def build_decoder_input(latent, robot, ang_vel_hist, jp_hist, jv_hist, act_hist, grav_hist):
    """Build 994-dim decoder input from latent + robot state history."""
    jp = get_sonic_jp(robot)
    jv = get_sonic_jv(robot)
    ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy().astype(np.float32)
    base_quat = robot.data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)
    gdir = gravity_dir(base_quat)
    jp_delta = jp - SONIC_DEFAULT

    # Update histories
    ang_vel_hist = np.roll(ang_vel_hist, -1, axis=0); ang_vel_hist[-1] = ang_vel
    jp_hist = np.roll(jp_hist, -1, axis=0); jp_hist[-1] = jp_delta
    jv_hist = np.roll(jv_hist, -1, axis=0); jv_hist[-1] = jv
    grav_hist = np.roll(grav_hist, -1, axis=0); grav_hist[-1] = gdir
    # act_hist updated after decoder output

    dec_obs = np.concatenate([
        latent.flatten()[:64],
        ang_vel_hist.ravel(),
        jp_hist.ravel(),
        jv_hist.ravel(),
        act_hist.ravel(),
        grav_hist.ravel(),
    ])[np.newaxis].astype(np.float32)
    return dec_obs, ang_vel_hist, jp_hist, jv_hist, grav_hist

# Initialize histories
init_jp = get_sonic_jp(robot)
init_jv = get_sonic_jv(robot)
av0 = robot.data.root_ang_vel_b[0].cpu().numpy().astype(np.float32)
bq0 = robot.data.root_state_w[0, 3:7].cpu().numpy().astype(np.float32)

ang_vel_hist = np.tile(av0, (HIST_LEN, 1))
jp_hist = np.tile(init_jp - SONIC_DEFAULT, (HIST_LEN, 1))
jv_hist = np.tile(init_jv, (HIST_LEN, 1))
act_hist = np.zeros((HIST_LEN, 29), dtype=np.float32)
grav_hist = np.tile(gravity_dir(bq0), (HIST_LEN, 1))

# Standing reference = SONIC default pose (delta = zeros)
REF_JP_DELTA = np.zeros(29, dtype=np.float32)
REF_JV_DELTA = np.zeros(29, dtype=np.float32)

# First latent from encoder
enc_in = build_encoder_input(robot, REF_JP_DELTA, REF_JV_DELTA)
latent = encoder.run(None, {enc_in_name: enc_in})[0]
flush(f"[sonic3] initial latent: shape={latent.shape} range=[{latent.min():.4f},{latent.max():.4f}] sum={np.abs(latent).sum():.4f}")
if args.run_decoder_without_input:
    flush("[sonic3] no live input: synthetic fixed-reference decoder experiment enabled")
else:
    flush("[sonic3] no live input: holding SONIC_DEFAULT; pass --run_decoder_without_input to opt into synthetic inference")

# Video recording
video_path = str(Path(PROJECT_ROOT) / "test_videos" / "real_scene_lab_sonic3.mp4")
Path(video_path).parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = None
frame_count = 0

t_loop = time.time()
for step in range(args.num_steps):
    if args.run_decoder_without_input:
        # This is a diagnostic-only mode. The fixed reference is not live
        # teleoperation input, so it must never be the default behavior.
        enc_in = build_encoder_input(robot, REF_JP_DELTA, REF_JV_DELTA)
        latent = encoder.run(None, {enc_in_name: enc_in})[0]
        dec_obs, ang_vel_hist, jp_hist, jv_hist, grav_hist = \
            build_decoder_input(latent, robot, ang_vel_hist, jp_hist, jv_hist, act_hist, grav_hist)
        raw = decoder.run(None, {dec_in_name: dec_obs})[0].flatten()[:29].astype(np.float32)
        target_body = raw * G1_ACTION_SCALE + SONIC_DEFAULT
    else:
        # No valid live command: keep issuing the absolute SONIC standing pose.
        raw = np.zeros(29, dtype=np.float32)
        target_body = SONIC_DEFAULT.copy()

    # Update action history
    act_hist = np.roll(act_hist, -1, axis=0); act_hist[-1] = raw

    action = sonic_target_to_env_action(target_body)

    obs, _, _, _, _ = env.step(action)

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
        except Exception: pass

    if (step+1) % 20 == 0 or step == 0:
        rp = robot.data.root_state_w[0, :3].cpu().numpy()
        flush(f"  step {step+1:3d}: root=[{rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f}] frames={frame_count} raw=[{raw.min():.3f},{raw.max():.3f}]")

if vid_writer: vid_writer.release()
flush(f"[sonic3] {args.num_steps} steps in {time.time()-t_loop:.1f}s, video: {video_path} ({frame_count}f)")
env.close()
simulation_app.close()
flush("[sonic3] DONE")
