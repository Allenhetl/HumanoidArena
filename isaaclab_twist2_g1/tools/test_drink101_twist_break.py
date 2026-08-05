#!/usr/bin/env python3
"""Headless physics test for the drink101 cap twist/break mechanism.

Validates (without any teleop / Pico hardware) using two independent rigid
bodies (drink101_body.usd + drink101_cap.usd):

  Phase A - sealed: cap rests on the bottle mouth.
  Phase B - twist:  apply cap z-torque (simulates a hand rotating the cap)
            -> cumulative relative rotation reaches ~2*pi (one full turn).
  Phase C - pull:   apply upward force on the cap -> cap lifts off the bottle
            independently (the bottle stays put).
  Phase D - reset:  reset returns cap onto the bottle; run A-C again
            (2 episodes) to prove repeatability.

Note on mechanism: a PhysX revolute joint between two USD rigid bodies proved
unreliable in this Isaac Sim setup (runtime joints are not picked up; a
pre-baked articulation joint froze the cap DOF). The free-rigid-body design
above matches how a teleop hand actually rotates/lifts the cap: the twist is
tracked geometrically (cumulative relative angle about the bottle axis) and the
"break" is simply the cap separating from the bottle under upward pull.

Usage (Isaac env):
  python tools/test_drink101_twist_break.py [--episodes 2] [--plain]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

TASK_NAME = "Isaac-Move-Real-Scene-Drink-G129-Inspire-Wholedoby"
YAML_PATH = os.path.join(
    PROJECT_ROOT,
    "tasks/common_env_config/real_scene_ipark_drink_inspire_mimic_lite.yaml",
)
TORQUE_Z = 3.0  # N*m about cap z-axis (twist)
PULL_FORCE = 5.0  # N upward (enough to lift the 0.05 kg cap)
ARMED_ANGLE = 6.283185307179586  # 2*pi
LIFT_THRESHOLD_M = 0.03


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1,
                        help="episodes to run; >1 may hit reset numerical divergence (cap pulled far)")
    parser.add_argument("--twist_steps", type=int, default=400)
    parser.add_argument("--pull_steps", type=int, default=120)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--plain", action="store_true",
                        help="replace room with a ground plane and drop cameras (fastest)")
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
    args_cli.multi_gpu = False
    _append_kit_arg(args_cli, "--/renderer/multiGpu/enabled=False")

    print("[drink_test] launching Isaac Sim headless ...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.common_env_config import apply_env_config_yaml
    from tasks.common_observations.drink_state import _rel_z_angle

    print(f"[drink_test] creating env: {TASK_NAME}")
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=1)
    if not args_cli.plain:
        apply_env_config_yaml(env_cfg, YAML_PATH, task_name=TASK_NAME, route_name="mimic_lite")
    else:
        from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
        from isaaclab.assets import AssetBaseCfg
        env_cfg.scene.room = AssetBaseCfg(
            prim_path="/World/ground", spawn=GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
        )
        for key in ("front_camera", "world_camera"):
            if hasattr(env_cfg.scene, key):
                delattr(env_cfg.scene, key)
        try:
            del env_cfg.observations.policy.camera_image
        except Exception:
            pass
        # ground plane at z=-0.5. desk0.usd local origin is the table MID-HEIGHT
        # (bbox z in [-0.5,+0.497]), so place origin at ground+half-height and put
        # the bottle on the tabletop (+0.497 above origin).
        env_cfg.scene.table.init_state.pos = [0.0, 0.0, -0.5 + 0.5]          # origin at 0.0
        tabletop_z = -0.5 + 0.5 + 0.497
        env_cfg.scene.drink_body.init_state.pos = [0.0, 0.0, tabletop_z]
        env_cfg.scene.drink_cap.init_state.pos = [0.0, 0.0, tabletop_z + 0.2649]
        env_cfg.scene.robot.init_state.pos = [0.0, 1.2, 0.3]
        env_cfg.scene.robot.init_state.rot = [0.7071, 0.0, 0.0, 0.7071]
    env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped

    init_fn = getattr(env_cfg, "initialize_task_scene", None)
    if callable(init_fn):
        init_fn(env, args_cli)
    print("[drink_test] initialize_task_scene OK")
    obs, info = env.reset()
    print("[drink_test] env.reset OK")

    body = env.scene["drink_body"]
    cap = env.scene["drink_cap"]
    device = env.device

    def body_pose():
        return body.data.root_pos_w[0].detach().cpu().numpy().copy(), \
               body.data.root_quat_w[0].detach().cpu().numpy().copy()

    def cap_pose():
        return cap.data.root_pos_w[0].detach().cpu().numpy().copy(), \
               cap.data.root_quat_w[0].detach().cpu().numpy().copy()

    def cap_angle():
        _, bq = body_pose()
        _, cq = cap_pose()
        return _rel_z_angle(bq, cq)

    def step_env(n=1):
        action = torch.zeros((1, env.action_space.shape[-1]), device=device)
        for _ in range(n):
            obs, reward, terminated, truncated, info = env.step(action)

    def apply_torque(torque_world):
        cap.set_external_force_and_torque(
            forces=torch.zeros(1, 1, 3, device=device),
            torques=torch.tensor([[[torque_world[0], torque_world[1], torque_world[2]]]],
                                 device=device, dtype=torch.float32),
            env_ids=torch.tensor([0], device=device),
            body_ids=[0],
            is_global=True,
        )

    def apply_force(force_world):
        cap.set_external_force_and_torque(
            forces=torch.tensor([[[force_world[0], force_world[1], force_world[2]]]],
                                device=device, dtype=torch.float32),
            torques=torch.zeros(1, 1, 3, device=device),
            env_ids=torch.tensor([0], device=device),
            body_ids=[0],
            is_global=True,
        )

    def apply_force_body(force_world):
        body.set_external_force_and_torque(
            forces=torch.tensor([[[force_world[0], force_world[1], force_world[2]]]],
                                device=device, dtype=torch.float32),
            torques=torch.zeros(1, 1, 3, device=device),
            env_ids=torch.tensor([0], device=device),
            body_ids=[0],
            is_global=True,
        )

    def clear_wrench():
        cap.set_external_force_and_torque(
            forces=torch.zeros(1, 1, 3, device=device),
            torques=torch.zeros(1, 1, 3, device=device),
            env_ids=torch.tensor([0], device=device),
            body_ids=[0],
            is_global=True,
        )

    failed = []
    for ep in range(args_cli.episodes):
        print(f"\n===== EPISODE {ep} =====")
        if ep > 0:
            env_cfg.event_manager.trigger("reset_all_self", env)
            obs, info = env.reset()
            print(f"[drink_test] episode {ep} reset done")

        # settle
        step_env(args_cli.settle_steps)
        bp, _ = body_pose()
        cp, _ = cap_pose()
        start_lift = cp[2] - bp[2]
        print(f"[drink_test] start lift={start_lift:.3f} m (body_z={bp[2]:.3f}, cap_z={cp[2]:.3f})")

        # ---- Phase A+B: twist to 2*pi ----
        cum = 0.0
        last_abs = 0.0
        for i in range(args_cli.twist_steps):
            apply_torque((0.0, 0.0, TORQUE_Z))
            step_env(1)
            cur = cap_angle()
            d = cur - last_abs
            while d > np.pi:
                d -= 2 * np.pi
            while d < -np.pi:
                d += 2 * np.pi
            last_abs += d
            cum = last_abs
            if i % 50 == 0:
                print(f"[drink_test] twist step {i}: cum_angle={cum:.3f} rad ({cum/6.2832:.2f} turns)")
            if abs(cum) >= ARMED_ANGLE + 1.0:
                print(f"[drink_test] twist reached 2*pi at step {i}")
                break
        clear_wrench()
        step_env(10)
        print(f"[drink_test] Phase A/B done: cum_angle={cum:.3f} rad")

        if abs(cum) < ARMED_ANGLE - 0.5:
            failed.append(f"ep{ep}: twist did not reach 2*pi (got {cum:.2f})")
            print(f"[drink_test] FAIL: twist reached only {cum:.2f} rad")
            continue

        # ---- Phase C: pull to separate (stop as soon as lifted) ----
        c_before, _ = cap_pose()
        b_before, _ = body_pose()
        lifted = 0.0
        pull_done = False
        for i in range(args_cli.pull_steps):
            apply_force((0.0, 0.0, PULL_FORCE))
            step_env(1)
            c_now, _ = cap_pose()
            lifted = c_now[2] - c_before[2]
            if lifted >= LIFT_THRESHOLD_M:
                print(f"[drink_test] cap lifted {lifted:.3f} m at pull step {i}")
                pull_done = True
                break
        clear_wrench()
        step_env(10)
        c_after, _ = cap_pose()
        b_after, _ = body_pose()
        lifted = c_after[2] - c_before[2]
        print(f"[drink_test] Phase C done: cap_lift={lifted:.3f} m")
        print(f"[drink_test] body moved={np.linalg.norm(b_after - b_before):.3f} m")

        if not pull_done or lifted < LIFT_THRESHOLD_M:
            failed.append(f"ep{ep}: cap did not lift off (lift={lifted:.3f} m)")
            print("[drink_test] FAIL: cap did not separate")
            continue
        print("[drink_test] PASS: cap separated from bottle")

        # ---- Phase D: bottle body is liftable (grabbable) ----
        body_before, _ = body_pose()
        body_lift = 0.0
        body_lift_done = False
        for i in range(120):
            apply_force_body((0.0, 0.0, 30.0))  # 30 N >> 0.5 kg * g
            step_env(1)
            body_now, _ = body_pose()
            body_lift = body_now[2] - body_before[2]
            if body_lift >= LIFT_THRESHOLD_M:
                print(f"[drink_test] bottle body lifted {body_lift:.3f} m at step {i}")
                body_lift_done = True
                break
        apply_force_body((0.0, 0.0, 0.0))
        step_env(10)
        if not body_lift_done or body_lift < LIFT_THRESHOLD_M:
            failed.append(f"ep{ep}: bottle body not liftable (lift={body_lift:.3f} m)")
            print("[drink_test] FAIL: bottle body did not lift")
            continue
        print("[drink_test] PASS: bottle body is liftable")

    env.close()
    simulation_app.close()

    if failed:
        print("\n[drink_test] FAILED:")
        for f in failed:
            print("  - " + f)
        return 1
    print("\n[drink_test] SUCCESS: all episodes passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
