# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Drink101 bottle cap twist/break monitoring.

The drink101 scene (move_real_scene_drink_inspire_wholedoby) connects the cap
rigid body to the bottle body with a breakable revolute joint (axis = bottle
z-axis, limits 0..2*pi).  This module tracks the relative rotation of the cap
about the body z-axis and reports the cap state:

  - "sealed"  : joint intact, relative rotation < 2*pi
  - "twisting": relative rotation increasing
  - "armed"   : relative rotation reached ~2*pi (one full turn) - pull to break
  - "opened"  : cap separated from the body (joint broken / cap lifted)

It is hardware-agnostic: the same code is used by the headless physics test
(tools/test_drink101_twist_break.py, which drives the cap with external
forces/torques) and by live teleop (called from sim_main with the same env).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

DRINK_ARMED_ANGLE = 6.283185307179586  # 2*pi, one full turn
DRINK_OPENED_LIFT_M = 0.03  # cap lifted this far above body => opened


def _normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_conj_wxyz(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul_wxyz(a, b):
    """Hamilton product, wxyz ordering."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _rel_z_angle(body_quat_w, cap_quat_w) -> float:
    """Angle (rad) of cap rotation about the body z-axis.

    Joint axis is the bottle z-axis; with both bodies starting at identity
    orientation the relative rotation's z-component is the cap twist angle.
    """
    body_q = _normalize_quat_wxyz(body_quat_w)
    cap_q = _normalize_quat_wxyz(cap_quat_w)
    rel = _quat_mul_wxyz(_quat_conj_wxyz(body_q), cap_q)
    # z-component angle from the relative quaternion (rotation about body z)
    w, x, y, z = rel
    # axis = (0,0,1) -> sin(theta/2) = z, cos(theta/2) = w
    return 2.0 * float(np.arctan2(z, w))


def _unwrap_angle(prev_abs, new_abs) -> float:
    """Accumulate signed total rotation across 2*pi wraps."""
    delta = new_abs - prev_abs
    while delta > np.pi:
        delta -= 2.0 * np.pi
    while delta < -np.pi:
        delta += 2.0 * np.pi
    return prev_abs + delta


def _init_state(env):
    if not hasattr(env, "_drink_twist_state"):
        env._drink_twist_state = {
            "state": "sealed",
            "total_angle": 0.0,
            "last_angle": 0.0,
            "body_pos_at_armed": None,
            "cap_pos_at_armed": None,
            "log_counter": 0,
        }
    return env._drink_twist_state


def update_drink_state(env, *, log_every: int = 25):
    """Update cap twist/break state. Safe to call every control step.

    Returns the state dict; prints on state transitions and periodically.
    """
    st = _init_state(env)
    st["log_counter"] += 1
    try:
        body = env.scene["drink_body"].data
        cap = env.scene["drink_cap"].data
    except Exception:
        return st

    body_pos = body.root_pos_w[0].detach().cpu().numpy().reshape(3)
    cap_pos = cap.root_pos_w[0].detach().cpu().numpy().reshape(3)
    body_quat = body.root_quat_w[0].detach().cpu().numpy().reshape(4)
    cap_quat = cap.root_quat_w[0].detach().cpu().numpy().reshape(4)

    angle_abs = _rel_z_angle(body_quat, cap_quat)
    st["last_angle"] = _unwrap_angle(st["last_angle"], angle_abs)
    total = st["last_angle"]
    lift = float(cap_pos[2] - body_pos[2])

    prev_state = st["state"]
    if prev_state == "opened":
        pass
    elif total >= DRINK_ARMED_ANGLE:
        if st["body_pos_at_armed"] is None:
            st["body_pos_at_armed"] = body_pos.copy()
            st["cap_pos_at_armed"] = cap_pos.copy()
        st["state"] = "armed"
        if (cap_pos[2] - st["cap_pos_at_armed"][2]) > DRINK_OPENED_LIFT_M:
            st["state"] = "opened"
    elif total > 0.1:
        st["state"] = "twisting"
    else:
        st["state"] = "sealed"

    if st["state"] != prev_state:
        print(
            f"[drink] cap state: {prev_state} -> {st['state']} "
            f"(total_angle={total:.2f} rad, lift={lift:.3f} m)"
        )
    elif st["log_counter"] % log_every == 0:
        print(
            f"[drink] cap: state={st['state']} total_angle={total:.2f} "
            f"lift={lift:.3f} m"
        )
    return st


def drink_is_opened(env) -> bool:
    st = _init_state(env)
    return st["state"] == "opened"


def drink_is_armed(env) -> bool:
    st = _init_state(env)
    return st["state"] in ("armed", "opened")
