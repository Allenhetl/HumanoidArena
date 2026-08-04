"""Frozen RH56E2 / Inspire FTP action-space mapping.

Training and hardware interfaces MUST use ``a_hw_6``.
``q_sim_12`` is kinematic state only and is derived via mimic coupling.
All lookups are name-based; never rely on anonymous array index order from
third-party optimizers without remapping through these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

A_HW_6_NAMES: tuple[str, ...] = (
    "pinky_flex",
    "ring_flex",
    "middle_flex",
    "index_flex",
    "thumb_flex",
    "thumb_rotation",
)

# Official Isaac Lab / Unitree Inspire FTP drive joints (per hand).
ACTUATOR_JOINT_SUFFIX: dict[str, str] = {
    "pinky_flex": "pinky_proximal_joint",
    "ring_flex": "ring_proximal_joint",
    "middle_flex": "middle_proximal_joint",
    "index_flex": "index_proximal_joint",
    "thumb_flex": "thumb_proximal_pitch_joint",
    "thumb_rotation": "thumb_proximal_yaw_joint",
}

# Mimic coefficients from retarget_inspire_white_*_hand.urdf
COUPLING = {
    "finger_intermediate_multiplier": 1.0,
    "thumb_intermediate_multiplier": 1.6,
    "thumb_distal_multiplier": 2.4,
}

JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "pinky_proximal_joint": (0.0, 1.7),
    "ring_proximal_joint": (0.0, 1.7),
    "middle_proximal_joint": (0.0, 1.7),
    "index_proximal_joint": (0.0, 1.7),
    "thumb_proximal_pitch_joint": (0.0, 0.5),
    "thumb_proximal_yaw_joint": (-0.1, 1.3),
}


@dataclass(frozen=True)
class HandSide:
    prefix: str  # "L_" or "R_"

    def joint(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"


RIGHT = HandSide("R_")
LEFT = HandSide("L_")


def q_sim_joint_names(side: HandSide = RIGHT) -> list[str]:
    """Canonical 12-DoF sim joint order for one hand."""
    return [
        side.joint("pinky_proximal_joint"),
        side.joint("ring_proximal_joint"),
        side.joint("middle_proximal_joint"),
        side.joint("index_proximal_joint"),
        side.joint("thumb_proximal_pitch_joint"),
        side.joint("thumb_proximal_yaw_joint"),
        side.joint("pinky_intermediate_joint"),
        side.joint("ring_intermediate_joint"),
        side.joint("middle_intermediate_joint"),
        side.joint("index_intermediate_joint"),
        side.joint("thumb_intermediate_joint"),
        side.joint("thumb_distal_joint"),
    ]


def actuator_joint_names(side: HandSide = RIGHT) -> list[str]:
    return [side.joint(ACTUATOR_JOINT_SUFFIX[n]) for n in A_HW_6_NAMES]


def normalize_hw_command(
    command: Sequence[float],
    *,
    source: str = "angle_set",
    clip: bool = True,
) -> np.ndarray:
    """Convert a hardware-facing command into normalized [0, 1] flex/rotation.

    Args:
        command: length-6 array in ``A_HW_6_NAMES`` order.
        source: ``angle_set`` (0-1000), ``normalized`` (0-1), or ``rad`` (sim rad).
        clip: whether to clip to valid range before normalizing.
    """
    arr = np.asarray(command, dtype=np.float64).reshape(6)
    if source == "normalized":
        out = arr
        if clip:
            out = np.clip(out, 0.0, 1.0)
        return out
    if source == "angle_set":
        out = arr / 1000.0
        if clip:
            out = np.clip(out, 0.0, 1.0)
        return out
    if source == "rad":
        out = np.zeros(6, dtype=np.float64)
        for i, name in enumerate(A_HW_6_NAMES):
            suffix = ACTUATOR_JOINT_SUFFIX[name]
            lo, hi = JOINT_LIMITS_RAD[suffix]
            span = max(hi - lo, 1e-8)
            val = (arr[i] - lo) / span
            out[i] = float(np.clip(val, 0.0, 1.0) if clip else val)
        return out
    raise ValueError(f"Unknown source unit: {source}")


def expand_a_hw_to_q_sim(
    a_hw: Sequence[float],
    *,
    side: HandSide = RIGHT,
    unit: str = "normalized",
) -> dict[str, float]:
    """Expand 6D hardware action to named 12D simulation joint positions (rad)."""
    norm = normalize_hw_command(a_hw, source=unit, clip=True)
    actuators: dict[str, float] = {}
    for i, name in enumerate(A_HW_6_NAMES):
        suffix = ACTUATOR_JOINT_SUFFIX[name]
        lo, hi = JOINT_LIMITS_RAD[suffix]
        actuators[suffix] = lo + norm[i] * (hi - lo)

    q: dict[str, float] = {}
    for suffix, value in actuators.items():
        q[side.joint(suffix)] = value

    # Finger proximal -> intermediate mimic
    for finger in ("pinky", "ring", "middle", "index"):
        proximal = actuators[f"{finger}_proximal_joint"]
        q[side.joint(f"{finger}_intermediate_joint")] = (
            proximal * COUPLING["finger_intermediate_multiplier"]
        )

    thumb_pitch = actuators["thumb_proximal_pitch_joint"]
    q[side.joint("thumb_intermediate_joint")] = thumb_pitch * COUPLING["thumb_intermediate_multiplier"]
    q[side.joint("thumb_distal_joint")] = thumb_pitch * COUPLING["thumb_distal_multiplier"]
    return q


def fold_q_sim_to_a_hw(
    q_sim: dict[str, float] | Sequence[float],
    *,
    side: HandSide = RIGHT,
    joint_order: Iterable[str] | None = None,
    out_unit: str = "normalized",
) -> np.ndarray:
    """Project simulation joints back to 6D hardware action."""
    if not isinstance(q_sim, dict):
        names = list(joint_order) if joint_order is not None else q_sim_joint_names(side)
        q_sim = {n: float(v) for n, v in zip(names, q_sim, strict=True)}

    rad = np.zeros(6, dtype=np.float64)
    for i, name in enumerate(A_HW_6_NAMES):
        joint = side.joint(ACTUATOR_JOINT_SUFFIX[name])
        if joint not in q_sim:
            raise KeyError(f"Missing joint '{joint}' while folding q_sim -> a_hw")
        rad[i] = float(q_sim[joint])

    if out_unit == "rad":
        return rad
    if out_unit == "normalized":
        return normalize_hw_command(rad, source="rad", clip=True)
    if out_unit == "angle_set":
        return normalize_hw_command(rad, source="rad", clip=True) * 1000.0
    raise ValueError(f"Unknown out_unit: {out_unit}")


def remap_named_joints(
    values: dict[str, float],
    target_names: Sequence[str],
    *,
    default: float = 0.0,
) -> np.ndarray:
    """Build a dense vector for ``target_names`` from a name->value map."""
    return np.asarray([values.get(n, default) for n in target_names], dtype=np.float64)
