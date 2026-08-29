"""Pure validation helpers for SONIC provider execution identity."""

from __future__ import annotations

import math
import numbers
import re
from typing import Any

import numpy as np

SEMANTIC_ACTION_DIM = 40
SONIC_BODY_ACTION_DIM = 29
HAND_INDICES = (38, 39)
HAND_BINARY_THRESHOLD = 0.5


def validate_finite_float32_vector(
    value: Any,
    *,
    expected_dim: int,
    field: str,
) -> np.ndarray:
    """Return a private float32 copy or reject malformed/non-finite input."""
    raw = np.asarray(value)
    if raw.shape != (expected_dim,):
        raise ValueError(
            f"{field} must have shape ({expected_dim},), got {raw.shape}."
        )
    try:
        result = raw.astype(np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must contain numeric values.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must contain only finite values.")
    return np.ascontiguousarray(result)


def validate_provider_semantic40(value: Any, *, field: str) -> np.ndarray:
    return validate_finite_float32_vector(
        value,
        expected_dim=SEMANTIC_ACTION_DIM,
        field=field,
    )


def validate_sonic_body_action(value: Any, *, field: str) -> np.ndarray:
    return validate_finite_float32_vector(
        value,
        expected_dim=SONIC_BODY_ACTION_DIM,
        field=field,
    )


def validate_hand_binary_threshold(
    value: Any,
    *,
    require_provider_identity: bool,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("lerobot_gripper_threshold must be a finite number.")
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("lerobot_gripper_threshold must be a finite number.") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "lerobot_gripper_threshold must be finite and within [0, 1], "
            f"got {value!r}."
        )
    if require_provider_identity and threshold != HAND_BINARY_THRESHOLD:
        raise ValueError(
            "SONIC semantic40 VLA execution requires lerobot_gripper_threshold "
            f"== {HAND_BINARY_THRESHOLD}, got {threshold}."
        )
    return threshold


def resolve_binary_hand_states(
    value: Any,
    *,
    threshold: float,
    field: str,
) -> tuple[bool, bool]:
    hand = validate_finite_float32_vector(value, expected_dim=2, field=field)
    validated_threshold = validate_hand_binary_threshold(
        threshold,
        require_provider_identity=False,
    )
    return (
        bool(hand[0] >= validated_threshold),
        bool(hand[1] >= validated_threshold),
    )


def canonicalize_provider_semantic40(
    value: Any,
    *,
    left_closed: bool,
    right_closed: bool,
    field: str,
) -> np.ndarray:
    """Bind semantic40 hand channels to the binary targets actually materialized."""
    action = validate_provider_semantic40(value, field=field)
    action[HAND_INDICES[0]] = float(bool(left_closed))
    action[HAND_INDICES[1]] = float(bool(right_closed))
    return action


def parse_sonic_output_delay_steps(
    value: Any,
    *,
    require_zero: bool,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("sonic_output_delay_steps must be a non-negative integer.")
    if isinstance(value, numbers.Integral):
        delay_steps = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        delay_steps = int(value.strip())
    else:
        raise ValueError("sonic_output_delay_steps must be a non-negative integer.")
    if delay_steps < 0:
        raise ValueError(
            "sonic_output_delay_steps must be non-negative, "
            f"got {delay_steps}."
        )
    if require_zero and delay_steps != 0:
        raise ValueError(
            "SONIC VLA execution identity requires sonic_output_delay_steps == 0, "
            f"got {delay_steps}."
        )
    return delay_steps


def validate_source_control_step(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Integral
    ):
        raise TypeError(f"{field} must be a non-negative integer, got {value!r}.")
    control_step = int(value)
    if control_step < 0:
        raise ValueError(f"{field} must be non-negative, got {control_step}.")
    return control_step
