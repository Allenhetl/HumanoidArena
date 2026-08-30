"""Explicit recovery-state hooks for the semantic40 SONIC VLA provider."""

from __future__ import annotations

import copy
import math
from collections import deque
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch

SONIC_PROVIDER_RECOVERY_SCHEMA_VERSION = 1

_MODE_FIELDS = (
    "_use_lerobot_vla",
    "_use_vla_latent64",
    "_vla_action_format",
    "_replay_enabled",
    "_sonic_output_delay_steps",
)
_ARRAY_FIELDS = (
    "_smpl_joints_buf",
    "_smpl_pose_buf",
    "_body_rot6d_buf",
    "_ref_smpl_joints_window",
    "_ref_body_quat_window",
    "_ref_joint_pos_window",
    "_robot_joint_pos_hist",
    "_robot_joint_vel_hist",
    "_motion_joint_pos_hist",
    "_motion_joint_vel_hist",
    "_motion_root_z_hist",
    "_motion_anchor_rot6d_hist",
    "_ang_vel_hist",
    "_grav_dir_hist",
    "_last_action_hist",
    "_left_hand_target",
    "_right_hand_target",
    "_vr_3pt_position",
    "_vr_3pt_orientation",
    "_anchor_init_base_quat_wxyz",
    "_anchor_init_ref_quat_wxyz",
    "_anchor_heading_align_quat_wxyz",
    "_tracking_target_buffer",
    "_latest_canonical_action_raw",
    "_latest_canonical_action",
    "_sonic_last_executed_target",
    "_latest_executed_canonical_action_raw",
    "_latest_executed_canonical_action",
    "_latest_encoder_input",
    "_latest_smpl_joint_window",
    "_latest_anchor_window",
    "_latest_wrist_window",
    "_latest_decoder_obs",
    "_latest_decoder_raw_action",
    "_latest_decoder_target",
    "_latest_decoder_body_effort",
    "_latest_aligned_body_quat_wxyz",
    "_latest_consumed_anchor_rot6d",
)
_OPTIONAL_ARRAY_FIELDS = (
    "_vla_initial_robot_quat_wxyz",
    "_vla_prev_root_rot6d_action",
    "_latest_vla_action",
)
_BOOL_FIELDS = (
    "_ref_window_valid",
    "_left_hand_binary_state",
    "_right_hand_binary_state",
    "_smpl_data_valid",
    "_anchor_heading_initialized",
    "_anchor_use_heading_align",
    "_latest_consumed_new_this_step",
    "_effort_mode_runtime_configured",
    "_position_mode_runtime_configured",
)
_INT_FIELDS = (
    "_vla_semantic_history_fill",
    "_frame_count",
    "_smpl_history_fill",
    "_stream_window_start",
    "_stream_current_frame",
    "_stream_frame_step",
    "_latest_frame_index",
    "_latest_consumed_control_step",
    "_latest_executed_source_frame_index",
    "_latest_executed_source_control_step",
    "_raw_input_frame_index",
    "_last_raw_frame_index",
)
_OPTIONAL_INT_FIELDS = ("_stream_playback_frame_idx",)
_FLOAT_FIELDS = (
    "_latest_timestamp_realtime",
    "_latest_timestamp_monotonic",
    "_latest_heading_increment",
    "_latest_executed_source_timestamp_realtime",
    "_latest_executed_source_timestamp_monotonic",
    "_raw_input_timestamp_realtime",
    "_raw_input_timestamp_monotonic",
)
_DYNAMIC_FIELDS = (
    "_stream_ref_frames",
    "_stream_ref_indices",
    "_raw_pose_payload",
    "_latest_pose_payload",
    "_latest_human_smplx_frame",
    "_latent",
)
_VLA_RUNTIME_FIELDS = (
    "_body_xy_world",
    "_body_z_world",
    "_prev_target_root_quat_wxyz",
    "_episode_ref_to_world_heading_quat_wxyz",
    "_prev_root_quat_wxyz",
    "_prev_action_rel_quat_wxyz",
    "_prev_joint_pos_canonical_29",
    "_last_selected_root_rot6d_layout",
)
_BUNDLE_KEYS = {
    "body_action_29dof",
    "canonical_action_raw",
    "canonical_action_aligned",
    "source_frame_index",
    "source_timestamp_realtime",
    "source_timestamp_monotonic",
    "source_control_step",
}


def _type_identity(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _require_attribute(owner: Any, name: str) -> Any:
    if not hasattr(owner, name):
        raise ValueError(f"SONIC recovery field is unavailable: {name}")
    return getattr(owner, name)


def _clone(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _finite(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.inexact) or bool(
            np.isfinite(value).all()
        )
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(
            torch.isfinite(value).all()
        )
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, Real):
        return math.isfinite(float(value))
    return True


def _mode_identity(provider: Any) -> dict[str, Any]:
    identity = {
        name: _clone(_require_attribute(provider, name)) for name in _MODE_FIELDS
    }
    if identity != {
        "_use_lerobot_vla": True,
        "_use_vla_latent64": False,
        "_vla_action_format": "semantic_v3",
        "_replay_enabled": False,
        "_sonic_output_delay_steps": 0,
    }:
        raise ValueError(
            "SONIC exact recovery supports only live semantic40 VLA with zero output delay"
        )
    return identity


def _validate_array(name: str, saved: Any, current: Any) -> None:
    if not isinstance(saved, np.ndarray) or not isinstance(current, np.ndarray):
        raise TypeError(f"SONIC recovery {name} must be a NumPy array")
    if (
        saved.shape != current.shape
        or saved.dtype != current.dtype
        or not _finite(saved)
    ):
        raise ValueError(f"SONIC recovery {name} schema mismatch")


def _validate_optional_array(name: str, saved: Any) -> None:
    expected_shapes = {
        "_vla_initial_robot_quat_wxyz": (4,),
        "_vla_prev_root_rot6d_action": (6,),
        "_latest_vla_action": (40,),
    }
    if saved is None:
        return
    if not isinstance(saved, np.ndarray) or (
        saved.shape != expected_shapes[name]
        or saved.dtype != np.float32
        or not _finite(saved)
    ):
        raise ValueError(f"SONIC recovery {name} schema mismatch")


def _validate_bundle(bundle: Any, *, path: str) -> None:
    if not isinstance(bundle, Mapping) or set(bundle) != _BUNDLE_KEYS:
        raise ValueError(f"SONIC recovery {path} bundle schema mismatch")
    for name, shape in (
        ("body_action_29dof", (29,)),
        ("canonical_action_raw", (40,)),
        ("canonical_action_aligned", (40,)),
    ):
        value = bundle[name]
        if not (
            isinstance(value, np.ndarray)
            and value.shape == shape
            and value.dtype == np.float32
            and _finite(value)
        ):
            raise ValueError(f"SONIC recovery {path}.{name} schema mismatch")
    for name in ("source_frame_index", "source_control_step"):
        if isinstance(bundle[name], bool) or not isinstance(bundle[name], Integral):
            raise TypeError(f"SONIC recovery {path}.{name} schema mismatch")
    for name in ("source_timestamp_realtime", "source_timestamp_monotonic"):
        if not isinstance(bundle[name], Real) or not math.isfinite(float(bundle[name])):
            raise ValueError(f"SONIC recovery {path}.{name} schema mismatch")


def _validate_queue(queue: Any) -> None:
    if not isinstance(queue, tuple) or not 1 <= len(queue) <= 40:
        raise ValueError(
            "SONIC recovery committed_action_queue must contain 1..40 rows"
        )
    for row in queue:
        if not (
            isinstance(row, np.ndarray)
            and row.shape == (40,)
            and row.dtype == np.float32
            and _finite(row)
        ):
            raise ValueError(
                "SONIC recovery committed_action_queue row must be finite float32[40]"
            )


def _validate_vla_runtime(provider: Any, state: Any) -> None:
    runtime = _require_attribute(provider, "_lerobot_vla_runtime")
    if not isinstance(state, Mapping) or set(state) != set(_VLA_RUNTIME_FIELDS):
        raise ValueError("SONIC recovery vla_runtime schema mismatch")
    optional_shapes = {
        "_body_xy_world": (2,),
        "_prev_target_root_quat_wxyz": (4,),
        "_episode_ref_to_world_heading_quat_wxyz": (4,),
        "_prev_root_quat_wxyz": (4,),
        "_prev_action_rel_quat_wxyz": (4,),
        "_prev_joint_pos_canonical_29": (29,),
    }
    for name, shape in optional_shapes.items():
        value = state[name]
        if value is not None and not (
            isinstance(value, np.ndarray)
            and value.shape == shape
            and value.dtype == np.float32
            and _finite(value)
        ):
            raise ValueError(f"SONIC recovery vla_runtime.{name} schema mismatch")
        _require_attribute(runtime, name)
    body_z = state["_body_z_world"]
    if body_z is not None and (
        not isinstance(body_z, Real) or not math.isfinite(float(body_z))
    ):
        raise ValueError("SONIC recovery vla_runtime._body_z_world schema mismatch")
    if state["_last_selected_root_rot6d_layout"] not in {"row", "col"}:
        raise ValueError(
            "SONIC recovery vla_runtime._last_selected_root_rot6d_layout mismatch"
        )


def capture_sonic_recovery_state(provider: Any) -> dict[str, Any]:
    """Capture the complete supported semantic40 provider continuation state."""

    mode = _mode_identity(provider)
    live_queue = tuple(_require_attribute(provider, "_lerobot_action_chunk_queue"))
    _validate_queue(live_queue)
    queue = tuple(row.copy() for row in live_queue)
    fields = {
        name: _clone(_require_attribute(provider, name))
        for name in (
            _ARRAY_FIELDS
            + _OPTIONAL_ARRAY_FIELDS
            + _BOOL_FIELDS
            + _INT_FIELDS
            + _OPTIONAL_INT_FIELDS
            + _FLOAT_FIELDS
            + _DYNAMIC_FIELDS
        )
    }
    runtime = _require_attribute(provider, "_lerobot_vla_runtime")
    state = {
        "schema_version": SONIC_PROVIDER_RECOVERY_SCHEMA_VERSION,
        "provider_type": _type_identity(provider),
        "mode": mode,
        "committed_action_queue": queue,
        "fields": fields,
        "vla_runtime_type": _type_identity(runtime),
        "vla_runtime": {
            name: _clone(_require_attribute(runtime, name))
            for name in _VLA_RUNTIME_FIELDS
        },
        "output_delay_queue": _clone(
            _require_attribute(provider, "_sonic_output_delay_queue")
        ),
        "last_executed_bundle": _clone(
            _require_attribute(provider, "_sonic_last_executed_bundle")
        ),
    }
    validate_sonic_recovery_state(provider, state)
    return state


def sonic_recovery_state_supported(provider: Any) -> bool:
    """Return whether this live provider is at a restorable mid-chunk boundary."""

    try:
        _mode_identity(provider)
        queue = _require_attribute(provider, "_lerobot_action_chunk_queue")
        _validate_queue(tuple(queue))
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def validate_sonic_recovery_state(provider: Any, state: Any) -> None:
    """Validate without mutating provider state."""

    expected_keys = {
        "schema_version",
        "provider_type",
        "mode",
        "committed_action_queue",
        "fields",
        "vla_runtime_type",
        "vla_runtime",
        "output_delay_queue",
        "last_executed_bundle",
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise ValueError("SONIC recovery root schema mismatch")
    if state["schema_version"] != SONIC_PROVIDER_RECOVERY_SCHEMA_VERSION:
        raise ValueError("SONIC recovery schema version mismatch")
    if state["provider_type"] != _type_identity(provider):
        raise ValueError("SONIC recovery provider type mismatch")
    if dict(state["mode"]) != _mode_identity(provider):
        raise ValueError("SONIC recovery mode identity mismatch")
    _validate_queue(state["committed_action_queue"])
    fields = state["fields"]
    expected_fields = set(
        _ARRAY_FIELDS
        + _OPTIONAL_ARRAY_FIELDS
        + _BOOL_FIELDS
        + _INT_FIELDS
        + _OPTIONAL_INT_FIELDS
        + _FLOAT_FIELDS
        + _DYNAMIC_FIELDS
    )
    if not isinstance(fields, Mapping) or set(fields) != expected_fields:
        raise ValueError("SONIC recovery fields schema mismatch")
    for name in _ARRAY_FIELDS:
        _validate_array(name, fields[name], _require_attribute(provider, name))
    for name in _OPTIONAL_ARRAY_FIELDS:
        _validate_optional_array(name, fields[name])
        _require_attribute(provider, name)
    for name in _BOOL_FIELDS:
        if type(fields[name]) is not bool:
            raise ValueError(f"SONIC recovery {name} schema mismatch")
    for name in _INT_FIELDS:
        if type(fields[name]) is not int:
            raise ValueError(f"SONIC recovery {name} schema mismatch")
    for name in _OPTIONAL_INT_FIELDS:
        if fields[name] is not None and type(fields[name]) is not int:
            raise ValueError(f"SONIC recovery {name} schema mismatch")
    for name in _FLOAT_FIELDS:
        if not isinstance(fields[name], Real) or not math.isfinite(float(fields[name])):
            raise ValueError(f"SONIC recovery {name} schema mismatch")
    if not isinstance(fields["_stream_ref_frames"], Mapping) or not isinstance(
        fields["_stream_ref_indices"], list
    ):
        raise TypeError("SONIC recovery streamed-reference schema mismatch")
    if not _finite(fields):
        raise ValueError("SONIC recovery fields contain non-finite values")
    runtime = _require_attribute(provider, "_lerobot_vla_runtime")
    if state["vla_runtime_type"] != _type_identity(runtime):
        raise ValueError("SONIC recovery VLA runtime type mismatch")
    _validate_vla_runtime(provider, state["vla_runtime"])
    delay_queue = state["output_delay_queue"]
    if not isinstance(delay_queue, list) or len(delay_queue) != 0:
        raise ValueError("SONIC recovery output_delay_queue must be empty in VLA mode")
    _validate_bundle(state["last_executed_bundle"], path="last_executed_bundle")


def _restore_value(owner: Any, name: str, value: Any) -> None:
    current = _require_attribute(owner, name)
    if isinstance(current, np.ndarray) and isinstance(value, np.ndarray):
        np.copyto(current, value)
    elif isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
        current.copy_(value)
    else:
        setattr(owner, name, _clone(value))


def restore_sonic_recovery_state(provider: Any, state: Any) -> None:
    """Restore a preflighted semantic40 provider snapshot."""

    validate_sonic_recovery_state(provider, state)
    queue = _require_attribute(provider, "_lerobot_action_chunk_queue")
    if not isinstance(queue, deque):
        raise TypeError("SONIC recovery live committed queue is not a deque")
    queue.clear()
    queue.extend(_clone(state["committed_action_queue"]))
    for name, value in state["fields"].items():
        _restore_value(provider, name, value)
    runtime = _require_attribute(provider, "_lerobot_vla_runtime")
    for name, value in state["vla_runtime"].items():
        _restore_value(runtime, name, value)
    provider._sonic_output_delay_queue = _clone(state["output_delay_queue"])
    provider._sonic_last_executed_bundle = _clone(state["last_executed_bundle"])
