from __future__ import annotations

import zlib
import time
from typing import Any

import numpy as np
import torch


_ROOT_STATE_FIELDS = (
    ("position", slice(0, 3), "_position", np.zeros(3, dtype=np.float32)),
    ("orientation", slice(3, 7), "_orientation", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
    ("linear_velocity", slice(7, 10), "_linear_velocity", np.zeros(3, dtype=np.float32)),
    ("angular_velocity", slice(10, 13), "_angular_velocity", np.zeros(3, dtype=np.float32)),
)
_POSE_RANGE_AXES = {"x": 0, "y": 1, "z": 2}


def get_recordable_env_object_specs(env_cfg: Any) -> list[dict[str, Any]]:
    merged_specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for attr_name in ("recordable_env_objects", "deterministic_object_resets"):
        raw_specs = getattr(env_cfg, attr_name, []) or []
        if not isinstance(raw_specs, list):
            continue
        for raw_spec in raw_specs:
            if not isinstance(raw_spec, dict):
                continue
            scene_keys = raw_spec.get("scene_keys")
            if scene_keys is None:
                scene_keys = [raw_spec.get("scene_key")]
            scene_keys = [str(value) for value in scene_keys if value]
            if not scene_keys:
                continue
            record_name = str(raw_spec.get("record_name") or scene_keys[0])
            if record_name in seen_names:
                continue
            merged_specs.append(
                {
                    "record_name": record_name,
                    "scene_keys": scene_keys,
                    "pose_range": raw_spec.get("pose_range", {}) or {},
                    "zero_velocity_on_reset": bool(raw_spec.get("zero_velocity_on_reset", True)),
                }
            )
            seen_names.add(record_name)

    return merged_specs


def resolve_env_object_scene_key(env, env_cfg: Any, object_name: str) -> str | None:
    for spec in get_recordable_env_object_specs(env_cfg):
        if spec["record_name"] != object_name:
            continue
        for scene_key in spec["scene_keys"]:
            if scene_key in env.scene.keys():
                return scene_key
        return None

    for fallback_key in (object_name,):
        if fallback_key in env.scene.keys():
            return fallback_key
    return None


def collect_recordable_env_object_states(env, env_cfg: Any) -> dict[str, dict[str, np.ndarray] | None]:
    env_state: dict[str, dict[str, np.ndarray] | None] = {}

    for spec in get_recordable_env_object_specs(env_cfg):
        record_name = spec["record_name"]
        scene_key = resolve_env_object_scene_key(env, env_cfg, record_name)
        if scene_key is None:
            env_state[record_name] = None
            continue

        try:
            obj = env.scene[scene_key]
            root_state = obj.data.root_state_w
            env_state[record_name] = {
                field_name: root_state[0, field_slice].detach().cpu().numpy().astype(np.float32).copy()
                for field_name, field_slice, _, _ in _ROOT_STATE_FIELDS
            }
        except Exception:
            env_state[record_name] = None

    return env_state


def add_env_object_frame_arrays(organized: dict[str, Any], data_buffer: list[dict[str, Any]]) -> None:
    if not data_buffer:
        return

    object_names: set[str] = set()
    for frame in data_buffer:
        env_obj = frame.get("env_obj", {})
        if isinstance(env_obj, dict):
            object_names.update(env_obj.keys())

    for object_name in sorted(object_names):
        states = [
            frame.get("env_obj", {}).get(object_name) if isinstance(frame.get("env_obj", {}), dict) else None
            for frame in data_buffer
        ]
        if not any(state is not None for state in states):
            continue
        for field_name, _, suffix, default_value in _ROOT_STATE_FIELDS:
            organized[f"env_obj_{object_name}{suffix}"] = np.array(
                [
                    np.asarray(state[field_name], dtype=np.float32) if state is not None else default_value.copy()
                    for state in states
                ],
                dtype=np.float32,
            )


def add_episode_init_env_object_fields(organized: dict[str, Any], episode_init_env: dict[str, Any] | None) -> None:
    if not isinstance(episode_init_env, dict):
        return
    for object_name, state in episode_init_env.items():
        if state is None:
            continue
        for field_name, _, suffix, default_value in _ROOT_STATE_FIELDS:
            organized[f"episode_init_env_obj_{object_name}{suffix}"] = np.asarray(
                state.get(field_name, default_value),
                dtype=np.float32,
            )


def get_current_episode_object_seed_info(env_cfg: Any) -> dict[str, Any]:
    seed_value = getattr(env_cfg, "_current_episode_object_seed", None)
    seed_source = getattr(env_cfg, "_current_episode_object_seed_source", "")
    return {
        "seed": None if seed_value is None else int(seed_value),
        "source": str(seed_source or ""),
    }


def _next_episode_object_seed(env_cfg: Any) -> tuple[int, str]:
    seed_source = str(getattr(env_cfg, "object_reset_seed_source", "time") or "time").strip().lower()
    if seed_source == "time":
        episode_seed = int(time.time_ns() & 0xFFFFFFFFFFFFFFFF)
        return episode_seed, seed_source

    if seed_source == "env_seed":
        reset_counter = int(getattr(env_cfg, "_episode_object_seed_counter", 0))
        setattr(env_cfg, "_episode_object_seed_counter", reset_counter + 1)
        base_seed = int(getattr(env_cfg, "seed", 0) or 0) & 0xFFFFFFFFFFFFFFFF
        episode_seed = (
            base_seed
            ^ ((reset_counter + 1) * 0x9E3779B185EBCA87)
        ) & 0xFFFFFFFFFFFFFFFF
        return episode_seed, seed_source

    raise ValueError(f"Unsupported object_reset_seed_source: {seed_source}")


def _make_local_spawn_rng(episode_seed: int, record_name: str, env_index: int) -> np.random.Generator:
    name_seed = zlib.crc32(record_name.encode("utf-8")) & 0xFFFFFFFF
    mixed_seed = (
        (int(episode_seed) & 0xFFFFFFFFFFFFFFFF)
        ^ name_seed
        ^ ((env_index + 1) * 0x85EBCA77)
    ) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(mixed_seed)


def apply_deterministic_object_resets(env_cfg: Any, env, *, selected_record_names: set[str] | None = None) -> list[str]:
    if getattr(env_cfg, "_replay_initial_env_state_active", False):
        return []

    specs = get_recordable_env_object_specs(env_cfg)
    if not specs:
        return []

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    applied: list[str] = []
    episode_seed, seed_source = _next_episode_object_seed(env_cfg)
    setattr(env_cfg, "_current_episode_object_seed", episode_seed)
    setattr(env_cfg, "_current_episode_object_seed_source", seed_source)

    for spec in specs:
        record_name = spec["record_name"]
        if selected_record_names is not None and record_name not in selected_record_names:
            continue

        scene_key = resolve_env_object_scene_key(env, env_cfg, record_name)
        if scene_key is None:
            continue

        obj = env.scene[scene_key]
        try:
            root_state = obj.data.default_root_state.clone()
        except Exception:
            root_state = obj.data.root_state_w.clone()

        pose_range = spec.get("pose_range", {}) or {}

        for env_offset, env_id in enumerate(env_ids.tolist()):
            rng = _make_local_spawn_rng(episode_seed, record_name, env_offset)
            for axis_name, axis_idx in _POSE_RANGE_AXES.items():
                axis_range = pose_range.get(axis_name)
                if axis_range is None:
                    continue
                low, high = [float(v) for v in axis_range]
                low, high = min(low, high), max(low, high)
                position = float(rng.uniform(low, high)) if high > low else low
                root_state[env_id, axis_idx] = position
            if spec.get("zero_velocity_on_reset", True):
                root_state[env_id, 7:13] = 0.0

        obj.write_root_state_to_sim(root_state, env_ids=env_ids)
        applied.append(
            f"{record_name}->{scene_key}:episode_seed={episode_seed}:seed_source={seed_source}:pos="
            f"{root_state[0, 0:3].detach().cpu().numpy().tolist()}"
        )

    if applied:
        env.scene.write_data_to_sim()
    return applied
