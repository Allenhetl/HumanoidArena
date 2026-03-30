"""Helpers for loading per-run environment overrides from YAML."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


COMMON_ENV_CONFIG_DIR = Path(__file__).resolve().parent


def resolve_env_config_yaml_path(config_path: str | None) -> Path | None:
    """Resolve a YAML path from CLI input.

    If ``config_path`` is relative, this first checks it relative to the current
    working directory and then relative to ``tasks/common_env_config``.
    """

    if not config_path:
        return None

    candidate = Path(config_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    common_candidate = COMMON_ENV_CONFIG_DIR / config_path
    if common_candidate.is_file():
        return common_candidate.resolve()

    raise FileNotFoundError(
        f"Environment config YAML not found: {config_path}. "
        f"Tried '{candidate}' and '{common_candidate}'."
    )


def _deep_merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _flatten_overrides(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_overrides(value, prefix=path))
        else:
            flat[path] = value
    return flat


def _coerce_value_like(current_value: Any, new_value: Any) -> Any:
    if current_value is None:
        return new_value
    if isinstance(current_value, tuple) and isinstance(new_value, list):
        return tuple(new_value)
    if isinstance(current_value, list) and isinstance(new_value, tuple):
        return list(new_value)
    if isinstance(current_value, bool) and isinstance(new_value, bool):
        return new_value
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(new_value)
    if isinstance(current_value, float):
        return float(new_value)
    return new_value


def _set_cfg_value(cfg_root: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = cfg_root
    for part in parts[:-1]:
        if isinstance(target, dict):
            if part not in target:
                raise AttributeError(f"Missing config key '{part}' in path '{dotted_path}'")
            target = target[part]
        else:
            if not hasattr(target, part):
                raise AttributeError(f"Missing config attr '{part}' in path '{dotted_path}'")
            target = getattr(target, part)

    leaf = parts[-1]
    if isinstance(target, dict):
        if leaf not in target:
            raise AttributeError(f"Missing config key '{leaf}' in path '{dotted_path}'")
        target[leaf] = _coerce_value_like(target[leaf], value)
    else:
        if not hasattr(target, leaf):
            raise AttributeError(f"Missing config attr '{leaf}' in path '{dotted_path}'")
        current_value = getattr(target, leaf)
        setattr(target, leaf, _coerce_value_like(current_value, value))


def _sync_derived_env_fields(env_cfg: Any, changed_paths: set[str]) -> None:
    if "sim.dt" in changed_paths:
        try:
            contact_forces = getattr(getattr(env_cfg, "scene", None), "contact_forces", None)
            if contact_forces is not None and hasattr(contact_forces, "update_period"):
                contact_forces.update_period = env_cfg.sim.dt
        except Exception:
            pass

    if ("decimation" in changed_paths) or ("sim.dt" in changed_paths):
        try:
            if hasattr(env_cfg.sim, "render_interval"):
                env_cfg.sim.render_interval = env_cfg.decimation
        except Exception:
            pass


def _collect_yaml_overrides(
    raw_cfg: dict[str, Any], task_name: str | None = None, route_name: str | None = None
) -> dict[str, Any]:
    base_overrides = raw_cfg.get("overrides", raw_cfg)
    if not isinstance(base_overrides, dict):
        raise ValueError("YAML 'overrides' must be a mapping")

    merged = deepcopy(base_overrides)

    route_overrides = raw_cfg.get("routes", {})
    if route_name and isinstance(route_overrides, dict):
        selected_route = route_overrides.get(route_name)
        if selected_route is not None:
            if not isinstance(selected_route, dict):
                raise ValueError(f"YAML route override '{route_name}' must be a mapping")
            merged = _deep_merge_dict(merged, selected_route)

    task_overrides = raw_cfg.get("tasks", {})
    if task_name and isinstance(task_overrides, dict):
        selected_task = task_overrides.get(task_name)
        if selected_task is not None:
            if not isinstance(selected_task, dict):
                raise ValueError(f"YAML task override '{task_name}' must be a mapping")
            merged = _deep_merge_dict(merged, selected_task)

    route_task_overrides = raw_cfg.get("route_tasks", {})
    if route_name and task_name and isinstance(route_task_overrides, dict):
        selected_route_task = route_task_overrides.get(route_name, {})
        if isinstance(selected_route_task, dict) and task_name in selected_route_task:
            task_data = selected_route_task[task_name]
            if not isinstance(task_data, dict):
                raise ValueError(
                    f"YAML route_tasks override '{route_name}.{task_name}' must be a mapping"
                )
            merged = _deep_merge_dict(merged, task_data)

    return merged


def apply_env_config_yaml(
    env_cfg: Any,
    config_path: str | None,
    *,
    task_name: str | None = None,
    route_name: str | None = None,
) -> Path | None:
    """Apply YAML overrides to an IsaacLab env cfg before env creation."""

    resolved_path = resolve_env_config_yaml_path(config_path)
    if resolved_path is None:
        return None

    with resolved_path.open("r", encoding="utf-8") as file:
        raw_cfg = yaml.safe_load(file) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Environment config YAML must contain a mapping: {resolved_path}")

    merged_overrides = _collect_yaml_overrides(
        raw_cfg,
        task_name=task_name,
        route_name=route_name,
    )
    flat_overrides = _flatten_overrides(merged_overrides)
    changed_paths: set[str] = set()

    print(f"[env_config_yaml] loaded: {resolved_path}")
    for path, value in flat_overrides.items():
        _set_cfg_value(env_cfg, path, value)
        changed_paths.add(path)
        print(f"[env_config_yaml] apply {path}={value}")

    _sync_derived_env_fields(env_cfg, changed_paths)
    if "sim.dt" in changed_paths or "decimation" in changed_paths:
        try:
            control_hz = 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.decimation))
            print(
                f"[env_config_yaml] effective sim.dt={env_cfg.sim.dt}, "
                f"decimation={env_cfg.decimation}, control_hz={control_hz:.2f}"
            )
        except Exception:
            pass

    return resolved_path
