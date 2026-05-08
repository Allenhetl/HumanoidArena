"""Reusable task-side scene adjustment helpers."""

from __future__ import annotations

import zlib
from typing import Any


def apply_scene_filter_from_cfg(env_cfg: Any) -> None:
    scene_deactivate_keywords = tuple(getattr(env_cfg, "scene_deactivate_keywords", ()))
    scene_deactivate_exclude_keywords = tuple(
        getattr(env_cfg, "scene_deactivate_exclude_keywords", ())
    )
    if not scene_deactivate_keywords:
        return
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        deactivated_paths = []
        keywords = tuple(k.lower() for k in scene_deactivate_keywords)
        exclude_keywords = tuple(k.lower() for k in scene_deactivate_exclude_keywords)
        for prim in stage.Traverse():
            if not prim.IsActive():
                continue
            prim_path = prim.GetPath().pathString
            prim_name = prim.GetName()
            path_lower = prim_path.lower()
            name_lower = prim_name.lower()
            if any((k in path_lower) or (k in name_lower) for k in exclude_keywords):
                continue
            if any((k in path_lower) or (k in name_lower) for k in keywords):
                prim.SetActive(False)
                deactivated_paths.append(prim_path)
        print(
            f"[scene_filter] deactivate keywords={scene_deactivate_keywords}, "
            f"exclude={scene_deactivate_exclude_keywords}, count={len(deactivated_paths)}"
        )
        for prim_path in deactivated_paths[:20]:
            print(f"[scene_filter] deactivated: {prim_path}")
    except Exception as exc:
        print(f"[scene_filter] deactivate failed: {exc}")


def apply_scene_reposition_from_cfg(env_cfg: Any) -> None:
    scene_reposition_rules = tuple(getattr(env_cfg, "scene_reposition_rules", ()))
    if not scene_reposition_rules:
        return
    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        moved_items = []
        all_active_prims = [prim for prim in stage.Traverse() if prim.IsActive()]
        for rule in scene_reposition_rules:
            keywords = tuple(str(k).lower() for k in rule.get("keywords", ()))
            if not keywords:
                continue
            rule_name = str(rule.get("name", "rule"))
            safe_rule_name = "".join(ch if ch.isalnum() else "_" for ch in rule_name.lower())
            offset = tuple(float(v) for v in rule.get("offset", (0.0, 0.0, 0.0)))
            if abs(offset[0]) + abs(offset[1]) + abs(offset[2]) < 1e-9:
                continue
            candidates = []
            for prim in all_active_prims:
                prim_path = prim.GetPath().pathString
                prim_name = prim.GetName()
                path_lower = prim_path.lower()
                name_lower = prim_name.lower()
                if "joint" in path_lower or "/materials" in path_lower:
                    continue
                if not any((k in name_lower) or (k in path_lower) for k in keywords):
                    continue
                depth = prim_path.count("/")
                candidates.append((depth, len(prim_path), prim_path))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (item[0], item[1]))
            target_path = candidates[0][2]
            target_prim = stage.GetPrimAtPath(target_path)
            if not target_prim.IsValid():
                continue
            marker_attr = target_prim.GetAttribute(
                f"userProperties:humanoidarena_reposition_applied_{safe_rule_name}"
            )
            if marker_attr.IsValid() and bool(marker_attr.Get()):
                continue
            xformable = UsdGeom.Xformable(target_prim)
            translate_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            current_t = translate_op.Get()
            if current_t is None:
                current_t = Gf.Vec3d(0.0, 0.0, 0.0)
            new_t = Gf.Vec3d(
                float(current_t[0]) + offset[0],
                float(current_t[1]) + offset[1],
                float(current_t[2]) + offset[2],
            )
            translate_op.Set(new_t)
            if not marker_attr.IsValid():
                marker_attr = target_prim.CreateAttribute(
                    f"userProperties:humanoidarena_reposition_applied_{safe_rule_name}",
                    Sdf.ValueTypeNames.Bool,
                    custom=True,
                )
            marker_attr.Set(True)
            moved_items.append((target_path, rule_name, offset))
        print(
            f"[scene_filter] reposition rules={len(scene_reposition_rules)}, moved={len(moved_items)}"
        )
        for prim_path, rule_name, offset in moved_items[:20]:
            print(f"[scene_filter] moved({rule_name}): {prim_path}, offset={offset}")
    except Exception as exc:
        print(f"[scene_filter] reposition failed: {exc}")


def apply_optional_runtime_augments(args_cli: Any) -> None:
    if getattr(args_cli, "modify_light", False):
        try:
            from tools.augmentation_utils import update_light

            update_light(
                prim_path="/World/light",
                color=(0.75, 0.75, 0.75),
                intensity=500.0,
                radius=0.1,
                enabled=True,
                cast_shadows=True,
            )
        except Exception as exc:
            print(f"[env_runtime] modify_light failed: {exc}")

    if getattr(args_cli, "modify_camera", False):
        try:
            from tools.augmentation_utils import batch_augment_cameras_by_name

            batch_augment_cameras_by_name(
                names=["front_cam"],
                focal_length=3.0,
                horizontal_aperture=22.0,
                vertical_aperture=16.0,
                exposure=0.8,
                focus_distance=1.2,
            )
        except Exception as exc:
            print(f"[env_runtime] modify_camera failed: {exc}")


def _get_enabled_vision_randomization_cfg(env_cfg: Any) -> dict[str, Any] | None:
    cfg = getattr(env_cfg, "vision_randomization", None)
    if not isinstance(cfg, dict):
        return None
    if not bool(cfg.get("enabled", False)):
        return None
    return cfg


def _tuple_from_cfg(value, default):
    if value is None:
        value = default
    return tuple(float(v) for v in value)


def _string_tuple_from_cfg(value, default=()):
    if value is None:
        value = default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _derive_vision_light_seed(episode_seed: int) -> int:
    namespace_seed = zlib.crc32(b"vision_light") & 0xFFFFFFFF
    return (int(episode_seed) ^ namespace_seed) & 0xFFFFFFFFFFFFFFFF


def _derive_vision_room_rect_light_seed(episode_seed: int) -> int:
    namespace_seed = zlib.crc32(b"vision_room_rect_lights") & 0xFFFFFFFF
    return (int(episode_seed) ^ namespace_seed) & 0xFFFFFFFFFFFFFFFF


def setup_vision_test_light(
    prim_path: str = "/World/light",
    color=(0.75, 0.75, 0.75),
    intensity: float = 5000.0,
    angle: float = 15.0,
    position=(-4.0, -1.0, 18.0),
    rotation=(0.0, 0.0, 0.0),
) -> bool:
    """Replace the scene light with a DistantLight for direction randomization."""

    try:
        from tools.augmentation_utils import replace_light_with_distant

        replace_light_with_distant(
            prim_path=prim_path,
            color=tuple(color),
            intensity=float(intensity),
            angle=float(angle),
            position=tuple(position),
            rotation=tuple(rotation),
        )
        print(f"[vision_randomization] DistantLight setup complete: {prim_path}")
        return True
    except Exception as exc:
        print(f"[vision_randomization] setup_vision_test_light failed: {exc}")
        return False


def setup_vision_test_light_from_cfg(env_cfg: Any) -> bool:
    cfg = _get_enabled_vision_randomization_cfg(env_cfg)
    if cfg is None:
        return False
    global_light_keys = {
        "setup",
        "prim_path",
        "base_color",
        "base_intensity",
        "position",
        "rotation",
        "rotation_ranges",
    }
    if not any(key in cfg for key in global_light_keys):
        return False
    if bool(getattr(env_cfg, "_vision_light_setup_done", False)):
        return True

    setup_cfg = cfg.get("setup", {}) if isinstance(cfg.get("setup", {}), dict) else {}
    ok = setup_vision_test_light(
        prim_path=str(cfg.get("prim_path", "/World/light")),
        color=_tuple_from_cfg(setup_cfg.get("color", cfg.get("base_color")), (0.75, 0.75, 0.75)),
        intensity=float(setup_cfg.get("intensity", cfg.get("base_intensity", 5000.0))),
        angle=float(setup_cfg.get("angle", cfg.get("angle", 15.0))),
        position=_tuple_from_cfg(setup_cfg.get("position", cfg.get("position")), (-4.0, -1.0, 18.0)),
        rotation=_tuple_from_cfg(setup_cfg.get("rotation", cfg.get("base_rotation")), (0.0, 0.0, 0.0)),
    )
    setattr(env_cfg, "_vision_light_setup_done", bool(ok))
    return ok


def apply_vision_light_randomization(
    prim_path: str = "/World/light",
    episode_seed: int = 0,
    rotation_ranges: dict | None = None,
    intensity_range: list | tuple | None = None,
    color_range: dict | None = None,
    position_range: dict | None = None,
) -> dict[str, Any] | None:
    """Apply deterministic light randomization from one episode object seed."""

    if not rotation_ranges:
        return None

    try:
        from tools.augmentation_utils import randomize_light_from_range

        return randomize_light_from_range(
            prim_path=prim_path,
            seed=_derive_vision_light_seed(int(episode_seed)),
            rotation_ranges=rotation_ranges,
            intensity_range=intensity_range,
            color_range=color_range,
            position_range=position_range,
        )
    except Exception as exc:
        print(f"[vision_randomization] apply_vision_light_randomization failed: {exc}")
        return None


def apply_vision_room_rect_light_randomization(
    *,
    episode_seed: int,
    path_keywords=("cell_light_bars",),
    prim_paths=None,
    exclude_keywords=(),
    expected_count_per_group: int | None = 8,
    disable_count_per_group: int = 4,
    disabled_intensity: float = 0.0,
    enabled_intensity_scale_range=None,
    color_range: dict | None = None,
    baseline_cache: dict | None = None,
) -> dict[str, Any] | None:
    """Apply deterministic 8-choose-4 RectLight shutdown per env group."""

    try:
        from tools.augmentation_utils import randomize_rect_lights_by_path_keywords

        return randomize_rect_lights_by_path_keywords(
            seed=_derive_vision_room_rect_light_seed(int(episode_seed)),
            path_keywords=path_keywords,
            prim_paths=prim_paths,
            exclude_keywords=exclude_keywords,
            expected_count_per_group=expected_count_per_group,
            disable_count_per_group=int(disable_count_per_group),
            disabled_intensity=float(disabled_intensity),
            enabled_intensity_scale_range=enabled_intensity_scale_range,
            color_range=color_range,
            baseline_cache=baseline_cache,
        )
    except Exception as exc:
        print(f"[vision_randomization] apply_vision_room_rect_light_randomization failed: {exc}")
        return None


def _resolve_episode_seed_once(env_cfg: Any, episode_seed: int | None, seed_source: str | None):
    if episode_seed is not None:
        return int(episode_seed), str(seed_source or "")

    from common_env_objects import ensure_current_episode_object_seed

    resolved_seed, resolved_source = ensure_current_episode_object_seed(env_cfg)
    return int(resolved_seed), str(resolved_source or "")


def _get_room_rect_light_cfg(cfg: dict[str, Any]) -> dict[str, Any] | None:
    room_cfg = cfg.get("room_rect_lights")
    if not isinstance(room_cfg, dict):
        return None
    if not bool(room_cfg.get("enabled", False)):
        return None
    return room_cfg


def apply_vision_light_randomization_from_cfg(
    env_cfg: Any,
    *,
    episode_seed: int | None = None,
    seed_source: str | None = None,
) -> bool:
    cfg = _get_enabled_vision_randomization_cfg(env_cfg)
    if cfg is None:
        return False

    applied = False
    resolved_seed = episode_seed
    resolved_source = seed_source

    rotation_ranges = cfg.get("rotation", cfg.get("rotation_ranges"))
    if isinstance(rotation_ranges, dict) and rotation_ranges:
        if not bool(getattr(env_cfg, "_vision_light_setup_done", False)):
            setup_vision_test_light_from_cfg(env_cfg)

        resolved_seed, resolved_source = _resolve_episode_seed_once(
            env_cfg,
            resolved_seed,
            resolved_source,
        )
        sampled = apply_vision_light_randomization(
            prim_path=str(cfg.get("prim_path", "/World/light")),
            episode_seed=int(resolved_seed),
            rotation_ranges=rotation_ranges,
            intensity_range=cfg.get("intensity_range"),
            color_range=cfg.get("color_range"),
            position_range=cfg.get("position_range"),
        )
        if sampled is not None:
            state = {
                "episode_object_seed": int(resolved_seed),
                "episode_object_seed_source": str(resolved_source or ""),
                **sampled,
            }
            setattr(env_cfg, "_last_vision_light_randomization", state)
            if bool(cfg.get("debug_logging", True)):
                print(
                    "[vision_randomization] "
                    f"episode_object_seed={resolved_seed} source={resolved_source or ''} "
                    f"prim={cfg.get('prim_path', '/World/light')} "
                    f"rotation_xyz_deg={sampled.get('rotation')} "
                    f"intensity={sampled.get('intensity')} color={sampled.get('color')} "
                    f"position={sampled.get('position')}"
                )
            applied = True

    room_cfg = _get_room_rect_light_cfg(cfg)
    if room_cfg is not None:
        resolved_seed, resolved_source = _resolve_episode_seed_once(
            env_cfg,
            resolved_seed,
            resolved_source,
        )
        baseline_cache = getattr(env_cfg, "_vision_room_rect_light_baselines", None)
        if not isinstance(baseline_cache, dict):
            baseline_cache = {}
            setattr(env_cfg, "_vision_room_rect_light_baselines", baseline_cache)

        path_keywords = room_cfg.get(
            "path_keywords",
            room_cfg.get("root_keywords", room_cfg.get("keywords", ("cell_light_bars",))),
        )
        sampled_room = apply_vision_room_rect_light_randomization(
            episode_seed=int(resolved_seed),
            path_keywords=_string_tuple_from_cfg(path_keywords, ("cell_light_bars",)),
            prim_paths=room_cfg.get("prim_paths"),
            exclude_keywords=_string_tuple_from_cfg(room_cfg.get("exclude_keywords"), ()),
            expected_count_per_group=room_cfg.get("expected_count_per_env", 8),
            disable_count_per_group=int(room_cfg.get("disable_count", 4)),
            disabled_intensity=float(room_cfg.get("disabled_intensity", 0.0)),
            enabled_intensity_scale_range=room_cfg.get("enabled_intensity_scale_range"),
            color_range=room_cfg.get("color_range"),
            baseline_cache=baseline_cache,
        )
        if sampled_room is not None:
            state = {
                "episode_object_seed": int(resolved_seed),
                "episode_object_seed_source": str(resolved_source or ""),
                **sampled_room,
            }
            setattr(env_cfg, "_last_vision_room_rect_light_randomization", state)
            if bool(cfg.get("debug_logging", True)) and bool(room_cfg.get("debug_logging", True)):
                print(
                    "[vision_randomization] room_rect_lights "
                    f"episode_object_seed={resolved_seed} source={resolved_source or ''} "
                    f"disabled_paths={sampled_room.get('disabled_paths')}"
                )
            applied = True

    return applied
