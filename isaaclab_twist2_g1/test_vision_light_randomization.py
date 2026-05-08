from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


THIS_DIR = Path(__file__).resolve().parent


def _load_env_runtime_hooks_module():
    module_path = THIS_DIR / "tasks" / "common_runtime" / "env_runtime_hooks.py"
    spec = importlib.util.spec_from_file_location("env_runtime_hooks", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_light_randomization_is_deterministic_and_uses_xyz_rotation_order():
    from tools.augmentation_utils import sample_light_randomization_from_range

    kwargs = {
        "seed": 1234,
        "rotation_ranges": {
            "yaw_deg": [30.0, 30.0],
            "pitch_deg": [20.0, 20.0],
            "roll_deg": [10.0, 10.0],
        },
        "intensity_range": [3000.0, 7000.0],
        "color_range": {
            "r": [0.65, 0.85],
            "g": [0.65, 0.85],
            "b": [0.65, 0.85],
        },
    }

    first = sample_light_randomization_from_range(**kwargs)
    second = sample_light_randomization_from_range(**kwargs)

    assert first == second
    assert first["rotation"] == (10.0, 20.0, 30.0)
    assert 3000.0 <= first["intensity"] <= 7000.0
    assert all(0.65 <= value <= 0.85 for value in first["color"])


def test_sample_rect_light_randomization_always_disables_exactly_four_per_env():
    from tools.augmentation_utils import sample_grouped_rect_light_bar_randomization

    paths = [
        f"/World/envs/env_{env_idx}/Room/cell_light_bars/RectLight_{light_idx}"
        for env_idx in range(2)
        for light_idx in range(8)
    ]

    first = sample_grouped_rect_light_bar_randomization(
        seed=1234,
        prim_paths=paths,
        disable_count_per_group=4,
        enabled_intensity_scale_range=[0.65, 1.25],
        color_range={
            "r": [0.70, 1.00],
            "g": [0.70, 1.00],
            "b": [0.65, 0.95],
        },
    )
    second = sample_grouped_rect_light_bar_randomization(
        seed=1234,
        prim_paths=paths,
        disable_count_per_group=4,
        enabled_intensity_scale_range=[0.65, 1.25],
        color_range={
            "r": [0.70, 1.00],
            "g": [0.70, 1.00],
            "b": [0.65, 0.95],
        },
    )

    assert first == second
    assert len(first["groups"]) == 2
    assert len(first["disabled_paths"]) == 8
    for group in first["groups"]:
        assert group["total_count"] == 8
        assert group["disable_count"] == 4
        assert len(group["disabled_paths"]) == 4
        assert len(group["enabled_paths"]) == 4
        assert all(
            light["intensity_scale"] == 0.0
            for light in group["lights"]
            if not light["enabled"]
        )


def test_setup_vision_test_light_from_cfg_calls_distant_light_replacement(monkeypatch):
    env_runtime_hooks = _load_env_runtime_hooks_module()
    import tools.augmentation_utils as augmentation_utils

    calls = []

    def fake_replace_light_with_distant(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        augmentation_utils,
        "replace_light_with_distant",
        fake_replace_light_with_distant,
    )
    env_cfg = SimpleNamespace(
        vision_randomization={
            "enabled": True,
            "prim_path": "/World/light",
            "setup": {
                "color": [0.75, 0.75, 0.75],
                "intensity": 5000.0,
                "angle": 15.0,
                "position": [-4.0, -1.0, 18.0],
                "rotation": [0.0, 0.0, 0.0],
            },
        }
    )

    assert env_runtime_hooks.setup_vision_test_light_from_cfg(env_cfg) is True

    assert env_cfg._vision_light_setup_done is True
    assert calls == [
        {
            "prim_path": "/World/light",
            "color": (0.75, 0.75, 0.75),
            "intensity": 5000.0,
            "angle": 15.0,
            "position": (-4.0, -1.0, 18.0),
            "rotation": (0.0, 0.0, 0.0),
        }
    ]


def test_apply_vision_light_randomization_reuses_current_episode_seed(monkeypatch):
    env_runtime_hooks = _load_env_runtime_hooks_module()
    import tools.augmentation_utils as augmentation_utils

    calls = []

    def fake_randomize_light_from_range(**kwargs):
        calls.append(kwargs)
        return {
            "rotation": (1.0, 2.0, 3.0),
            "intensity": 4000.0,
            "color": (0.7, 0.8, 0.75),
            "position": None,
        }

    monkeypatch.setattr(
        augmentation_utils,
        "randomize_light_from_range",
        fake_randomize_light_from_range,
    )
    env_cfg = SimpleNamespace(
        _current_episode_object_seed=1234,
        _current_episode_object_seed_source="recorded",
        _vision_light_setup_done=True,
        vision_randomization={
            "enabled": True,
            "debug_logging": False,
            "prim_path": "/World/light",
            "rotation": {
                "yaw_deg": [-180.0, 180.0],
                "pitch_deg": [-25.0, 25.0],
                "roll_deg": [-10.0, 10.0],
            },
            "intensity_range": [3000.0, 7000.0],
            "color_range": {
                "r": [0.65, 0.85],
                "g": [0.65, 0.85],
                "b": [0.65, 0.85],
            },
        },
    )

    assert env_runtime_hooks.apply_vision_light_randomization_from_cfg(env_cfg) is True

    assert calls[0]["prim_path"] == "/World/light"
    assert calls[0]["seed"] != 1234
    assert calls[0]["rotation_ranges"] == env_cfg.vision_randomization["rotation"]
    assert env_cfg._last_vision_light_randomization["episode_object_seed"] == 1234
    assert env_cfg._last_vision_light_randomization["episode_object_seed_source"] == "recorded"


def test_apply_vision_light_randomization_can_apply_room_rect_lights_without_global_light(
    monkeypatch,
):
    env_runtime_hooks = _load_env_runtime_hooks_module()
    import tools.augmentation_utils as augmentation_utils

    calls = []

    def fake_randomize_rect_lights_by_path_keywords(**kwargs):
        calls.append(kwargs)
        return {
            "seed": kwargs["seed"],
            "total_count": 8,
            "disable_count_per_group": kwargs["disable_count_per_group"],
            "disabled_paths": [
                "/World/envs/env_0/Room/cell_light_bars/RectLight_0",
                "/World/envs/env_0/Room/cell_light_bars/RectLight_1",
                "/World/envs/env_0/Room/cell_light_bars/RectLight_2",
                "/World/envs/env_0/Room/cell_light_bars/RectLight_3",
            ],
            "enabled_paths": [
                "/World/envs/env_0/Room/cell_light_bars/RectLight_4",
                "/World/envs/env_0/Room/cell_light_bars/RectLight_5",
                "/World/envs/env_0/Room/cell_light_bars/RectLight_6",
                "/World/envs/env_0/Room/cell_light_bars/RectLight_7",
            ],
            "groups": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        augmentation_utils,
        "randomize_rect_lights_by_path_keywords",
        fake_randomize_rect_lights_by_path_keywords,
    )
    env_cfg = SimpleNamespace(
        _current_episode_object_seed=4321,
        _current_episode_object_seed_source="recorded",
        vision_randomization={
            "enabled": True,
            "debug_logging": False,
            "room_rect_lights": {
                "enabled": True,
                "root_keywords": ["Room", "cell_light_bars"],
                "expected_count_per_env": 8,
                "disable_count": 4,
                "disabled_intensity": 0.0,
                "enabled_intensity_scale_range": [0.65, 1.25],
                "color_range": {
                    "r": [0.70, 1.00],
                    "g": [0.70, 1.00],
                    "b": [0.65, 0.95],
                },
            },
        },
    )

    assert env_runtime_hooks.apply_vision_light_randomization_from_cfg(env_cfg) is True

    assert calls[0]["seed"] != 4321
    assert calls[0]["path_keywords"] == ("Room", "cell_light_bars")
    assert calls[0]["expected_count_per_group"] == 8
    assert calls[0]["disable_count_per_group"] == 4
    assert isinstance(calls[0]["baseline_cache"], dict)
    assert env_cfg._last_vision_room_rect_light_randomization["episode_object_seed"] == 4321
    assert (
        len(env_cfg._last_vision_room_rect_light_randomization["disabled_paths"])
        == 4
    )
