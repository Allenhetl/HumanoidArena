from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent
_MODULE_PATH = (
    _PROJECT_ROOT
    / "tasks"
    / "g1_tasks"
    / "move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody"
    / "obstacle_layout.py"
)
_SPEC = importlib.util.spec_from_file_location("small_warehouse_obstacle_layout", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load obstacle layout module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

OBSTACLE_RECORD_NAMES = _MODULE.OBSTACLE_RECORD_NAMES
build_obstacle_layout_states = _MODULE.build_obstacle_layout_states


def _default_states() -> dict[str, dict[str, np.ndarray]]:
    return {
        record_name: {
            "position": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            "orientation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "linear_velocity": np.zeros(3, dtype=np.float32),
            "angular_velocity": np.zeros(3, dtype=np.float32),
        }
        for record_name in OBSTACLE_RECORD_NAMES
    }


def _legacy_slot_index(
    position_xy: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> int:
    x_mid = 0.5 * (float(x_range[0]) + float(x_range[1]))
    y_mid = 0.5 * (float(y_range[0]) + float(y_range[1]))
    x_idx = 0 if float(position_xy[0]) < x_mid else 1
    y_idx = 0 if float(position_xy[1]) < y_mid else 1
    return y_idx * 2 + x_idx


def test_build_obstacle_layout_states_uses_fixed_unique_z_values() -> None:
    x_ranges = {
        "obstacle_01_a": (-2.2, -1.8),
        "obstacle_01_b": (-1.7, -1.3),
        "obstacle_01_c": (-1.2, -0.8),
        "obstacle_02_a": (-0.7, -0.3),
        "obstacle_02_b": (-0.2, 0.2),
        "obstacle_02_c": (0.3, 0.7),
    }
    y_range = (-0.8, 1.5)
    z_ranges = {
        "obstacle_01_a": (0.16, 0.16),
        "obstacle_01_b": (0.36, 0.36),
        "obstacle_01_c": (0.56, 0.56),
        "obstacle_02_a": (0.76, 0.76),
        "obstacle_02_b": (0.96, 0.96),
        "obstacle_02_c": (1.16, 1.16),
    }

    states = build_obstacle_layout_states(
        default_states=_default_states(),
        episode_seed=123,
        x_ranges=x_ranges,
        y_range=y_range,
        z_ranges=z_ranges,
        yaw_range=(0.0, 0.0),
    )

    assert set(states.keys()) == set(OBSTACLE_RECORD_NAMES)
    sampled_z_values = []
    for record_name, state in states.items():
        position = state["position"]
        assert x_ranges[record_name][0] <= float(position[0]) <= x_ranges[record_name][1]
        assert y_range[0] <= float(position[1]) <= y_range[1]
        np.testing.assert_allclose(
            position[2],
            np.asarray(z_ranges[record_name][0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            state["orientation"],
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        sampled_z_values.append(float(position[2]))

    assert len(set(sampled_z_values)) == len(OBSTACLE_RECORD_NAMES)


def test_build_obstacle_layout_states_allows_multiple_obstacles_in_same_legacy_slot() -> None:
    x_ranges = {
        record_name: (-2.2, 2.5)
        for record_name in OBSTACLE_RECORD_NAMES
    }
    x_range = (-2.2, 2.5)
    y_range = (-0.8, 1.5)
    z_ranges = {
        "obstacle_01_a": (0.16, 0.16),
        "obstacle_01_b": (0.36, 0.36),
        "obstacle_01_c": (0.56, 0.56),
        "obstacle_02_a": (0.76, 0.76),
        "obstacle_02_b": (0.96, 0.96),
        "obstacle_02_c": (1.16, 1.16),
    }

    duplicate_slot_found = False
    for seed in range(20):
        states = build_obstacle_layout_states(
            default_states=_default_states(),
            episode_seed=seed,
            x_ranges=x_ranges,
            y_range=y_range,
            z_ranges=z_ranges,
            yaw_range=(0.0, 0.0),
        )
        slot_indices = [
            _legacy_slot_index(
                states[record_name]["position"][:2],
                x_range=x_range,
                y_range=y_range,
            )
            for record_name in OBSTACLE_RECORD_NAMES
        ]
        if len(set(slot_indices)) < len(slot_indices):
            duplicate_slot_found = True
            break

    assert duplicate_slot_found
