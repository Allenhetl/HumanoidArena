from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
project_root_str = str(_PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from common_env_objects import add_env_object_frame_arrays


def _football_state(position: list[float]) -> dict[str, np.ndarray]:
    return {
        "position": np.asarray(position, dtype=np.float32),
        "orientation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "linear_velocity": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "angular_velocity": np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
    }


def test_add_env_object_frame_arrays_accepts_legacy_env_obj_layout() -> None:
    organized: dict[str, np.ndarray] = {}
    data_buffer = [
        {"env_obj": {"football": _football_state([1.0, 2.0, 3.0])}},
        {"env_obj": {"football": _football_state([4.0, 5.0, 6.0])}},
    ]

    add_env_object_frame_arrays(organized, data_buffer)

    np.testing.assert_allclose(
        organized["env_obj_football_position"],
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        organized["env_obj_football_linear_velocity"],
        np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=np.float32),
    )


def test_add_env_object_frame_arrays_accepts_sonic_env_layout() -> None:
    organized: dict[str, np.ndarray] = {}
    data_buffer = [
        {
            "env": {
                "football": _football_state([0.0, 1.0, 2.0]),
                "vision": {"rgb": None, "depth": None},
            }
        },
        {
            "env": {
                "football": _football_state([3.0, 4.0, 5.0]),
                "vision": {"rgb": None, "depth": None},
            }
        },
    ]

    add_env_object_frame_arrays(organized, data_buffer)

    np.testing.assert_allclose(
        organized["env_obj_football_position"],
        np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        organized["env_obj_football_angular_velocity"],
        np.asarray([[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]], dtype=np.float32),
    )
