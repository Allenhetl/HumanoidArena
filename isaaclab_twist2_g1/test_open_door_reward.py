from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


_PROJECT_ROOT = Path(__file__).resolve().parent
_MODULE_PATH = (
    _PROJECT_ROOT
    / "tasks"
    / "g1_tasks"
    / "move_open_door_g1_29dof_dex3_wholebody"
    / "mdp"
    / "rewards.py"
)
_SPEC = importlib.util.spec_from_file_location("open_door_rewards", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load reward module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_reward_open_door = _MODULE.compute_reward_open_door
compute_success_mask = _MODULE.compute_success_mask


class _DummyRobot:
    def __init__(self, root_states: torch.Tensor) -> None:
        self.data = SimpleNamespace(root_state_w=root_states)


class _DummyDoor:
    def __init__(self, root_states: torch.Tensor) -> None:
        self.data = SimpleNamespace(root_state_w=root_states)


class _Scene(dict):
    def keys(self):
        return super().keys()


def _make_root_state(
    *,
    x: float,
    y: float,
    z: float,
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> list[float]:
    return [x, y, z, *quat_wxyz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _make_env(
    robot_root_states: list[list[float]],
    *,
    door_root_states: list[list[float]] | None = None,
) -> SimpleNamespace:
    robot_root_state_w = torch.tensor(robot_root_states, dtype=torch.float32)
    if door_root_states is None:
        door_root_states = [_make_root_state(x=-1.614, y=2.314, z=0.002) for _ in robot_root_states]
    door_root_state_w = torch.tensor(door_root_states, dtype=torch.float32)
    scene = _Scene(
        {
            "robot": _DummyRobot(robot_root_state_w),
            "door": _DummyDoor(door_root_state_w),
        }
    )
    return SimpleNamespace(
        num_envs=len(robot_root_states),
        device="cpu",
        scene=scene,
    )


def test_open_door_reward_is_binary() -> None:
    env = _make_env(
        [
            _make_root_state(x=-1.6, y=2.40, z=0.78),
            _make_root_state(x=-1.6, y=2.10, z=0.78),
        ]
    )

    success = compute_success_mask(env)
    reward = compute_reward_open_door(env)

    assert success.tolist() == [True, False]
    assert reward.tolist() == [1.0, 0.0]


def test_open_door_reward_uses_detected_door_pose() -> None:
    env = _make_env(
        [
            _make_root_state(x=-0.90, y=1.90, z=0.78),
            _make_root_state(x=-0.90, y=1.70, z=0.78),
        ],
        door_root_states=[
            _make_root_state(x=-1.20, y=1.80, z=0.002),
            _make_root_state(x=-1.20, y=1.80, z=0.002),
        ],
    )

    success = compute_success_mask(env)
    reward = compute_reward_open_door(env)

    assert success.tolist() == [True, False]
    assert reward.tolist() == [1.0, 0.0]


def test_open_door_reward_requires_root_to_pass_inside_frame_width() -> None:
    env = _make_env(
        [
            _make_root_state(x=-0.70, y=2.45, z=0.78),
            _make_root_state(x=-0.50, y=2.45, z=0.78),
        ],
        door_root_states=[
            _make_root_state(x=-1.05, y=2.30, z=0.002),
            _make_root_state(x=-1.20, y=2.30, z=0.002),
        ],
    )

    success = compute_success_mask(env)
    reward = compute_reward_open_door(env)

    assert success.tolist() == [True, False]
    assert reward.tolist() == [1.0, 0.0]


def test_open_door_reward_requires_robot_to_remain_standing() -> None:
    env = _make_env(
        [
            _make_root_state(x=-1.6, y=2.40, z=0.78),
            _make_root_state(
                x=-1.6,
                y=2.40,
                z=0.78,
                quat_wxyz=(0.70710677, 0.70710677, 0.0, 0.0),
            ),
            _make_root_state(x=-1.6, y=2.40, z=0.30),
        ]
    )

    success = compute_success_mask(env)
    reward = compute_reward_open_door(env)

    assert success.tolist() == [True, False, False]
    assert reward.tolist() == [1.0, 0.0, 0.0]
