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
    / "move_boxing_bag_g1_29dof_dex3_wholebody"
    / "mdp"
    / "rewards.py"
)
_SPEC = importlib.util.spec_from_file_location("boxing_bag_rewards", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load reward module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_reward = _MODULE.compute_reward
compute_success_mask = _MODULE.compute_success_mask


class _DummyRobot:
    def __init__(self, *, body_names: list[str], body_pos_w: torch.Tensor) -> None:
        self.data = SimpleNamespace(
            body_names=body_names,
            body_pos_w=body_pos_w,
        )


class _DummyTarget:
    def __init__(self, *, root_pos_w: torch.Tensor) -> None:
        self.data = SimpleNamespace(root_pos_w=root_pos_w)


class _Scene(dict):
    def keys(self):
        return super().keys()


def _make_env(
    *,
    body_names: list[str],
    body_positions: list[list[list[float]]],
    target_positions: list[list[float]],
) -> SimpleNamespace:
    scene = _Scene(
        {
            "robot": _DummyRobot(
                body_names=body_names,
                body_pos_w=torch.tensor(body_positions, dtype=torch.float32),
            ),
            "boxing_target": _DummyTarget(
                root_pos_w=torch.tensor(target_positions, dtype=torch.float32),
            ),
        }
    )
    return SimpleNamespace(
        num_envs=len(body_positions),
        device="cpu",
        scene=scene,
    )


def test_boxing_bag_reward_is_binary_for_palm_hit() -> None:
    env = _make_env(
        body_names=["left_hand_palm_link", "right_hand_palm_link", "torso_link"],
        body_positions=[
            [[0.05, 0.0, 0.0], [0.40, 0.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.40, 0.0, 0.0], [0.50, 0.0, 0.0], [0.0, 0.0, 0.5]],
        ],
        target_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )

    success = compute_success_mask(env)
    reward = compute_reward(env)

    assert success.tolist() == [True, False]
    assert reward.tolist() == [1.0, 0.0]


def test_boxing_bag_reward_falls_back_to_wrist_links() -> None:
    env = _make_env(
        body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        body_positions=[
            [[0.04, 0.0, 0.0], [0.30, 0.0, 0.0]],
        ],
        target_positions=[
            [0.0, 0.0, 0.0],
        ],
    )

    success = compute_success_mask(env)
    reward = compute_reward(env)

    assert success.tolist() == [True]
    assert reward.tolist() == [1.0]
