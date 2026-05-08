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
    / "move_pickplace_box_g1_29dof_dex3_wholedoby"
    / "mdp"
    / "rewards.py"
)
_SPEC = importlib.util.spec_from_file_location("pickplace_box_rewards", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load reward module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_reward_pickplace_box = _MODULE.compute_reward_pickplace_box
compute_success_mask = _MODULE.compute_success_mask


class _DummyBox:
    def __init__(self, positions: list[list[float]]) -> None:
        root_state_w = torch.zeros((len(positions), 13), dtype=torch.float32)
        root_state_w[:, 0:3] = torch.tensor(positions, dtype=torch.float32)
        self.data = SimpleNamespace(root_state_w=root_state_w)


class _Scene(dict):
    def keys(self):
        return super().keys()


def _make_env(box_positions: list[list[float]], *, scene_key: str = "box") -> SimpleNamespace:
    scene = _Scene({scene_key: _DummyBox(box_positions)})
    scene.env_origins = torch.zeros((len(box_positions), 3), dtype=torch.float32)
    return SimpleNamespace(
        num_envs=len(box_positions),
        device="cpu",
        scene=scene,
    )


def test_pickplace_box_reward_is_binary(monkeypatch) -> None:
    env = _make_env(
        [
            [0.0, 0.0, 1.0],
            [1.5, 1.5, 1.0],
        ]
    )

    monkeypatch.setattr(
        _MODULE,
        "_get_shelf_support_surfaces_world",
        lambda _env: [
            [(-0.8, 0.8, -0.4, 0.4, 0.3563)],
            [(-0.8, 0.8, -0.4, 0.4, 0.3563)],
        ],
    )

    success = compute_success_mask(env)
    reward = compute_reward_pickplace_box(env)

    assert success.tolist() == [True, False]
    assert reward.tolist() == [1.0, 0.0]


def test_pickplace_box_reward_requires_box_bottom_to_match_support_surface(monkeypatch) -> None:
    env = _make_env([[0.0, 0.0, 0.85]], scene_key="Box")

    monkeypatch.setattr(
        _MODULE,
        "_get_shelf_support_surfaces_world",
        lambda _env: [[(-0.8, 0.8, -0.4, 0.4, 0.3563)]],
    )

    success = compute_success_mask(env)
    reward = compute_reward_pickplace_box(env)

    assert success.tolist() == [False]
    assert reward.tolist() == [0.0]
