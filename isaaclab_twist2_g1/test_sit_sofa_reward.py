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
    / "move_sit_sofa_g1_29dof_dex3_wholebody"
    / "mdp"
    / "rewards.py"
)
_SPEC = importlib.util.spec_from_file_location("sit_sofa_rewards", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load reward module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compute_reward = _MODULE.compute_reward
compute_success_mask = _MODULE.compute_success_mask


class _DummyRobot:
    def __init__(
        self,
        *,
        body_names: list[str],
        body_pos_w: torch.Tensor,
        body_net_contact_force_w: torch.Tensor,
    ) -> None:
        self.data = SimpleNamespace(
            body_names=body_names,
            body_pos_w=body_pos_w,
            body_net_contact_force_w=body_net_contact_force_w,
        )


class _DummyContactSensor:
    def __init__(self, *, body_names: list[str], net_forces_w: torch.Tensor) -> None:
        self.body_names = body_names
        self.data = SimpleNamespace(net_forces_w=net_forces_w)


class _Scene(dict):
    def keys(self):
        return super().keys()


def _make_env(
    *,
    body_names: list[str],
    body_positions: list[list[list[float]]],
    body_contact_forces: list[list[list[float]]],
) -> SimpleNamespace:
    scene = _Scene(
        {
            "robot": _DummyRobot(
                body_names=body_names,
                body_pos_w=torch.tensor(body_positions, dtype=torch.float32),
                body_net_contact_force_w=torch.tensor(body_contact_forces, dtype=torch.float32),
            )
        }
    )
    return SimpleNamespace(
        num_envs=len(body_positions),
        device="cpu",
        scene=scene,
    )


def test_sit_sofa_reward_is_binary(monkeypatch) -> None:
    env = _make_env(
        body_names=["waist_yaw_link", "torso_link"],
        body_positions=[
            [[1.0, 2.0, 0.55], [1.0, 2.0, 0.90]],
            [[3.0, 4.0, 0.55], [3.0, 4.0, 0.90]],
        ],
        body_contact_forces=[
            [[0.0, 0.0, 30.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 30.0], [0.0, 0.0, 0.0]],
        ],
    )

    monkeypatch.setattr(
        _MODULE,
        "_get_sofa_seat_boxes_world",
        lambda _env: [
            (0.5, 1.5, 1.5, 2.5, 0.35, 0.65),
            (0.5, 1.5, 1.5, 2.5, 0.35, 0.65),
        ],
    )

    success = compute_success_mask(env)
    reward = compute_reward(env)

    assert success.tolist() == [True, False]
    assert reward.tolist() == [1.0, 0.0]


def test_sit_sofa_reward_requires_contact(monkeypatch) -> None:
    env = _make_env(
        body_names=["waist_yaw_link"],
        body_positions=[[[1.0, 2.0, 0.55]]],
        body_contact_forces=[[[0.0, 0.0, 1.0]]],
    )

    monkeypatch.setattr(
        _MODULE,
        "_get_sofa_seat_boxes_world",
        lambda _env: [(0.5, 1.5, 1.5, 2.5, 0.35, 0.65)],
    )

    success = compute_success_mask(env)
    reward = compute_reward(env)

    assert success.tolist() == [False]
    assert reward.tolist() == [0.0]


def test_sit_sofa_reward_requires_body_to_be_inside_seat_region(monkeypatch) -> None:
    env = _make_env(
        body_names=["waist_yaw_link"],
        body_positions=[[[2.2, 2.0, 0.55]]],
        body_contact_forces=[[[0.0, 0.0, 30.0]]],
    )

    monkeypatch.setattr(
        _MODULE,
        "_get_sofa_seat_boxes_world",
        lambda _env: [(0.5, 1.5, 1.5, 2.5, 0.35, 0.65)],
    )

    success = compute_success_mask(env)
    reward = compute_reward(env)

    assert success.tolist() == [False]
    assert reward.tolist() == [0.0]


def test_sit_sofa_reward_accepts_refactored_sofa_seat_names() -> None:
    sofa_score = _MODULE._candidate_sofa_seat_score(
        "r_seat",
        "/World/envs/env_0/Room/bl_sofaz9A_tea/R_seat",
    )
    chair_score = _MODULE._candidate_sofa_seat_score(
        "r_seat",
        "/World/envs/env_0/Room/office_chair/R_seat",
    )

    assert _MODULE._is_sofa_seat_candidate(
        "r_seat",
        "/World/envs/env_0/Room/bl_sofaz9A_tea/R_seat",
    )
    assert sofa_score > chair_score


def test_sit_sofa_reward_falls_back_to_seated_geometry_without_contact(monkeypatch) -> None:
    env = _make_env(
        body_names=["pelvis", "left_hip_pitch_link", "torso_link"],
        body_positions=[
            [
                [1.00, 2.00, 0.60],
                [1.08, 2.00, 0.56],
                [1.00, 2.00, 0.92],
            ]
        ],
        body_contact_forces=[
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ],
    )

    monkeypatch.setattr(
        _MODULE,
        "_get_sofa_seat_boxes_world",
        lambda _env: [(0.5, 1.5, 1.5, 2.5, 0.35, 0.65)],
    )

    success = compute_success_mask(env)
    reward = compute_reward(env)

    assert success.tolist() == [True]
    assert reward.tolist() == [1.0]


def test_sit_sofa_reward_falls_back_to_contact_sensor_forces(monkeypatch) -> None:
    body_names = ["waist_yaw_link", "torso_link"]
    scene = _Scene(
        {
            "robot": _DummyRobot(
                body_names=body_names,
                body_pos_w=torch.tensor(
                    [
                        [[1.0, 2.0, 0.55], [1.0, 2.0, 0.90]],
                    ],
                    dtype=torch.float32,
                ),
                body_net_contact_force_w=None,
            ),
            "contact_forces": _DummyContactSensor(
                body_names=body_names,
                net_forces_w=torch.tensor(
                    [
                        [[0.0, 0.0, 30.0], [0.0, 0.0, 0.0]],
                    ],
                    dtype=torch.float32,
                ),
            ),
        }
    )
    env = SimpleNamespace(num_envs=1, device="cpu", scene=scene)

    monkeypatch.setattr(
        _MODULE,
        "_get_sofa_seat_boxes_world",
        lambda _env: [(0.5, 1.5, 1.5, 2.5, 0.35, 0.65)],
    )

    success = compute_success_mask(env)
    reward = compute_reward(env)

    assert success.tolist() == [True]
    assert reward.tolist() == [1.0]
