from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


THIS_DIR = Path(__file__).resolve().parent
RECOVERY_STATE_PATH = (
    THIS_DIR
    / "tasks"
    / "g1_tasks"
    / "move_pickplace_box_g1_29dof_dex3_wholedoby"
    / "mdp"
    / "recovery_state.py"
)


def _load_recovery_state_module():
    spec = importlib.util.spec_from_file_location("pp_box_recovery_state", RECOVERY_STATE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Scene:
    def __init__(self, env: "_FakeEnv") -> None:
        self._env = env

    def get_state(self) -> dict[str, object]:
        return {
            "robot": {"joint_pos": self._env.joint_pos},
            "box": {"root_state": self._env.box_root_state},
        }


class _FakeEnv:
    def __init__(self) -> None:
        self.joint_pos = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        self.box_root_state = np.arange(13, dtype=np.float32).reshape(1, 13)
        self.episode_length_buf = torch.tensor([7], dtype=torch.int64)
        self.common_step_counter = 11
        self.reset_buf = torch.tensor([False])
        self.episode_count = 3
        self.reset_count = torch.tensor([2], dtype=torch.int64)
        self.recovery_task_state = {
            "fsm_stage": "grasp",
            "hands_materialized": torch.tensor([[True, False]]),
        }
        self.task_generator = torch.Generator().manual_seed(101)
        self.wrapper_rng = np.random.default_rng(202)
        self.scene = _Scene(self)
        self.restore_events: list[str] = []

    def reset_to(self, state: dict[str, object]) -> None:
        self.restore_events.append("scene")
        self.joint_pos = state["robot"]["joint_pos"].clone()
        self.box_root_state = state["box"]["root_state"].copy()
        # Real reset paths may consume RNG and mutate counters before callers restore them.
        random.random()
        np.random.random()
        torch.rand(1)
        torch.rand(1, generator=self.task_generator)
        self.wrapper_rng.random()
        self.episode_length_buf.add_(100)

    def restore_recovery_task_state(self, state: dict[str, object]) -> None:
        self.restore_events.append("task")
        assert int(self.episode_length_buf.item()) == 7
        self.recovery_task_state = state

    def deterministic_step(self) -> tuple[object, ...]:
        self.joint_pos.add_(0.25)
        self.episode_length_buf.add_(1)
        return (
            self.joint_pos.clone(),
            self.episode_length_buf.clone(),
            random.random(),
            float(np.random.random()),
            torch.rand(2),
            torch.rand(2, generator=self.task_generator),
            self.wrapper_rng.random(2),
        )


class _CacheEnv(_FakeEnv):
    def capture_physics_solver_cache(self) -> dict[str, int]:
        return {"solver_epoch": 4}

    def restore_physics_solver_cache(self, state: dict[str, int]) -> None:
        self.restore_events.append(f"solver:{state['solver_epoch']}")


@pytest.fixture()
def recovery_state():
    return _load_recovery_state_module()


def test_snapshot_schema_is_versioned_and_task_bound(recovery_state) -> None:
    env = _FakeEnv()

    snapshot = recovery_state.capture_recovery_state(env)

    assert snapshot.schema_version == recovery_state.RECOVERY_STATE_SCHEMA_VERSION == 1
    assert snapshot.task_identity == recovery_state.PP_BOX_TASK_IDENTITY
    assert snapshot.capabilities.schema_version == 1


def test_capture_deep_clones_tensor_and_numpy_scene_state(recovery_state) -> None:
    env = _FakeEnv()

    snapshot = recovery_state.capture_recovery_state(env)
    env.joint_pos.fill_(9.0)
    env.box_root_state.fill(9.0)
    env.recovery_task_state["hands_materialized"].fill_(False)

    torch.testing.assert_close(
        snapshot.scene_state["robot"]["joint_pos"],
        torch.tensor([[0.1, 0.2]], dtype=torch.float32),
    )
    np.testing.assert_array_equal(
        snapshot.scene_state["box"]["root_state"],
        np.arange(13, dtype=np.float32).reshape(1, 13),
    )
    torch.testing.assert_close(
        snapshot.task_state["hands_materialized"],
        torch.tensor([[True, False]]),
    )


def test_global_python_numpy_and_torch_rng_roundtrip(recovery_state) -> None:
    env = _FakeEnv()
    random.seed(10)
    np.random.seed(20)
    torch.manual_seed(30)
    snapshot = recovery_state.capture_recovery_state(env)

    expected = (random.random(), np.random.random(), torch.rand(3))
    recovery_state.restore_recovery_state(env, snapshot)
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_required_unavailable_capabilities_fail_with_typed_diagnostics(recovery_state) -> None:
    env = _FakeEnv()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.capture_recovery_state(
            env,
            required_capabilities={"physics_solver_cache", "contact_cache"},
        )

    assert exc_info.value.missing_capabilities == (
        "contact_cache",
        "physics_solver_cache",
    )
    assert "contact_cache" in str(exc_info.value)
    assert "physics_solver_cache" in str(exc_info.value)


def test_restore_rejects_missing_required_cache_payload_before_reset(recovery_state) -> None:
    env = _CacheEnv()
    snapshot = recovery_state.capture_recovery_state(
        env,
        required_capabilities={"physics_solver_cache"},
    )
    incomplete = replace(snapshot, runtime_state={})

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.restore_recovery_state(
            env,
            incomplete,
            required_capabilities={"physics_solver_cache"},
        )

    assert exc_info.value.missing_capabilities == ("physics_solver_cache",)
    assert env.restore_events == []


def test_capture_records_scene_task_counters_and_optional_task_state(recovery_state) -> None:
    env = _FakeEnv()

    snapshot = recovery_state.capture_recovery_state(env)

    assert snapshot.capabilities.available["scene_state"] is True
    assert snapshot.capabilities.available["task_counters"] is True
    assert snapshot.capabilities.available["task_state"] is True
    assert snapshot.capabilities.available["task_local_rng"] is True
    assert snapshot.capabilities.available["wrapper_rng"] is True
    assert snapshot.capabilities.available["physics_solver_cache"] is False
    assert snapshot.capabilities.available["contact_cache"] is False
    torch.testing.assert_close(snapshot.task_counters["episode_length_buf"], torch.tensor([7]))
    torch.testing.assert_close(snapshot.task_counters["reset_buf"], torch.tensor([False]))
    assert snapshot.task_counters["common_step_counter"] == 11
    assert snapshot.task_counters["episode_count"] == 3
    torch.testing.assert_close(snapshot.task_counters["reset_count"], torch.tensor([2]))


def test_restore_uses_scene_reset_then_counters_task_state_and_rng(recovery_state) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env.episode_length_buf.fill_(99)
    env.common_step_counter = 99
    env.reset_buf.fill_(True)
    env.episode_count = 99
    env.reset_count.fill_(99)
    env.recovery_task_state = {"fsm_stage": "corrupted"}

    recovery_state.restore_recovery_state(env, snapshot)

    assert env.restore_events == ["scene", "task"]
    torch.testing.assert_close(env.episode_length_buf, torch.tensor([7]))
    assert env.common_step_counter == 11
    torch.testing.assert_close(env.reset_buf, torch.tensor([False]))
    assert env.episode_count == 3
    torch.testing.assert_close(env.reset_count, torch.tensor([2]))
    assert env.recovery_task_state["fsm_stage"] == "grasp"


def test_restore_rejects_snapshot_from_unknown_schema(recovery_state) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    incompatible = recovery_state.RecoveryStateSnapshot(
        schema_version=999,
        task_identity=snapshot.task_identity,
        capabilities=snapshot.capabilities,
        scene_state=snapshot.scene_state,
        task_counters=snapshot.task_counters,
        task_state=snapshot.task_state,
        rng_state=snapshot.rng_state,
    )

    with pytest.raises(recovery_state.RecoveryStateSchemaError, match="schema version"):
        recovery_state.restore_recovery_state(env, incompatible)


def test_restore_replays_the_same_next_observable_step(recovery_state) -> None:
    env = _FakeEnv()
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    snapshot = recovery_state.capture_recovery_state(env)

    expected = env.deterministic_step()
    env.deterministic_step()
    recovery_state.restore_recovery_state(env, snapshot)
    actual = env.deterministic_step()

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert actual[2] == expected[2]
    assert actual[3] == expected[3]
    torch.testing.assert_close(actual[4], expected[4])
    torch.testing.assert_close(actual[5], expected[5])
    np.testing.assert_array_equal(actual[6], expected[6])
