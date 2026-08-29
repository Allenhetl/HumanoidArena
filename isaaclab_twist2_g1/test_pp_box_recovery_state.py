from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

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
            "debug_score": 5,
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

    def capture_recovery_task_state(self) -> dict[str, object]:
        return self.recovery_task_state

    def restore_recovery_task_state(self, state: dict[str, object]) -> None:
        self.restore_events.append("task")
        assert int(self.episode_length_buf.item()) == 7
        self.recovery_task_state = state

    def deterministic_step(self) -> dict[str, Any]:
        python_draw = random.random()
        numpy_draw = float(np.random.random())
        torch_draw = torch.rand(2)
        task_draw = torch.rand(2, generator=self.task_generator)
        wrapper_draw = self.wrapper_rng.random(2)

        self.joint_pos.add_(0.25 + python_draw * 0.01)
        self.box_root_state[:, :3] += numpy_draw * 0.01
        self.episode_length_buf.add_(1)
        self.common_step_counter += 1
        self.episode_count += 1
        self.reset_count.add_(self.reset_buf.to(torch.int64))
        self.recovery_task_state["debug_score"] += 1
        hands = self.recovery_task_state["hands_materialized"]
        observation = torch.tensor(
            [
                float(self.joint_pos.sum()),
                float(self.box_root_state[:, :3].sum()),
                float(self.episode_length_buf.item()),
                float(self.common_step_counter),
                float(self.episode_count),
                float(self.reset_count.item()),
                float(self.recovery_task_state["debug_score"]),
                float(hands.to(torch.float32).sum()),
                python_draw,
                numpy_draw,
                float(torch_draw.sum()),
                float(task_draw.sum()),
                float(wrapper_draw.sum()),
            ],
            dtype=torch.float64,
        )
        reward = float(observation.sum())
        terminal = (
            "success"
            if self.recovery_task_state["fsm_stage"] == "grasp"
            and bool(hands[0, 0])
            and not bool(self.reset_buf[0])
            else "running"
        )
        return {
            "scene": {
                "joint_pos": self.joint_pos.clone(),
                "box_root_state": self.box_root_state.copy(),
            },
            "counters": {
                "episode_length_buf": self.episode_length_buf.clone(),
                "common_step_counter": self.common_step_counter,
                "reset_buf": self.reset_buf.clone(),
                "episode_count": self.episode_count,
                "reset_count": self.reset_count.clone(),
            },
            "task_state": {
                "fsm_stage": self.recovery_task_state["fsm_stage"],
                "debug_score": self.recovery_task_state["debug_score"],
                "hands_materialized": hands.clone(),
            },
            "rng": {
                "python": python_draw,
                "numpy": numpy_draw,
                "torch": torch_draw,
                "task": task_draw,
                "wrapper": wrapper_draw,
            },
            "observation": observation,
            "reward": reward,
            "terminal": terminal,
        }


class _CacheEnv(_FakeEnv):
    def capture_physics_solver_cache(self) -> dict[str, int]:
        return {"solver_epoch": 4}

    def restore_physics_solver_cache(self, state: dict[str, int]) -> None:
        self.restore_events.append(f"solver:{state['solver_epoch']}")


class _BothCacheEnv(_CacheEnv):
    def capture_contact_cache(self) -> dict[str, int]:
        return {"contact_epoch": 6}

    def restore_contact_cache(self, state: dict[str, int]) -> None:
        self.restore_events.append(f"contact:{state['contact_epoch']}")


class _CaptureOnlyTaskStateEnv(_FakeEnv):
    restore_recovery_task_state = None

    def __init__(self) -> None:
        super().__init__()
        del self.recovery_task_state

    def capture_recovery_task_state(self) -> dict[str, str]:
        return {"fsm_stage": "capture-only"}


class _DirectTaskStateEnv(_FakeEnv):
    capture_recovery_task_state = None
    restore_recovery_task_state = None


def _global_rng_state() -> tuple[object, tuple[object, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_global_rng_state_equal(
    actual: tuple[object, tuple[object, ...], torch.Tensor],
    expected: tuple[object, tuple[object, ...], torch.Tensor],
) -> None:
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    torch.testing.assert_close(actual[2], expected[2])


def _env_state(env: _FakeEnv) -> dict[str, Any]:
    task_state = getattr(env, "recovery_task_state", None)
    if task_state is not None:
        task_state = {
            "fsm_stage": task_state["fsm_stage"],
            "debug_score": task_state["debug_score"],
            "hands_materialized": task_state["hands_materialized"].clone(),
        }
    return {
        "joint_pos": env.joint_pos.clone(),
        "box_root_state": env.box_root_state.copy(),
        "episode_length_buf": env.episode_length_buf.clone(),
        "common_step_counter": env.common_step_counter,
        "reset_buf": env.reset_buf.clone(),
        "episode_count": env.episode_count,
        "reset_count": env.reset_count.clone(),
        "task_state": task_state,
        "task_rng": env.task_generator.get_state().clone(),
        "wrapper_rng": dict(env.wrapper_rng.bit_generator.state),
        "restore_events": list(env.restore_events),
    }


def _assert_env_state_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    torch.testing.assert_close(actual["joint_pos"], expected["joint_pos"])
    np.testing.assert_array_equal(actual["box_root_state"], expected["box_root_state"])
    torch.testing.assert_close(actual["episode_length_buf"], expected["episode_length_buf"])
    assert actual["common_step_counter"] == expected["common_step_counter"]
    torch.testing.assert_close(actual["reset_buf"], expected["reset_buf"])
    assert actual["episode_count"] == expected["episode_count"]
    torch.testing.assert_close(actual["reset_count"], expected["reset_count"])
    if expected["task_state"] is None:
        assert actual["task_state"] is None
    else:
        assert actual["task_state"]["fsm_stage"] == expected["task_state"]["fsm_stage"]
        assert actual["task_state"]["debug_score"] == expected["task_state"]["debug_score"]
        torch.testing.assert_close(
            actual["task_state"]["hands_materialized"],
            expected["task_state"]["hands_materialized"],
        )
    torch.testing.assert_close(actual["task_rng"], expected["task_rng"])
    assert actual["wrapper_rng"] == expected["wrapper_rng"]
    assert actual["restore_events"] == expected["restore_events"]


def _assert_transition_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    torch.testing.assert_close(actual["scene"]["joint_pos"], expected["scene"]["joint_pos"])
    np.testing.assert_array_equal(
        actual["scene"]["box_root_state"], expected["scene"]["box_root_state"]
    )
    for name in ("episode_length_buf", "reset_buf", "reset_count"):
        torch.testing.assert_close(actual["counters"][name], expected["counters"][name])
    for name in ("common_step_counter", "episode_count"):
        assert actual["counters"][name] == expected["counters"][name]
    assert actual["task_state"]["fsm_stage"] == expected["task_state"]["fsm_stage"]
    assert actual["task_state"]["debug_score"] == expected["task_state"]["debug_score"]
    torch.testing.assert_close(
        actual["task_state"]["hands_materialized"],
        expected["task_state"]["hands_materialized"],
    )
    assert actual["rng"]["python"] == expected["rng"]["python"]
    assert actual["rng"]["numpy"] == expected["rng"]["numpy"]
    torch.testing.assert_close(actual["rng"]["torch"], expected["rng"]["torch"])
    torch.testing.assert_close(actual["rng"]["task"], expected["rng"]["task"])
    np.testing.assert_array_equal(actual["rng"]["wrapper"], expected["rng"]["wrapper"])
    torch.testing.assert_close(actual["observation"], expected["observation"])
    assert actual["reward"] == expected["reward"]
    assert actual["terminal"] == expected["terminal"]


@pytest.fixture()
def recovery_state():
    return _load_recovery_state_module()


def test_snapshot_schema_is_versioned_and_task_bound(recovery_state) -> None:
    env = _FakeEnv()

    snapshot = recovery_state.capture_recovery_state(env)

    assert snapshot.schema_version == recovery_state.RECOVERY_STATE_SCHEMA_VERSION == 1
    assert snapshot.task_identity == recovery_state.PP_BOX_TASK_IDENTITY
    assert snapshot.capabilities.schema_version == 1


def test_capability_mapping_is_structurally_immutable(recovery_state) -> None:
    snapshot = recovery_state.capture_recovery_state(_FakeEnv())

    with pytest.raises(TypeError):
        snapshot.capabilities.available["scene_state"] = False


def test_restore_rejects_unknown_capability_schema_before_mutation(recovery_state) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    incompatible = replace(
        snapshot,
        capabilities=replace(snapshot.capabilities, schema_version=999),
    )
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateSchemaError, match="capability schema"):
        recovery_state.restore_recovery_state(env, incompatible)

    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


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


def test_capture_only_task_state_hook_is_not_a_capability(recovery_state) -> None:
    env = _CaptureOnlyTaskStateEnv()

    capabilities = recovery_state.discover_recovery_state_capabilities(env)

    assert capabilities.available["task_state"] is False
    assert capabilities.task_state_mode is None
    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.capture_recovery_state(env, required_capabilities={"task_state"})
    assert exc_info.value.missing_capabilities == ("task_state",)


def test_restore_never_creates_fallback_task_state_attribute(recovery_state) -> None:
    source = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(source)
    target = _CaptureOnlyTaskStateEnv()
    env_before = _env_state(target)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.restore_recovery_state(target, snapshot)

    assert exc_info.value.missing_capabilities == ("task_state",)
    assert not hasattr(target, "recovery_task_state")
    _assert_env_state_equal(_env_state(target), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_direct_task_state_attribute_uses_symmetric_direct_mode(recovery_state) -> None:
    env = _DirectTaskStateEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env.recovery_task_state = {
        "fsm_stage": "corrupted",
        "debug_score": -1,
        "hands_materialized": torch.tensor([[False, False]]),
    }

    recovery_state.restore_recovery_state(env, snapshot)

    assert snapshot.capabilities.task_state_mode == "direct"
    assert env.recovery_task_state["fsm_stage"] == "grasp"
    assert env.recovery_task_state["debug_score"] == 5
    torch.testing.assert_close(
        env.recovery_task_state["hands_materialized"],
        torch.tensor([[True, False]]),
    )


def test_snapshot_without_task_state_can_restore_into_richer_runtime(recovery_state) -> None:
    snapshot = recovery_state.capture_recovery_state(_CaptureOnlyTaskStateEnv())
    target = _FakeEnv()

    recovery_state.restore_recovery_state(target, snapshot)

    assert target.restore_events == ["scene"]
    assert target.recovery_task_state["fsm_stage"] == "grasp"


@pytest.mark.parametrize(
    "payload_name",
    [
        "scene_state",
        "scene_key:robot",
        "scene_key:box",
        "scene_shape",
        "scene_nonfinite",
        "task_counters",
        "counter_shape",
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
        "task_local_rng",
        "wrapper_rng",
        "task_state",
        "task_state_shape",
        "episode_length_buf",
        "common_step_counter",
        "reset_buf",
        "episode_count",
        "reset_count",
    ],
)
def test_incomplete_mandatory_payload_is_rejected_before_any_mutation(
    recovery_state,
    payload_name: str,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    if payload_name == "scene_state":
        incomplete = replace(snapshot, scene_state={})
        expected_missing = "scene_state"
    elif payload_name.startswith("scene_key:"):
        scene_key = payload_name.split(":", maxsplit=1)[1]
        scene_state = dict(snapshot.scene_state)
        scene_state.pop(scene_key)
        incomplete = replace(snapshot, scene_state=scene_state)
        expected_missing = payload_name
    elif payload_name == "scene_shape":
        scene_state = dict(snapshot.scene_state)
        scene_state["robot"] = {"joint_pos": torch.zeros(3, dtype=torch.float32)}
        incomplete = replace(snapshot, scene_state=scene_state)
        expected_missing = "scene_state"
    elif payload_name == "scene_nonfinite":
        scene_state = dict(snapshot.scene_state)
        box_state = snapshot.scene_state["box"]["root_state"].copy()
        box_state[0, 0] = np.nan
        scene_state["box"] = {"root_state": box_state}
        incomplete = replace(snapshot, scene_state=scene_state)
        expected_missing = "scene_state"
    elif payload_name == "task_counters":
        incomplete = replace(snapshot, task_counters=None)
        expected_missing = "task_counters"
    elif payload_name == "counter_shape":
        counters = dict(snapshot.task_counters)
        counters["episode_length_buf"] = torch.tensor([7, 8], dtype=torch.int64)
        incomplete = replace(snapshot, task_counters=counters)
        expected_missing = "task_counter:episode_length_buf"
    elif payload_name == "python_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(snapshot.rng_state, python=("invalid",)),
        )
        expected_missing = "python_rng"
    elif payload_name == "numpy_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(snapshot.rng_state, numpy=("invalid",)),
        )
        expected_missing = "numpy_rng"
    elif payload_name == "torch_cpu_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(
                snapshot.rng_state,
                torch_cpu=torch.empty(0, dtype=torch.uint8),
            ),
        )
        expected_missing = "torch_cpu_rng"
    elif payload_name == "task_local_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(
                snapshot.rng_state,
                task_local=replace(
                    snapshot.rng_state.task_local,
                    state=torch.empty(0, dtype=torch.uint8),
                ),
            ),
        )
        expected_missing = "task_local_rng"
    elif payload_name == "wrapper_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(
                snapshot.rng_state,
                wrapper=replace(snapshot.rng_state.wrapper, state={"invalid": True}),
            ),
        )
        expected_missing = "wrapper_rng"
    elif payload_name == "task_state":
        incomplete = replace(snapshot, task_state=None)
        expected_missing = "task_state"
    elif payload_name == "task_state_shape":
        task_state = dict(snapshot.task_state)
        task_state["hands_materialized"] = torch.tensor([True, False])
        incomplete = replace(snapshot, task_state=task_state)
        expected_missing = "task_state"
    else:
        counters = dict(snapshot.task_counters)
        counters.pop(payload_name)
        incomplete = replace(snapshot, task_counters=counters)
        expected_missing = f"task_counter:{payload_name}"
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.restore_recovery_state(env, incomplete)

    assert expected_missing in exc_info.value.missing_capabilities
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_missing_rng_record_reports_every_advertised_rng_before_mutation(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    incomplete = replace(snapshot, rng_state=None)
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.restore_recovery_state(env, incomplete)

    assert set(exc_info.value.missing_capabilities) >= {
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
        "task_local_rng",
        "wrapper_rng",
    }
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


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


@pytest.mark.parametrize(
    "runtime_state",
    [None, ["physics_solver_cache", "contact_cache"]],
)
def test_restore_rejects_non_mapping_runtime_cache_state_before_any_mutation(
    recovery_state,
    runtime_state: object,
) -> None:
    env = _BothCacheEnv()
    snapshot = recovery_state.capture_recovery_state(
        env,
        required_capabilities={"physics_solver_cache", "contact_cache"},
    )
    incomplete = replace(snapshot, runtime_state=runtime_state)
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.restore_recovery_state(env, incomplete)

    assert exc_info.value.missing_capabilities == (
        "contact_cache",
        "physics_solver_cache",
    )
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


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
    assert snapshot.capabilities.task_state_mode == "hooks"
    assert snapshot.capabilities.scene_state_keys == ("robot", "box")
    assert snapshot.capabilities.counter_names == (
        "episode_length_buf",
        "common_step_counter",
        "reset_buf",
        "episode_count",
        "reset_count",
    )
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
    env.joint_pos.fill_(-10.0)
    env.box_root_state.fill(-20.0)
    env.episode_length_buf.fill_(90)
    env.common_step_counter = 91
    env.reset_buf.fill_(True)
    env.episode_count = 92
    env.reset_count.fill_(93)
    env.recovery_task_state = {
        "fsm_stage": "corrupted",
        "debug_score": -94,
        "hands_materialized": torch.tensor([[False, False]]),
    }
    recovery_state.restore_recovery_state(env, snapshot)
    actual = env.deterministic_step()

    _assert_transition_equal(actual, expected)
