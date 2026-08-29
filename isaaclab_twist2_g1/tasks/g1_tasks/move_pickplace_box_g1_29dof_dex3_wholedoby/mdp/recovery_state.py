"""Versioned recovery snapshots for the HOI pick-and-place box task.

This module intentionally depends only on NumPy and Torch. Isaac Lab objects are
accessed through their runtime protocols so the snapshot contract remains
testable without launching the simulator.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch


__all__ = [
    "PP_BOX_TASK_IDENTITY",
    "RECOVERY_STATE_SCHEMA_VERSION",
    "RecoveryRngState",
    "RecoveryStateCapabilities",
    "RecoveryStateIncompleteError",
    "RecoveryStateSchemaError",
    "RecoveryStateSnapshot",
    "RngSourceState",
    "capture_recovery_state",
    "clone_recovery_value",
    "discover_recovery_state_capabilities",
    "restore_recovery_state",
]

RECOVERY_STATE_SCHEMA_VERSION = 1
PP_BOX_TASK_IDENTITY = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"

_CORE_TASK_COUNTER_NAMES = (
    "episode_length_buf",
    "common_step_counter",
    "reset_buf",
)
_OPTIONAL_TASK_COUNTER_NAMES = (
    "episode_count",
    "episode_counter",
    "reset_count",
    "reset_counter",
)
_DEFAULT_REQUIRED_CAPABILITIES = frozenset(
    {
        "scene_state",
        "task_counters",
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
    }
)
_KNOWN_CAPABILITIES = (
    "scene_state",
    "task_counters",
    "task_state",
    "python_rng",
    "numpy_rng",
    "torch_cpu_rng",
    "torch_cuda_rng",
    "task_local_rng",
    "wrapper_rng",
    "physics_solver_cache",
    "contact_cache",
)
_RUNTIME_CACHE_NAMES = ("physics_solver_cache", "contact_cache")


class RecoveryStateSchemaError(ValueError):
    """Raised when a snapshot does not match the supported schema or task."""


class RecoveryStateIncompleteError(RuntimeError):
    """Raised when required runtime state cannot be captured or restored."""

    def __init__(
        self,
        missing_capabilities: set[str] | frozenset[str] | tuple[str, ...],
        *,
        operation: str,
        available: Mapping[str, bool],
    ) -> None:
        self.missing_capabilities = tuple(sorted(missing_capabilities))
        self.operation = operation
        self.available = dict(available)
        missing = ", ".join(self.missing_capabilities)
        super().__init__(
            f"recovery state {operation} is incomplete; missing capabilities: {missing}"
        )


@dataclass(frozen=True)
class RecoveryStateCapabilities:
    """Versioned capability report for one simulator environment instance."""

    schema_version: int
    available: Mapping[str, bool]


@dataclass(frozen=True)
class RngSourceState:
    """Serialized state for one task-local or wrapper RNG object."""

    kind: str
    state: Any


@dataclass(frozen=True)
class RecoveryRngState:
    """Global and environment-local RNG states needed for continuation."""

    python: Any
    numpy: Any
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...] | None
    task_local: RngSourceState | None
    wrapper: RngSourceState | None


@dataclass(frozen=True)
class RecoveryStateSnapshot:
    """A task-bound simulator snapshot with an explicit schema version."""

    schema_version: int
    task_identity: str
    capabilities: RecoveryStateCapabilities
    scene_state: Mapping[str, Any]
    task_counters: Mapping[str, Any]
    task_state: Any
    rng_state: RecoveryRngState
    runtime_state: Mapping[str, Any] = field(default_factory=dict)


def clone_recovery_value(value: Any) -> Any:
    """Deep-clone nested state while preserving tensor and array values."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {
            clone_recovery_value(key): clone_recovery_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [clone_recovery_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_recovery_value(item) for item in value)
    if isinstance(value, set):
        return {clone_recovery_value(item) for item in value}
    return copy.deepcopy(value)


def _has_paired_hooks(env: Any, name: str) -> bool:
    return callable(getattr(env, f"capture_{name}", None)) and callable(
        getattr(env, f"restore_{name}", None)
    )


def _task_rng(env: Any) -> Any | None:
    for name in ("task_generator", "generator"):
        source = getattr(env, name, None)
        if source is not None:
            return source
    return None


def _wrapper_rng(env: Any) -> Any | None:
    source = getattr(env, "wrapper_rng", None)
    if source is not None:
        return source
    wrapped = getattr(env, "env", None)
    if wrapped is not None:
        return getattr(wrapped, "np_random", None)
    return None


def _supports_rng(source: Any | None) -> bool:
    return isinstance(
        source,
        (torch.Generator, np.random.Generator, np.random.RandomState, random.Random),
    )


def discover_recovery_state_capabilities(env: Any) -> RecoveryStateCapabilities:
    """Report state that this runtime exposes without importing Isaac Lab."""

    scene = getattr(env, "scene", None)
    has_scene = callable(getattr(scene, "get_state", None)) and callable(
        getattr(env, "reset_to", None)
    )
    task_state = callable(getattr(env, "capture_recovery_task_state", None)) or hasattr(
        env, "recovery_task_state"
    )
    cuda_rng = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    available = {name: False for name in _KNOWN_CAPABILITIES}
    available.update(
        {
            "scene_state": has_scene,
            "task_counters": all(hasattr(env, name) for name in _CORE_TASK_COUNTER_NAMES),
            "task_state": task_state,
            "python_rng": True,
            "numpy_rng": True,
            "torch_cpu_rng": True,
            "torch_cuda_rng": cuda_rng,
            "task_local_rng": _supports_rng(_task_rng(env)),
            "wrapper_rng": _supports_rng(_wrapper_rng(env)),
            "physics_solver_cache": _has_paired_hooks(env, "physics_solver_cache"),
            "contact_cache": _has_paired_hooks(env, "contact_cache"),
        }
    )
    return RecoveryStateCapabilities(
        schema_version=RECOVERY_STATE_SCHEMA_VERSION,
        available=available,
    )


def _capture_rng_source(source: Any | None) -> RngSourceState | None:
    if isinstance(source, torch.Generator):
        return RngSourceState("torch", source.get_state().clone())
    if isinstance(source, np.random.Generator):
        return RngSourceState("numpy_generator", clone_recovery_value(source.bit_generator.state))
    if isinstance(source, np.random.RandomState):
        return RngSourceState("numpy_random_state", clone_recovery_value(source.get_state()))
    if isinstance(source, random.Random):
        return RngSourceState("python_random", clone_recovery_value(source.getstate()))
    return None


def _restore_rng_source(
    source: Any | None,
    snapshot: RngSourceState | None,
    *,
    capability: str,
) -> None:
    if snapshot is None:
        return
    if snapshot.kind == "torch" and isinstance(source, torch.Generator):
        source.set_state(snapshot.state.clone())
        return
    if snapshot.kind == "numpy_generator" and isinstance(source, np.random.Generator):
        source.bit_generator.state = clone_recovery_value(snapshot.state)
        return
    if snapshot.kind == "numpy_random_state" and isinstance(source, np.random.RandomState):
        source.set_state(clone_recovery_value(snapshot.state))
        return
    if snapshot.kind == "python_random" and isinstance(source, random.Random):
        source.setstate(clone_recovery_value(snapshot.state))
        return
    raise RecoveryStateIncompleteError(
        {capability},
        operation="restore",
        available={capability: False},
    )


def _require_capabilities(
    capabilities: RecoveryStateCapabilities,
    required_capabilities: set[str] | frozenset[str] | None,
    *,
    operation: str,
) -> frozenset[str]:
    required = _DEFAULT_REQUIRED_CAPABILITIES | frozenset(required_capabilities or ())
    missing = {name for name in required if not capabilities.available.get(name, False)}
    if missing:
        raise RecoveryStateIncompleteError(
            missing,
            operation=operation,
            available=capabilities.available,
        )
    return required


def capture_recovery_state(
    env: Any,
    *,
    required_capabilities: set[str] | frozenset[str] | None = None,
    task_identity: str = PP_BOX_TASK_IDENTITY,
) -> RecoveryStateSnapshot:
    """Capture state needed to replay the task's next observable transition."""

    capabilities = discover_recovery_state_capabilities(env)
    _require_capabilities(capabilities, required_capabilities, operation="capture")

    task_state = None
    if capabilities.available["task_state"]:
        capture_task_state = getattr(env, "capture_recovery_task_state", None)
        if callable(capture_task_state):
            task_state = capture_task_state()
        else:
            task_state = getattr(env, "recovery_task_state")

    runtime_state: dict[str, Any] = {}
    for name in _RUNTIME_CACHE_NAMES:
        if capabilities.available[name]:
            runtime_state[name] = getattr(env, f"capture_{name}")()

    torch_cuda = None
    if capabilities.available["torch_cuda_rng"]:
        torch_cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    return RecoveryStateSnapshot(
        schema_version=RECOVERY_STATE_SCHEMA_VERSION,
        task_identity=task_identity,
        capabilities=capabilities,
        scene_state=clone_recovery_value(env.scene.get_state()),
        task_counters={
            name: clone_recovery_value(getattr(env, name))
            for name in _CORE_TASK_COUNTER_NAMES + _OPTIONAL_TASK_COUNTER_NAMES
            if hasattr(env, name)
        },
        task_state=clone_recovery_value(task_state),
        rng_state=RecoveryRngState(
            python=clone_recovery_value(random.getstate()),
            numpy=clone_recovery_value(np.random.get_state()),
            torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=torch_cuda,
            task_local=_capture_rng_source(_task_rng(env)),
            wrapper=_capture_rng_source(_wrapper_rng(env)),
        ),
        runtime_state=clone_recovery_value(runtime_state),
    )


def _restore_attribute(owner: Any, name: str, saved: Any) -> None:
    current = getattr(owner, name, None)
    if isinstance(current, torch.Tensor) and isinstance(saved, torch.Tensor):
        current.copy_(saved)
    elif isinstance(current, np.ndarray) and isinstance(saved, np.ndarray):
        np.copyto(current, saved)
    else:
        setattr(owner, name, clone_recovery_value(saved))


def _missing_snapshot_payloads(snapshot: RecoveryStateSnapshot) -> set[str]:
    missing: set[str] = set()
    if not isinstance(snapshot.rng_state.torch_cpu, torch.Tensor):
        missing.add("torch_cpu_rng")
    if snapshot.capabilities.available.get("torch_cuda_rng", False):
        if snapshot.rng_state.torch_cuda is None:
            missing.add("torch_cuda_rng")
    if snapshot.capabilities.available.get("task_local_rng", False):
        if snapshot.rng_state.task_local is None:
            missing.add("task_local_rng")
    if snapshot.capabilities.available.get("wrapper_rng", False):
        if snapshot.rng_state.wrapper is None:
            missing.add("wrapper_rng")
    for name in _RUNTIME_CACHE_NAMES:
        if snapshot.capabilities.available.get(name, False) and name not in snapshot.runtime_state:
            missing.add(name)
    return missing


def restore_recovery_state(
    env: Any,
    snapshot: RecoveryStateSnapshot,
    *,
    required_capabilities: set[str] | frozenset[str] | None = None,
    task_identity: str = PP_BOX_TASK_IDENTITY,
) -> None:
    """Restore a snapshot in scene, task, then RNG order."""

    if snapshot.schema_version != RECOVERY_STATE_SCHEMA_VERSION:
        raise RecoveryStateSchemaError(
            f"unsupported recovery state schema version {snapshot.schema_version}; "
            f"expected {RECOVERY_STATE_SCHEMA_VERSION}"
        )
    if snapshot.task_identity != task_identity:
        raise RecoveryStateSchemaError(
            f"recovery state task identity {snapshot.task_identity!r} does not match {task_identity!r}"
        )

    capabilities = discover_recovery_state_capabilities(env)
    snapshot_capabilities = {
        name for name, is_available in snapshot.capabilities.available.items() if is_available
    }
    required = _require_capabilities(
        capabilities,
        snapshot_capabilities | frozenset(required_capabilities or ()),
        operation="restore",
    )
    missing_from_snapshot = {
        name for name in required if not snapshot.capabilities.available.get(name, False)
    }
    missing_from_snapshot.update(_missing_snapshot_payloads(snapshot))
    if missing_from_snapshot:
        raise RecoveryStateIncompleteError(
            missing_from_snapshot,
            operation="restore snapshot",
            available=snapshot.capabilities.available,
        )

    env.reset_to(clone_recovery_value(snapshot.scene_state))
    for name, value in snapshot.task_counters.items():
        _restore_attribute(env, name, value)

    if snapshot.capabilities.available.get("task_state", False):
        restore_task_state = getattr(env, "restore_recovery_task_state", None)
        if callable(restore_task_state):
            restore_task_state(clone_recovery_value(snapshot.task_state))
        else:
            env.recovery_task_state = clone_recovery_value(snapshot.task_state)

    for name in _RUNTIME_CACHE_NAMES:
        if snapshot.capabilities.available.get(name, False):
            getattr(env, f"restore_{name}")(clone_recovery_value(snapshot.runtime_state[name]))

    random.setstate(clone_recovery_value(snapshot.rng_state.python))
    np.random.set_state(clone_recovery_value(snapshot.rng_state.numpy))
    torch.set_rng_state(snapshot.rng_state.torch_cpu.clone())
    if snapshot.rng_state.torch_cuda is not None:
        torch.cuda.set_rng_state_all([state.clone() for state in snapshot.rng_state.torch_cuda])
    _restore_rng_source(
        _task_rng(env),
        snapshot.rng_state.task_local,
        capability="task_local_rng",
    )
    _restore_rng_source(
        _wrapper_rng(env),
        snapshot.rng_state.wrapper,
        capability="wrapper_rng",
    )
