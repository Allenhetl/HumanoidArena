"""Versioned recovery snapshots for the HOI pick-and-place box task.

This module intentionally depends only on NumPy and Torch. Isaac Lab objects are
accessed through their runtime protocols so the snapshot contract remains
testable without launching the simulator.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import random
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

__all__ = [
    "PP_BOX_TASK_IDENTITY",
    "RECOVERY_DIGEST_SCHEMA_VERSION",
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
    "recovery_state_digest",
    "recovery_value_digest",
    "restore_recovery_state",
]

RECOVERY_DIGEST_SCHEMA_VERSION = 1
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
    scene_state_keys: tuple[str, ...] = ()
    scene_state_schema: Any = None
    counter_names: tuple[str, ...] = ()
    counter_schemas: tuple[tuple[str, Any], ...] = ()
    task_state_mode: str | None = None
    task_state_schema: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", MappingProxyType(dict(self.available)))
        object.__setattr__(self, "scene_state_keys", tuple(self.scene_state_keys))
        object.__setattr__(self, "counter_names", tuple(self.counter_names))
        object.__setattr__(self, "counter_schemas", tuple(self.counter_schemas))


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


def _digest_blob(hasher: Any, tag: bytes, payload: bytes) -> None:
    hasher.update(struct.pack("!H", len(tag)))
    hasher.update(tag)
    hasher.update(struct.pack("!Q", len(payload)))
    hasher.update(payload)


def _mapping_key_sort_key(value: Any) -> bytes:
    if value is None:
        return b"n"
    if type(value) is bool:
        return b"b1" if value else b"b0"
    if type(value) is int:
        return b"i" + str(value).encode("ascii")
    if type(value) is float:
        return b"f" + struct.pack("!d", value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"s" + struct.pack("!Q", len(raw)) + raw
    if isinstance(value, bytes):
        return b"y" + struct.pack("!Q", len(value)) + value
    if isinstance(value, tuple):
        items = tuple(_mapping_key_sort_key(item) for item in value)
        return b"t" + b"".join(struct.pack("!Q", len(item)) + item for item in items)
    raise RecoveryStateSchemaError(
        f"unsupported recovery mapping key type: {type(value).__name__}"
    )


def _update_recovery_digest(hasher: Any, value: Any, active_ids: set[int]) -> None:
    if value is None:
        _digest_blob(hasher, b"none", b"")
        return
    if type(value) is bool:
        _digest_blob(hasher, b"bool", b"1" if value else b"0")
        return
    if type(value) is int:
        _digest_blob(hasher, b"int", str(value).encode("ascii"))
        return
    if type(value) is float:
        _digest_blob(hasher, b"float64", struct.pack("!d", value))
        return
    if isinstance(value, str):
        _digest_blob(hasher, b"str", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _digest_blob(hasher, b"bytes", value)
        return
    if isinstance(value, np.generic):
        _update_recovery_digest(hasher, np.asarray(value), active_ids)
        return
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise RecoveryStateSchemaError(
                "recovery digest requires a dense non-quantized tensor"
            )
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        metadata = (
            str(value.dtype),
            tuple(value.shape),
        )
        _digest_blob(hasher, b"torch-metadata", repr(metadata).encode("ascii"))
        _digest_blob(hasher, b"torch-data", raw)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise RecoveryStateSchemaError(
                "recovery digest does not accept object-dtype arrays"
            )
        array = np.ascontiguousarray(value)
        metadata = (array.dtype.str, tuple(array.shape))
        _digest_blob(hasher, b"numpy-metadata", repr(metadata).encode("ascii"))
        _digest_blob(hasher, b"numpy-data", array.tobytes(order="C"))
        return

    is_recursive = is_dataclass(value) or isinstance(
        value,
        (Mapping, list, tuple, set, frozenset),
    )
    value_id = id(value)
    if is_recursive:
        if value_id in active_ids:
            raise RecoveryStateSchemaError(
                "recovery digest does not accept cyclic values"
            )
        active_ids.add(value_id)
    try:
        if is_dataclass(value):
            _digest_blob(
                hasher,
                b"dataclass",
                type(value).__qualname__.encode("utf-8"),
            )
            for item in fields(value):
                _digest_blob(hasher, b"field", item.name.encode("utf-8"))
                _update_recovery_digest(hasher, getattr(value, item.name), active_ids)
            return
        if isinstance(value, Mapping):
            _digest_blob(hasher, b"mapping-size", str(len(value)).encode("ascii"))
            for key in sorted(value, key=_mapping_key_sort_key):
                _update_recovery_digest(hasher, key, active_ids)
                _update_recovery_digest(hasher, value[key], active_ids)
            return
        if isinstance(value, list):
            _digest_blob(hasher, b"list-size", str(len(value)).encode("ascii"))
            for item in value:
                _update_recovery_digest(hasher, item, active_ids)
            return
        if isinstance(value, tuple):
            _digest_blob(hasher, b"tuple-size", str(len(value)).encode("ascii"))
            for item in value:
                _update_recovery_digest(hasher, item, active_ids)
            return
        if isinstance(value, (set, frozenset)):
            tag = b"frozenset" if isinstance(value, frozenset) else b"set"
            _digest_blob(hasher, tag + b"-size", str(len(value)).encode("ascii"))
            for item in sorted(value, key=_mapping_key_sort_key):
                _update_recovery_digest(hasher, item, active_ids)
            return
    finally:
        if is_recursive:
            active_ids.remove(value_id)
    raise RecoveryStateSchemaError(
        f"unsupported recovery digest value type: {type(value).__name__}"
    )


def _recovery_digest(value: Any, *, domain: bytes) -> str:
    hasher = hashlib.sha256()
    _digest_blob(
        hasher,
        b"digest-schema",
        str(RECOVERY_DIGEST_SCHEMA_VERSION).encode("ascii"),
    )
    _digest_blob(hasher, b"domain", domain)
    _update_recovery_digest(hasher, value, set())
    return hasher.hexdigest()


def recovery_value_digest(value: Any) -> str:
    """Return a stable typed SHA-256 digest for recovery continuation payloads."""

    return _recovery_digest(value, domain=b"recovery-value")


def recovery_state_digest(
    snapshot: RecoveryStateSnapshot,
    *,
    task_identity: str = PP_BOX_TASK_IDENTITY,
) -> str:
    """Bind a digest to every versioned snapshot payload before restoration."""

    if not isinstance(snapshot, RecoveryStateSnapshot):
        raise RecoveryStateSchemaError("recovery state digest requires a snapshot")
    _validate_snapshot_schema(snapshot, task_identity)
    return _recovery_digest(snapshot, domain=b"recovery-state-snapshot")


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


def _has_writable_direct_task_state(env: Any) -> bool:
    name = "recovery_task_state"
    try:
        inspect.getattr_static(env, name)
    except AttributeError:
        return False

    descriptor = inspect.getattr_static(type(env), name, None)
    if isinstance(descriptor, property):
        return descriptor.fset is not None
    if descriptor is not None and hasattr(descriptor, "__set__"):
        return True
    try:
        return name in vars(env)
    except TypeError:
        return False


def _task_state_mode(env: Any) -> str | None:
    if _has_paired_hooks(env, "recovery_task_state"):
        return "hooks"
    if _has_writable_direct_task_state(env):
        return "direct"
    return None


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
    task_state_mode = _task_state_mode(env)
    cuda_rng = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    counter_names = tuple(
        name
        for name in _CORE_TASK_COUNTER_NAMES + _OPTIONAL_TASK_COUNTER_NAMES
        if hasattr(env, name)
    )
    available = {name: False for name in _KNOWN_CAPABILITIES}
    available.update(
        {
            "scene_state": has_scene,
            "task_counters": all(hasattr(env, name) for name in _CORE_TASK_COUNTER_NAMES),
            "task_state": task_state_mode is not None,
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
        counter_names=counter_names,
        task_state_mode=task_state_mode,
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
    scene_state = clone_recovery_value(env.scene.get_state())
    scene_state_keys = tuple(scene_state) if isinstance(scene_state, Mapping) else ()

    task_state = None
    if capabilities.task_state_mode == "hooks":
        task_state = env.capture_recovery_task_state()
    elif capabilities.task_state_mode == "direct":
        task_state = env.recovery_task_state
    task_state = clone_recovery_value(task_state)
    task_counters = {
        name: clone_recovery_value(getattr(env, name)) for name in capabilities.counter_names
    }
    capabilities = replace(
        capabilities,
        scene_state_keys=scene_state_keys,
        scene_state_schema=_payload_schema(scene_state),
        counter_schemas=tuple(
            (name, _payload_schema(value)) for name, value in task_counters.items()
        ),
        task_state_schema=(
            _payload_schema(task_state) if capabilities.available["task_state"] else None
        ),
    )

    runtime_state: dict[str, Any] = {}
    for name in _RUNTIME_CACHE_NAMES:
        if capabilities.available[name]:
            runtime_state[name] = getattr(env, f"capture_{name}")()

    torch_cuda = None
    if capabilities.available["torch_cuda_rng"]:
        torch_cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    snapshot = RecoveryStateSnapshot(
        schema_version=RECOVERY_STATE_SCHEMA_VERSION,
        task_identity=task_identity,
        capabilities=capabilities,
        scene_state=scene_state,
        task_counters=task_counters,
        task_state=task_state,
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
    missing = _missing_snapshot_payloads(snapshot, env)
    if missing:
        raise RecoveryStateIncompleteError(
            missing,
            operation="capture payload",
            available=capabilities.available,
        )
    return snapshot


def _restore_attribute(owner: Any, name: str, saved: Any) -> None:
    if not hasattr(owner, name):
        raise RecoveryStateIncompleteError(
            {name},
            operation="restore",
            available={name: False},
        )
    current = getattr(owner, name)
    if isinstance(current, torch.Tensor) and isinstance(saved, torch.Tensor):
        current.copy_(saved)
    elif isinstance(current, np.ndarray) and isinstance(saved, np.ndarray):
        np.copyto(current, saved)
    else:
        setattr(owner, name, clone_recovery_value(saved))


def _payload_schema(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return ("torch", tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, np.ndarray):
        return ("numpy", tuple(value.shape), str(value.dtype))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple((key, _payload_schema(item)) for key, item in value.items()),
        )
    if isinstance(value, list):
        return ("list", tuple(_payload_schema(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_payload_schema(item) for item in value))
    return ("scalar", type(value).__module__, type(value).__qualname__)


def _payload_is_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return bool(torch.isfinite(value).all())
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact):
            return bool(np.isfinite(value).all())
        return True
    if isinstance(value, Mapping):
        return all(_payload_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_payload_is_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _payload_matches_schema(value: Any, schema: Any) -> bool:
    return _payload_schema(value) == schema and _payload_is_finite(value)


def _valid_python_rng_state(state: Any) -> bool:
    try:
        random.Random().setstate(clone_recovery_value(state))
    except (TypeError, ValueError):
        return False
    return True


def _valid_numpy_rng_state(state: Any) -> bool:
    try:
        np.random.RandomState().set_state(clone_recovery_value(state))
    except (TypeError, ValueError):
        return False
    return True


def _valid_torch_cpu_rng_state(state: Any) -> bool:
    if not isinstance(state, torch.Tensor):
        return False
    if state.device.type != "cpu" or state.dtype != torch.uint8 or state.ndim != 1:
        return False
    if state.numel() == 0:
        return False
    try:
        torch.Generator(device="cpu").set_state(state.clone())
    except RuntimeError:
        return False
    return True


def _valid_torch_cuda_rng_states(states: Any) -> bool:
    if not isinstance(states, (tuple, list)) or not states:
        return False
    if len(states) != torch.cuda.device_count():
        return False
    try:
        for device_index, state in enumerate(states):
            if not (
                isinstance(state, torch.Tensor)
                and state.device.type == "cpu"
                and state.dtype == torch.uint8
                and state.ndim == 1
                and state.numel() > 0
            ):
                return False
            torch.Generator(device=f"cuda:{device_index}").set_state(state.clone())
    except RuntimeError:
        return False
    return True


def _valid_rng_source_state(source: Any | None, snapshot: Any) -> bool:
    if not isinstance(snapshot, RngSourceState):
        return False
    try:
        if snapshot.kind == "torch" and isinstance(source, torch.Generator):
            torch.Generator(device=source.device).set_state(snapshot.state.clone())
            return True
        if snapshot.kind == "numpy_generator" and isinstance(source, np.random.Generator):
            bit_generator = copy.deepcopy(source.bit_generator)
            bit_generator.state = clone_recovery_value(snapshot.state)
            return True
        if snapshot.kind == "numpy_random_state" and isinstance(source, np.random.RandomState):
            np.random.RandomState().set_state(clone_recovery_value(snapshot.state))
            return True
        if snapshot.kind == "python_random" and isinstance(source, random.Random):
            random.Random().setstate(clone_recovery_value(snapshot.state))
            return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    return False


def _missing_snapshot_payloads(snapshot: RecoveryStateSnapshot, env: Any) -> set[str]:
    missing: set[str] = set()
    if not isinstance(snapshot.scene_state, Mapping) or not snapshot.scene_state:
        missing.add("scene_state")
    else:
        missing.update(
            f"scene_key:{name}"
            for name in snapshot.capabilities.scene_state_keys
            if name not in snapshot.scene_state
        )
        if not _payload_matches_schema(
            snapshot.scene_state,
            snapshot.capabilities.scene_state_schema,
        ):
            missing.add("scene_state")
        elif _payload_schema(env.scene.get_state()) != snapshot.capabilities.scene_state_schema:
            missing.add("scene_state")

    counter_names = snapshot.capabilities.counter_names
    counter_schemas = dict(snapshot.capabilities.counter_schemas)
    if snapshot.capabilities.available.get("task_counters", False):
        if not isinstance(snapshot.task_counters, Mapping):
            missing.add("task_counters")
        else:
            for name in dict.fromkeys(_CORE_TASK_COUNTER_NAMES + counter_names):
                schema = counter_schemas.get(name)
                if (
                    name not in snapshot.task_counters
                    or schema is None
                    or not _payload_matches_schema(snapshot.task_counters[name], schema)
                    or _payload_schema(getattr(env, name, None)) != schema
                ):
                    missing.add(f"task_counter:{name}")

    if not isinstance(snapshot.rng_state, RecoveryRngState):
        missing.update({"python_rng", "numpy_rng", "torch_cpu_rng"})
        for name in ("torch_cuda_rng", "task_local_rng", "wrapper_rng"):
            if snapshot.capabilities.available.get(name, False):
                missing.add(name)
    else:
        if not _valid_python_rng_state(snapshot.rng_state.python):
            missing.add("python_rng")
        if not _valid_numpy_rng_state(snapshot.rng_state.numpy):
            missing.add("numpy_rng")
        if not _valid_torch_cpu_rng_state(snapshot.rng_state.torch_cpu):
            missing.add("torch_cpu_rng")
        if snapshot.capabilities.available.get("torch_cuda_rng", False) and not (
            _valid_torch_cuda_rng_states(snapshot.rng_state.torch_cuda)
        ):
            missing.add("torch_cuda_rng")
        if snapshot.capabilities.available.get("task_local_rng", False) and not (
            _valid_rng_source_state(_task_rng(env), snapshot.rng_state.task_local)
        ):
            missing.add("task_local_rng")
        if snapshot.capabilities.available.get("wrapper_rng", False) and not (
            _valid_rng_source_state(_wrapper_rng(env), snapshot.rng_state.wrapper)
        ):
            missing.add("wrapper_rng")
    if snapshot.capabilities.available.get("task_state", False):
        if snapshot.task_state is None or (
            isinstance(snapshot.task_state, Mapping) and not snapshot.task_state
        ):
            missing.add("task_state")
        elif not _payload_matches_schema(
            snapshot.task_state,
            snapshot.capabilities.task_state_schema,
        ):
            missing.add("task_state")
    advertised_caches = {
        name
        for name in _RUNTIME_CACHE_NAMES
        if snapshot.capabilities.available.get(name, False)
    }
    if advertised_caches and not isinstance(snapshot.runtime_state, Mapping):
        missing.update(advertised_caches)
    else:
        missing.update(name for name in advertised_caches if name not in snapshot.runtime_state)
    return missing


def _validate_snapshot_schema(snapshot: RecoveryStateSnapshot, task_identity: str) -> None:
    if snapshot.schema_version != RECOVERY_STATE_SCHEMA_VERSION:
        raise RecoveryStateSchemaError(
            f"unsupported recovery state schema version {snapshot.schema_version}; "
            f"expected {RECOVERY_STATE_SCHEMA_VERSION}"
        )
    if snapshot.capabilities.schema_version != RECOVERY_STATE_SCHEMA_VERSION:
        raise RecoveryStateSchemaError(
            f"unsupported recovery capability schema version "
            f"{snapshot.capabilities.schema_version}; expected {RECOVERY_STATE_SCHEMA_VERSION}"
        )
    if snapshot.task_identity != task_identity:
        raise RecoveryStateSchemaError(
            f"recovery state task identity {snapshot.task_identity!r} does not match {task_identity!r}"
        )
    task_state_available = snapshot.capabilities.available.get("task_state", False)
    if task_state_available != (snapshot.capabilities.task_state_mode in {"hooks", "direct"}):
        raise RecoveryStateSchemaError("recovery capability task-state mode is inconsistent")


def restore_recovery_state(
    env: Any,
    snapshot: RecoveryStateSnapshot,
    *,
    required_capabilities: set[str] | frozenset[str] | None = None,
    task_identity: str = PP_BOX_TASK_IDENTITY,
) -> None:
    """Restore a snapshot in scene, task, then RNG order."""

    _validate_snapshot_schema(snapshot, task_identity)

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
    if snapshot.capabilities.available.get("task_state", False) and (
        snapshot.capabilities.task_state_mode != capabilities.task_state_mode
    ):
        missing_from_snapshot.add("task_state")
    current_counter_names = set(capabilities.counter_names)
    missing_from_snapshot.update(
        f"task_counter:{name}"
        for name in snapshot.capabilities.counter_names
        if name not in current_counter_names
    )
    missing_from_snapshot.update(_missing_snapshot_payloads(snapshot, env))
    if missing_from_snapshot:
        raise RecoveryStateIncompleteError(
            missing_from_snapshot,
            operation="restore snapshot",
            available=snapshot.capabilities.available,
        )

    env.reset_to(clone_recovery_value(snapshot.scene_state))
    for name in snapshot.capabilities.counter_names:
        _restore_attribute(env, name, snapshot.task_counters[name])

    if snapshot.capabilities.task_state_mode == "hooks":
        env.restore_recovery_task_state(clone_recovery_value(snapshot.task_state))
    elif snapshot.capabilities.task_state_mode == "direct":
        _restore_attribute(env, "recovery_task_state", snapshot.task_state)

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
