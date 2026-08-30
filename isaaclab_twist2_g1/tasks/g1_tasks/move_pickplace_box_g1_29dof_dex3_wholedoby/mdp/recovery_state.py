"""Versioned recovery snapshots for the HOI pick-and-place box task.

This module intentionally depends only on NumPy and Torch. Isaac Lab objects are
accessed through their runtime protocols so the snapshot contract remains
testable without launching the simulator.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import inspect
import random
import struct
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType, MethodType
from typing import Any

import numpy as np
import torch

__all__ = [
    "PP_BOX_TASK_IDENTITY",
    "PP_BOX_TASK_STATE_SCHEMA_VERSION",
    "RECOVERY_DIGEST_SCHEMA_VERSION",
    "RECOVERY_STATE_SCHEMA_VERSION",
    "RecoveryRngState",
    "RecoveryStateCapabilities",
    "RecoveryStateCoordinator",
    "RecoveryStateDigestMismatchError",
    "RecoveryStateGlobalRngBusyError",
    "RecoveryStateIncompleteError",
    "RecoveryStatePreflightError",
    "RecoveryStateRestoreEvidence",
    "RecoveryStateSchemaError",
    "RecoveryStateScope",
    "RecoveryStateSnapshot",
    "RecoveryStateTransactionError",
    "RngSourceState",
    "capture_recovery_state",
    "clone_recovery_value",
    "discover_recovery_state_capabilities",
    "install_pp_box_recovery_task_state_hooks",
    "recovery_state_digest",
    "recovery_value_digest",
    "restore_recovery_state",
]

RECOVERY_DIGEST_SCHEMA_VERSION = 1
RECOVERY_STATE_SCHEMA_VERSION = 3
PP_BOX_TASK_IDENTITY = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"
PP_BOX_TASK_STATE_SCHEMA_VERSION = 3
PP_BOX_ACTION_MANAGER_STATE_SCHEMA_VERSION = 1

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
        "process_global_rng_exclusive",
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
    "process_global_rng_exclusive",
    "task_local_rng",
    "wrapper_rng",
    "physics_solver_cache",
    "contact_cache",
    "action_manager_state",
    "action_provider_state",
    "controller_state",
)
_RUNTIME_CACHE_NAMES = ("physics_solver_cache", "contact_cache")
_CONTROL_PARTICIPANTS = (
    ("action_manager_state", "action_manager", "recovery_action_manager_state"),
    ("action_provider_state", "action_provider", "recovery_provider_state"),
    ("controller_state", "recovery_controller", "recovery_controller_state"),
)
_EXACT_CONTINUATION_CAPABILITIES = frozenset(
    {
        "task_state",
        "task_local_rng",
        "wrapper_rng",
        "physics_solver_cache",
        "contact_cache",
        "action_manager_state",
        "action_provider_state",
        "controller_state",
    }
)
_FIDELITY_TIERS = frozenset({"state_only", "exact_continuation"})
_PROCESS_GLOBAL_RNG_LOCK = threading.Lock()
_PP_BOX_CFG_STATE_NAMES = (
    "_episode_runtime_seed",
    "_episode_object_seed_counter",
    "_current_episode_object_seed",
    "_current_episode_object_seed_source",
    "_replay_initial_env_state_active",
)
_PP_BOX_PASSIVE_MANAGER_NAMES = (
    "command_manager",
    "curriculum_manager",
    "event_manager",
    "recorder_manager",
)
_PP_BOX_BASE_OBSERVATION_TERMS = (
    "robot_joint_state",
    "robot_gipper_state",
    "camera_image",
)
_PP_BOX_RLINF_OBSERVATION_TERMS = _PP_BOX_BASE_OBSERVATION_TERMS + (
    "vla_state64",
    "vla_front_rgb",
    "up_alignment",
    "critical_contact_force_max",
    "critical_contact_force_available",
)
_PP_BOX_OBSERVATION_TERM_CONTRACTS = frozenset(
    {_PP_BOX_BASE_OBSERVATION_TERMS, _PP_BOX_RLINF_OBSERVATION_TERMS}
)


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


class RecoveryStateDigestMismatchError(RecoveryStateSchemaError):
    """Raised before mutation when the caller-bound snapshot digest differs."""


class RecoveryStateGlobalRngBusyError(RuntimeError):
    """Raised when another recovery operation owns process-global RNG state."""


class RecoveryStatePreflightError(RuntimeError):
    """Raised when a restore hook rejects state before any environment mutation."""

    def __init__(self, capability: str, cause: Exception) -> None:
        self.capability = capability
        self.cause_type = type(cause).__name__
        self.cause_message = str(cause)
        super().__init__(
            f"recovery restore preflight failed for {capability}: "
            f"{self.cause_type}: {self.cause_message}"
        )


@dataclass(frozen=True)
class RecoveryStateRestoreEvidence:
    """Structured evidence for a failed target restore and its rollback attempt."""

    schema_version: int
    target_snapshot_digest: str
    rollback_snapshot_digest: str
    failure_phase: str
    failure_type: str
    failure_message: str
    rollback_succeeded: bool
    rollback_failure_type: str | None = None
    rollback_failure_message: str | None = None


class RecoveryStateTransactionError(RuntimeError):
    """Raised after a mutating restore fails, with rollback outcome attached."""

    def __init__(self, evidence: RecoveryStateRestoreEvidence) -> None:
        self.evidence = evidence
        rollback = "succeeded" if evidence.rollback_succeeded else "failed"
        super().__init__(
            f"recovery restore transaction failed during {evidence.failure_phase}; "
            f"rollback {rollback}; target={evidence.target_snapshot_digest}; "
            f"rollback_snapshot={evidence.rollback_snapshot_digest}"
        )


@contextmanager
def _exclusive_process_global_rng(operation: str):
    if not _PROCESS_GLOBAL_RNG_LOCK.acquire(blocking=False):
        raise RecoveryStateGlobalRngBusyError(
            f"process-global RNG state is already owned during recovery {operation}"
        )
    try:
        yield
    finally:
        _PROCESS_GLOBAL_RNG_LOCK.release()


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
class RecoveryStateScope:
    """Digest-bound Isaac Lab vector scope for process-global recovery state."""

    schema_version: int
    num_envs: int
    env_ids: tuple[int, ...]
    is_relative: bool
    process_global_rng_scope: str


@dataclass(frozen=True)
class RecoveryStateSnapshot:
    """A task-bound simulator snapshot with an explicit schema version."""

    schema_version: int
    task_identity: str
    scope: RecoveryStateScope
    fidelity_tier: str
    capabilities: RecoveryStateCapabilities
    scene_state: Mapping[str, Any]
    task_counters: Mapping[str, Any]
    task_state: Any
    rng_state: RecoveryRngState
    runtime_state: Mapping[str, Any] = field(default_factory=dict)


class RecoveryStateCoordinator:
    """Task-bound production entry point for transactional recovery snapshots."""

    __slots__ = ("_coordinator_type", "_env", "_task_identity")

    SCHEMA_VERSION = 1

    def __init__(self, env: Any, *, task_identity: str) -> None:
        self._env = env
        self._task_identity = task_identity
        self._coordinator_type = _type_identity(self)

    def _validate_binding(self) -> None:
        expected_type = _type_identity(self)
        if (
            type(self) is not RecoveryStateCoordinator
            or self._coordinator_type != expected_type
            or getattr(self._env, "recovery_state_coordinator", None) is not self
        ):
            raise RecoveryStateSchemaError(
                "recovery state coordinator binding mismatch"
            )
        _validate_env_task_identity(self._env, self._task_identity)

    @property
    def binding_identity(self) -> Mapping[str, Any]:
        self._validate_binding()
        return MappingProxyType(
            {
                "schema_version": self.SCHEMA_VERSION,
                "coordinator_type": self._coordinator_type,
                "task_identity": self._task_identity,
            }
        )

    def capture(
        self,
        *,
        fidelity_tier: str,
        required_capabilities: set[str] | frozenset[str] | None = None,
    ) -> RecoveryStateSnapshot:
        self._validate_binding()
        return capture_recovery_state(
            self._env,
            required_capabilities=required_capabilities,
            task_identity=self._task_identity,
            fidelity_tier=fidelity_tier,
        )

    def digest(self, snapshot: RecoveryStateSnapshot) -> str:
        self._validate_binding()
        _validate_snapshot_schema(snapshot, self._task_identity)
        return recovery_state_digest(snapshot, task_identity=self._task_identity)

    def preflight(
        self,
        snapshot: RecoveryStateSnapshot,
        *,
        snapshot_digest: str,
        required_capabilities: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._validate_binding()
        _validate_bound_snapshot_digest(
            snapshot,
            snapshot_digest,
            task_identity=self._task_identity,
        )
        with _exclusive_process_global_rng("preflight"):
            _preflight_recovery_restore(
                self._env,
                snapshot,
                required_capabilities=required_capabilities,
                task_identity=self._task_identity,
            )

    def restore(
        self,
        snapshot: RecoveryStateSnapshot,
        *,
        snapshot_digest: str,
        required_capabilities: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._validate_binding()
        restore_recovery_state(
            self._env,
            snapshot,
            snapshot_digest=snapshot_digest,
            required_capabilities=required_capabilities,
            task_identity=self._task_identity,
        )


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


def _has_transactional_hooks(env: Any, name: str) -> bool:
    return all(
        callable(getattr(env, method_name, None))
        for method_name in (
            f"capture_{name}",
            f"preflight_restore_{name}",
            f"restore_{name}",
        )
    )


def _control_participant(env: Any, owner_name: str) -> Any | None:
    return _optional_runtime_attribute(env, owner_name)


def _required_for_fidelity(fidelity_tier: str) -> frozenset[str]:
    if fidelity_tier not in _FIDELITY_TIERS:
        raise RecoveryStateSchemaError(
            f"unsupported recovery fidelity tier {fidelity_tier!r}"
        )
    if fidelity_tier == "exact_continuation":
        return _EXACT_CONTINUATION_CAPABILITIES
    return frozenset()


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
    if _has_transactional_hooks(env, "recovery_task_state"):
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


def _optional_runtime_attribute(owner: Any, name: str) -> Any | None:
    try:
        return getattr(owner, name)
    except (AttributeError, RecursionError):
        return None


def _unwrapped_env(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        unwrapped = _optional_runtime_attribute(current, "unwrapped")
        if unwrapped is None or unwrapped is current or id(unwrapped) in visited:
            return current
        current = unwrapped
    return current


def _nested_envs(env: Any):
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = _optional_runtime_attribute(current, "env")


def _wrapper_rng(env: Any) -> Any | None:
    unwrapped = _unwrapped_env(env)
    for name in ("np_random", "wrapper_rng"):
        source = _optional_runtime_attribute(unwrapped, name)
        if _supports_rng(source):
            return source
    for nested in _nested_envs(env):
        if nested is unwrapped:
            continue
        for name in ("np_random", "wrapper_rng"):
            source = _optional_runtime_attribute(nested, name)
            if _supports_rng(source):
                return source
    return None


def _supports_rng(source: Any | None) -> bool:
    return isinstance(
        source,
        (torch.Generator, np.random.Generator, np.random.RandomState, random.Random),
    )


def _single_env_ids(env: Any, *, operation: str) -> torch.Tensor:
    missing: set[str] = set()
    num_envs = _optional_runtime_attribute(env, "num_envs")
    if (
        isinstance(num_envs, bool)
        or not isinstance(num_envs, (int, np.integer))
        or int(num_envs) != 1
    ):
        missing.add("num_envs")

    env_ids = None
    device = _optional_runtime_attribute(env, "device")
    if device is None:
        missing.add("device")
    else:
        try:
            torch_device = torch.device(device)
            if torch_device.type not in {"cpu", "cuda"}:
                raise ValueError(
                    f"unsupported recovery device type: {torch_device.type}"
                )
            env_ids = torch.tensor([0], dtype=torch.long, device=torch_device)
        except (RuntimeError, TypeError, ValueError):
            missing.add("device")

    if missing:
        raise RecoveryStateIncompleteError(
            missing,
            operation=operation,
            available={
                "num_envs": "num_envs" not in missing,
                "device": "device" not in missing,
            },
        )
    assert env_ids is not None
    return env_ids


def _single_env_scope(
    env: Any, *, operation: str
) -> tuple[RecoveryStateScope, torch.Tensor]:
    env_ids = _single_env_ids(env, operation=operation)
    return (
        RecoveryStateScope(
            schema_version=RECOVERY_STATE_SCHEMA_VERSION,
            num_envs=1,
            env_ids=(0,),
            is_relative=True,
            process_global_rng_scope="exclusive_process_single_env",
        ),
        env_ids,
    )


def _scene_state(env: Any, *, is_relative: bool) -> Any:
    return env.scene.get_state(is_relative=is_relative)


def _validate_env_task_identity(env: Any, task_identity: str) -> None:
    runtime_env = _unwrapped_env(env)
    cfg = _optional_runtime_attribute(runtime_env, "cfg")
    if cfg is None:
        raise RecoveryStateSchemaError("recovery runtime cfg is unavailable")
    env_name = _optional_runtime_attribute(cfg, "env_name")
    if env_name != task_identity:
        raise RecoveryStateSchemaError(
            f"recovery runtime cfg.env_name {env_name!r} does not match {task_identity!r}"
        )
    configured_identity = _optional_runtime_attribute(cfg, "recovery_task_identity")
    if configured_identity != task_identity:
        raise RecoveryStateSchemaError(
            "recovery runtime cfg.recovery_task_identity "
            f"{configured_identity!r} does not match {task_identity!r}"
        )


def _validate_reset_to_signature(env: Any, env_ids: torch.Tensor) -> None:
    reset_to = _optional_runtime_attribute(env, "reset_to")
    if not callable(reset_to):
        missing = {"reset_to_signature"}
    else:
        try:
            inspect.signature(reset_to).bind({}, env_ids=env_ids, is_relative=True)
            missing = set()
        except (TypeError, ValueError):
            missing = {"reset_to_signature"}
    if missing:
        raise RecoveryStateIncompleteError(
            missing,
            operation="restore preflight",
            available={"reset_to_signature": False},
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
            "task_counters": all(
                hasattr(env, name) for name in _CORE_TASK_COUNTER_NAMES
            ),
            "task_state": task_state_mode is not None,
            "python_rng": True,
            "numpy_rng": True,
            "torch_cpu_rng": True,
            "torch_cuda_rng": cuda_rng,
            "process_global_rng_exclusive": (
                getattr(env, "recovery_process_global_rng_exclusive", None) is True
            ),
            "task_local_rng": _supports_rng(_task_rng(env)),
            "wrapper_rng": _supports_rng(_wrapper_rng(env)),
            "physics_solver_cache": _has_transactional_hooks(
                env, "physics_solver_cache"
            ),
            "contact_cache": _has_transactional_hooks(env, "contact_cache"),
        }
    )
    for capability, owner_name, hook_name in _CONTROL_PARTICIPANTS:
        owner = _control_participant(env, owner_name)
        supported = owner is not None and _has_transactional_hooks(owner, hook_name)
        support_probe = (
            None if owner is None else getattr(owner, f"{hook_name}_supported", None)
        )
        if supported and callable(support_probe):
            try:
                supported = support_probe() is True
            except Exception:  # noqa: BLE001 - capability discovery must fail closed.
                supported = False
        available[capability] = supported
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
        return RngSourceState(
            "numpy_generator", clone_recovery_value(source.bit_generator.state)
        )
    if isinstance(source, np.random.RandomState):
        return RngSourceState(
            "numpy_random_state", clone_recovery_value(source.get_state())
        )
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
    if snapshot.kind == "numpy_random_state" and isinstance(
        source, np.random.RandomState
    ):
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
    fidelity_tier: str = "state_only",
) -> RecoveryStateSnapshot:
    """Capture a digest-bound state tier; only exact tier claims continuation."""

    with _exclusive_process_global_rng("capture"):
        return _capture_recovery_state_unlocked(
            env,
            required_capabilities=required_capabilities,
            task_identity=task_identity,
            fidelity_tier=fidelity_tier,
        )


def _capture_recovery_state_unlocked(
    env: Any,
    *,
    required_capabilities: set[str] | frozenset[str] | None = None,
    task_identity: str = PP_BOX_TASK_IDENTITY,
    fidelity_tier: str = "state_only",
) -> RecoveryStateSnapshot:
    """Capture while the caller owns ``_PROCESS_GLOBAL_RNG_LOCK``."""

    _validate_env_task_identity(env, task_identity)
    scope, _ = _single_env_scope(env, operation="capture single-env selection")
    capabilities = discover_recovery_state_capabilities(env)
    fidelity_required = _required_for_fidelity(fidelity_tier)
    _require_capabilities(
        capabilities,
        fidelity_required | frozenset(required_capabilities or ()),
        operation=f"capture {fidelity_tier}",
    )
    scene_state = clone_recovery_value(_scene_state(env, is_relative=scope.is_relative))
    scene_state_keys = tuple(scene_state) if isinstance(scene_state, Mapping) else ()

    task_state = None
    if capabilities.task_state_mode == "hooks":
        task_state = env.capture_recovery_task_state()
    elif capabilities.task_state_mode == "direct":
        task_state = env.recovery_task_state
    task_state = clone_recovery_value(task_state)
    task_counters = {
        name: clone_recovery_value(getattr(env, name))
        for name in capabilities.counter_names
    }
    capabilities = replace(
        capabilities,
        scene_state_keys=scene_state_keys,
        scene_state_schema=_payload_schema(scene_state),
        counter_schemas=tuple(
            (name, _payload_schema(value)) for name, value in task_counters.items()
        ),
        task_state_schema=(
            _payload_schema(task_state)
            if capabilities.available["task_state"]
            else None
        ),
    )

    runtime_state: dict[str, Any] = {}
    for name in _RUNTIME_CACHE_NAMES:
        if capabilities.available[name]:
            runtime_state[name] = getattr(env, f"capture_{name}")()
    for capability, owner_name, hook_name in _CONTROL_PARTICIPANTS:
        if capabilities.available[capability]:
            owner = _control_participant(env, owner_name)
            assert owner is not None
            try:
                runtime_state[capability] = getattr(owner, f"capture_{hook_name}")()
            except Exception as exc:
                raise RecoveryStateIncompleteError(
                    {capability},
                    operation="capture control participant",
                    available=capabilities.available,
                ) from exc

    torch_cuda = None
    if capabilities.available["torch_cuda_rng"]:
        torch_cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    snapshot = RecoveryStateSnapshot(
        schema_version=RECOVERY_STATE_SCHEMA_VERSION,
        task_identity=task_identity,
        scope=scope,
        fidelity_tier=fidelity_tier,
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


def _resolve_restore_attribute_owner(owner: Any, name: str) -> Any:
    if isinstance(owner, dict):
        return owner
    try:
        if (
            name in vars(owner)
            or inspect.getattr_static(type(owner), name, None) is not None
        ):
            return owner
    except TypeError:
        pass
    unwrapped = _unwrapped_env(owner)
    if unwrapped is not owner and hasattr(unwrapped, name):
        return unwrapped
    return owner


def _attribute_is_restorable(owner: Any, name: str, saved: Any) -> bool:
    owner = _resolve_restore_attribute_owner(owner, name)
    if isinstance(owner, dict):
        return name in owner
    if not hasattr(owner, name):
        return False
    current = getattr(owner, name)
    if isinstance(current, torch.Tensor) and isinstance(saved, torch.Tensor):
        return True
    if isinstance(current, np.ndarray) and isinstance(saved, np.ndarray):
        return True
    descriptor = inspect.getattr_static(type(owner), name, None)
    if isinstance(descriptor, property):
        return descriptor.fset is not None
    if descriptor is not None and hasattr(descriptor, "__set__"):
        return True
    try:
        return name in vars(owner)
    except TypeError:
        return False


def _restore_attribute(owner: Any, name: str, saved: Any) -> None:
    owner = _resolve_restore_attribute_owner(owner, name)
    is_dict = isinstance(owner, dict)
    if (is_dict and name not in owner) or (not is_dict and not hasattr(owner, name)):
        raise RecoveryStateIncompleteError(
            {name},
            operation="restore",
            available={name: False},
        )
    current = owner[name] if is_dict else getattr(owner, name)
    if isinstance(current, torch.Tensor) and isinstance(saved, torch.Tensor):
        current.copy_(saved)
    elif isinstance(current, np.ndarray) and isinstance(saved, np.ndarray):
        np.copyto(current, saved)
    elif is_dict:
        owner[name] = clone_recovery_value(saved)
    else:
        setattr(owner, name, clone_recovery_value(saved))


def _raise_missing_task_state(path: str, *, operation: str) -> None:
    raise RecoveryStateIncompleteError(
        {f"task_state:{path}"},
        operation=operation,
        available={f"task_state:{path}": False},
    )


def _require_runtime_attribute(
    owner: Any,
    name: str,
    *,
    path: str,
    operation: str,
) -> Any:
    if owner is None or not hasattr(owner, name):
        _raise_missing_task_state(path, operation=operation)
    return getattr(owner, name)


def _type_identity(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalized_active_terms(manager: Any, *, path: str) -> Any:
    active = _require_runtime_attribute(
        manager,
        "active_terms",
        path=f"{path}.active_terms",
        operation="capture task state",
    )
    if isinstance(active, Mapping):
        return tuple((str(mode), tuple(names)) for mode, names in active.items())
    return tuple(active)


def _pp_box_robot(env: Any, *, operation: str) -> Any:
    scene = _require_runtime_attribute(
        env,
        "scene",
        path="scene",
        operation=operation,
    )
    try:
        return scene["robot"]
    except (KeyError, TypeError):
        robot = getattr(scene, "robot", None)
        if robot is None:
            _raise_missing_task_state("scene.robot", operation=operation)
        return robot


def _pp_box_runtime_identity(env: Any) -> Mapping[str, Any]:
    operation = "capture task state"
    cfg = _require_runtime_attribute(env, "cfg", path="cfg", operation=operation)
    identity = getattr(cfg, "recovery_task_identity", None)
    if identity != PP_BOX_TASK_IDENTITY:
        raise RecoveryStateSchemaError(
            f"PP-box recovery task identity {identity!r} does not match "
            f"{PP_BOX_TASK_IDENTITY!r}"
        )

    action_manager = _require_runtime_attribute(
        env, "action_manager", path="action_manager", operation=operation
    )
    action_names = tuple(
        _require_runtime_attribute(
            action_manager,
            "_term_names",
            path="action_manager._term_names",
            operation=operation,
        )
    )
    action_terms = _require_runtime_attribute(
        action_manager,
        "_terms",
        path="action_manager._terms",
        operation=operation,
    )
    if action_names != ("joint_pos",) or tuple(action_terms) != action_names:
        _raise_missing_task_state("action_manager.term_contract", operation=operation)
    action_term = action_terms["joint_pos"]
    if type(action_term).__name__ != "JointPositionAction":
        _raise_missing_task_state("action_manager.joint_pos_type", operation=operation)

    reward_manager = _require_runtime_attribute(
        env, "reward_manager", path="reward_manager", operation=operation
    )
    reward_names = tuple(
        _require_runtime_attribute(
            reward_manager,
            "_term_names",
            path="reward_manager._term_names",
            operation=operation,
        )
    )
    if reward_names != ("reward",):
        _raise_missing_task_state("reward_manager.term_contract", operation=operation)
    if tuple(getattr(reward_manager, "_class_term_cfgs", ())) != ():
        _raise_missing_task_state("reward_manager.class_terms", operation=operation)

    termination_manager = _require_runtime_attribute(
        env, "termination_manager", path="termination_manager", operation=operation
    )
    termination_names = tuple(
        _require_runtime_attribute(
            termination_manager,
            "_term_names",
            path="termination_manager._term_names",
            operation=operation,
        )
    )
    if (
        termination_names != ()
        or tuple(getattr(termination_manager, "_class_term_cfgs", ())) != ()
    ):
        _raise_missing_task_state(
            "termination_manager.term_contract", operation=operation
        )

    observation_manager = _require_runtime_attribute(
        env, "observation_manager", path="observation_manager", operation=operation
    )
    observation_terms = _require_runtime_attribute(
        observation_manager,
        "_group_obs_term_names",
        path="observation_manager._group_obs_term_names",
        operation=operation,
    )
    normalized_observation_terms = {
        str(group): tuple(names) for group, names in observation_terms.items()
    }
    if (
        set(normalized_observation_terms) != {"policy"}
        or normalized_observation_terms["policy"]
        not in _PP_BOX_OBSERVATION_TERM_CONTRACTS
    ):
        _raise_missing_task_state(
            "observation_manager.term_contract", operation=operation
        )
    history_buffers = _require_runtime_attribute(
        observation_manager,
        "_group_obs_term_history_buffer",
        path="observation_manager._group_obs_term_history_buffer",
        operation=operation,
    )
    if any(dict(group_buffers) for group_buffers in history_buffers.values()):
        _raise_missing_task_state("observation_manager.history", operation=operation)
    class_terms = _require_runtime_attribute(
        observation_manager,
        "_group_obs_class_term_cfgs",
        path="observation_manager._group_obs_class_term_cfgs",
        operation=operation,
    )
    if any(tuple(group_terms) for group_terms in class_terms.values()) or tuple(
        getattr(observation_manager, "_group_obs_class_instances", ())
    ):
        _raise_missing_task_state(
            "observation_manager.class_terms", operation=operation
        )

    passive_managers: dict[str, Any] = {}
    for manager_name in _PP_BOX_PASSIVE_MANAGER_NAMES:
        manager = _require_runtime_attribute(
            env, manager_name, path=manager_name, operation=operation
        )
        active = _normalized_active_terms(manager, path=manager_name)
        if isinstance(active, tuple) and any(
            (bool(item[1]) if isinstance(item, tuple) and len(item) == 2 else True)
            for item in active
        ):
            _raise_missing_task_state(
                f"{manager_name}.mutable_terms", operation=operation
            )
        passive_managers[manager_name] = {
            "type": _type_identity(manager),
            "active_terms": active,
        }

    robot = _pp_box_robot(env, operation=operation)
    return MappingProxyType(
        {
            "env_type": _type_identity(env),
            "task_identity": identity,
            "action_manager_type": _type_identity(action_manager),
            "action_terms": tuple(
                (
                    name,
                    _type_identity(action_terms[name]),
                    int(action_terms[name].action_dim),
                )
                for name in action_names
            ),
            "reward_manager_type": _type_identity(reward_manager),
            "reward_terms": reward_names,
            "termination_manager_type": _type_identity(termination_manager),
            "termination_terms": termination_names,
            "observation_manager_type": _type_identity(observation_manager),
            "observation_terms": normalized_observation_terms,
            "passive_managers": passive_managers,
            "robot_type": _type_identity(robot),
        }
    )


def _require_alias_if_present(
    env: Any, env_name: str, owner: Any, owner_name: str
) -> None:
    """Reject an independent driver cache without inventing optional aliases."""

    driver_value = _optional_runtime_attribute(env, env_name)
    if driver_value is not None and driver_value is not getattr(
        owner, owner_name, None
    ):
        _raise_missing_task_state(
            f"driver_alias:{env_name}->{owner_name}",
            operation="capture task state",
        )


def _capture_driver_observation_state(
    env: Any, observation_manager: Any
) -> tuple[str, Any]:
    manager_value = _require_runtime_attribute(
        observation_manager,
        "_obs_buffer",
        path="observation_manager._obs_buffer",
        operation="capture task state",
    )
    driver_value = _optional_runtime_attribute(env, "obs_buf")
    if driver_value is None:
        return "absent", None
    mode = "manager_alias" if driver_value is manager_value else "independent"
    return mode, clone_recovery_value(driver_value)


def _capture_pp_box_recovery_task_state(env: Any) -> Mapping[str, Any]:
    runtime_identity = _pp_box_runtime_identity(env)
    operation = "capture task state"
    action_manager = env.action_manager
    reward_manager = env.reward_manager
    termination_manager = env.termination_manager
    observation_manager = env.observation_manager
    robot = _pp_box_robot(env, operation=operation)
    robot_data = _require_runtime_attribute(
        robot, "data", path="scene.robot.data", operation=operation
    )

    _require_alias_if_present(env, "reward_buf", reward_manager, "_reward_buf")
    _require_alias_if_present(
        env, "reset_terminated", termination_manager, "_terminated_buf"
    )
    _require_alias_if_present(
        env, "reset_time_outs", termination_manager, "_truncated_buf"
    )
    obs_buf_mode, driver_obs_buf = _capture_driver_observation_state(
        env, observation_manager
    )

    action_terms: dict[str, Any] = {}
    for name in action_manager._term_names:
        term = action_manager._terms[name]
        action_terms[name] = {
            "raw_actions": clone_recovery_value(
                _require_runtime_attribute(
                    term,
                    "_raw_actions",
                    path=f"action_manager.{name}._raw_actions",
                    operation=operation,
                )
            ),
            "processed_actions": clone_recovery_value(
                _require_runtime_attribute(
                    term,
                    "_processed_actions",
                    path=f"action_manager.{name}._processed_actions",
                    operation=operation,
                )
            ),
        }

    cfg_state = {
        name: clone_recovery_value(
            _require_runtime_attribute(
                env.cfg,
                name,
                path=f"cfg.{name}",
                operation=operation,
            )
        )
        for name in _PP_BOX_CFG_STATE_NAMES
    }
    payload = {
        "schema_version": PP_BOX_TASK_STATE_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "runtime_identity": dict(runtime_identity),
        "driver": {
            "sim_step_counter": clone_recovery_value(
                _require_runtime_attribute(
                    env,
                    "_sim_step_counter",
                    path="driver._sim_step_counter",
                    operation=operation,
                )
            ),
            "extras": clone_recovery_value(
                _require_runtime_attribute(
                    env, "extras", path="driver.extras", operation=operation
                )
            ),
            "obs_buf_mode": obs_buf_mode,
            "obs_buf": driver_obs_buf,
        },
        "action_manager": {
            "action": clone_recovery_value(action_manager._action),
            "prev_action": clone_recovery_value(action_manager._prev_action),
            "terms": action_terms,
        },
        "reward_manager": {
            "episode_sums": clone_recovery_value(reward_manager._episode_sums),
            "reward_buf": clone_recovery_value(reward_manager._reward_buf),
            "step_reward": clone_recovery_value(reward_manager._step_reward),
        },
        "termination_manager": {
            "term_dones": clone_recovery_value(termination_manager._term_dones),
            "truncated_buf": clone_recovery_value(termination_manager._truncated_buf),
            "terminated_buf": clone_recovery_value(termination_manager._terminated_buf),
        },
        "observation_manager": {
            "obs_buffer": clone_recovery_value(observation_manager._obs_buffer),
        },
        "robot_targets": {
            "joint_pos": clone_recovery_value(robot_data.joint_pos_target),
            "joint_vel": clone_recovery_value(robot_data.joint_vel_target),
            "joint_effort": clone_recovery_value(robot_data.joint_effort_target),
        },
        "task_cfg": cfg_state,
    }
    if not _payload_is_finite(payload):
        _raise_missing_task_state("nonfinite_payload", operation=operation)
    return payload


def _require_exact_mapping_keys(
    value: Any, expected: set[str], *, path: str
) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != expected:
        saved_keys = (
            tuple(sorted(value, key=_mapping_key_sort_key))
            if isinstance(value, Mapping)
            else (f"<{type(value).__module__}.{type(value).__qualname__}>",)
        )
        expected_keys = tuple(sorted(expected, key=_mapping_key_sort_key))
        raise RecoveryStateSchemaError(
            f"PP-box recovery task state {path} keys do not match schema: "
            f"saved_keys={saved_keys!r}, runtime_keys={expected_keys!r}"
        )
    return value


def _require_same_payload_schema(current: Any, saved: Any, *, path: str) -> None:
    if not _payload_matches_schema(saved, _payload_schema(current)):
        raise RecoveryStateSchemaError(
            f"PP-box recovery task state {path} does not match runtime schema"
        )


def _preflight_pp_box_recovery_task_state(env: Any, state: Any) -> None:
    top = _require_exact_mapping_keys(
        state,
        {
            "schema_version",
            "task_identity",
            "runtime_identity",
            "driver",
            "action_manager",
            "reward_manager",
            "termination_manager",
            "observation_manager",
            "robot_targets",
            "task_cfg",
        },
        path="root",
    )
    if top["schema_version"] != PP_BOX_TASK_STATE_SCHEMA_VERSION:
        raise RecoveryStateSchemaError("unsupported PP-box recovery task state schema")
    if top["task_identity"] != PP_BOX_TASK_IDENTITY:
        raise RecoveryStateSchemaError(
            "PP-box recovery task state task identity mismatch"
        )
    if dict(top["runtime_identity"]) != dict(_pp_box_runtime_identity(env)):
        raise RecoveryStateSchemaError("PP-box recovery runtime identity mismatch")

    driver = _require_exact_mapping_keys(
        top["driver"],
        {"sim_step_counter", "extras", "obs_buf_mode", "obs_buf"},
        path="driver",
    )
    if not isinstance(driver["sim_step_counter"], (int, np.integer)):
        raise RecoveryStateSchemaError("PP-box driver sim step counter is invalid")
    if not isinstance(driver["extras"], Mapping) or not _payload_is_finite(
        driver["extras"]
    ):
        raise RecoveryStateSchemaError("PP-box driver extras are invalid")
    obs_buf_mode = driver["obs_buf_mode"]
    driver_obs_buf = driver["obs_buf"]
    if obs_buf_mode not in {"absent", "manager_alias", "independent"}:
        raise RecoveryStateSchemaError("PP-box driver obs_buf mode is invalid")
    if obs_buf_mode == "absent":
        if driver_obs_buf is not None:
            raise RecoveryStateSchemaError(
                "PP-box absent driver obs_buf must not carry a payload"
            )
    else:
        if driver_obs_buf is None or not _payload_is_finite(driver_obs_buf):
            raise RecoveryStateSchemaError("PP-box driver obs_buf payload is invalid")
        current_obs_buf = (
            env.observation_manager._obs_buffer
            if obs_buf_mode == "manager_alias"
            else _optional_runtime_attribute(env, "obs_buf")
        )
        if current_obs_buf is None:
            raise RecoveryStateSchemaError(
                "PP-box driver obs_buf is absent from the live runtime"
            )
        _require_same_payload_schema(
            current_obs_buf,
            driver_obs_buf,
            path="driver.obs_buf",
        )

    action = _require_exact_mapping_keys(
        top["action_manager"], {"action", "prev_action", "terms"}, path="action_manager"
    )
    _require_same_payload_schema(
        env.action_manager._action, action["action"], path="action_manager.action"
    )
    _require_same_payload_schema(
        env.action_manager._prev_action,
        action["prev_action"],
        path="action_manager.prev_action",
    )
    terms = _require_exact_mapping_keys(
        action["terms"],
        set(env.action_manager._term_names),
        path="action_manager.terms",
    )
    for name in env.action_manager._term_names:
        term_state = _require_exact_mapping_keys(
            terms[name],
            {"raw_actions", "processed_actions"},
            path=f"action_manager.{name}",
        )
        term = env.action_manager._terms[name]
        _require_same_payload_schema(
            term._raw_actions,
            term_state["raw_actions"],
            path=f"action_manager.{name}.raw_actions",
        )
        _require_same_payload_schema(
            term._processed_actions,
            term_state["processed_actions"],
            path=f"action_manager.{name}.processed_actions",
        )

    reward = _require_exact_mapping_keys(
        top["reward_manager"],
        {"episode_sums", "reward_buf", "step_reward"},
        path="reward_manager",
    )
    episode_sums = _require_exact_mapping_keys(
        reward["episode_sums"],
        set(env.reward_manager._episode_sums),
        path="reward_manager.episode_sums",
    )
    for name, current in env.reward_manager._episode_sums.items():
        _require_same_payload_schema(
            current, episode_sums[name], path=f"reward_manager.episode_sums.{name}"
        )
    _require_same_payload_schema(
        env.reward_manager._reward_buf,
        reward["reward_buf"],
        path="reward_manager.reward_buf",
    )
    _require_same_payload_schema(
        env.reward_manager._step_reward,
        reward["step_reward"],
        path="reward_manager.step_reward",
    )

    termination = _require_exact_mapping_keys(
        top["termination_manager"],
        {"term_dones", "truncated_buf", "terminated_buf"},
        path="termination_manager",
    )
    _require_same_payload_schema(
        env.termination_manager._term_dones,
        termination["term_dones"],
        path="termination_manager.term_dones",
    )
    _require_same_payload_schema(
        env.termination_manager._truncated_buf,
        termination["truncated_buf"],
        path="termination_manager.truncated_buf",
    )
    _require_same_payload_schema(
        env.termination_manager._terminated_buf,
        termination["terminated_buf"],
        path="termination_manager.terminated_buf",
    )

    observation = _require_exact_mapping_keys(
        top["observation_manager"], {"obs_buffer"}, path="observation_manager"
    )
    _require_same_payload_schema(
        env.observation_manager._obs_buffer,
        observation["obs_buffer"],
        path="observation_manager.obs_buffer",
    )

    targets = _require_exact_mapping_keys(
        top["robot_targets"],
        {"joint_pos", "joint_vel", "joint_effort"},
        path="robot_targets",
    )
    robot = _pp_box_robot(env, operation="restore task state preflight")
    _require_same_payload_schema(
        robot.data.joint_pos_target,
        targets["joint_pos"],
        path="robot_targets.joint_pos",
    )
    _require_same_payload_schema(
        robot.data.joint_vel_target,
        targets["joint_vel"],
        path="robot_targets.joint_vel",
    )
    _require_same_payload_schema(
        robot.data.joint_effort_target,
        targets["joint_effort"],
        path="robot_targets.joint_effort",
    )

    task_cfg = _require_exact_mapping_keys(
        top["task_cfg"], set(_PP_BOX_CFG_STATE_NAMES), path="task_cfg"
    )
    if not _payload_is_finite(task_cfg):
        raise RecoveryStateSchemaError("PP-box recovery task cfg state is invalid")


def _restore_pp_box_recovery_task_state(env: Any, state: Any) -> None:
    _preflight_pp_box_recovery_task_state(env, state)
    action = state["action_manager"]
    _restore_attribute(env.action_manager, "_action", action["action"])
    _restore_attribute(env.action_manager, "_prev_action", action["prev_action"])
    for name, term_state in action["terms"].items():
        term = env.action_manager._terms[name]
        _restore_attribute(term, "_raw_actions", term_state["raw_actions"])
        _restore_attribute(term, "_processed_actions", term_state["processed_actions"])

    reward = state["reward_manager"]
    for name, value in reward["episode_sums"].items():
        _restore_attribute(env.reward_manager._episode_sums, name, value)
    _restore_attribute(env.reward_manager, "_reward_buf", reward["reward_buf"])
    _restore_attribute(env.reward_manager, "_step_reward", reward["step_reward"])
    env.reward_buf = env.reward_manager._reward_buf

    termination = state["termination_manager"]
    _restore_attribute(
        env.termination_manager, "_term_dones", termination["term_dones"]
    )
    _restore_attribute(
        env.termination_manager, "_truncated_buf", termination["truncated_buf"]
    )
    _restore_attribute(
        env.termination_manager, "_terminated_buf", termination["terminated_buf"]
    )
    env.reset_terminated = env.termination_manager._terminated_buf
    env.reset_time_outs = env.termination_manager._truncated_buf

    observation = clone_recovery_value(state["observation_manager"]["obs_buffer"])
    env.observation_manager._obs_buffer = observation
    driver = state["driver"]
    if driver["obs_buf_mode"] == "absent":
        try:
            local_attributes = vars(env)
        except TypeError:
            local_attributes = {}
        if "obs_buf" in local_attributes:
            delattr(env, "obs_buf")
        elif _optional_runtime_attribute(env, "obs_buf") is not None:
            _raise_missing_task_state(
                "driver.obs_buf:cannot_restore_absence",
                operation="restore task state",
            )
    elif driver["obs_buf_mode"] == "manager_alias":
        env.obs_buf = observation
    else:
        env.obs_buf = clone_recovery_value(driver["obs_buf"])
    env._sim_step_counter = int(state["driver"]["sim_step_counter"])
    env.extras = clone_recovery_value(state["driver"]["extras"])

    robot = _pp_box_robot(env, operation="restore task state")
    env_ids = _single_env_ids(env, operation="restore task state targets")
    targets = state["robot_targets"]
    robot.set_joint_position_target(
        clone_recovery_value(targets["joint_pos"]), env_ids=env_ids
    )
    robot.set_joint_velocity_target(
        clone_recovery_value(targets["joint_vel"]), env_ids=env_ids
    )
    robot.set_joint_effort_target(
        clone_recovery_value(targets["joint_effort"]), env_ids=env_ids
    )
    for name, value in state["task_cfg"].items():
        setattr(env.cfg, name, clone_recovery_value(value))


def _capture_pp_box_recovery_task_state_bound(env: Any) -> Mapping[str, Any]:
    return _capture_pp_box_recovery_task_state(env)


def _preflight_pp_box_recovery_task_state_bound(env: Any, state: Any) -> None:
    _preflight_pp_box_recovery_task_state(env, state)


def _restore_pp_box_recovery_task_state_bound(env: Any, state: Any) -> None:
    _restore_pp_box_recovery_task_state(env, state)


def _capture_pp_box_action_manager_state(manager: Any) -> Mapping[str, Any]:
    term_names = tuple(
        _require_runtime_attribute(
            manager,
            "_term_names",
            path="action_manager._term_names",
            operation="capture action-manager state",
        )
    )
    terms = _require_runtime_attribute(
        manager,
        "_terms",
        path="action_manager._terms",
        operation="capture action-manager state",
    )
    if tuple(terms) != term_names:
        _raise_missing_task_state(
            "action_manager.term_contract",
            operation="capture action-manager state",
        )
    return {
        "schema_version": PP_BOX_ACTION_MANAGER_STATE_SCHEMA_VERSION,
        "manager_type": _type_identity(manager),
        "term_identity": tuple(
            (name, _type_identity(terms[name]), int(terms[name].action_dim))
            for name in term_names
        ),
        "action": clone_recovery_value(
            _require_runtime_attribute(
                manager,
                "_action",
                path="action_manager._action",
                operation="capture action-manager state",
            )
        ),
        "prev_action": clone_recovery_value(
            _require_runtime_attribute(
                manager,
                "_prev_action",
                path="action_manager._prev_action",
                operation="capture action-manager state",
            )
        ),
        "terms": {
            name: {
                "raw_actions": clone_recovery_value(terms[name]._raw_actions),
                "processed_actions": clone_recovery_value(
                    terms[name]._processed_actions
                ),
            }
            for name in term_names
        },
    }


def _preflight_pp_box_action_manager_state(manager: Any, state: Any) -> None:
    top = _require_exact_mapping_keys(
        state,
        {
            "schema_version",
            "manager_type",
            "term_identity",
            "action",
            "prev_action",
            "terms",
        },
        path="action_manager_runtime",
    )
    if top["schema_version"] != PP_BOX_ACTION_MANAGER_STATE_SCHEMA_VERSION:
        raise RecoveryStateSchemaError("unsupported action-manager recovery schema")
    current = _capture_pp_box_action_manager_state(manager)
    for identity_name in ("manager_type", "term_identity"):
        if top[identity_name] != current[identity_name]:
            raise RecoveryStateSchemaError(
                f"action-manager recovery {identity_name} mismatch"
            )
    _require_same_payload_schema(
        manager._action, top["action"], path="action_manager_runtime.action"
    )
    _require_same_payload_schema(
        manager._prev_action,
        top["prev_action"],
        path="action_manager_runtime.prev_action",
    )
    terms = _require_exact_mapping_keys(
        top["terms"], set(manager._term_names), path="action_manager_runtime.terms"
    )
    for name in manager._term_names:
        saved = _require_exact_mapping_keys(
            terms[name],
            {"raw_actions", "processed_actions"},
            path=f"action_manager_runtime.{name}",
        )
        term = manager._terms[name]
        _require_same_payload_schema(
            term._raw_actions,
            saved["raw_actions"],
            path=f"action_manager_runtime.{name}.raw_actions",
        )
        _require_same_payload_schema(
            term._processed_actions,
            saved["processed_actions"],
            path=f"action_manager_runtime.{name}.processed_actions",
        )


def _restore_pp_box_action_manager_state(manager: Any, state: Any) -> None:
    _preflight_pp_box_action_manager_state(manager, state)
    _restore_attribute(manager, "_action", state["action"])
    _restore_attribute(manager, "_prev_action", state["prev_action"])
    for name, saved in state["terms"].items():
        term = manager._terms[name]
        _restore_attribute(term, "_raw_actions", saved["raw_actions"])
        _restore_attribute(term, "_processed_actions", saved["processed_actions"])


def _initialize_pp_box_pre_step_driver_aliases(env: Any) -> None:
    """Fill Isaac Lab driver aliases that are first assigned by ``step``."""

    reward_manager = _require_runtime_attribute(
        env,
        "reward_manager",
        path="reward_manager",
        operation="install task state hooks",
    )
    termination_manager = _require_runtime_attribute(
        env,
        "termination_manager",
        path="termination_manager",
        operation="install task state hooks",
    )
    _require_runtime_attribute(
        env,
        "observation_manager",
        path="observation_manager",
        operation="install task state hooks",
    )
    aliases = {
        "reward_buf": _require_runtime_attribute(
            reward_manager,
            "_reward_buf",
            path="reward_manager._reward_buf",
            operation="install task state hooks",
        ),
        "reset_terminated": _require_runtime_attribute(
            termination_manager,
            "_terminated_buf",
            path="termination_manager._terminated_buf",
            operation="install task state hooks",
        ),
        "reset_time_outs": _require_runtime_attribute(
            termination_manager,
            "_truncated_buf",
            path="termination_manager._truncated_buf",
            operation="install task state hooks",
        ),
    }
    terminated = aliases["reset_terminated"]
    truncated = aliases["reset_time_outs"]
    if (
        not isinstance(terminated, torch.Tensor)
        or not isinstance(truncated, torch.Tensor)
        or terminated.dtype != torch.bool
        or truncated.dtype != torch.bool
        or terminated.shape != truncated.shape
        or terminated.device != truncated.device
    ):
        _raise_missing_task_state(
            "termination_manager:buffer_contract",
            operation="install task state hooks",
        )
    aliases["reset_buf"] = torch.logical_or(terminated, truncated)
    for name, value in aliases.items():
        if not hasattr(env, name):
            setattr(env, name, value)


def install_pp_box_recovery_task_state_hooks(env: Any) -> None:
    """Install task-specific state hooks on the production PP-box environment."""

    _single_env_ids(env, operation="install PP-box recovery task-state hooks")
    _validate_env_task_identity(env, PP_BOX_TASK_IDENTITY)
    marker = getattr(env, "_pp_box_recovery_task_state_hooks_version", None)
    if marker == PP_BOX_TASK_STATE_SCHEMA_VERSION:
        coordinator = getattr(env, "recovery_state_coordinator", None)
        if type(coordinator) is not RecoveryStateCoordinator:
            _raise_missing_task_state(
                "recovery_state_coordinator",
                operation="validate installed task state hooks",
            )
        coordinator._validate_binding()
        return
    for name in (
        "capture_recovery_task_state",
        "preflight_restore_recovery_task_state",
        "restore_recovery_task_state",
    ):
        if callable(getattr(env, name, None)):
            _raise_missing_task_state(
                f"conflicting_hook:{name}", operation="install task state hooks"
            )
    action_manager = _require_runtime_attribute(
        env,
        "action_manager",
        path="action_manager",
        operation="install task state hooks",
    )
    for name in (
        "capture_recovery_action_manager_state",
        "preflight_restore_recovery_action_manager_state",
        "restore_recovery_action_manager_state",
    ):
        if callable(getattr(action_manager, name, None)):
            _raise_missing_task_state(
                f"conflicting_hook:action_manager.{name}",
                operation="install task state hooks",
            )
    if getattr(env, "recovery_state_coordinator", None) is not None:
        _raise_missing_task_state(
            "conflicting_hook:recovery_state_coordinator",
            operation="install task state hooks",
        )
    _initialize_pp_box_pre_step_driver_aliases(env)
    coordinator = RecoveryStateCoordinator(
        env,
        task_identity=PP_BOX_TASK_IDENTITY,
    )

    env.capture_recovery_task_state = MethodType(
        _capture_pp_box_recovery_task_state_bound, env
    )
    env.preflight_restore_recovery_task_state = MethodType(
        _preflight_pp_box_recovery_task_state_bound, env
    )
    env.restore_recovery_task_state = MethodType(
        _restore_pp_box_recovery_task_state_bound, env
    )
    action_manager.capture_recovery_action_manager_state = MethodType(
        _capture_pp_box_action_manager_state, action_manager
    )
    action_manager.preflight_restore_recovery_action_manager_state = MethodType(
        _preflight_pp_box_action_manager_state, action_manager
    )
    action_manager.restore_recovery_action_manager_state = MethodType(
        _restore_pp_box_action_manager_state, action_manager
    )
    env.recovery_state_coordinator = coordinator
    env.recovery_process_global_rng_exclusive = True
    env._pp_box_recovery_task_state_hooks_version = PP_BOX_TASK_STATE_SCHEMA_VERSION


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
        if snapshot.kind == "numpy_generator" and isinstance(
            source, np.random.Generator
        ):
            bit_generator = copy.deepcopy(source.bit_generator)
            bit_generator.state = clone_recovery_value(snapshot.state)
            return True
        if snapshot.kind == "numpy_random_state" and isinstance(
            source, np.random.RandomState
        ):
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
        ) or (
            _payload_schema(_scene_state(env, is_relative=snapshot.scope.is_relative))
            != snapshot.capabilities.scene_state_schema
        ):
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
    if snapshot.capabilities.available.get("task_state", False) and (
        snapshot.task_state is None
        or (isinstance(snapshot.task_state, Mapping) and not snapshot.task_state)
        or not _payload_matches_schema(
            snapshot.task_state,
            snapshot.capabilities.task_state_schema,
        )
    ):
        missing.add("task_state")
    advertised_runtime = {
        name
        for name in _RUNTIME_CACHE_NAMES
        + tuple(capability for capability, _, _ in _CONTROL_PARTICIPANTS)
        if snapshot.capabilities.available.get(name, False)
    }
    if advertised_runtime and not isinstance(snapshot.runtime_state, Mapping):
        missing.update(advertised_runtime)
    else:
        missing.update(
            name for name in advertised_runtime if name not in snapshot.runtime_state
        )
    return missing


def _validate_snapshot_schema(
    snapshot: RecoveryStateSnapshot, task_identity: str
) -> None:
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
    _required_for_fidelity(snapshot.fidelity_tier)
    scope = snapshot.scope
    if not isinstance(scope, RecoveryStateScope) or (
        scope.schema_version != RECOVERY_STATE_SCHEMA_VERSION
        or scope.num_envs != 1
        or scope.env_ids != (0,)
        or scope.is_relative is not True
        or scope.process_global_rng_scope != "exclusive_process_single_env"
    ):
        raise RecoveryStateSchemaError(
            "recovery snapshot scope must be exclusive num_envs=1, "
            "env_ids=(0,), is_relative=True"
        )
    task_state_available = snapshot.capabilities.available.get("task_state", False)
    if task_state_available != (
        snapshot.capabilities.task_state_mode in {"hooks", "direct"}
    ):
        raise RecoveryStateSchemaError(
            "recovery capability task-state mode is inconsistent"
        )


def _preflight_recovery_restore(
    env: Any,
    snapshot: RecoveryStateSnapshot,
    *,
    required_capabilities: set[str] | frozenset[str] | None,
    task_identity: str,
) -> torch.Tensor:
    _validate_snapshot_schema(snapshot, task_identity)
    _validate_env_task_identity(env, task_identity)
    env_ids = _single_env_ids(env, operation="restore single-env selection")
    _validate_reset_to_signature(env, env_ids)
    capabilities = discover_recovery_state_capabilities(env)
    snapshot_capabilities = {
        name
        for name, is_available in snapshot.capabilities.available.items()
        if is_available
    }
    required = _require_capabilities(
        capabilities,
        snapshot_capabilities
        | _required_for_fidelity(snapshot.fidelity_tier)
        | frozenset(required_capabilities or ()),
        operation="restore",
    )
    missing_from_snapshot = {
        name
        for name in required
        if not snapshot.capabilities.available.get(name, False)
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

    non_restorable_counters = {
        f"task_counter:{name}"
        for name in snapshot.capabilities.counter_names
        if not _attribute_is_restorable(env, name, snapshot.task_counters[name])
    }
    if non_restorable_counters:
        raise RecoveryStateIncompleteError(
            non_restorable_counters,
            operation="restore preflight",
            available=snapshot.capabilities.available,
        )

    if snapshot.capabilities.task_state_mode == "hooks":
        try:
            env.preflight_restore_recovery_task_state(
                clone_recovery_value(snapshot.task_state)
            )
        except Exception as exc:
            raise RecoveryStatePreflightError("task_state", exc) from exc
    elif snapshot.capabilities.task_state_mode == "direct" and not (
        _has_writable_direct_task_state(env)
    ):
        raise RecoveryStateIncompleteError(
            {"task_state"},
            operation="restore preflight",
            available=capabilities.available,
        )

    for name in _RUNTIME_CACHE_NAMES:
        if snapshot.capabilities.available.get(name, False):
            try:
                getattr(env, f"preflight_restore_{name}")(
                    clone_recovery_value(snapshot.runtime_state[name])
                )
            except Exception as exc:
                raise RecoveryStatePreflightError(name, exc) from exc
    for capability, owner_name, hook_name in _CONTROL_PARTICIPANTS:
        if snapshot.capabilities.available.get(capability, False):
            owner = _control_participant(env, owner_name)
            assert owner is not None
            try:
                getattr(owner, f"preflight_restore_{hook_name}")(
                    clone_recovery_value(snapshot.runtime_state[capability])
                )
            except Exception as exc:
                raise RecoveryStatePreflightError(capability, exc) from exc
    return env_ids


def _apply_recovery_restore(
    env: Any,
    snapshot: RecoveryStateSnapshot,
    *,
    env_ids: torch.Tensor,
) -> None:
    env.reset_to(
        clone_recovery_value(snapshot.scene_state),
        env_ids=env_ids,
        is_relative=snapshot.scope.is_relative,
    )
    for name in snapshot.capabilities.counter_names:
        _restore_attribute(env, name, snapshot.task_counters[name])

    if snapshot.capabilities.task_state_mode == "hooks":
        env.restore_recovery_task_state(clone_recovery_value(snapshot.task_state))
    elif snapshot.capabilities.task_state_mode == "direct":
        _restore_attribute(env, "recovery_task_state", snapshot.task_state)

    for name in _RUNTIME_CACHE_NAMES:
        if snapshot.capabilities.available.get(name, False):
            getattr(env, f"restore_{name}")(
                clone_recovery_value(snapshot.runtime_state[name])
            )
    for capability, owner_name, hook_name in _CONTROL_PARTICIPANTS:
        if snapshot.capabilities.available.get(capability, False):
            owner = _control_participant(env, owner_name)
            assert owner is not None
            getattr(owner, f"restore_{hook_name}")(
                clone_recovery_value(snapshot.runtime_state[capability])
            )

    random.setstate(clone_recovery_value(snapshot.rng_state.python))
    np.random.set_state(clone_recovery_value(snapshot.rng_state.numpy))
    torch.set_rng_state(snapshot.rng_state.torch_cpu.clone())
    if snapshot.rng_state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(
            [state.clone() for state in snapshot.rng_state.torch_cuda]
        )
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


def _project_snapshot_for_verification(
    current: RecoveryStateSnapshot,
    target: RecoveryStateSnapshot,
) -> RecoveryStateSnapshot:
    available = target.capabilities.available
    current_rng = current.rng_state
    return RecoveryStateSnapshot(
        schema_version=target.schema_version,
        task_identity=target.task_identity,
        scope=target.scope,
        fidelity_tier=target.fidelity_tier,
        capabilities=target.capabilities,
        scene_state=clone_recovery_value(current.scene_state),
        task_counters={
            name: clone_recovery_value(current.task_counters[name])
            for name in target.capabilities.counter_names
        },
        task_state=(
            clone_recovery_value(current.task_state)
            if available.get("task_state", False)
            else clone_recovery_value(target.task_state)
        ),
        rng_state=RecoveryRngState(
            python=clone_recovery_value(current_rng.python),
            numpy=clone_recovery_value(current_rng.numpy),
            torch_cpu=current_rng.torch_cpu.clone(),
            torch_cuda=(
                clone_recovery_value(current_rng.torch_cuda)
                if available.get("torch_cuda_rng", False)
                else None
            ),
            task_local=(
                clone_recovery_value(current_rng.task_local)
                if available.get("task_local_rng", False)
                else None
            ),
            wrapper=(
                clone_recovery_value(current_rng.wrapper)
                if available.get("wrapper_rng", False)
                else None
            ),
        ),
        runtime_state={
            name: clone_recovery_value(current.runtime_state[name])
            for name in _RUNTIME_CACHE_NAMES
            + tuple(capability for capability, _, _ in _CONTROL_PARTICIPANTS)
            if available.get(name, False)
        },
    )


def _verify_recovery_restore(
    env: Any,
    target: RecoveryStateSnapshot,
    *,
    target_digest: str,
    task_identity: str,
) -> None:
    advertised = {
        name for name, value in target.capabilities.available.items() if value
    }
    current = _capture_recovery_state_unlocked(
        env,
        required_capabilities=advertised,
        task_identity=task_identity,
        fidelity_tier=target.fidelity_tier,
    )
    projected = _project_snapshot_for_verification(current, target)
    actual_digest = recovery_state_digest(projected, task_identity=task_identity)
    if not hmac.compare_digest(actual_digest, target_digest):
        raise RecoveryStateDigestMismatchError(
            f"post-restore snapshot digest {actual_digest} does not match "
            f"target {target_digest}"
        )


def _validate_bound_snapshot_digest(
    snapshot: RecoveryStateSnapshot,
    snapshot_digest: str,
    *,
    task_identity: str,
) -> str:
    _validate_snapshot_schema(snapshot, task_identity)
    actual = recovery_state_digest(snapshot, task_identity=task_identity)
    if not (
        isinstance(snapshot_digest, str)
        and len(snapshot_digest) == 64
        and set(snapshot_digest) <= set("0123456789abcdef")
        and hmac.compare_digest(snapshot_digest, actual)
    ):
        raise RecoveryStateDigestMismatchError(
            f"caller-bound recovery snapshot digest {snapshot_digest!r} "
            f"does not match {actual}"
        )
    return actual


def restore_recovery_state(
    env: Any,
    snapshot: RecoveryStateSnapshot,
    *,
    snapshot_digest: str,
    required_capabilities: set[str] | frozenset[str] | None = None,
    task_identity: str = PP_BOX_TASK_IDENTITY,
) -> None:
    """Restore a caller-digest-bound snapshot as a verified transaction."""

    target_digest = _validate_bound_snapshot_digest(
        snapshot,
        snapshot_digest,
        task_identity=task_identity,
    )
    with _exclusive_process_global_rng("restore"):
        env_ids = _preflight_recovery_restore(
            env,
            snapshot,
            required_capabilities=required_capabilities,
            task_identity=task_identity,
        )
        try:
            rollback_snapshot = _capture_recovery_state_unlocked(
                env,
                required_capabilities=None,
                task_identity=task_identity,
                fidelity_tier=snapshot.fidelity_tier,
            )
            rollback_digest = recovery_state_digest(
                rollback_snapshot, task_identity=task_identity
            )
        except Exception as exc:
            raise RecoveryStatePreflightError("rollback_snapshot", exc) from exc

        failure_phase = "apply_target"
        try:
            _apply_recovery_restore(env, snapshot, env_ids=env_ids)
            failure_phase = "verify_target"
            _verify_recovery_restore(
                env,
                snapshot,
                target_digest=target_digest,
                task_identity=task_identity,
            )
            return
        except Exception as target_failure:
            rollback_failure: Exception | None = None
            try:
                rollback_env_ids = _preflight_recovery_restore(
                    env,
                    rollback_snapshot,
                    required_capabilities=None,
                    task_identity=task_identity,
                )
                _apply_recovery_restore(
                    env,
                    rollback_snapshot,
                    env_ids=rollback_env_ids,
                )
                _verify_recovery_restore(
                    env,
                    rollback_snapshot,
                    target_digest=rollback_digest,
                    task_identity=task_identity,
                )
            except Exception as exc:  # noqa: BLE001 - preserve arbitrary hook evidence.
                rollback_failure = exc

            evidence = RecoveryStateRestoreEvidence(
                schema_version=1,
                target_snapshot_digest=target_digest,
                rollback_snapshot_digest=rollback_digest,
                failure_phase=failure_phase,
                failure_type=type(target_failure).__name__,
                failure_message=str(target_failure),
                rollback_succeeded=rollback_failure is None,
                rollback_failure_type=(
                    None
                    if rollback_failure is None
                    else type(rollback_failure).__name__
                ),
                rollback_failure_message=(
                    None if rollback_failure is None else str(rollback_failure)
                ),
            )
            raise RecoveryStateTransactionError(evidence) from target_failure
