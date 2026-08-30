"""Deterministic failure catalog for HOI pick-and-place recovery.

The module is intentionally Isaac-free. It describes failure truth and delegates
all simulator mutation to explicit runtime hooks so unsupported runtimes fail
closed instead of reaching into raw assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from . import recovery_state, rewards
from .recovery_telemetry import (
    RECOVERY_TELEMETRY_SCHEMA_VERSION,
    DriverTerminalContext,
    PrivilegedRecoveryTelemetry,
)

PP_BOX_TASK_IDENTITY = recovery_state.PP_BOX_TASK_IDENTITY
RECOVERY_FAILURE_DESCRIPTOR_SCHEMA_VERSION = 1
RECOVERY_ATTEMPT_SCHEMA_VERSION = 1
RECOVERY_ACTIVATION_SCHEMA_VERSION = 1
RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION = 1
RECOVERY_INJECTION_PLAN_SCHEMA_VERSION = 1
RECOVERY_RAW_RECEIPT_SCHEMA_VERSION = 1
RECOVERY_CATALOG_QUALIFICATION_SCHEMA_VERSION = 1

DECLARED_FAILURE_CATEGORIES = (
    "dropped",
    "failed-grasp",
    "misaligned",
    "near-shelf-misplaced",
)
_CATEGORY_INITIAL_STAGE = {
    "dropped": "approach",
    "failed-grasp": "acquire",
    "misaligned": "place",
    "near-shelf-misplaced": "approach",
}
_DESCRIPTOR_ENTITIES = ("box", "shelf_target", "bimanual_ee")
_REWARD_TERMS = ("distance", "grasp", "placement")
_CONTINUATION_REWARD_TERMS = (
    "articulation",
    "distance",
    "grasp",
    "placement",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
BOX_HALF_EXTENTS_M = rewards.BOX_HALF_EXTENTS_M
BOX_HALF_EXTENT_M = BOX_HALF_EXTENTS_M[0]
PLACEMENT_Z_TOLERANCE_M = rewards.BOX_BOTTOM_SURFACE_TOLERANCE_M

_COMMON_RUNTIME_CAPABILITIES = frozenset(
    {
        "recovery_state_snapshot",
        "recovery_state_continuation",
        "task_counters",
        "recovery_task_state",
        "episode_rng",
        "selected_env_box_writer",
        "explicit_failure_writer",
        "deterministic_settle",
        "privileged_readback",
        "live_target_shelf_geometry",
        "exact_bimanual_ee_pose",
        "pairwise_bimanual_box_contact",
        "authoritative_terminal_context",
    }
)
_CATEGORY_RUNTIME_CAPABILITIES = {
    "dropped": frozenset({"live_ground_support", "target_disjoint_ground_region"}),
    "failed-grasp": frozenset({"verified_pickup_anchor"}),
    "misaligned": frozenset({"verified_grasp_preserving_anchor"}),
    "near-shelf-misplaced": frozenset({"live_target_bounds"}),
}
_KNOWN_RUNTIME_CAPABILITIES = frozenset(
    _COMMON_RUNTIME_CAPABILITIES
    | frozenset().union(*_CATEGORY_RUNTIME_CAPABILITIES.values())
)
_SNAPSHOT_RESTORE_REQUIRED_CAPABILITIES = frozenset({"task_state", "wrapper_rng"})
_QUALIFICATION_OPERATION_TRACE = (
    "restore",
    "write",
    "settle",
    "readback",
    "primitive_step",
    "readback",
)
_ISSUED_RECEIPT_DIGESTS: dict[object, str] = {}
_ISSUED_QUALIFICATION_DIGESTS: dict[object, str] = {}
_RECOVERY_ACTIVATION_FACTORY_TOKEN = object()


class RecoveryFailureError(RuntimeError):
    """Base class for fail-closed recovery failure errors."""


class RecoveryFailureSchemaError(RecoveryFailureError, ValueError):
    """Raised when versioned recovery input is incomplete or inconsistent."""


class RecoveryFailurePredicateConflictError(RecoveryFailureError):
    """Raised when mutually exclusive physical predicates are simultaneously true."""


class RecoveryFailureCapabilityError(RecoveryFailureError):
    """Raised when a runtime cannot prove an injection requirement."""

    def __init__(self, category: str, missing_capabilities: Sequence[str]) -> None:
        self.category = category
        self.missing_capabilities = tuple(sorted(set(missing_capabilities)))
        super().__init__(
            f"{category} recovery injection is missing runtime capabilities: "
            + ", ".join(self.missing_capabilities)
        )


class RecoveryFailureVerificationError(RecoveryFailureError):
    """Raised when post-injection readback does not prove the requested failure."""

    def __init__(
        self,
        *,
        category: str,
        failure_seed: int,
        observed_category: str | None,
        detail: str,
    ) -> None:
        self.category = category
        self.failure_seed = failure_seed
        self.observed_category = observed_category
        super().__init__(
            f"{category} failure readback verification failed for seed {failure_seed}: {detail}"
        )


class RecoveryFailureSnapshotDigestError(RecoveryFailureError):
    """Raised before injection when a descriptor does not bind the actual snapshot."""

    def __init__(self, *, expected_digest: str, actual_digest: str) -> None:
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest
        super().__init__(
            "recovery snapshot digest mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )


def required_runtime_capabilities(category: str) -> frozenset[str]:
    normalized = _category(category)
    assert normalized is not None
    return _COMMON_RUNTIME_CAPABILITIES | _CATEGORY_RUNTIME_CAPABILITIES[normalized]


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        requirement = (
            "a non-negative integer" if minimum == 0 else f"an integer >= {minimum}"
        )
        raise RecoveryFailureSchemaError(f"{name} must be {requirement}")
    return value


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise RecoveryFailureSchemaError(f"{name} must be a boolean")
    return value


def _digest(value: object, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecoveryFailureSchemaError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _category(value: object, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or value not in DECLARED_FAILURE_CATEGORIES:
        raise RecoveryFailureSchemaError(
            f"unknown recovery failure category: {value!r}"
        )
    return value


def _exact_mapping(
    value: object,
    *,
    required: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != required:
        raise RecoveryFailureSchemaError(
            f"{name} schema must contain exactly: {', '.join(sorted(required))}"
        )
    return value


def _canonical_json_value(value: object, *, name: str) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecoveryFailureSchemaError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise RecoveryFailureSchemaError(f"{name} keys must be non-empty strings")
        return {
            key: _canonical_json_value(value[key], name=f"{name}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical_json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RecoveryFailureSchemaError(
        f"{name} contains unsupported value type {type(value).__name__}"
    )


def _canonical_json_bytes(value: object, *, name: str) -> bytes:
    canonical = _canonical_json_value(value, name=name)
    return json.dumps(
        canonical,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _freeze_json_value(value: object, *, name: str) -> object:
    canonical = _canonical_json_value(value, name=name)

    def freeze(item: object) -> object:
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(canonical)


def _finite_tuple(value: object, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise RecoveryFailureSchemaError(f"{name} must be a length-{length} sequence")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise RecoveryFailureSchemaError(f"{name} must contain finite values")
    return result


def _freeze_recovery_value(value: object) -> object:
    """Clone nested runtime evidence while keeping tensor payloads digestible."""

    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {freeze(key): freeze(child) for key, child in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        if isinstance(item, set):
            return frozenset(freeze(child) for child in item)
        return recovery_state.clone_recovery_value(item)

    frozen = freeze(value)
    recovery_state.recovery_value_digest(frozen)
    return frozen


@dataclass(frozen=True)
class RecoveryRewardBinding:
    term: str
    stage: str
    entity_roles: Mapping[str, str]
    telemetry_binding: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entity_roles", MappingProxyType(dict(self.entity_roles))
        )


_RECOVERY_REWARD_BINDINGS = (
    RecoveryRewardBinding(
        term="distance",
        stage="approach",
        entity_roles={"source": "bimanual_ee", "target": "box"},
        telemetry_binding="max_bimanual_ee_box_distance_m",
    ),
    RecoveryRewardBinding(
        term="grasp",
        stage="acquire",
        entity_roles={"end_effector": "bimanual_ee", "object": "box"},
        telemetry_binding="bimanual_grasp",
    ),
    RecoveryRewardBinding(
        term="placement",
        stage="place",
        entity_roles={"object": "box", "target": "shelf_target"},
        telemetry_binding="hypot_xy_z_mismatch",
    ),
)


def recovery_reward_bindings() -> tuple[RecoveryRewardBinding, ...]:
    """Bind compiler potentials only to deterministic task telemetry."""

    return _RECOVERY_REWARD_BINDINGS


def recovery_reward_component_scope(binding: RecoveryRewardBinding) -> str:
    """Return the shared compiler's stable scope key without importing its code."""

    if not isinstance(binding, RecoveryRewardBinding):
        raise RecoveryFailureSchemaError(
            "reward component scope requires a RecoveryRewardBinding"
        )
    ordered_roles = [[role, entity] for role, entity in binding.entity_roles.items()]
    return json.dumps(
        [binding.term, ordered_roles],
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class RecoveryActivationBaseline:
    """Eq. 8 normalizers captured once for one environment lane."""

    env_index: int
    distance_d_init_m: float
    placement_d_init_m: float

    def __post_init__(self) -> None:
        env_index = _strict_int(self.env_index, name="activation baseline env index")
        normalized: dict[str, float] = {}
        for name in ("distance_d_init_m", "placement_d_init_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecoveryFailureSchemaError(
                    f"activation baseline {name} must be finite and non-negative"
                )
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise RecoveryFailureSchemaError(
                    f"activation baseline {name} must be finite and non-negative"
                )
            normalized[name] = number
        object.__setattr__(self, "env_index", env_index)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, init=False)
class RecoveryActivationContext:
    """Explicit immutable activation state carried across primitive steps."""

    schema_version: int
    task_identity: str
    activation_id: str
    baselines: Mapping[int, RecoveryActivationBaseline]
    activation_digest: str
    _issuance_token: object = field(repr=False, compare=False)


def _normalized_activation_baselines(
    baselines: object,
) -> dict[int, RecoveryActivationBaseline]:
    if not isinstance(baselines, Mapping) or not baselines:
        raise RecoveryFailureSchemaError(
            "recovery activation baselines must be a non-empty mapping"
        )
    normalized: dict[int, RecoveryActivationBaseline] = {}
    for raw_env_index, baseline in baselines.items():
        env_index = _strict_int(
            raw_env_index,
            name="recovery activation env index",
        )
        if not isinstance(baseline, RecoveryActivationBaseline):
            raise RecoveryFailureSchemaError(
                "recovery activation baseline has the wrong type"
            )
        if baseline.env_index != env_index:
            raise RecoveryFailureSchemaError(
                "recovery activation baseline key does not match its env index"
            )
        normalized[env_index] = baseline
    return normalized


def _recovery_activation_digest(context: RecoveryActivationContext) -> str:
    return recovery_state.recovery_value_digest(
        (
            "recovery-activation-context",
            context.schema_version,
            context.task_identity,
            context.activation_id,
            context.baselines,
        )
    )


def _issue_recovery_activation_context(
    *,
    activation_id: str,
    baselines: Mapping[int, RecoveryActivationBaseline],
) -> RecoveryActivationContext:
    if not isinstance(activation_id, str) or not activation_id.strip():
        raise RecoveryFailureSchemaError(
            "recovery activation id must be a non-empty string"
        )
    normalized = _normalized_activation_baselines(baselines)
    context = object.__new__(RecoveryActivationContext)
    values = {
        "schema_version": RECOVERY_ACTIVATION_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "activation_id": activation_id,
        "baselines": MappingProxyType(normalized),
        "_issuance_token": _RECOVERY_ACTIVATION_FACTORY_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(context, name, value)
    object.__setattr__(
        context,
        "activation_digest",
        _recovery_activation_digest(context),
    )
    return context


def _validate_recovery_activation(
    value: object,
) -> RecoveryActivationContext:
    if type(value) is not RecoveryActivationContext:
        raise RecoveryFailureSchemaError(
            "reward resolution requires a recovery activation context"
        )
    if (
        value.schema_version != RECOVERY_ACTIVATION_SCHEMA_VERSION
        or value.task_identity != PP_BOX_TASK_IDENTITY
        or not isinstance(value.activation_id, str)
        or not value.activation_id.strip()
        or value._issuance_token is not _RECOVERY_ACTIVATION_FACTORY_TOKEN
    ):
        raise RecoveryFailureSchemaError(
            "recovery activation context was not issued by begin_recovery_activation"
        )
    _normalized_activation_baselines(value.baselines)
    if value.activation_digest != _recovery_activation_digest(value):
        raise RecoveryFailureSchemaError(
            "recovery activation context digest does not match its baselines"
        )
    return value


def _reward_potential_distances(
    telemetry: PrivilegedRecoveryTelemetry,
) -> tuple[float, float]:
    state = _validate_privileged_telemetry(telemetry)
    left_distance = math.dist(state.left_ee_pose_w[:3], state.box_center_w)
    right_distance = math.dist(state.right_ee_pose_w[:3], state.box_center_w)
    return (
        max(left_distance, right_distance),
        math.hypot(state.xy_mismatch_m, state.z_mismatch_m),
    )


def begin_recovery_activation(
    *,
    activation_id: str,
    telemetry_by_env: Mapping[int, PrivilegedRecoveryTelemetry],
    active_context: RecoveryActivationContext | None = None,
) -> RecoveryActivationContext:
    """Freeze Eq. 8 normalizers once; repeated begin returns the active context."""

    if not isinstance(activation_id, str) or not activation_id.strip():
        raise RecoveryFailureSchemaError(
            "recovery activation id must be a non-empty string"
        )
    if active_context is not None:
        active = _validate_recovery_activation(active_context)
        if active.activation_id == activation_id:
            return active
    if not isinstance(telemetry_by_env, Mapping) or not telemetry_by_env:
        raise RecoveryFailureSchemaError(
            "recovery activation telemetry must be a non-empty env mapping"
        )
    baselines: dict[int, RecoveryActivationBaseline] = {}
    for raw_env_index, telemetry in telemetry_by_env.items():
        env_index = _strict_int(raw_env_index, name="recovery activation env index")
        state = _validate_privileged_telemetry(telemetry)
        if state.env_index != env_index:
            raise RecoveryFailureSchemaError(
                "recovery activation telemetry env index does not match its lane"
            )
        distance, placement_distance = _reward_potential_distances(state)
        baselines[env_index] = RecoveryActivationBaseline(
            env_index=env_index,
            distance_d_init_m=distance,
            placement_d_init_m=placement_distance,
        )
    return _issue_recovery_activation_context(
        activation_id=activation_id,
        baselines=baselines,
    )


def _recovery_gate_truth(
    telemetry: PrivilegedRecoveryTelemetry,
) -> Mapping[str, bool]:
    running_safe = bool(
        telemetry.terminal_reason == "running" and not telemetry.fall_candidate
    )
    pose_evidence = telemetry.grasp_evidence.pose_evidence
    if type(pose_evidence) is not bool:
        raise RecoveryFailureSchemaError(
            "bimanual pose evidence must be an exact boolean"
        )
    truth = {
        "running_and_not_grasp": running_safe and not telemetry.grasp,
        "bimanual_pose_evidence": pose_evidence,
        "running_and_not_grasp_and_bimanual_pose": (
            running_safe and not telemetry.grasp and pose_evidence
        ),
        "grasp": telemetry.grasp,
        "lost_bimanual_pose": running_safe and not pose_evidence,
        "running_and_grasp": running_safe and telemetry.grasp,
        "placement": telemetry.placement,
        "lost_grasp_with_bimanual_pose": (
            running_safe and not telemetry.grasp and pose_evidence
        ),
        "lost_grasp_without_bimanual_pose": (
            running_safe and not telemetry.grasp and not pose_evidence
        ),
    }
    return MappingProxyType(truth)


def resolve_recovery_reward_telemetry(
    telemetry: PrivilegedRecoveryTelemetry,
    *,
    activation_context: RecoveryActivationContext | None = None,
    env_index: int,
) -> Mapping[str, object]:
    """Resolve one primitive step using activation-frozen Eq. 8 normalizers."""

    state = _validate_privileged_telemetry(telemetry)
    activation = _validate_recovery_activation(activation_context)
    lane = _strict_int(env_index, name="reward telemetry env index")
    if state.env_index != lane:
        raise RecoveryFailureSchemaError(
            "reward telemetry env index does not match the requested lane"
        )
    try:
        baseline = activation.baselines[lane]
    except KeyError as exc:
        raise RecoveryFailureSchemaError(
            f"recovery activation context has no env index {lane}"
        ) from exc
    distance, placement_distance = _reward_potential_distances(state)
    gate_truth = _recovery_gate_truth(state)

    potential_values: Mapping[str, Mapping[str, float]] = {
        "distance": MappingProxyType(
            {"distance": distance, "d_init": baseline.distance_d_init_m}
        ),
        "grasp": MappingProxyType({"q_grasp": 1.0 if state.grasp else 0.0}),
        "placement": MappingProxyType(
            {
                "distance": placement_distance,
                "d_init": baseline.placement_d_init_m,
            }
        ),
    }
    components = {
        recovery_reward_component_scope(binding): potential_values[binding.term]
        for binding in _RECOVERY_REWARD_BINDINGS
    }

    return MappingProxyType(
        {
            "activation_id": activation.activation_id,
            "activation_digest": activation.activation_digest,
            "components": MappingProxyType(components),
            "gate_truth": gate_truth,
        }
    )


@dataclass(frozen=True)
class RecoveryFailureCatalogEntry:
    category: str
    initial_stage: str
    reward_bindings: tuple[RecoveryRewardBinding, ...]
    required_capabilities: frozenset[str]
    declared: bool = True
    recoverable: bool = True


_DECLARED_CATALOG = MappingProxyType(
    {
        category: RecoveryFailureCatalogEntry(
            category=category,
            initial_stage=_CATEGORY_INITIAL_STAGE[category],
            reward_bindings=_RECOVERY_REWARD_BINDINGS,
            required_capabilities=required_runtime_capabilities(category),
        )
        for category in DECLARED_FAILURE_CATEGORIES
    }
)


def declared_failure_catalog() -> Mapping[str, RecoveryFailureCatalogEntry]:
    """Return all source-decidable candidates, independent of runtime support."""

    return _DECLARED_CATALOG


def effective_failure_catalog(
    qualifications: Mapping[str, object] | None = None,
) -> Mapping[str, RecoveryFailureCatalogEntry]:
    """Return only entries proven by two factory-executed raw receipts."""

    if qualifications is None:
        return {}
    if not isinstance(qualifications, Mapping):
        raise RecoveryFailureSchemaError("catalog qualifications must be a mapping")
    unknown = set(qualifications) - set(DECLARED_FAILURE_CATEGORIES)
    if unknown:
        raise RecoveryFailureSchemaError(
            "unknown catalog qualification categories: " + ", ".join(sorted(unknown))
        )
    effective: dict[str, RecoveryFailureCatalogEntry] = {}
    for category, value in qualifications.items():
        if not isinstance(value, FailureCatalogQualification):
            raise RecoveryFailureSchemaError(
                f"catalog qualification for {category!r} has the wrong type"
            )
        if value.category != category:
            raise RecoveryFailureSchemaError(
                f"catalog qualification key {category!r} does not match its category"
            )
        if _qualification_is_effective(value):
            effective[category] = _DECLARED_CATALOG[category]
    return MappingProxyType(effective)


@dataclass(frozen=True)
class RecoveryFailureDescriptor:
    """Strict task-bound request used to build one deterministic injection."""

    schema_version: int
    task_identity: str
    category: str
    stage: str
    entities: tuple[str, ...]
    confidence: float
    reward_mask: Mapping[str, bool]
    failure_seed: int
    snapshot_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_FAILURE_DESCRIPTOR_SCHEMA_VERSION:
            raise RecoveryFailureSchemaError(
                f"unsupported descriptor schema version {self.schema_version}"
            )
        if self.task_identity != PP_BOX_TASK_IDENTITY:
            raise RecoveryFailureSchemaError(
                "descriptor task identity is not HOI_pp_box"
            )
        category = _category(self.category)
        assert category is not None
        expected_stage = _CATEGORY_INITIAL_STAGE[category]
        if self.stage != expected_stage:
            raise RecoveryFailureSchemaError(
                f"descriptor stage {self.stage!r} does not match {category!r} stage "
                f"{expected_stage!r}"
            )
        if (
            not isinstance(self.entities, (tuple, list))
            or tuple(self.entities) != _DESCRIPTOR_ENTITIES
        ):
            raise RecoveryFailureSchemaError(
                f"descriptor entities must be exactly {_DESCRIPTOR_ENTITIES!r}"
            )
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise RecoveryFailureSchemaError(
                "descriptor confidence must be finite within [0, 1]"
            )
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RecoveryFailureSchemaError(
                "descriptor confidence must be finite within [0, 1]"
            )
        if not isinstance(self.reward_mask, Mapping) or set(self.reward_mask) != set(
            _REWARD_TERMS
        ):
            raise RecoveryFailureSchemaError(
                "descriptor reward mask must define distance, grasp, and placement"
            )
        if any(type(value) is not bool for value in self.reward_mask.values()):
            raise RecoveryFailureSchemaError(
                "descriptor reward mask values must be booleans"
            )
        if not any(self.reward_mask.values()):
            raise RecoveryFailureSchemaError(
                "descriptor reward mask must enable at least one term"
            )
        failure_seed = _strict_int(self.failure_seed, name="failure seed")
        snapshot_digest = _digest(self.snapshot_digest, name="snapshot digest")
        assert snapshot_digest is not None
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self, "reward_mask", MappingProxyType(dict(self.reward_mask))
        )
        object.__setattr__(self, "failure_seed", failure_seed)
        object.__setattr__(self, "snapshot_digest", snapshot_digest)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RecoveryFailureDescriptor:
        required = {field.name for field in fields(cls)}
        raw = _exact_mapping(value, required=required, name="descriptor")
        entities = raw["entities"]
        return cls(
            schema_version=raw["schema_version"],
            task_identity=raw["task_identity"],
            category=raw["category"],
            stage=raw["stage"],
            entities=tuple(entities) if isinstance(entities, (tuple, list)) else (),
            confidence=raw["confidence"],
            reward_mask=raw["reward_mask"],
            failure_seed=raw["failure_seed"],
            snapshot_digest=raw["snapshot_digest"],
        )


@dataclass(frozen=True)
class RecoveryAttemptEvidence:
    """Checkpointable primitive-step evidence required by failure predicates."""

    schema_version: int
    trigger_kind: str
    anchor_id: str | None
    anchor_digest: str | None
    failure_seed: int
    phase: str
    phase_enter_step: int
    attempt_count: int
    pickup_attempted: bool
    place_attempted: bool
    release_attempted: bool
    last_progress_step: int
    no_progress_steps: int
    stall_confirm_steps: int
    stable_steps: int
    stable_confirm_steps: int
    injected_category: str | None
    transform_digest: str | None

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_ATTEMPT_SCHEMA_VERSION:
            raise RecoveryFailureSchemaError(
                f"unsupported attempt evidence schema version {self.schema_version}"
            )
        if self.trigger_kind not in {
            "pickup-attempt",
            "place-attempt",
            "post-release",
            "injection",
        }:
            raise RecoveryFailureSchemaError(
                f"unknown trigger kind: {self.trigger_kind!r}"
            )
        if self.anchor_id is not None and (
            not isinstance(self.anchor_id, str) or not self.anchor_id.strip()
        ):
            raise RecoveryFailureSchemaError(
                "anchor id must be non-empty when supplied"
            )
        anchor_digest = _digest(self.anchor_digest, name="anchor digest", optional=True)
        failure_seed = _strict_int(self.failure_seed, name="failure seed")
        if self.phase not in {"approach", "acquire", "place"}:
            raise RecoveryFailureSchemaError(f"unknown recovery phase: {self.phase!r}")
        phase_enter_step = _strict_int(self.phase_enter_step, name="phase enter step")
        attempt_count = _strict_int(self.attempt_count, name="attempt count")
        last_progress_step = _strict_int(
            self.last_progress_step, name="last progress step"
        )
        if last_progress_step < phase_enter_step:
            raise RecoveryFailureSchemaError(
                "last progress step must not precede phase enter step"
            )
        counters = {
            "no progress steps": self.no_progress_steps,
            "stall confirm steps": self.stall_confirm_steps,
            "stable steps": self.stable_steps,
            "stable confirm steps": self.stable_confirm_steps,
        }
        normalized_counters = {
            name: _strict_int(value, name=name) for name, value in counters.items()
        }
        if normalized_counters["stall confirm steps"] == 0:
            raise RecoveryFailureSchemaError("stall confirm steps must be positive")
        if normalized_counters["stable confirm steps"] == 0:
            raise RecoveryFailureSchemaError("stable confirm steps must be positive")
        for field_name in (
            "pickup_attempted",
            "place_attempted",
            "release_attempted",
        ):
            _strict_bool(getattr(self, field_name), name=field_name.replace("_", " "))
        injected_category = _category(self.injected_category, optional=True)
        transform_digest = _digest(
            self.transform_digest,
            name="transform digest",
            optional=True,
        )
        if (injected_category is None) != (transform_digest is None):
            raise RecoveryFailureSchemaError(
                "injected category and transform digest must be supplied together"
            )
        object.__setattr__(self, "anchor_digest", anchor_digest)
        object.__setattr__(self, "failure_seed", failure_seed)
        object.__setattr__(self, "phase_enter_step", phase_enter_step)
        object.__setattr__(self, "attempt_count", attempt_count)
        object.__setattr__(self, "last_progress_step", last_progress_step)
        object.__setattr__(
            self, "no_progress_steps", normalized_counters["no progress steps"]
        )
        object.__setattr__(
            self, "stall_confirm_steps", normalized_counters["stall confirm steps"]
        )
        object.__setattr__(self, "stable_steps", normalized_counters["stable steps"])
        object.__setattr__(
            self, "stable_confirm_steps", normalized_counters["stable confirm steps"]
        )
        object.__setattr__(self, "injected_category", injected_category)
        object.__setattr__(self, "transform_digest", transform_digest)

    @property
    def stalled(self) -> bool:
        return self.no_progress_steps >= self.stall_confirm_steps

    @property
    def stable(self) -> bool:
        return self.stable_steps >= self.stable_confirm_steps

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RecoveryAttemptEvidence:
        required = {field.name for field in fields(cls)}
        raw = _exact_mapping(value, required=required, name="attempt evidence")
        return cls(**raw)


@dataclass(frozen=True)
class LiveSupportGeometry:
    """Runtime-resolved support AABB; configured fallbacks are not accepted."""

    schema_version: int
    geometry_id: str
    geometry_digest: str
    bounds_w: tuple[float, float, float, float]
    top_z_m: float
    source: str
    target_disjoint: bool

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION:
            raise RecoveryFailureSchemaError(
                f"unsupported support geometry schema version {self.schema_version}"
            )
        if not isinstance(self.geometry_id, str) or not self.geometry_id.strip():
            raise RecoveryFailureSchemaError("support geometry id must be non-empty")
        geometry_digest = _digest(self.geometry_digest, name="geometry digest")
        bounds = _finite_tuple(self.bounds_w, length=4, name="support bounds")
        x_lo, x_hi, y_lo, y_hi = bounds
        if x_hi - x_lo <= 2.0 * BOX_HALF_EXTENT_M:
            raise RecoveryFailureSchemaError("support x span cannot contain the box")
        if y_hi - y_lo <= 2.0 * BOX_HALF_EXTENT_M:
            raise RecoveryFailureSchemaError("support y span cannot contain the box")
        if isinstance(self.top_z_m, bool) or not isinstance(self.top_z_m, (int, float)):
            raise RecoveryFailureSchemaError("support top z must be finite")
        top_z = float(self.top_z_m)
        if not math.isfinite(top_z):
            raise RecoveryFailureSchemaError("support top z must be finite")
        if self.source != "live-stage":
            raise RecoveryFailureSchemaError(
                "support geometry must come from the live stage"
            )
        target_disjoint = _strict_bool(
            self.target_disjoint,
            name="support target disjoint",
        )
        assert geometry_digest is not None
        object.__setattr__(self, "geometry_digest", geometry_digest)
        object.__setattr__(self, "bounds_w", bounds)
        object.__setattr__(self, "top_z_m", top_z)
        object.__setattr__(self, "target_disjoint", target_disjoint)


@dataclass(frozen=True)
class VerifiedFailureAnchor:
    """A runtime-validated anchor; arbitrary synthesized poses are prohibited."""

    schema_version: int
    category: str
    anchor_id: str
    anchor_digest: str
    kind: str
    state_transform: Mapping[str, object]
    predicate_verified: bool

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION:
            raise RecoveryFailureSchemaError(
                f"unsupported failure anchor schema version {self.schema_version}"
            )
        category = _category(self.category)
        if category not in {"failed-grasp", "misaligned"}:
            raise RecoveryFailureSchemaError(
                "verified anchors are only valid for failed-grasp or misaligned"
            )
        expected_kind = {
            "failed-grasp": "verified-pickup-anchor",
            "misaligned": "verified-grasp-preserving-anchor",
        }[category]
        if self.kind != expected_kind:
            raise RecoveryFailureSchemaError(
                f"{category} anchor kind must be {expected_kind!r}"
            )
        if not isinstance(self.anchor_id, str) or not self.anchor_id.strip():
            raise RecoveryFailureSchemaError("failure anchor id must be non-empty")
        anchor_digest = _digest(self.anchor_digest, name="failure anchor digest")
        if not isinstance(self.state_transform, Mapping) or not self.state_transform:
            raise RecoveryFailureSchemaError(
                "failure anchor transform must be non-empty"
            )
        frozen_transform = _freeze_json_value(
            self.state_transform,
            name="failure anchor transform",
        )
        predicate_verified = _strict_bool(
            self.predicate_verified,
            name="anchor predicate verified",
        )
        assert anchor_digest is not None
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "anchor_digest", anchor_digest)
        object.__setattr__(self, "state_transform", frozen_transform)
        object.__setattr__(self, "predicate_verified", predicate_verified)


@dataclass(frozen=True)
class FailureRuntimeCapabilityEvidence:
    """Immutable per-category runtime evidence controlling catalog activation."""

    schema_version: int
    task_identity: str
    category: str
    validated_capabilities: frozenset[str]
    evidence_id: str
    target_shelf: LiveSupportGeometry
    ground_support: LiveSupportGeometry | None
    verified_anchor: VerifiedFailureAnchor | None
    evidence_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION:
            raise RecoveryFailureSchemaError(
                f"unsupported runtime capability schema version {self.schema_version}"
            )
        if self.task_identity != PP_BOX_TASK_IDENTITY:
            raise RecoveryFailureSchemaError(
                "runtime evidence task identity is not HOI_pp_box"
            )
        category = _category(self.category)
        assert category is not None
        if not isinstance(self.validated_capabilities, (set, frozenset, tuple, list)):
            raise RecoveryFailureSchemaError(
                "validated capabilities must be a sequence of capability names"
            )
        capabilities = frozenset(self.validated_capabilities)
        if any(not isinstance(name, str) or not name for name in capabilities):
            raise RecoveryFailureSchemaError(
                "validated capability names must be non-empty strings"
            )
        unknown = capabilities - _KNOWN_RUNTIME_CAPABILITIES
        if unknown:
            raise RecoveryFailureSchemaError(
                "unknown runtime capabilities: " + ", ".join(sorted(unknown))
            )
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise RecoveryFailureSchemaError("runtime evidence id must be non-empty")
        if not isinstance(self.target_shelf, LiveSupportGeometry):
            raise RecoveryFailureSchemaError(
                "runtime evidence requires live target shelf geometry"
            )
        if self.target_shelf.target_disjoint:
            raise RecoveryFailureSchemaError(
                "target shelf geometry cannot be target-disjoint"
            )
        if self.ground_support is not None and not isinstance(
            self.ground_support,
            LiveSupportGeometry,
        ):
            raise RecoveryFailureSchemaError("ground support evidence is invalid")
        if self.verified_anchor is not None and not isinstance(
            self.verified_anchor,
            VerifiedFailureAnchor,
        ):
            raise RecoveryFailureSchemaError("verified failure anchor is invalid")
        evidence_digest = _runtime_evidence_digest(self, category, capabilities)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "validated_capabilities", capabilities)
        object.__setattr__(self, "evidence_digest", evidence_digest)


def _runtime_evidence_digest(
    evidence: FailureRuntimeCapabilityEvidence,
    category: str | None = None,
    capabilities: frozenset[str] | None = None,
) -> str:
    normalized_category = evidence.category if category is None else category
    normalized_capabilities = (
        frozenset(evidence.validated_capabilities)
        if capabilities is None
        else capabilities
    )
    payload = (
        "runtime-capability-evidence",
        evidence.schema_version,
        evidence.task_identity,
        normalized_category,
        tuple(sorted(normalized_capabilities)),
        evidence.evidence_id,
        evidence.target_shelf,
        evidence.ground_support,
        evidence.verified_anchor,
    )
    return recovery_state.recovery_value_digest(payload)


@dataclass(frozen=True)
class FailureInjectionPlan:
    """Versioned writer-neutral deterministic state transformation."""

    schema_version: int
    task_identity: str
    category: str
    failure_seed: int
    category_seed: int
    snapshot_digest: str
    runtime_evidence_digest: str
    transform_kind: str
    state_transform: Mapping[str, object]
    anchor_id: str | None
    anchor_digest: str | None
    transform_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_INJECTION_PLAN_SCHEMA_VERSION:
            raise RecoveryFailureSchemaError(
                f"unsupported injection plan schema version {self.schema_version}"
            )
        if self.task_identity != PP_BOX_TASK_IDENTITY:
            raise RecoveryFailureSchemaError(
                "injection plan task identity is not HOI_pp_box"
            )
        category = _category(self.category)
        failure_seed = _strict_int(self.failure_seed, name="failure seed")
        category_seed = _strict_int(self.category_seed, name="category seed")
        snapshot_digest = _digest(self.snapshot_digest, name="snapshot digest")
        evidence_digest = _digest(
            self.runtime_evidence_digest,
            name="runtime evidence digest",
        )
        if not isinstance(self.transform_kind, str) or not self.transform_kind:
            raise RecoveryFailureSchemaError("transform kind must be non-empty")
        if not isinstance(self.state_transform, Mapping) or not self.state_transform:
            raise RecoveryFailureSchemaError("state transform must be non-empty")
        frozen_transform = _freeze_json_value(
            self.state_transform,
            name="state transform",
        )
        if self.anchor_id is not None and (
            not isinstance(self.anchor_id, str) or not self.anchor_id.strip()
        ):
            raise RecoveryFailureSchemaError("plan anchor id must be non-empty")
        anchor_digest = _digest(
            self.anchor_digest, name="plan anchor digest", optional=True
        )
        if (self.anchor_id is None) != (anchor_digest is None):
            raise RecoveryFailureSchemaError(
                "plan anchor id and digest must be supplied together"
            )
        transform_digest = _digest(self.transform_digest, name="transform digest")
        assert category is not None
        assert snapshot_digest is not None
        assert evidence_digest is not None
        assert transform_digest is not None
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "failure_seed", failure_seed)
        object.__setattr__(self, "category_seed", category_seed)
        object.__setattr__(self, "snapshot_digest", snapshot_digest)
        object.__setattr__(self, "runtime_evidence_digest", evidence_digest)
        object.__setattr__(self, "state_transform", frozen_transform)
        object.__setattr__(self, "anchor_digest", anchor_digest)
        object.__setattr__(self, "transform_digest", transform_digest)


@dataclass(frozen=True)
class FailureContinuationRaw:
    """Full primitive-step continuation captured after one real injection."""

    schema_version: int
    env_index: int
    runtime_identity_digest: str
    fixed_action40: tuple[float, ...]
    applied_action40: tuple[float, ...]
    observation_before: Mapping[str, object]
    observation_after: Mapping[str, object]
    reward: float
    reward_terms: Mapping[str, float]
    terminated: bool
    truncated: bool
    terminal_context: DriverTerminalContext
    task_state_before: Mapping[str, object]
    task_state_after: Mapping[str, object]
    rng_before: recovery_state.RecoveryRngState
    rng_after: recovery_state.RecoveryRngState
    contact16_before: tuple[Mapping[str, object], ...]
    contact16_after: tuple[Mapping[str, object], ...]
    live_fall_evidence_before: Mapping[str, object]
    live_fall_evidence_after: Mapping[str, object]
    continuation_readback: FailurePredicateContext
    raw_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_RAW_RECEIPT_SCHEMA_VERSION:
            raise RecoveryFailureSchemaError(
                f"unsupported raw receipt schema version {self.schema_version}"
            )
        env_index = _strict_int(self.env_index, name="continuation env index")
        runtime_digest = _digest(
            self.runtime_identity_digest,
            name="continuation runtime identity digest",
        )
        fixed_action = _finite_tuple(
            self.fixed_action40,
            length=40,
            name="continuation fixed action40",
        )
        applied_action = _finite_tuple(
            self.applied_action40,
            length=40,
            name="continuation applied action40",
        )
        observations: dict[str, object] = {}
        for name in ("observation_before", "observation_after"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise RecoveryFailureSchemaError(
                    f"continuation {name.replace('_', ' ')} must be a non-empty mapping"
                )
            observations[name] = _freeze_recovery_value(value)
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise RecoveryFailureSchemaError("continuation reward must be finite")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise RecoveryFailureSchemaError("continuation reward must be finite")
        if not isinstance(self.reward_terms, Mapping) or set(self.reward_terms) != set(
            _CONTINUATION_REWARD_TERMS
        ):
            raise RecoveryFailureSchemaError(
                "continuation reward terms must contain exactly articulation, "
                "distance, grasp, placement"
            )
        reward_terms: dict[str, float] = {}
        for name in _CONTINUATION_REWARD_TERMS:
            value = self.reward_terms[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecoveryFailureSchemaError(
                    f"continuation reward term {name!r} must be finite"
                )
            normalized = float(value)
            if not math.isfinite(normalized):
                raise RecoveryFailureSchemaError(
                    f"continuation reward term {name!r} must be finite"
                )
            reward_terms[name] = normalized
        terminated = _strict_bool(self.terminated, name="continuation terminated")
        truncated = _strict_bool(self.truncated, name="continuation truncated")
        if not isinstance(self.terminal_context, DriverTerminalContext):
            raise RecoveryFailureSchemaError(
                "continuation terminal context must be DriverTerminalContext"
            )
        try:
            self.terminal_context.validate()
        except (TypeError, ValueError) as exc:
            raise RecoveryFailureSchemaError(
                f"continuation terminal context is invalid: {exc}"
            ) from exc
        task_states: dict[str, object] = {}
        for name in ("task_state_before", "task_state_after"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise RecoveryFailureSchemaError(
                    f"continuation {name.replace('_', ' ')} must be a non-empty mapping"
                )
            task_states[name] = _freeze_recovery_value(value)
        rng_states: dict[str, recovery_state.RecoveryRngState] = {}
        for name in ("rng_before", "rng_after"):
            value = getattr(self, name)
            if not isinstance(value, recovery_state.RecoveryRngState):
                raise RecoveryFailureSchemaError(
                    f"continuation {name.replace('_', ' ')} is invalid"
                )
            rng_states[name] = _freeze_recovery_value(value)  # type: ignore[assignment]
        contacts: dict[str, tuple[Mapping[str, object], ...]] = {}
        for name in ("contact16_before", "contact16_after"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 16:
                raise RecoveryFailureSchemaError(
                    f"continuation {name.replace('_', ' ')} must contain 16 rows"
                )
            indices: list[int] = []
            frozen_rows: list[Mapping[str, object]] = []
            for row_index, row in enumerate(value):
                if not isinstance(row, Mapping) or not row:
                    raise RecoveryFailureSchemaError(
                        f"continuation {name} row {row_index} must be a non-empty mapping"
                    )
                if "sensor_index" not in row or "sensor_scene_key" not in row:
                    raise RecoveryFailureSchemaError(
                        f"continuation {name} row {row_index} lacks sensor identity"
                    )
                indices.append(
                    _strict_int(
                        row["sensor_index"],
                        name=f"continuation {name} sensor index",
                    )
                )
                scene_key = row["sensor_scene_key"]
                if not isinstance(scene_key, str) or not scene_key:
                    raise RecoveryFailureSchemaError(
                        f"continuation {name} sensor scene key must be non-empty"
                    )
                frozen_rows.append(_freeze_recovery_value(row))  # type: ignore[arg-type]
            if set(indices) != set(range(16)):
                raise RecoveryFailureSchemaError(
                    f"continuation {name.replace('_', ' ')} must cover sensor indices 0..15"
                )
            contacts[name] = tuple(frozen_rows)
        fall_evidence: dict[str, object] = {}
        for name in ("live_fall_evidence_before", "live_fall_evidence_after"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise RecoveryFailureSchemaError(
                    f"continuation {name.replace('_', ' ')} must be a non-empty mapping"
                )
            fall_evidence[name] = _freeze_recovery_value(value)
        if not isinstance(self.continuation_readback, FailurePredicateContext):
            raise RecoveryFailureSchemaError(
                "continuation readback must be FailurePredicateContext"
            )
        continuation_readback = _freeze_recovery_value(self.continuation_readback)
        assert runtime_digest is not None
        object.__setattr__(self, "env_index", env_index)
        object.__setattr__(self, "runtime_identity_digest", runtime_digest)
        object.__setattr__(self, "fixed_action40", fixed_action)
        object.__setattr__(self, "applied_action40", applied_action)
        for name, value in observations.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "reward_terms", MappingProxyType(reward_terms))
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        for name, value in task_states.items():
            object.__setattr__(self, name, value)
        for name, value in rng_states.items():
            object.__setattr__(self, name, value)
        for name, value in contacts.items():
            object.__setattr__(self, name, value)
        for name, value in fall_evidence.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "continuation_readback", continuation_readback)
        object.__setattr__(self, "raw_digest", _continuation_raw_digest(self))


def _continuation_raw_payload(value: FailureContinuationRaw) -> tuple[object, ...]:
    return (
        "failure-continuation-raw",
        value.schema_version,
        value.env_index,
        value.runtime_identity_digest,
        value.fixed_action40,
        value.applied_action40,
        value.observation_before,
        value.observation_after,
        value.reward,
        value.reward_terms,
        value.terminated,
        value.truncated,
        value.terminal_context,
        value.task_state_before,
        value.task_state_after,
        value.rng_before,
        value.rng_after,
        value.contact16_before,
        value.contact16_after,
        value.live_fall_evidence_before,
        value.live_fall_evidence_after,
        value.continuation_readback,
    )


def _continuation_raw_digest(value: FailureContinuationRaw) -> str:
    return recovery_state.recovery_value_digest(_continuation_raw_payload(value))


@dataclass(frozen=True, init=False)
class RawInjectorExecutionReceipt:
    """Factory-issued canonical receipt for one injector plus primitive step."""

    schema_version: int
    repeat_index: int
    task_identity: str
    category: str
    env_index: int
    runtime_identity_digest: str
    source_snapshot_digest: str
    plan: FailureInjectionPlan
    operation_trace: tuple[str, ...]
    injection_readback: FailurePredicateContext
    continuation: FailureContinuationRaw
    injection_readback_digest: str
    continuation_digest: str
    execution_digest: str
    receipt_digest: str
    _issuance_token: object = field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class FailureCatalogQualification:
    """Factory-issued proof that exactly two real executions were identical."""

    schema_version: int
    task_identity: str
    category: str
    runtime_evidence: FailureRuntimeCapabilityEvidence
    receipts: tuple[RawInjectorExecutionReceipt, RawInjectorExecutionReceipt]
    qualification_digest: str
    _issuance_token: object = field(repr=False, compare=False)


def _missing_runtime_capabilities(
    evidence: FailureRuntimeCapabilityEvidence,
) -> set[str]:
    required = required_runtime_capabilities(evidence.category)
    missing = set(required - evidence.validated_capabilities)
    if evidence.category == "dropped":
        if evidence.ground_support is None:
            missing.add("live_ground_support")
        elif not evidence.ground_support.target_disjoint:
            missing.add("target_disjoint_ground_region")
    elif evidence.category == "failed-grasp":
        anchor = evidence.verified_anchor
        if (
            anchor is None
            or anchor.category != "failed-grasp"
            or anchor.kind != "verified-pickup-anchor"
            or not anchor.predicate_verified
        ):
            missing.add("verified_pickup_anchor")
    elif evidence.category == "misaligned":
        anchor = evidence.verified_anchor
        if (
            anchor is None
            or anchor.category != "misaligned"
            or anchor.kind != "verified-grasp-preserving-anchor"
            or not anchor.predicate_verified
        ):
            missing.add("verified_grasp_preserving_anchor")
    return missing


def derive_category_seed(category: str, failure_seed: int, snapshot_digest: str) -> int:
    """Derive an isolated seed without consuming any global RNG state."""

    normalized = _category(category)
    seed = _strict_int(failure_seed, name="failure seed")
    digest = _digest(snapshot_digest, name="snapshot digest")
    assert normalized is not None
    assert digest is not None
    payload = {
        "schema_version": RECOVERY_INJECTION_PLAN_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "category": normalized,
        "failure_seed": seed,
        "snapshot_digest": digest,
    }
    raw_digest = hashlib.sha256(
        _canonical_json_bytes(payload, name="category seed payload")
    ).digest()
    return int.from_bytes(raw_digest[:8], byteorder="big", signed=False)


def _box_root_transform(position_w: tuple[float, float, float]) -> dict[str, object]:
    return {
        "box_position_w": position_w,
        "box_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
        "box_linear_velocity_w": (0.0, 0.0, 0.0),
        "box_angular_velocity_w": (0.0, 0.0, 0.0),
    }


def _near_shelf_transform(
    geometry: LiveSupportGeometry,
    rng: random.Random,
) -> dict[str, object]:
    x_lo, x_hi, y_lo, y_hi = geometry.bounds_w
    inner_x_lo = x_lo + BOX_HALF_EXTENT_M
    inner_x_hi = x_hi - BOX_HALF_EXTENT_M
    inner_y_lo = y_lo + BOX_HALF_EXTENT_M
    inner_y_hi = y_hi - BOX_HALF_EXTENT_M
    overhang = BOX_HALF_EXTENT_M * (0.2 + 0.6 * rng.random())
    side = rng.randrange(4)
    if side == 0:
        x = inner_x_lo - overhang
        y = rng.uniform(inner_y_lo, inner_y_hi)
    elif side == 1:
        x = inner_x_hi + overhang
        y = rng.uniform(inner_y_lo, inner_y_hi)
    elif side == 2:
        x = rng.uniform(inner_x_lo, inner_x_hi)
        y = inner_y_lo - overhang
    else:
        x = rng.uniform(inner_x_lo, inner_x_hi)
        y = inner_y_hi + overhang
    return _box_root_transform((x, y, geometry.top_z_m + BOX_HALF_EXTENT_M))


def _ground_transform(
    geometry: LiveSupportGeometry,
    rng: random.Random,
) -> dict[str, object]:
    x_lo, x_hi, y_lo, y_hi = geometry.bounds_w
    x = rng.uniform(x_lo + BOX_HALF_EXTENT_M, x_hi - BOX_HALF_EXTENT_M)
    y = rng.uniform(y_lo + BOX_HALF_EXTENT_M, y_hi - BOX_HALF_EXTENT_M)
    return _box_root_transform((x, y, geometry.top_z_m + BOX_HALF_EXTENT_M))


def build_failure_injection_plan(
    descriptor: RecoveryFailureDescriptor,
    evidence: FailureRuntimeCapabilityEvidence,
) -> FailureInjectionPlan:
    """Build exactly one deterministic plan or raise without retrying the seed."""

    if not isinstance(descriptor, RecoveryFailureDescriptor):
        raise RecoveryFailureSchemaError("injection descriptor is invalid")
    if not isinstance(evidence, FailureRuntimeCapabilityEvidence):
        raise RecoveryFailureSchemaError("runtime capability evidence is invalid")
    if descriptor.category != evidence.category:
        raise RecoveryFailureSchemaError(
            "descriptor category does not match runtime capability evidence"
        )
    missing = _missing_runtime_capabilities(evidence)
    if missing:
        raise RecoveryFailureCapabilityError(descriptor.category, missing)

    category_seed = derive_category_seed(
        descriptor.category,
        descriptor.failure_seed,
        descriptor.snapshot_digest,
    )
    rng = random.Random(category_seed)
    anchor_id = None
    anchor_digest = None
    if descriptor.category == "near-shelf-misplaced":
        transform_kind = "live-target-near-shelf-box-root"
        state_transform = _near_shelf_transform(evidence.target_shelf, rng)
    elif descriptor.category == "dropped":
        assert evidence.ground_support is not None
        transform_kind = "live-ground-box-root"
        state_transform = _ground_transform(evidence.ground_support, rng)
    else:
        assert evidence.verified_anchor is not None
        transform_kind = evidence.verified_anchor.kind
        state_transform = evidence.verified_anchor.state_transform
        anchor_id = evidence.verified_anchor.anchor_id
        anchor_digest = evidence.verified_anchor.anchor_digest

    plan_identity = {
        "schema_version": RECOVERY_INJECTION_PLAN_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "category": descriptor.category,
        "failure_seed": descriptor.failure_seed,
        "category_seed": category_seed,
        "snapshot_digest": descriptor.snapshot_digest,
        "runtime_evidence_digest": evidence.evidence_digest,
        "transform_kind": transform_kind,
        "state_transform": state_transform,
        "anchor_id": anchor_id,
        "anchor_digest": anchor_digest,
    }
    transform_digest = hashlib.sha256(
        _canonical_json_bytes(plan_identity, name="failure injection plan")
    ).hexdigest()
    return FailureInjectionPlan(
        **plan_identity,
        transform_digest=transform_digest,
    )


def _validate_privileged_telemetry(value: object) -> PrivilegedRecoveryTelemetry:
    if not isinstance(value, PrivilegedRecoveryTelemetry):
        raise RecoveryFailureSchemaError(
            "failure predicates require PrivilegedRecoveryTelemetry"
        )
    if value.schema_version != RECOVERY_TELEMETRY_SCHEMA_VERSION:
        raise RecoveryFailureSchemaError(
            "unsupported privileged telemetry schema version"
        )
    if value.task_identity != PP_BOX_TASK_IDENTITY:
        raise RecoveryFailureSchemaError(
            "privileged telemetry task identity is not HOI_pp_box"
        )
    for field_name in (
        "grasp",
        "placement",
        "success",
        "fall_candidate",
        "fall",
        "time_limit",
    ):
        _strict_bool(getattr(value, field_name), name=f"telemetry {field_name}")
    if value.grasp is not value.grasp_evidence.bimanual_grasp:
        raise RecoveryFailureSchemaError("grasp telemetry contradicts its evidence")
    if value.placement is not value.success:
        raise RecoveryFailureSchemaError(
            "placement and task success telemetry disagree"
        )
    expected_terminal = (
        "success"
        if value.success
        else "fall"
        if value.fall
        else "time_limit"
        if value.time_limit
        else "running"
    )
    if value.terminal_reason != expected_terminal:
        raise RecoveryFailureSchemaError("task success and terminal telemetry disagree")
    for name in ("xy_mismatch_m", "z_mismatch_m"):
        mismatch = getattr(value, name)
        if not isinstance(mismatch, (int, float)) or isinstance(mismatch, bool):
            raise RecoveryFailureSchemaError(f"{name} must be finite and non-negative")
        if not math.isfinite(float(mismatch)) or float(mismatch) < 0.0:
            raise RecoveryFailureSchemaError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class FailurePredicateContext:
    """Source truth needed to classify one primitive control step."""

    telemetry: PrivilegedRecoveryTelemetry
    attempt: RecoveryAttemptEvidence
    ground_supported: bool
    target_disjoint: bool
    box_axis_aligned: bool

    def __post_init__(self) -> None:
        _validate_privileged_telemetry(self.telemetry)
        if not isinstance(self.attempt, RecoveryAttemptEvidence):
            raise RecoveryFailureSchemaError(
                "failure predicates require versioned attempt evidence"
            )
        for name in ("ground_supported", "target_disjoint", "box_axis_aligned"):
            _strict_bool(getattr(self, name), name=name.replace("_", " "))


def _predicate_facts(context: FailurePredicateContext) -> dict[str, bool]:
    if not isinstance(context, FailurePredicateContext):
        raise RecoveryFailureSchemaError("failure predicate context is invalid")
    telemetry = _validate_privileged_telemetry(context.telemetry)
    attempt = context.attempt
    running_safe = (
        telemetry.terminal_reason == "running" and not telemetry.fall_candidate
    )
    pose_evidence = bool(
        telemetry.grasp_evidence.left_pose_valid
        and telemetry.grasp_evidence.right_pose_valid
    )
    near_shelf = bool(
        attempt.stable
        and context.box_axis_aligned
        and not telemetry.placement
        and 0.0 < telemetry.xy_mismatch_m < BOX_HALF_EXTENT_M
        and telemetry.z_mismatch_m <= PLACEMENT_Z_TOLERANCE_M
    )
    dropped = bool(
        attempt.stable and context.ground_supported and context.target_disjoint
    )
    return {
        "running_safe": running_safe,
        "pose_evidence": pose_evidence,
        "near_shelf": near_shelf,
        "dropped": dropped,
    }


def evaluate_failure_predicates(
    context: FailurePredicateContext,
) -> Mapping[str, bool]:
    """Evaluate the disjoint catalog predicates; unknown states match nothing."""

    telemetry = context.telemetry
    attempt = context.attempt
    facts = _predicate_facts(context)
    running = facts["running_safe"]
    grasp = telemetry.grasp
    dropped = facts["dropped"]
    near = facts["near_shelf"]
    conflict = dropped and near
    matches = {
        "dropped": bool(running and not grasp and dropped and not near),
        "failed-grasp": bool(
            running
            and not grasp
            and not dropped
            and not near
            and facts["pose_evidence"]
            and attempt.pickup_attempted
            and attempt.stalled
        ),
        "misaligned": bool(
            running
            and grasp
            and not telemetry.placement
            and attempt.place_attempted
            and attempt.stalled
        ),
        "near-shelf-misplaced": bool(
            running and not grasp and not dropped and near and attempt.release_attempted
        ),
    }
    if conflict:
        return MappingProxyType({category: False for category in matches})
    return MappingProxyType(matches)


def classify_recoverable_failure(context: FailurePredicateContext) -> str | None:
    """Return the sole recoverable category, or fail closed on conflict/unknown."""

    facts = _predicate_facts(context)
    if not facts["running_safe"]:
        return None
    if facts["dropped"] and facts["near_shelf"]:
        raise RecoveryFailurePredicateConflictError(
            "failure predicate conflict: box is both ground-supported and near-shelf"
        )
    matches = evaluate_failure_predicates(context)
    matched = tuple(category for category, active in matches.items() if active)
    if len(matched) > 1:
        raise RecoveryFailurePredicateConflictError(
            "failure predicate conflict: " + ", ".join(matched)
        )
    return matched[0] if matched else None


def _derive_failure_execution_truth(
    plan: object,
    readback: object,
    continuation_readback: object,
) -> dict[str, object]:
    if not isinstance(plan, FailureInjectionPlan):
        raise RecoveryFailureSchemaError("execution receipt plan is invalid")
    contexts = {
        "readback": readback,
        "continuation": continuation_readback,
    }
    for name, context in contexts.items():
        if not isinstance(context, FailurePredicateContext):
            raise RecoveryFailureSchemaError(
                f"execution physical {name} must be FailurePredicateContext"
            )
        attempt = context.attempt
        if not isinstance(attempt, RecoveryAttemptEvidence):
            raise RecoveryFailureSchemaError(
                f"execution physical {name} attempt evidence is invalid"
            )
        if not (
            attempt.failure_seed == plan.failure_seed
            and attempt.injected_category == plan.category
            and attempt.transform_digest == plan.transform_digest
        ):
            raise RecoveryFailureSchemaError(
                f"execution physical {name} attempt metadata does not bind its plan"
            )

    observed: dict[str, str | None] = {}
    for name, context in contexts.items():
        try:
            observed[name] = classify_recoverable_failure(context)
        except RecoveryFailurePredicateConflictError:
            observed[name] = None
    readback_category = observed["readback"]
    continuation_category = observed["continuation"]
    predicate_passed = readback_category == plan.category
    category_passed = bool(predicate_passed and continuation_category == plan.category)
    return {
        "readback_state_digest": recovery_state.recovery_value_digest(readback),
        "continuation_digest": recovery_state.recovery_value_digest(
            continuation_readback
        ),
        "predicate_passed": predicate_passed,
        "observed_category": readback_category,
        "continuation_category": continuation_category,
        "category_passed": category_passed,
    }


def _clone_failure_continuation(
    value: FailureContinuationRaw,
) -> FailureContinuationRaw:
    if not isinstance(value, FailureContinuationRaw):
        raise RecoveryFailureSchemaError(
            "primitive continuation runner must return FailureContinuationRaw"
        )
    init_values = {
        item.name: getattr(value, item.name)
        for item in fields(FailureContinuationRaw)
        if item.init
    }
    return FailureContinuationRaw(**init_values)


def _receipt_execution_payload(
    *,
    task_identity: str,
    category: str,
    env_index: int,
    runtime_identity_digest: str,
    source_snapshot_digest: str,
    plan: FailureInjectionPlan,
    operation_trace: tuple[str, ...],
    injection_readback: FailurePredicateContext,
    continuation: FailureContinuationRaw,
) -> tuple[object, ...]:
    return (
        "raw-injector-execution",
        RECOVERY_RAW_RECEIPT_SCHEMA_VERSION,
        task_identity,
        category,
        env_index,
        runtime_identity_digest,
        source_snapshot_digest,
        plan,
        operation_trace,
        injection_readback,
        _continuation_raw_payload(continuation),
    )


def _canonical_receipt_payload(
    receipt: RawInjectorExecutionReceipt,
) -> tuple[object, ...]:
    return (
        "raw-injector-execution-receipt",
        receipt.schema_version,
        receipt.repeat_index,
        *_receipt_execution_payload(
            task_identity=receipt.task_identity,
            category=receipt.category,
            env_index=receipt.env_index,
            runtime_identity_digest=receipt.runtime_identity_digest,
            source_snapshot_digest=receipt.source_snapshot_digest,
            plan=receipt.plan,
            operation_trace=receipt.operation_trace,
            injection_readback=receipt.injection_readback,
            continuation=receipt.continuation,
        ),
    )


def _issue_raw_injector_receipt(
    result: FailureInjectionResult,
    *,
    repeat_index: int,
    fixed_action40: tuple[float, ...],
    continuation: FailureContinuationRaw,
) -> RawInjectorExecutionReceipt:
    if not isinstance(result, FailureInjectionResult):
        raise RecoveryFailureSchemaError(
            "raw receipt requires a FailureInjectionResult"
        )
    if result.readback_passed is not True or result.category != result.plan.category:
        raise RecoveryFailureSchemaError(
            "raw receipt requires a verified injection result"
        )
    repeat = _strict_int(repeat_index, name="receipt repeat index")
    normalized_continuation = _clone_failure_continuation(continuation)
    if normalized_continuation.fixed_action40 != fixed_action40:
        raise RecoveryFailureSchemaError(
            "continuation fixed action40 does not match the qualification action"
        )
    truth = _derive_failure_execution_truth(
        result.plan,
        result.readback,
        normalized_continuation.continuation_readback,
    )
    if (
        truth["predicate_passed"] is not True
        or truth["observed_category"] != result.category
        or truth["continuation_category"] != result.category
        or truth["category_passed"] is not True
        or normalized_continuation.terminated
        or normalized_continuation.truncated
    ):
        raise RecoveryFailureVerificationError(
            category=result.category,
            failure_seed=result.plan.failure_seed,
            observed_category=truth["continuation_category"],  # type: ignore[arg-type]
            detail="fixed-action continuation did not preserve the injected failure",
        )
    injection_readback = _freeze_recovery_value(result.readback)
    injection_digest = recovery_state.recovery_value_digest(injection_readback)
    continuation_digest = _continuation_raw_digest(normalized_continuation)
    execution_payload = _receipt_execution_payload(
        task_identity=PP_BOX_TASK_IDENTITY,
        category=result.category,
        env_index=normalized_continuation.env_index,
        runtime_identity_digest=normalized_continuation.runtime_identity_digest,
        source_snapshot_digest=result.plan.snapshot_digest,
        plan=result.plan,
        operation_trace=_QUALIFICATION_OPERATION_TRACE,
        injection_readback=injection_readback,  # type: ignore[arg-type]
        continuation=normalized_continuation,
    )
    execution_digest = recovery_state.recovery_value_digest(execution_payload)
    token = object()
    receipt = object.__new__(RawInjectorExecutionReceipt)
    values = {
        "schema_version": RECOVERY_RAW_RECEIPT_SCHEMA_VERSION,
        "repeat_index": repeat,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "category": result.category,
        "env_index": normalized_continuation.env_index,
        "runtime_identity_digest": normalized_continuation.runtime_identity_digest,
        "source_snapshot_digest": result.plan.snapshot_digest,
        "plan": result.plan,
        "operation_trace": _QUALIFICATION_OPERATION_TRACE,
        "injection_readback": injection_readback,
        "continuation": normalized_continuation,
        "injection_readback_digest": injection_digest,
        "continuation_digest": continuation_digest,
        "execution_digest": execution_digest,
        "_issuance_token": token,
    }
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    receipt_digest = recovery_state.recovery_value_digest(
        _canonical_receipt_payload(receipt)
    )
    object.__setattr__(receipt, "receipt_digest", receipt_digest)
    _ISSUED_RECEIPT_DIGESTS[token] = receipt_digest
    return receipt


def _issued_receipt_is_valid(receipt: object) -> bool:
    if type(receipt) is not RawInjectorExecutionReceipt:
        return False
    try:
        if (
            receipt.schema_version != RECOVERY_RAW_RECEIPT_SCHEMA_VERSION
            or receipt.task_identity != PP_BOX_TASK_IDENTITY
            or receipt.operation_trace != _QUALIFICATION_OPERATION_TRACE
            or receipt.category != receipt.plan.category
            or receipt.env_index != receipt.continuation.env_index
            or receipt.runtime_identity_digest
            != receipt.continuation.runtime_identity_digest
            or receipt.source_snapshot_digest != receipt.plan.snapshot_digest
            or receipt.continuation.terminated
            or receipt.continuation.truncated
        ):
            return False
        _strict_int(receipt.repeat_index, name="receipt repeat index")
        if not isinstance(receipt.plan, FailureInjectionPlan):
            return False
        if not isinstance(receipt.injection_readback, FailurePredicateContext):
            return False
        if not isinstance(receipt.continuation, FailureContinuationRaw):
            return False
        continuation_digest = _continuation_raw_digest(receipt.continuation)
        injection_digest = recovery_state.recovery_value_digest(
            receipt.injection_readback
        )
        truth = _derive_failure_execution_truth(
            receipt.plan,
            receipt.injection_readback,
            receipt.continuation.continuation_readback,
        )
        if (
            receipt.continuation.raw_digest != continuation_digest
            or receipt.continuation_digest != continuation_digest
            or receipt.injection_readback_digest != injection_digest
            or truth["predicate_passed"] is not True
            or truth["observed_category"] != receipt.category
            or truth["continuation_category"] != receipt.category
            or truth["category_passed"] is not True
        ):
            return False
        execution_digest = recovery_state.recovery_value_digest(
            _receipt_execution_payload(
                task_identity=receipt.task_identity,
                category=receipt.category,
                env_index=receipt.env_index,
                runtime_identity_digest=receipt.runtime_identity_digest,
                source_snapshot_digest=receipt.source_snapshot_digest,
                plan=receipt.plan,
                operation_trace=receipt.operation_trace,
                injection_readback=receipt.injection_readback,
                continuation=receipt.continuation,
            )
        )
        receipt_digest = recovery_state.recovery_value_digest(
            _canonical_receipt_payload(receipt)
        )
        return bool(
            receipt.execution_digest == execution_digest
            and receipt.receipt_digest == receipt_digest
            and _ISSUED_RECEIPT_DIGESTS.get(receipt._issuance_token)
            == receipt_digest
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        RecoveryFailureError,
        recovery_state.RecoveryStateSchemaError,
    ):
        return False


def _qualification_payload(
    qualification: FailureCatalogQualification,
) -> tuple[object, ...]:
    return (
        "failure-catalog-qualification",
        qualification.schema_version,
        qualification.task_identity,
        qualification.category,
        qualification.runtime_evidence,
        tuple(
            _canonical_receipt_payload(receipt)
            for receipt in qualification.receipts
        ),
    )


def _qualification_is_effective(qualification: object) -> bool:
    if type(qualification) is not FailureCatalogQualification:
        return False
    try:
        evidence = qualification.runtime_evidence
        receipts = qualification.receipts
        if (
            qualification.schema_version
            != RECOVERY_CATALOG_QUALIFICATION_SCHEMA_VERSION
            or qualification.task_identity != PP_BOX_TASK_IDENTITY
            or not isinstance(evidence, FailureRuntimeCapabilityEvidence)
            or qualification.category != evidence.category
            or evidence.evidence_digest != _runtime_evidence_digest(evidence)
            or _missing_runtime_capabilities(evidence)
            or not isinstance(receipts, tuple)
            or len(receipts) != 2
            or tuple(receipt.repeat_index for receipt in receipts) != (0, 1)
            or not all(_issued_receipt_is_valid(receipt) for receipt in receipts)
        ):
            return False
        first, second = receipts
        if (
            first.category != qualification.category
            or second.category != qualification.category
            or first.plan != second.plan
            or first.plan.runtime_evidence_digest != evidence.evidence_digest
            or first.execution_digest != second.execution_digest
            or first.env_index != second.env_index
            or first.runtime_identity_digest != second.runtime_identity_digest
            or first.continuation.fixed_action40
            != second.continuation.fixed_action40
        ):
            return False
        descriptor = RecoveryFailureDescriptor(
            schema_version=RECOVERY_FAILURE_DESCRIPTOR_SCHEMA_VERSION,
            task_identity=PP_BOX_TASK_IDENTITY,
            category=qualification.category,
            stage=_CATEGORY_INITIAL_STAGE[qualification.category],
            entities=_DESCRIPTOR_ENTITIES,
            confidence=1.0,
            reward_mask={term: True for term in _REWARD_TERMS},
            failure_seed=first.plan.failure_seed,
            snapshot_digest=first.plan.snapshot_digest,
        )
        if first.plan != build_failure_injection_plan(descriptor, evidence):
            return False
        qualification_digest = recovery_state.recovery_value_digest(
            _qualification_payload(qualification)
        )
        return bool(
            qualification.qualification_digest == qualification_digest
            and _ISSUED_QUALIFICATION_DIGESTS.get(
                qualification._issuance_token
            )
            == qualification_digest
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        RecoveryFailureError,
        recovery_state.RecoveryStateSchemaError,
    ):
        return False


def recovery_succeeded(telemetry: PrivilegedRecoveryTelemetry) -> bool:
    """Recovery completion is exactly the task's authoritative success terminal."""

    state = _validate_privileged_telemetry(telemetry)
    return state.terminal_reason == "success"


@runtime_checkable
class RecoveryFailureWriter(Protocol):
    """Writes one validated plan through the simulator adapter's selected-env API."""

    def write_recovery_failure(self, plan: FailureInjectionPlan) -> None: ...


@runtime_checkable
class RecoveryFailureSettler(Protocol):
    """Performs the runtime-validated deterministic settle procedure."""

    def settle_recovery_failure(self, plan: FailureInjectionPlan) -> None: ...


@runtime_checkable
class RecoveryFailureReader(Protocol):
    """Reads privileged source truth after settle, outside actor observations."""

    def read_recovery_failure(
        self, plan: FailureInjectionPlan
    ) -> FailurePredicateContext: ...


@runtime_checkable
class RecoveryFailureContinuationRunner(Protocol):
    """Executes one fixed primitive action and returns the complete raw state."""

    def run_failure_continuation(
        self,
        plan: FailureInjectionPlan,
        fixed_action40: tuple[float, ...],
    ) -> FailureContinuationRaw: ...


@dataclass(frozen=True)
class FailureInjectionResult:
    plan: FailureInjectionPlan
    category: str
    readback_passed: bool
    readback: FailurePredicateContext


def inject_recovery_failure(
    env: object,
    snapshot: object,
    descriptor: RecoveryFailureDescriptor,
    evidence: FailureRuntimeCapabilityEvidence,
    *,
    writer: RecoveryFailureWriter,
    settler: RecoveryFailureSettler,
    reader: RecoveryFailureReader,
) -> FailureInjectionResult:
    """Restore then perform one write/settle/readback attempt with no resampling."""

    if not isinstance(descriptor, RecoveryFailureDescriptor):
        raise RecoveryFailureSchemaError("injection descriptor is invalid")
    actual_snapshot_digest = recovery_state.recovery_state_digest(snapshot)
    if descriptor.snapshot_digest != actual_snapshot_digest:
        raise RecoveryFailureSnapshotDigestError(
            expected_digest=descriptor.snapshot_digest,
            actual_digest=actual_snapshot_digest,
        )
    plan = build_failure_injection_plan(descriptor, evidence)
    missing_hooks = []
    if not isinstance(writer, RecoveryFailureWriter):
        missing_hooks.append("explicit_failure_writer")
    if not isinstance(settler, RecoveryFailureSettler):
        missing_hooks.append("deterministic_settle")
    if not isinstance(reader, RecoveryFailureReader):
        missing_hooks.append("privileged_readback")
    if missing_hooks:
        raise RecoveryFailureCapabilityError(plan.category, missing_hooks)

    recovery_state.restore_recovery_state(
        env,
        snapshot,
        snapshot_digest=actual_snapshot_digest,
        required_capabilities=_SNAPSHOT_RESTORE_REQUIRED_CAPABILITIES,
        task_identity=PP_BOX_TASK_IDENTITY,
    )
    writer.write_recovery_failure(plan)
    settler.settle_recovery_failure(plan)
    readback = reader.read_recovery_failure(plan)
    if not isinstance(readback, FailurePredicateContext):
        raise RecoveryFailureVerificationError(
            category=plan.category,
            failure_seed=plan.failure_seed,
            observed_category=None,
            detail="readback did not return FailurePredicateContext",
        )
    attempt = readback.attempt
    metadata_matches = bool(
        attempt.failure_seed == plan.failure_seed
        and attempt.injected_category == plan.category
        and attempt.transform_digest == plan.transform_digest
    )
    try:
        observed = classify_recoverable_failure(readback)
    except RecoveryFailurePredicateConflictError as exc:
        raise RecoveryFailureVerificationError(
            category=plan.category,
            failure_seed=plan.failure_seed,
            observed_category=None,
            detail=str(exc),
        ) from exc
    if not metadata_matches or observed != plan.category:
        detail = (
            "readback metadata does not match the fixed plan"
            if not metadata_matches
            else f"readback classified {observed!r}"
        )
        raise RecoveryFailureVerificationError(
            category=plan.category,
            failure_seed=plan.failure_seed,
            observed_category=observed,
            detail=detail,
        )
    return FailureInjectionResult(
        plan=plan,
        category=plan.category,
        readback_passed=True,
        readback=readback,
    )


def qualify_failure_catalog_entry(
    *,
    env: object,
    snapshot: object,
    descriptor: RecoveryFailureDescriptor,
    runtime_evidence: FailureRuntimeCapabilityEvidence,
    writer: RecoveryFailureWriter,
    settler: RecoveryFailureSettler,
    reader: RecoveryFailureReader,
    continuation_runner: RecoveryFailureContinuationRunner,
    fixed_action40: Sequence[float],
) -> FailureCatalogQualification:
    """Issue a catalog proof from exactly two real injector/step executions."""

    if not isinstance(continuation_runner, RecoveryFailureContinuationRunner):
        category = getattr(descriptor, "category", "unknown")
        raise RecoveryFailureCapabilityError(
            str(category),
            ("fixed_primitive_continuation",),
        )
    fixed_action = _finite_tuple(
        fixed_action40,
        length=40,
        name="catalog qualification fixed action40",
    )
    receipts: list[RawInjectorExecutionReceipt] = []
    for repeat_index in range(2):
        result = inject_recovery_failure(
            env,
            snapshot,
            descriptor,
            runtime_evidence,
            writer=writer,
            settler=settler,
            reader=reader,
        )
        continuation = continuation_runner.run_failure_continuation(
            result.plan,
            fixed_action,
        )
        receipts.append(
            _issue_raw_injector_receipt(
                result,
                repeat_index=repeat_index,
                fixed_action40=fixed_action,
                continuation=continuation,
            )
        )
    first, second = receipts
    if first.execution_digest != second.execution_digest:
        raise RecoveryFailureVerificationError(
            category=descriptor.category,
            failure_seed=descriptor.failure_seed,
            observed_category=second.category,
            detail="two canonical injector executions were not identical",
        )
    token = object()
    qualification = object.__new__(FailureCatalogQualification)
    values = {
        "schema_version": RECOVERY_CATALOG_QUALIFICATION_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "category": descriptor.category,
        "runtime_evidence": runtime_evidence,
        "receipts": (first, second),
        "_issuance_token": token,
    }
    for name, value in values.items():
        object.__setattr__(qualification, name, value)
    qualification_digest = recovery_state.recovery_value_digest(
        _qualification_payload(qualification)
    )
    object.__setattr__(qualification, "qualification_digest", qualification_digest)
    _ISSUED_QUALIFICATION_DIGESTS[token] = qualification_digest
    if not _qualification_is_effective(qualification):
        raise RecoveryFailureVerificationError(
            category=descriptor.category,
            failure_seed=descriptor.failure_seed,
            observed_category=None,
            detail="factory-issued catalog qualification failed canonical validation",
        )
    return qualification


__all__ = [
    "BOX_HALF_EXTENTS_M",
    "BOX_HALF_EXTENT_M",
    "DECLARED_FAILURE_CATEGORIES",
    "PLACEMENT_Z_TOLERANCE_M",
    "PP_BOX_TASK_IDENTITY",
    "RECOVERY_ACTIVATION_SCHEMA_VERSION",
    "RECOVERY_ATTEMPT_SCHEMA_VERSION",
    "RECOVERY_CATALOG_QUALIFICATION_SCHEMA_VERSION",
    "RECOVERY_FAILURE_DESCRIPTOR_SCHEMA_VERSION",
    "RECOVERY_INJECTION_PLAN_SCHEMA_VERSION",
    "RECOVERY_RAW_RECEIPT_SCHEMA_VERSION",
    "RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION",
    "FailureCatalogQualification",
    "FailureContinuationRaw",
    "FailureInjectionPlan",
    "FailureInjectionResult",
    "FailurePredicateContext",
    "FailureRuntimeCapabilityEvidence",
    "LiveSupportGeometry",
    "RawInjectorExecutionReceipt",
    "RecoveryActivationBaseline",
    "RecoveryActivationContext",
    "RecoveryAttemptEvidence",
    "RecoveryFailureCapabilityError",
    "RecoveryFailureCatalogEntry",
    "RecoveryFailureContinuationRunner",
    "RecoveryFailureDescriptor",
    "RecoveryFailureError",
    "RecoveryFailurePredicateConflictError",
    "RecoveryFailureReader",
    "RecoveryFailureSchemaError",
    "RecoveryFailureSettler",
    "RecoveryFailureSnapshotDigestError",
    "RecoveryFailureVerificationError",
    "RecoveryFailureWriter",
    "RecoveryRewardBinding",
    "VerifiedFailureAnchor",
    "begin_recovery_activation",
    "build_failure_injection_plan",
    "classify_recoverable_failure",
    "declared_failure_catalog",
    "derive_category_seed",
    "effective_failure_catalog",
    "evaluate_failure_predicates",
    "inject_recovery_failure",
    "qualify_failure_catalog_entry",
    "recovery_reward_bindings",
    "recovery_reward_component_scope",
    "recovery_state",
    "recovery_succeeded",
    "required_runtime_capabilities",
    "resolve_recovery_reward_telemetry",
]
