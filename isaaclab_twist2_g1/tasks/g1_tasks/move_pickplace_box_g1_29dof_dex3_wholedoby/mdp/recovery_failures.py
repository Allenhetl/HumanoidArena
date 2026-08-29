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
    PrivilegedRecoveryTelemetry,
)

PP_BOX_TASK_IDENTITY = recovery_state.PP_BOX_TASK_IDENTITY
RECOVERY_FAILURE_DESCRIPTOR_SCHEMA_VERSION = 1
RECOVERY_ATTEMPT_SCHEMA_VERSION = 1
RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION = 1
RECOVERY_INJECTION_PLAN_SCHEMA_VERSION = 1
RECOVERY_REPLAY_EVIDENCE_SCHEMA_VERSION = 1

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


@dataclass(frozen=True)
class RecoveryFallbackTransition:
    predicate: str
    target_stage: str


@dataclass(frozen=True)
class RecoveryStageSpec:
    name: str
    activation_predicate: str
    completion_predicate: str
    fallbacks: tuple[RecoveryFallbackTransition, ...] = ()


_RECOVERY_STAGE_FSM = (
    RecoveryStageSpec(
        name="approach",
        activation_predicate="running_and_not_grasp",
        completion_predicate="bimanual_pose_evidence",
    ),
    RecoveryStageSpec(
        name="acquire",
        activation_predicate="running_and_not_grasp_and_bimanual_pose",
        completion_predicate="grasp",
        fallbacks=(RecoveryFallbackTransition("lost_bimanual_pose", "approach"),),
    ),
    RecoveryStageSpec(
        name="place",
        activation_predicate="running_and_grasp",
        completion_predicate="placement",
        fallbacks=(
            RecoveryFallbackTransition("lost_grasp_with_bimanual_pose", "acquire"),
            RecoveryFallbackTransition("lost_grasp_without_bimanual_pose", "approach"),
        ),
    ),
)


def recovery_stage_fsm() -> tuple[RecoveryStageSpec, ...]:
    """Return the task FSM without importing an algorithm-side gate."""

    return _RECOVERY_STAGE_FSM


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
) -> Mapping[str, object]:
    """Resolve task truth into component-scoped reward and stage-gate telemetry."""

    state = _validate_privileged_telemetry(telemetry)
    left_distance = math.dist(state.left_ee_pose_w[:3], state.box_center_w)
    right_distance = math.dist(state.right_ee_pose_w[:3], state.box_center_w)
    distance = max(left_distance, right_distance)
    placement_distance = math.hypot(state.xy_mismatch_m, state.z_mismatch_m)
    gate_truth = _recovery_gate_truth(state)

    components: dict[str, Mapping[str, object]] = {}
    stage_gates: dict[str, Mapping[str, object]] = {}
    potential_values: Mapping[str, Mapping[str, float]] = {
        "distance": MappingProxyType({"distance": distance, "d_init": distance}),
        "grasp": MappingProxyType({"q_grasp": 1.0 if state.grasp else 0.0}),
        "placement": MappingProxyType(
            {"distance": placement_distance, "d_init": placement_distance}
        ),
    }
    bindings_by_stage = {
        binding.stage: binding for binding in _RECOVERY_REWARD_BINDINGS
    }
    for stage in _RECOVERY_STAGE_FSM:
        fallback_truth = {
            transition.predicate: gate_truth[transition.predicate]
            for transition in stage.fallbacks
        }
        stage_gates[stage.name] = MappingProxyType(
            {
                "activation_predicate": stage.activation_predicate,
                "activation": gate_truth[stage.activation_predicate],
                "completion_predicate": stage.completion_predicate,
                "completion": gate_truth[stage.completion_predicate],
                "fallbacks": MappingProxyType(fallback_truth),
            }
        )
        binding = bindings_by_stage[stage.name]
        scoped: dict[str, object] = dict(potential_values[binding.term])
        scoped[stage.activation_predicate] = gate_truth[stage.activation_predicate]
        scoped[stage.completion_predicate] = gate_truth[stage.completion_predicate]
        scoped.update(fallback_truth)
        components[recovery_reward_component_scope(binding)] = MappingProxyType(scoped)

    return MappingProxyType(
        {
            "components": MappingProxyType(components),
            "stage_gates": MappingProxyType(stage_gates),
        }
    )


@dataclass(frozen=True)
class RecoveryFailureCatalogEntry:
    category: str
    initial_stage: str
    stage_fsm: tuple[RecoveryStageSpec, ...]
    reward_bindings: tuple[RecoveryRewardBinding, ...]
    required_capabilities: frozenset[str]
    declared: bool = True
    recoverable: bool = True


_DECLARED_CATALOG = MappingProxyType(
    {
        category: RecoveryFailureCatalogEntry(
            category=category,
            initial_stage=_CATEGORY_INITIAL_STAGE[category],
            stage_fsm=_RECOVERY_STAGE_FSM,
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
    runtime_evidence: Mapping[str, object] | None = None,
) -> Mapping[str, RecoveryFailureCatalogEntry]:
    """Return only runtime-proven entries; no evidence means no effective expert."""

    if runtime_evidence is None:
        return {}
    if not isinstance(runtime_evidence, Mapping):
        raise RecoveryFailureSchemaError("runtime evidence catalog must be a mapping")
    unknown = set(runtime_evidence) - set(DECLARED_FAILURE_CATEGORIES)
    if unknown:
        raise RecoveryFailureSchemaError(
            "unknown runtime evidence categories: " + ", ".join(sorted(unknown))
        )
    effective: dict[str, RecoveryFailureCatalogEntry] = {}
    for category, value in runtime_evidence.items():
        if not isinstance(value, FailureRuntimeCapabilityEvidence):
            raise RecoveryFailureSchemaError(
                f"runtime evidence for {category!r} has the wrong type"
            )
        if value.category != category:
            raise RecoveryFailureSchemaError(
                f"runtime evidence key {category!r} does not match its category"
            )
        if not _missing_runtime_capabilities(value) and _runtime_replays_are_effective(
            value
        ):
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
class FailureReplayRecord:
    """One content-bound deterministic injection/readback/continuation replay."""

    schema_version: int
    repeat_index: int
    category: str
    failure_seed: int
    snapshot_digest: str
    category_seed: int
    plan_transform_digest: str
    runtime_evidence_digest: str
    readback_state_digest: str
    continuation_digest: str
    predicate_passed: bool
    observed_category: str | None
    category_passed: bool

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERY_REPLAY_EVIDENCE_SCHEMA_VERSION:
            raise RecoveryFailureSchemaError(
                f"unsupported replay evidence schema version {self.schema_version}"
            )
        repeat_index = _strict_int(self.repeat_index, name="repeat index")
        category = _category(self.category)
        failure_seed = _strict_int(self.failure_seed, name="failure seed")
        snapshot_digest = _digest(self.snapshot_digest, name="snapshot digest")
        category_seed = _strict_int(self.category_seed, name="category seed")
        plan_digest = _digest(
            self.plan_transform_digest,
            name="plan transform digest",
        )
        runtime_digest = _digest(
            self.runtime_evidence_digest,
            name="runtime evidence digest",
        )
        readback_digest = _digest(
            self.readback_state_digest,
            name="readback state digest",
        )
        continuation_digest = _digest(
            self.continuation_digest,
            name="continuation digest",
        )
        predicate_passed = _strict_bool(
            self.predicate_passed,
            name="predicate passed",
        )
        observed_category = _category(self.observed_category, optional=True)
        category_passed = _strict_bool(
            self.category_passed,
            name="category passed",
        )
        assert category is not None
        assert snapshot_digest is not None
        assert plan_digest is not None
        assert runtime_digest is not None
        assert readback_digest is not None
        assert continuation_digest is not None
        object.__setattr__(self, "repeat_index", repeat_index)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "failure_seed", failure_seed)
        object.__setattr__(self, "snapshot_digest", snapshot_digest)
        object.__setattr__(self, "category_seed", category_seed)
        object.__setattr__(self, "plan_transform_digest", plan_digest)
        object.__setattr__(self, "runtime_evidence_digest", runtime_digest)
        object.__setattr__(self, "readback_state_digest", readback_digest)
        object.__setattr__(self, "continuation_digest", continuation_digest)
        object.__setattr__(self, "predicate_passed", predicate_passed)
        object.__setattr__(self, "observed_category", observed_category)
        object.__setattr__(self, "category_passed", category_passed)


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
    replay_records: tuple[FailureReplayRecord, ...] = ()
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
        if not isinstance(self.replay_records, (tuple, list)) or any(
            not isinstance(record, FailureReplayRecord)
            for record in self.replay_records
        ):
            raise RecoveryFailureSchemaError(
                "runtime replay records must be FailureReplayRecord values"
            )
        replay_records = tuple(self.replay_records)
        evidence_payload = (
            "runtime-capability-evidence",
            self.schema_version,
            self.task_identity,
            category,
            tuple(sorted(capabilities)),
            self.evidence_id,
            self.target_shelf,
            self.ground_support,
            self.verified_anchor,
        )
        evidence_digest = recovery_state.recovery_value_digest(evidence_payload)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "validated_capabilities", capabilities)
        object.__setattr__(self, "replay_records", replay_records)
        object.__setattr__(self, "evidence_digest", evidence_digest)


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


def _runtime_replays_are_effective(
    evidence: FailureRuntimeCapabilityEvidence,
) -> bool:
    records = evidence.replay_records
    if len(records) < 2:
        return False
    repeat_indices = {record.repeat_index for record in records}
    if len(repeat_indices) != len(records):
        return False

    first = records[0]
    stable_fields = (
        "category",
        "failure_seed",
        "snapshot_digest",
        "category_seed",
        "plan_transform_digest",
        "runtime_evidence_digest",
        "readback_state_digest",
        "continuation_digest",
    )
    if any(
        getattr(record, name) != getattr(first, name)
        for record in records[1:]
        for name in stable_fields
    ):
        return False
    if any(
        record.category != evidence.category
        or record.runtime_evidence_digest != evidence.evidence_digest
        or not record.predicate_passed
        or record.observed_category != evidence.category
        or not record.category_passed
        for record in records
    ):
        return False

    descriptor = RecoveryFailureDescriptor(
        schema_version=RECOVERY_FAILURE_DESCRIPTOR_SCHEMA_VERSION,
        task_identity=PP_BOX_TASK_IDENTITY,
        category=evidence.category,
        stage=_CATEGORY_INITIAL_STAGE[evidence.category],
        entities=_DESCRIPTOR_ENTITIES,
        confidence=1.0,
        reward_mask={term: True for term in _REWARD_TERMS},
        failure_seed=first.failure_seed,
        snapshot_digest=first.snapshot_digest,
    )
    try:
        expected_plan = build_failure_injection_plan(descriptor, evidence)
    except RecoveryFailureError:
        return False
    return bool(
        first.category_seed == expected_plan.category_seed
        and first.plan_transform_digest == expected_plan.transform_digest
        and first.runtime_evidence_digest == expected_plan.runtime_evidence_digest
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


@dataclass(frozen=True)
class FailureInjectionResult:
    plan: FailureInjectionPlan
    category: str
    readback_passed: bool
    readback: FailurePredicateContext


def build_failure_replay_record(
    result: FailureInjectionResult,
    *,
    repeat_index: int,
    continuation_state: object,
) -> FailureReplayRecord:
    """Bind one verified injection and its continued state into replay evidence."""

    if not isinstance(result, FailureInjectionResult):
        raise RecoveryFailureSchemaError(
            "replay evidence requires a FailureInjectionResult"
        )
    if not result.readback_passed:
        raise RecoveryFailureSchemaError(
            "replay evidence requires a passing injection readback"
        )
    observed = classify_recoverable_failure(result.readback)
    category_passed = bool(observed == result.plan.category == result.category)
    return FailureReplayRecord(
        schema_version=RECOVERY_REPLAY_EVIDENCE_SCHEMA_VERSION,
        repeat_index=repeat_index,
        category=result.plan.category,
        failure_seed=result.plan.failure_seed,
        snapshot_digest=result.plan.snapshot_digest,
        category_seed=result.plan.category_seed,
        plan_transform_digest=result.plan.transform_digest,
        runtime_evidence_digest=result.plan.runtime_evidence_digest,
        readback_state_digest=recovery_state.recovery_value_digest(result.readback),
        continuation_digest=recovery_state.recovery_value_digest(continuation_state),
        predicate_passed=bool(result.readback_passed),
        observed_category=observed,
        category_passed=category_passed,
    )


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


__all__ = [
    "BOX_HALF_EXTENTS_M",
    "BOX_HALF_EXTENT_M",
    "DECLARED_FAILURE_CATEGORIES",
    "PLACEMENT_Z_TOLERANCE_M",
    "PP_BOX_TASK_IDENTITY",
    "RECOVERY_ATTEMPT_SCHEMA_VERSION",
    "RECOVERY_FAILURE_DESCRIPTOR_SCHEMA_VERSION",
    "RECOVERY_INJECTION_PLAN_SCHEMA_VERSION",
    "RECOVERY_REPLAY_EVIDENCE_SCHEMA_VERSION",
    "RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION",
    "FailureInjectionPlan",
    "FailureInjectionResult",
    "FailurePredicateContext",
    "FailureReplayRecord",
    "FailureRuntimeCapabilityEvidence",
    "LiveSupportGeometry",
    "RecoveryAttemptEvidence",
    "RecoveryFailureCapabilityError",
    "RecoveryFailureCatalogEntry",
    "RecoveryFailureDescriptor",
    "RecoveryFailureError",
    "RecoveryFailurePredicateConflictError",
    "RecoveryFailureReader",
    "RecoveryFailureSchemaError",
    "RecoveryFailureSettler",
    "RecoveryFailureSnapshotDigestError",
    "RecoveryFailureVerificationError",
    "RecoveryFailureWriter",
    "RecoveryFallbackTransition",
    "RecoveryRewardBinding",
    "RecoveryStageSpec",
    "VerifiedFailureAnchor",
    "build_failure_injection_plan",
    "build_failure_replay_record",
    "classify_recoverable_failure",
    "declared_failure_catalog",
    "derive_category_seed",
    "effective_failure_catalog",
    "evaluate_failure_predicates",
    "inject_recovery_failure",
    "recovery_reward_bindings",
    "recovery_reward_component_scope",
    "recovery_stage_fsm",
    "recovery_state",
    "recovery_succeeded",
    "required_runtime_capabilities",
    "resolve_recovery_reward_telemetry",
]
