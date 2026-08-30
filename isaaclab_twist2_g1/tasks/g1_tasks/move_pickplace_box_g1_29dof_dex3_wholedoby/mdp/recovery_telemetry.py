"""Privileged recovery telemetry for the HOI pick-place-box task.

This module stays importable without Isaac Lab. Runtime access is duck-typed so
the recovery adapter can keep privileged state outside the policy observation
manager while pure tests exercise the same predicates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any, Literal

import torch

from .recovery_state import RecoveryStateCoordinator

RECOVERY_TELEMETRY_SCHEMA_VERSION = 2
LIVE_FALL_EVIDENCE_SCHEMA_VERSION = 1
EVALUATOR_TERMINAL_EVIDENCE_SCHEMA_VERSION = 1
RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION = 3
RUNTIME_CONTACT_SENSOR_REPORT_SCHEMA_VERSION = 2
RESIDUAL_ACTOR_OBSERVATION_SCHEMA_VERSION = 1
PP_BOX_TASK_IDENTITY = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"

DEFAULT_CONTACT_FORCE_THRESHOLD_N = 1.0
DEFAULT_MAX_EE_BOX_DISTANCE_M = 0.3
SOFT_FALL_UP_ALIGNMENT = math.cos(math.radians(60.0))
HARD_FALL_UP_ALIGNMENT = math.cos(math.radians(75.0))
CRITICAL_BODY_CONTACT_THRESHOLD_N = 50.0
EMPIRICAL_CONTACT_QUIET_MAX_N = 0.05
EMPIRICAL_CONTACT_TOUCH_MIN_N = 1.0
EMPIRICAL_CONTACT_TOUCH_GAP_M = 0.005
EMPIRICAL_CONTACT_TOUCH_VELOCITY_M_S = -1.0

_HAND_CONTACT_LINK_TOKENS = (
    "palm",
    "index_0",
    "index_1",
    "middle_0",
    "middle_1",
    "thumb_0",
    "thumb_1",
    "thumb_2",
)

_FALL_CRITICAL_BODY_TOKENS = (
    "pelvis",
    "torso",
    "waist",
    "hip",
    "thigh",
    "knee",
    "calf",
    "shin",
    "head",
)
_FALL_SUPPORTED_BODY_TOKENS = ("ankle", "foot")

TerminalReason = Literal["success", "fall", "time_limit", "running"]
_LIVE_FALL_PRODUCER_TOKEN = object()
_EVALUATOR_TERMINAL_EVIDENCE_TOKEN = object()
_RESIDUAL_ACTOR_OBSERVATION_TOKEN = object()
_CONTACT_CALIBRATION_RECEIPT_TOKEN = object()
_CONTACT_CALIBRATION_EXECUTOR_TOKEN = object()
_CONTACT_CALIBRATION_EXECUTORS: dict[int, tuple[object, object]] = {}
_CONTACT_CALIBRATION_RECEIPTS: dict[int, tuple[object, object]] = {}
_EVALUATOR_TERMINAL_EVIDENCE: dict[int, tuple[object, object]] = {}
_ACTOR_POLICY_TERM_ALLOWLIST = (
    "robot_joint_state",
    "robot_gipper_state",
    "camera_image",
)


class RecoveryTelemetryIncompleteError(RuntimeError):
    """Raised when required simulator truth is absent or structurally invalid."""

    def __init__(
        self,
        missing_capabilities: Sequence[str],
        *,
        runtime_evidence: Mapping[str, object] | None = None,
    ) -> None:
        self.missing_capabilities = tuple(sorted(set(missing_capabilities)))
        self.runtime_evidence = (
            None if runtime_evidence is None else dict(runtime_evidence)
        )
        detail = ", ".join(self.missing_capabilities) or "unknown"
        super().__init__(f"privileged recovery telemetry is incomplete: {detail}")


class PrivilegedObservationLeakError(RuntimeError):
    """Raised when privileged telemetry appears inside an actor observation."""

    def __init__(self, leak_paths: Sequence[str]) -> None:
        self.leak_paths = tuple(sorted(set(leak_paths)))
        super().__init__(
            "privileged actor-observation leak: " + ", ".join(self.leak_paths)
        )


@dataclass(frozen=True, init=False)
class ResidualActorObservation:
    """Factory-issued view of the exact PP-box policy observation group."""

    schema_version: int
    task_identity: str
    policy_term_names: tuple[str, ...]
    policy: Mapping[str, object]
    source_manager_id: int
    source_value_ids: tuple[int, ...]
    payload_digest: str
    _producer_token: object = dataclass_field(repr=False, compare=False)


@dataclass(frozen=True)
class PairwiseContactEvidence:
    force_w: tuple[float, float, float]
    magnitude_n: float
    threshold_n: float
    sensor_body: str
    filtered_body: str
    in_contact: bool


@dataclass(frozen=True)
class HandContactEvidence:
    side: Literal["left", "right"]
    links: tuple[PairwiseContactEvidence, ...]
    contacting_bodies: tuple[str, ...]
    resultant_force_w: tuple[float, float, float]
    total_magnitude_n: float
    in_contact: bool


@dataclass(frozen=True)
class BimanualGraspEvidence:
    left_ee_box_distance_m: float
    right_ee_box_distance_m: float
    max_ee_box_distance_m: float
    left_pose_valid: bool
    right_pose_valid: bool
    pose_evidence: bool
    pairwise_contact: bool
    bimanual_grasp: bool


@dataclass(frozen=True)
class DriverTerminalContext:
    """Authoritative control-driver terminal counters for one environment."""

    control_step_count: int
    max_control_steps: int
    fall_streak: int
    fall_confirm_steps: int
    time_limit: bool
    fall_confirmed: bool

    @classmethod
    def from_value(cls, value: object) -> DriverTerminalContext:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                return cls(
                    control_step_count=value["control_step_count"],
                    max_control_steps=value["max_control_steps"],
                    fall_streak=value["fall_streak"],
                    fall_confirm_steps=value["fall_confirm_steps"],
                    time_limit=value["time_limit"],
                    fall_confirmed=value["fall_confirmed"],
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing terminal context field: {exc.args[0]}"
                ) from exc
        raise TypeError(f"unsupported terminal context type: {type(value).__name__}")

    def validate(self) -> None:
        integer_fields = (
            "control_step_count",
            "max_control_steps",
            "fall_streak",
            "fall_confirm_steps",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise ValueError("driver terminal counters must be integers")
        if self.control_step_count < 0 or self.max_control_steps <= 0:
            raise ValueError("driver step limits must be non-negative and positive")
        if self.fall_streak < 0 or self.fall_confirm_steps <= 0:
            raise ValueError("fall streak and confirmation steps are invalid")
        if self.fall_streak > self.control_step_count:
            raise ValueError("fall streak exceeds the available control-step history")
        if type(self.time_limit) is not bool or type(self.fall_confirmed) is not bool:
            raise ValueError("driver terminal flags must be booleans")
        if self.time_limit != (self.control_step_count >= self.max_control_steps):
            raise ValueError("driver time-limit flag contradicts its counters")
        if self.fall_confirmed != (self.fall_streak >= self.fall_confirm_steps):
            raise ValueError("driver fall flag contradicts its streak")


@dataclass(frozen=True)
class EvaluatorFallDetectorConfig:
    """Normalized form of the existing evaluator's configurable fall detector."""

    enabled: bool
    soft_up_alignment: float
    hard_up_alignment: float
    contact_force_threshold_n: float
    confirm_steps: int
    critical_body_indices: tuple[int, ...]
    critical_body_names: tuple[str, ...]


@dataclass(frozen=True)
class LiveFallLaneEvidence:
    """One primitive step of root/contact fall truth for one environment lane."""

    env_index: int
    control_step_count: int
    root_quat_wxyz: tuple[float, float, float, float]
    root_up_alignment: float
    critical_body_contact: bool | None
    fall_candidate: bool
    detector_enabled: bool = True
    soft_up_alignment: float = SOFT_FALL_UP_ALIGNMENT
    hard_up_alignment: float = HARD_FALL_UP_ALIGNMENT

    def __post_init__(self) -> None:
        if type(self.env_index) is not int or self.env_index < 0:
            raise ValueError("live fall env index must be a non-negative integer")
        if type(self.control_step_count) is not int or self.control_step_count < 0:
            raise ValueError("live fall control step must be a non-negative integer")
        quaternion = _as_float_tuple(
            self.root_quat_wxyz,
            4,
            name="live fall root quaternion",
        )
        alignment = float(self.root_up_alignment)
        expected_alignment = compute_root_up_alignment(quaternion)
        if not math.isclose(alignment, expected_alignment, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("live fall root alignment contradicts its quaternion")
        if (
            self.critical_body_contact is not None
            and type(self.critical_body_contact) is not bool
        ):
            raise ValueError("live fall critical body contact must be boolean or None")
        if type(self.fall_candidate) is not bool:
            raise ValueError("live fall candidate must be boolean")
        expected_candidate = classify_fall(
            alignment,
            critical_body_contact=self.critical_body_contact,
            detector_enabled=self.detector_enabled,
            soft_up_alignment=self.soft_up_alignment,
            hard_up_alignment=self.hard_up_alignment,
        )
        if self.fall_candidate is not expected_candidate:
            raise ValueError("live fall candidate contradicts root/contact truth")
        object.__setattr__(self, "root_quat_wxyz", quaternion)
        object.__setattr__(self, "root_up_alignment", alignment)


@dataclass(frozen=True, init=False)
class LiveFallProducerEvidence:
    """Factory-issued, runtime-bound fall evidence consumed by driver and extractor."""

    schema_version: int
    task_identity: str
    runtime_identity_digest: str
    detector_config_digest: str
    lanes: tuple[LiveFallLaneEvidence, ...]
    evidence_digest: str
    _producer_token: object = dataclass_field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class EvaluatorTerminalEvidence:
    """Factory-issued terminal truth for one evaluator primitive step."""

    schema_version: int
    task_identity: str
    runtime_identity_digest: str
    detector_config: EvaluatorFallDetectorConfig
    detector_config_digest: str
    step_idx: int
    max_steps: int
    previous_evidence_digest: str
    contexts: tuple[DriverTerminalContext, ...]
    live_fall_evidence: LiveFallProducerEvidence
    evidence_digest: str
    _producer_token: object = dataclass_field(repr=False, compare=False)


@dataclass(frozen=True)
class PairwiseContactBinding:
    """Runtime identity for one-body-to-one-filter pairwise contact evidence."""

    sensor_scene_key: str
    sensor_body_name: str
    filtered_body_name: str

    @classmethod
    def from_value(cls, value: object) -> PairwiseContactBinding:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                return cls(
                    sensor_scene_key=str(value["sensor_scene_key"]),
                    sensor_body_name=str(value["sensor_body_name"]),
                    filtered_body_name=str(value["filtered_body_name"]),
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing contact binding field: {exc.args[0]}"
                ) from exc
        raise TypeError(f"unsupported contact binding type: {type(value).__name__}")

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not value:
                raise ValueError(f"contact binding {field.name} must be non-empty")


@dataclass(frozen=True)
class HandContactBinding:
    """Exact per-hand one-body contact sensors and palm pose binding."""

    side: Literal["left", "right"]
    ee_body_name: str
    sensors: tuple[PairwiseContactBinding, ...]

    @classmethod
    def from_value(cls, value: object) -> HandContactBinding:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                raw_sensors = value["sensors"]
                if not isinstance(raw_sensors, Sequence) or isinstance(
                    raw_sensors, (str, bytes, bytearray)
                ):
                    raise TypeError("hand contact sensors must be a sequence")
                return cls(
                    side=str(value["side"]),  # type: ignore[arg-type]
                    ee_body_name=str(value["ee_body_name"]),
                    sensors=tuple(
                        PairwiseContactBinding.from_value(sensor)
                        for sensor in raw_sensors
                    ),
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing hand contact binding field: {exc.args[0]}"
                ) from exc
        raise TypeError(
            f"unsupported hand contact binding type: {type(value).__name__}"
        )

    def validate(
        self, *, expected_side: Literal["left", "right"] | None = None
    ) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError(f"unknown hand side: {self.side!r}")
        if expected_side is not None and self.side != expected_side:
            raise ValueError(f"expected {expected_side} hand binding, got {self.side}")
        for sensor in self.sensors:
            sensor.validate()
        expected = _default_hand_contact_binding(self.side)
        if self != expected:
            raise ValueError(
                f"{self.side} hand binding does not match the verified leaf catalog"
            )


@dataclass(frozen=True, init=False)
class ContactCalibrationPhaseReceipt:
    """One executor-issued simulator step and its exact raw force bytes."""

    schema_version: int
    task_identity: str
    phase: Literal["baseline", "target_touch", "target_removed"]
    sensor_scene_key: str
    sensor_body_name: str
    source_snapshot_digest: str
    phase_state_digest: str
    runtime_identity_digest: str
    sensor_identity_digest: str
    control_step_before: int
    control_step_after: int
    force_shape: tuple[int, ...]
    force_dtype: str
    force_device: str
    force_byte_order: str
    raw_force_bytes: bytes
    raw_force_sha256: str
    receipt_digest: str
    _producer_token: object = dataclass_field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class ContactSensorCalibrationReceipt:
    """Three causal phase receipts for one one-body/one-filter sensor."""

    schema_version: int
    task_identity: str
    sensor_scene_key: str
    sensor_body_name: str
    target_asset_scene_key: str
    target_prim_paths: tuple[str, ...]
    source_snapshot_digest: str
    runtime_identity_digest: str
    sensor_identity_digest: str
    phases: tuple[ContactCalibrationPhaseReceipt, ...]
    receipt_digest: str
    _producer_token: object = dataclass_field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class ContactCalibrationExecutionReceipt:
    """Runtime-owned, registry-bound receipt for all 16 pairwise sensors."""

    schema_version: int
    task_identity: str
    source_snapshot_digest: str
    runtime_identity_digest: str
    snapshot_fidelity_tier: str
    coordinator_binding_identity: tuple[tuple[str, object], ...]
    coordinator_binding_digest: str
    sensor_receipts: tuple[ContactSensorCalibrationReceipt, ...]
    receipt_digest: str
    _producer_token: object = dataclass_field(repr=False, compare=False)


@dataclass(frozen=True)
class _ContactCalibrationExecutor:
    coordinator: object
    snapshot_fidelity_tier: str
    coordinator_binding_identity: tuple[tuple[str, object], ...]
    coordinator_binding_digest: str
    _executor_token: object = dataclass_field(repr=False, compare=False)


@dataclass(frozen=True)
class RuntimeContactSensorReport:
    """Materialized sensor identity plus separately proven filter identity."""

    schema_version: int
    side: Literal["left", "right"]
    sensor_scene_key: str
    sensor_prim_path_expression: str
    resolved_sensor_prim_paths: tuple[str, ...]
    resolved_sensor_body_names: tuple[str, ...]
    configured_filter_prim_path_expressions: tuple[str, ...]
    candidate_filter_asset_scene_key: str
    candidate_filter_asset_prim_path_expression: str
    candidate_filter_asset_prim_paths: tuple[str, ...]
    proven_filter_prim_paths: tuple[str, ...]
    proven_filter_body_name: str
    filter_mapping_proof_schema_version: int
    filter_mapping_proof_source: str
    filter_mapping_proof_digest: str
    num_bodies: int
    filter_count: int
    force_matrix_shape: tuple[int, ...]
    force_matrix_dtype: str
    force_matrix_device: str
    force_matrix_finite: bool


def _default_hand_contact_binding(side: Literal["left", "right"]) -> HandContactBinding:
    sensors = tuple(
        PairwiseContactBinding(
            sensor_scene_key=f"{side}_box_contact_{link_token}",
            sensor_body_name=f"{side}_hand_{link_token}_link",
            filtered_body_name="Box",
        )
        for link_token in _HAND_CONTACT_LINK_TOKENS
    )
    return HandContactBinding(
        side=side,
        ee_body_name=f"{side}_hand_palm_link",
        sensors=sensors,
    )


def default_hand_contact_bindings() -> dict[str, HandContactBinding]:
    """Return the USD-evidenced candidates; runtime identity is still required."""

    return {side: _default_hand_contact_binding(side) for side in ("left", "right")}


@dataclass(frozen=True)
class PrivilegedRecoveryTelemetry:
    schema_version: int
    task_identity: str
    env_index: int
    box_center_w: tuple[float, float, float]
    box_linear_velocity_w: tuple[float, float, float]
    box_angular_velocity_w: tuple[float, float, float]
    shelf_bounds_w: tuple[float, float, float, float]
    support_surface_z_m: float
    target_support_surface_z_m: float
    left_ee_pose_w: tuple[float, float, float, float, float, float, float]
    right_ee_pose_w: tuple[float, float, float, float, float, float, float]
    left_box_contact: HandContactEvidence
    right_box_contact: HandContactEvidence
    grasp_evidence: BimanualGraspEvidence
    grasp: bool
    xy_mismatch_m: float
    z_gap_m: float
    z_mismatch_m: float
    placement_distance_m: float
    placement: bool
    success: bool
    root_up_alignment: float
    control_step_count: int
    max_control_steps: int
    fall_candidate: bool
    fall_streak: int
    fall_confirm_steps: int
    fall: bool
    time_limit: bool
    terminal_reason: TerminalReason


_PRIVILEGED_FIELD_NAMES = frozenset(
    field.name
    for field in fields(PrivilegedRecoveryTelemetry)
    if field.name not in {"schema_version", "task_identity", "env_index"}
)


def _as_float_tuple(value: object, length: int, *, name: str) -> tuple[float, ...]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
    else:
        raise TypeError(f"{name} must be a length-{length} vector")
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    result = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def pairwise_contact_evidence(
    force_w: object,
    *,
    sensor_body: str,
    filtered_body: str,
    threshold_n: float = DEFAULT_CONTACT_FORCE_THRESHOLD_N,
) -> PairwiseContactEvidence:
    """Classify one explicitly identified sensor-body/filter-body force."""

    force = _as_float_tuple(force_w, 3, name="pairwise contact force")
    threshold = float(threshold_n)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("contact force threshold must be finite and positive")
    if not sensor_body or not filtered_body:
        raise ValueError("pairwise contact body identities must be non-empty")
    magnitude = math.sqrt(sum(component * component for component in force))
    return PairwiseContactEvidence(
        force_w=(force[0], force[1], force[2]),
        magnitude_n=magnitude,
        threshold_n=threshold,
        sensor_body=str(sensor_body),
        filtered_body=str(filtered_body),
        in_contact=magnitude >= threshold,
    )


def aggregate_hand_contact_evidence(
    side: Literal["left", "right"],
    links: Sequence[PairwiseContactEvidence],
) -> HandContactEvidence:
    if side not in {"left", "right"}:
        raise ValueError(f"unknown hand side: {side!r}")
    link_tuple = tuple(links)
    if not link_tuple:
        raise ValueError("hand contact evidence requires at least one link")
    body_names = tuple(link.sensor_body for link in link_tuple)
    if len(set(body_names)) != len(body_names):
        raise ValueError("hand contact evidence contains duplicate sensor bodies")
    expected_prefix = f"{side}_hand_"
    if any(not body_name.startswith(expected_prefix) for body_name in body_names):
        raise ValueError(
            f"{side} hand contact evidence contains a body from another side"
        )
    filtered_bodies = {link.filtered_body for link in link_tuple}
    if filtered_bodies != {"Box"}:
        raise ValueError("hand contact evidence must be filtered only to Box")

    resultant_force = tuple(
        sum(link.force_w[axis] for link in link_tuple) for axis in range(3)
    )
    contacting_bodies = tuple(
        link.sensor_body for link in link_tuple if link.in_contact
    )
    return HandContactEvidence(
        side=side,
        links=link_tuple,
        contacting_bodies=contacting_bodies,
        resultant_force_w=(
            float(resultant_force[0]),
            float(resultant_force[1]),
            float(resultant_force[2]),
        ),
        total_magnitude_n=sum(link.magnitude_n for link in link_tuple),
        in_contact=bool(contacting_bodies),
    )


def has_pairwise_bimanual_contact(
    left: PairwiseContactEvidence | HandContactEvidence,
    right: PairwiseContactEvidence | HandContactEvidence,
) -> bool:
    return bool(left.in_contact and right.in_contact)


def classify_bimanual_grasp(
    *,
    box_center_w: object,
    left_ee_pose_w: object,
    right_ee_pose_w: object,
    left_contact: bool,
    right_contact: bool,
    max_ee_box_distance_m: float = DEFAULT_MAX_EE_BOX_DISTANCE_M,
) -> BimanualGraspEvidence:
    """Require independent bilateral contact and bilateral EE pose evidence."""

    box_center = _as_float_tuple(box_center_w, 3, name="box center")
    left_pose = _as_float_tuple(left_ee_pose_w, 7, name="left EE pose")
    right_pose = _as_float_tuple(right_ee_pose_w, 7, name="right EE pose")
    max_distance = float(max_ee_box_distance_m)
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError("EE-to-box distance threshold must be finite and positive")

    left_distance = math.dist(left_pose[:3], box_center)
    right_distance = math.dist(right_pose[:3], box_center)
    left_pose_valid = left_distance <= max_distance
    right_pose_valid = right_distance <= max_distance
    pose_evidence = left_pose_valid and right_pose_valid
    pairwise_contact = bool(left_contact and right_contact)
    return BimanualGraspEvidence(
        left_ee_box_distance_m=left_distance,
        right_ee_box_distance_m=right_distance,
        max_ee_box_distance_m=max_distance,
        left_pose_valid=left_pose_valid,
        right_pose_valid=right_pose_valid,
        pose_evidence=pose_evidence,
        pairwise_contact=pairwise_contact,
        bimanual_grasp=bool(pose_evidence and pairwise_contact),
    )


def compute_root_up_alignment(root_quat_wxyz: object) -> float:
    quat = _as_float_tuple(root_quat_wxyz, 4, name="root quaternion")
    norm = math.sqrt(sum(component * component for component in quat))
    if norm <= 1e-12:
        raise ValueError("root quaternion norm must be positive")
    _w, x, y, _z = (component / norm for component in quat)
    return max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))


def classify_fall(
    root_up_alignment: float,
    *,
    critical_body_contact: bool | None,
    detector_enabled: bool = True,
    soft_up_alignment: float = SOFT_FALL_UP_ALIGNMENT,
    hard_up_alignment: float = HARD_FALL_UP_ALIGNMENT,
) -> bool:
    """Mirror the evaluator's instantaneous hard/soft tilt predicate."""

    alignment = float(root_up_alignment)
    if not math.isfinite(alignment) or alignment < -1.0 or alignment > 1.0:
        raise ValueError("root up-alignment must be finite and within [-1, 1]")
    if type(detector_enabled) is not bool:
        raise ValueError("fall-detector enabled flag must be boolean")
    if not detector_enabled:
        return False
    soft_threshold = float(soft_up_alignment)
    hard_threshold = float(hard_up_alignment)
    if (
        not math.isfinite(soft_threshold)
        or not math.isfinite(hard_threshold)
        or not -1.0 <= hard_threshold < soft_threshold <= 1.0
    ):
        raise ValueError("fall-detector alignment thresholds are invalid")
    if alignment < hard_threshold:
        return True
    if alignment >= soft_threshold:
        return False
    if critical_body_contact is None:
        raise RecoveryTelemetryIncompleteError(("critical_body_contact",))
    return bool(critical_body_contact)


def classify_terminal(*, success: bool, fall: bool, time_limit: bool) -> TerminalReason:
    if success:
        return "success"
    if fall:
        return "fall"
    if time_limit:
        return "time_limit"
    return "running"


def _validate_live_fall_lane(
    value: object,
    *,
    env_index: int,
    control_step_count: int,
) -> LiveFallLaneEvidence:
    if not isinstance(value, LiveFallLaneEvidence):
        raise RecoveryTelemetryIncompleteError(("live_fall_evidence",))
    try:
        normalized = LiveFallLaneEvidence(
            env_index=value.env_index,
            control_step_count=value.control_step_count,
            root_quat_wxyz=value.root_quat_wxyz,
            root_up_alignment=value.root_up_alignment,
            critical_body_contact=value.critical_body_contact,
            fall_candidate=value.fall_candidate,
            detector_enabled=value.detector_enabled,
            soft_up_alignment=value.soft_up_alignment,
            hard_up_alignment=value.hard_up_alignment,
        )
    except (TypeError, ValueError, RecoveryTelemetryIncompleteError) as exc:
        raise RecoveryTelemetryIncompleteError(("live_fall_evidence",)) from exc
    if (
        normalized != value
        or normalized.env_index != env_index
        or normalized.control_step_count != control_step_count
    ):
        raise RecoveryTelemetryIncompleteError(("live_fall_evidence",))
    return normalized


def build_privileged_telemetry(
    *,
    task_identity: str,
    env_index: int,
    box_center_w: object,
    box_linear_velocity_w: object,
    box_angular_velocity_w: object,
    support: object,
    left_ee_pose_w: object,
    right_ee_pose_w: object,
    left_contact: HandContactEvidence,
    right_contact: HandContactEvidence,
    live_fall_evidence: LiveFallLaneEvidence,
    terminal_context: DriverTerminalContext,
    max_ee_box_distance_m: float = DEFAULT_MAX_EE_BOX_DISTANCE_M,
) -> PrivilegedRecoveryTelemetry:
    if task_identity != PP_BOX_TASK_IDENTITY:
        raise ValueError("privileged telemetry task identity is not HOI_pp_box")
    if not isinstance(left_contact, HandContactEvidence) or left_contact.side != "left":
        raise ValueError("privileged telemetry requires aggregated left hand contact")
    if (
        not isinstance(right_contact, HandContactEvidence)
        or right_contact.side != "right"
    ):
        raise ValueError("privileged telemetry requires aggregated right hand contact")
    if not isinstance(terminal_context, DriverTerminalContext):
        raise TypeError(
            "privileged telemetry requires authoritative driver terminal context"
        )
    terminal_context.validate()
    fall_evidence = _validate_live_fall_lane(
        live_fall_evidence,
        env_index=int(env_index),
        control_step_count=terminal_context.control_step_count,
    )
    box_center = _as_float_tuple(box_center_w, 3, name="box center")
    box_linear_velocity = _as_float_tuple(
        box_linear_velocity_w,
        3,
        name="box linear velocity",
    )
    box_angular_velocity = _as_float_tuple(
        box_angular_velocity_w,
        3,
        name="box angular velocity",
    )
    left_pose = _as_float_tuple(left_ee_pose_w, 7, name="left EE pose")
    right_pose = _as_float_tuple(right_ee_pose_w, 7, name="right EE pose")
    support_bounds = _as_float_tuple(
        support.support_bounds_w,
        4,
        name="shelf support bounds",
    )
    grasp_evidence = classify_bimanual_grasp(
        box_center_w=box_center,
        left_ee_pose_w=left_pose,
        right_ee_pose_w=right_pose,
        left_contact=left_contact.in_contact,
        right_contact=right_contact.in_contact,
        max_ee_box_distance_m=max_ee_box_distance_m,
    )
    root_up_alignment = fall_evidence.root_up_alignment
    fall_candidate = fall_evidence.fall_candidate
    if fall_candidate != (terminal_context.fall_streak > 0):
        raise RecoveryTelemetryIncompleteError(("authoritative_terminal_context",))
    fall = terminal_context.fall_confirmed
    success = bool(support.placed)
    timeout = terminal_context.time_limit

    return PrivilegedRecoveryTelemetry(
        schema_version=RECOVERY_TELEMETRY_SCHEMA_VERSION,
        task_identity=task_identity,
        env_index=int(env_index),
        box_center_w=(box_center[0], box_center[1], box_center[2]),
        box_linear_velocity_w=(
            box_linear_velocity[0],
            box_linear_velocity[1],
            box_linear_velocity[2],
        ),
        box_angular_velocity_w=(
            box_angular_velocity[0],
            box_angular_velocity[1],
            box_angular_velocity[2],
        ),
        shelf_bounds_w=(
            support_bounds[0],
            support_bounds[1],
            support_bounds[2],
            support_bounds[3],
        ),
        support_surface_z_m=float(support.support_top_z_m),
        target_support_surface_z_m=float(support.target_support_top_z_m),
        left_ee_pose_w=(
            left_pose[0],
            left_pose[1],
            left_pose[2],
            left_pose[3],
            left_pose[4],
            left_pose[5],
            left_pose[6],
        ),
        right_ee_pose_w=(
            right_pose[0],
            right_pose[1],
            right_pose[2],
            right_pose[3],
            right_pose[4],
            right_pose[5],
            right_pose[6],
        ),
        left_box_contact=left_contact,
        right_box_contact=right_contact,
        grasp_evidence=grasp_evidence,
        grasp=grasp_evidence.bimanual_grasp,
        xy_mismatch_m=float(support.xy_mismatch_m),
        z_gap_m=float(support.z_gap_m),
        z_mismatch_m=float(support.z_mismatch_m),
        placement_distance_m=float(support.placement_distance_m),
        placement=success,
        success=success,
        root_up_alignment=root_up_alignment,
        control_step_count=terminal_context.control_step_count,
        max_control_steps=terminal_context.max_control_steps,
        fall_candidate=fall_candidate,
        fall_streak=terminal_context.fall_streak,
        fall_confirm_steps=terminal_context.fall_confirm_steps,
        fall=fall,
        time_limit=timeout,
        terminal_reason=classify_terminal(
            success=success, fall=fall, time_limit=timeout
        ),
    )


def _collect_privileged_object_ids(value: object) -> set[int]:
    object_ids: set[int] = set()
    visited: set[int] = set()

    def visit(item: object) -> None:
        if item is None or isinstance(
            item, (bool, int, float, complex, str, bytes, bytearray)
        ):
            return
        item_id = id(item)
        if item_id in visited:
            return
        visited.add(item_id)
        object_ids.add(item_id)
        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for nested in item:
                visit(nested)

    visit(value)
    return object_ids


def _actor_observation_digest(
    term_names: tuple[str, ...],
    policy: Mapping[str, object],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"pp-box-residual-actor-observation-v1\0")
    for name in term_names:
        value = policy[name]
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        if isinstance(value, torch.Tensor):
            metadata = (
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
                int(value.data_ptr()),
                int(getattr(value, "_version", 0)),
            )
            hasher.update(repr(metadata).encode("ascii"))
        else:
            hasher.update(type(value).__qualname__.encode("utf-8"))
            hasher.update(b":")
            hasher.update(str(id(value)).encode("ascii"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _policy_observation_source(env: object) -> tuple[object, Mapping[str, object]]:
    manager = getattr(env, "observation_manager", None)
    raw = getattr(manager, "_obs_buffer", None)
    if not isinstance(raw, Mapping):
        raw = getattr(env, "obs_buf", None)
    policy = raw.get("policy") if isinstance(raw, Mapping) else None
    if not isinstance(policy, Mapping):
        raise RecoveryTelemetryIncompleteError(("actor_observation_source",))
    return manager, policy


def issue_residual_actor_observation(env: object) -> ResidualActorObservation:
    """Read only allowlisted policy terms from the live observation manager."""

    _require_pp_box_task_identity(env)
    manager, policy = _policy_observation_source(env)
    configured = getattr(manager, "_group_obs_term_names", None)
    configured_policy = (
        configured.get("policy") if isinstance(configured, Mapping) else None
    )
    term_names = tuple(str(name) for name in (configured_policy or ()))
    if term_names != _ACTOR_POLICY_TERM_ALLOWLIST or set(policy) != set(
        _ACTOR_POLICY_TERM_ALLOWLIST
    ):
        raise RecoveryTelemetryIncompleteError(("actor_observation_allowlist",))
    issued_policy = MappingProxyType({name: policy[name] for name in term_names})
    observation = object.__new__(ResidualActorObservation)
    values = {
        "schema_version": RESIDUAL_ACTOR_OBSERVATION_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "policy_term_names": term_names,
        "policy": issued_policy,
        "source_manager_id": id(manager),
        "source_value_ids": tuple(id(issued_policy[name]) for name in term_names),
        "payload_digest": _actor_observation_digest(term_names, issued_policy),
        "_producer_token": _RESIDUAL_ACTOR_OBSERVATION_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(observation, name, value)
    return observation


def assert_actor_observation_isolated(
    actor_observation: object,
    privileged_state: PrivilegedRecoveryTelemetry,
    *,
    env: object,
) -> None:
    """Validate exact allowlist provenance before checking object alias leaks."""

    if (
        type(actor_observation) is not ResidualActorObservation
        or actor_observation._producer_token is not _RESIDUAL_ACTOR_OBSERVATION_TOKEN
        or actor_observation.schema_version != RESIDUAL_ACTOR_OBSERVATION_SCHEMA_VERSION
        or actor_observation.task_identity != PP_BOX_TASK_IDENTITY
    ):
        raise PrivilegedObservationLeakError(("$.provenance",))
    manager, current_policy = _policy_observation_source(env)
    term_names = actor_observation.policy_term_names
    if (
        term_names != _ACTOR_POLICY_TERM_ALLOWLIST
        or actor_observation.source_manager_id != id(manager)
        or set(actor_observation.policy) != set(_ACTOR_POLICY_TERM_ALLOWLIST)
        or tuple(id(current_policy[name]) for name in term_names)
        != actor_observation.source_value_ids
        or tuple(id(actor_observation.policy[name]) for name in term_names)
        != actor_observation.source_value_ids
        or actor_observation.payload_digest
        != _actor_observation_digest(term_names, actor_observation.policy)
    ):
        raise PrivilegedObservationLeakError(("$.provenance",))

    privileged_object_ids = _collect_privileged_object_ids(privileged_state)
    leak_paths: set[str] = set()
    visited: set[int] = set()

    def visit(item: object, path: str) -> None:
        if item is None or isinstance(
            item, (bool, int, float, complex, str, bytes, bytearray)
        ):
            return
        item_id = id(item)
        if item_id in privileged_object_ids:
            leak_paths.add(path)
            return
        if item_id in visited:
            return
        visited.add(item_id)

        if isinstance(item, Mapping):
            for key, nested in item.items():
                key_path = f"{path}.{key}"
                if isinstance(key, str) and key in _PRIVILEGED_FIELD_NAMES:
                    leak_paths.add(key_path)
                visit(nested, key_path)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                field_path = f"{path}.{field.name}"
                if field.name in _PRIVILEGED_FIELD_NAMES:
                    leak_paths.add(field_path)
                visit(getattr(item, field.name), field_path)

    visit(actor_observation.policy, "$.policy")
    if leak_paths:
        raise PrivilegedObservationLeakError(tuple(leak_paths))


def _scene_get(scene: object, key: str) -> object | None:
    if isinstance(scene, Mapping):
        return scene.get(key)
    try:
        return scene[key]  # type: ignore[index]
    except Exception:  # noqa: BLE001 - scene implementations use custom lookup errors.
        return getattr(scene, key, None)


def _require_pp_box_task_identity(env: object) -> str:
    cfg = getattr(env, "cfg", None)
    cfg_identity = getattr(cfg, "recovery_task_identity", None)
    runtime_env_name = getattr(cfg, "env_name", None)
    env_identity = getattr(env, "recovery_task_identity", cfg_identity)
    if (
        runtime_env_name != PP_BOX_TASK_IDENTITY
        or cfg_identity != PP_BOX_TASK_IDENTITY
        or env_identity != PP_BOX_TASK_IDENTITY
    ):
        raise RecoveryTelemetryIncompleteError(("task_identity",))
    return PP_BOX_TASK_IDENTITY


_MATERIALIZED_ENV_PATH = re.compile(
    r"^(?P<namespace>/World/envs/env_[0-9]+)(?P<relative>/.*)$"
)
_ENV_PATH_EXPRESSION = re.compile(r"^/World/envs/env_(?:\.\*|[0-9]+)")
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical_env_relative_path(path: object) -> str:
    value = str(path)
    if value.startswith("{ENV_REGEX_NS}"):
        return value[len("{ENV_REGEX_NS}") :]
    return _ENV_PATH_EXPRESSION.sub("", value, count=1)


def _materialized_env_paths(
    raw_paths: object,
    *,
    expected_expression: str,
    expected_leaf_name: str,
    num_envs: int,
    capability: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(raw_paths, Sequence) or isinstance(
        raw_paths, (str, bytes, bytearray)
    ):
        raise RecoveryTelemetryIncompleteError((capability,))
    paths = tuple(str(path) for path in raw_paths)
    if len(paths) != num_envs or len(set(paths)) != num_envs:
        raise RecoveryTelemetryIncompleteError((capability,))

    expected_relative = _canonical_env_relative_path(expected_expression)
    namespaces = []
    for path in paths:
        match = _MATERIALIZED_ENV_PATH.fullmatch(path)
        if (
            match is None
            or match.group("relative") != expected_relative
            or path.rsplit("/", 1)[-1] != expected_leaf_name
        ):
            raise RecoveryTelemetryIncompleteError((capability,))
        namespaces.append(match.group("namespace"))
    expected_namespaces = {f"/World/envs/env_{index}" for index in range(num_envs)}
    if set(namespaces) != expected_namespaces:
        raise RecoveryTelemetryIncompleteError((capability,))
    return paths, tuple(namespaces)


def _resolve_candidate_filter_asset(
    scene: object,
    *,
    num_envs: int,
) -> tuple[str, object, str, tuple[str, ...], tuple[str, ...]]:
    candidates = tuple(
        (scene_key, asset)
        for scene_key in ("box", "Box")
        if (asset := _scene_get(scene, scene_key)) is not None
    )
    if len(candidates) != 1:
        raise RecoveryTelemetryIncompleteError(("runtime_contact_filter_asset",))
    scene_key, asset = candidates[0]
    prim_path = str(getattr(getattr(asset, "cfg", None), "prim_path", ""))
    if not prim_path or prim_path.rsplit("/", 1)[-1] != "Box":
        raise RecoveryTelemetryIncompleteError(("runtime_contact_filter_asset",))
    view = getattr(asset, "root_physx_view", None)
    paths, namespaces = _materialized_env_paths(
        getattr(view, "prim_paths", None),
        expected_expression=prim_path,
        expected_leaf_name="Box",
        num_envs=num_envs,
        capability="runtime_contact_filter_asset",
    )
    return scene_key, asset, prim_path, paths, namespaces


def _canonical_json_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def _sensor_identity_payload(
    sensor: object,
    binding: PairwiseContactBinding,
    *,
    target_asset_scene_key: str,
    target_prim_paths: tuple[str, ...],
) -> dict[str, object]:
    force_matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
    cfg = getattr(sensor, "cfg", None)
    return {
        "sensor_scene_key": binding.sensor_scene_key,
        "sensor_body_name": binding.sensor_body_name,
        "filtered_body_name": binding.filtered_body_name,
        "sensor_prim_path_expression": str(getattr(cfg, "prim_path", "")),
        "sensor_prim_paths": tuple(
            str(path)
            for path in (
                getattr(getattr(sensor, "body_physx_view", None), "prim_paths", ())
                or ()
            )
        ),
        "sensor_body_names": tuple(
            str(name) for name in (getattr(sensor, "body_names", ()) or ())
        ),
        "num_bodies": getattr(sensor, "num_bodies", None),
        "filter_prim_paths": tuple(
            str(path) for path in (getattr(cfg, "filter_prim_paths_expr", ()) or ())
        ),
        "filter_count": getattr(
            getattr(sensor, "contact_physx_view", None), "filter_count", None
        ),
        "target_asset_scene_key": target_asset_scene_key,
        "target_prim_paths": target_prim_paths,
        "force_shape": tuple(force_matrix.shape)
        if isinstance(force_matrix, torch.Tensor)
        else None,
        "force_dtype": str(force_matrix.dtype)
        if isinstance(force_matrix, torch.Tensor)
        else None,
        "force_device": str(force_matrix.device)
        if isinstance(force_matrix, torch.Tensor)
        else None,
    }


def _runtime_contact_identities(
    env: object,
    bindings: Mapping[str, HandContactBinding],
) -> tuple[str, dict[str, str], str, tuple[str, ...]]:
    task_identity = _require_pp_box_task_identity(env)
    num_envs = int(getattr(env, "num_envs", 0))
    scene = getattr(env, "scene", None)
    if num_envs <= 0 or scene is None:
        raise RecoveryTelemetryIncompleteError(("scene",))
    (
        target_asset_scene_key,
        _target_asset,
        _target_expression,
        target_prim_paths,
        _target_namespaces,
    ) = _resolve_candidate_filter_asset(scene, num_envs=num_envs)
    sensor_identities: dict[str, str] = {}
    for side in ("left", "right"):
        for binding in bindings[side].sensors:
            sensor = _scene_get(scene, binding.sensor_scene_key)
            if sensor is None:
                raise RecoveryTelemetryIncompleteError(
                    (f"runtime_contact_sensor:{binding.sensor_scene_key}",)
                )
            sensor_identities[binding.sensor_scene_key] = _canonical_json_digest(
                _sensor_identity_payload(
                    sensor,
                    binding,
                    target_asset_scene_key=target_asset_scene_key,
                    target_prim_paths=target_prim_paths,
                )
            )
    registered_executor = _CONTACT_CALIBRATION_EXECUTORS.get(id(env))
    executor_identity = None
    if registered_executor is not None and registered_executor[0] is env:
        executor = registered_executor[1]
        if type(executor) is _ContactCalibrationExecutor:
            executor_identity = {
                "snapshot_fidelity_tier": executor.snapshot_fidelity_tier,
                "coordinator_binding_identity": executor.coordinator_binding_identity,
                "coordinator_binding_digest": executor.coordinator_binding_digest,
            }
    runtime_identity = _canonical_json_digest(
        {
            "task_identity": task_identity,
            "num_envs": num_envs,
            "device": str(getattr(env, "device", "")),
            "target_asset_scene_key": target_asset_scene_key,
            "target_prim_paths": target_prim_paths,
            "sensor_identities": sensor_identities,
            "snapshot_coordinator": executor_identity,
        }
    )
    return (
        runtime_identity,
        sensor_identities,
        target_asset_scene_key,
        target_prim_paths,
    )


def install_pp_box_contact_calibration_executor(
    env: object,
) -> None:
    """Bind the one-time calibration executor to the installed coordinator."""

    _require_pp_box_task_identity(env)
    coordinator = getattr(env, "recovery_state_coordinator", None)
    required_methods = ("capture", "digest", "preflight", "restore")
    if coordinator is None or any(
        not callable(getattr(coordinator, name, None)) for name in required_methods
    ):
        raise RecoveryTelemetryIncompleteError(("recovery_state_coordinator",))
    try:
        raw_binding = coordinator.binding_identity
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != {
            "schema_version",
            "coordinator_type",
            "task_identity",
        }:
            raise ValueError("invalid coordinator binding schema")
        coordinator_type = str(raw_binding["coordinator_type"])
        actual_type = f"{type(coordinator).__module__}.{type(coordinator).__qualname__}"
        if (
            type(raw_binding["schema_version"]) is not int
            or raw_binding["schema_version"] != 1
            or type(coordinator) is not RecoveryStateCoordinator
            or coordinator_type != actual_type
            or not coordinator_type.endswith(".RecoveryStateCoordinator")
            or raw_binding["task_identity"] != PP_BOX_TASK_IDENTITY
        ):
            raise ValueError("coordinator binding identity mismatch")
        binding_identity = tuple(sorted(raw_binding.items()))
        binding_digest = _canonical_json_digest(binding_identity)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RecoveryTelemetryIncompleteError(("recovery_state_coordinator",)) from exc
    executor = _ContactCalibrationExecutor(
        coordinator=coordinator,
        snapshot_fidelity_tier="state_only",
        coordinator_binding_identity=binding_identity,
        coordinator_binding_digest=binding_digest,
        _executor_token=_CONTACT_CALIBRATION_EXECUTOR_TOKEN,
    )
    _CONTACT_CALIBRATION_EXECUTORS[id(env)] = (env, executor)
    _CONTACT_CALIBRATION_RECEIPTS.pop(id(env), None)


def _installed_contact_calibration_executor(env: object) -> _ContactCalibrationExecutor:
    registered = _CONTACT_CALIBRATION_EXECUTORS.get(id(env))
    if registered is None or registered[0] is not env:
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_calibration_executor",)
        )
    executor = registered[1]
    if (
        type(executor) is not _ContactCalibrationExecutor
        or executor._executor_token is not _CONTACT_CALIBRATION_EXECUTOR_TOKEN
        or getattr(env, "recovery_state_coordinator", None) is not executor.coordinator
    ):
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_calibration_executor",)
        )
    try:
        current_binding = tuple(sorted(executor.coordinator.binding_identity.items()))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_calibration_executor",)
        ) from exc
    if (
        current_binding != executor.coordinator_binding_identity
        or _canonical_json_digest(current_binding)
        != executor.coordinator_binding_digest
    ):
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_calibration_executor",)
        )
    return executor


def _calibration_fixed_action(env: object) -> torch.Tensor:
    manager = getattr(env, "action_manager", None)
    action = getattr(manager, "_action", None)
    if action is None:
        action = getattr(manager, "action", None)
    if not isinstance(action, torch.Tensor) or action.ndim != 2:
        raise RecoveryTelemetryIncompleteError(("runtime_contact_calibration_action",))
    return action.detach().clone()


def _runtime_collision_local_bounds(
    root_prim_paths: Sequence[str],
) -> tuple[tuple[float, float, float, float, float, float], ...]:
    try:
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics
    except Exception as exc:
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_collision_geometry",)
        ) from exc
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RecoveryTelemetryIncompleteError(("runtime_contact_collision_geometry",))
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
            useExtentsHint=False,
        )
    except Exception as exc:
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_collision_geometry",)
        ) from exc

    result = []
    try:
        for path in root_prim_paths:
            if type(path) is not str or not path.startswith("/World/envs/env_"):
                raise ValueError("invalid collision root path")
            root = stage.GetPrimAtPath(path)
            if root is None or not root.IsValid() or not root.IsActive():
                raise ValueError("collision root is unavailable")
            minima = [math.inf, math.inf, math.inf]
            maxima = [-math.inf, -math.inf, -math.inf]
            collision_count = 0
            predicate = Usd.TraverseInstanceProxies()
            for prim in Usd.PrimRange(root, predicate):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if enabled is False:
                    continue
                aligned = bbox_cache.ComputeRelativeBound(
                    prim, root
                ).ComputeAlignedRange()
                lower = aligned.GetMin()
                upper = aligned.GetMax()
                values = tuple(float(lower[index]) for index in range(3)) + tuple(
                    float(upper[index]) for index in range(3)
                )
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("collision bounds are non-finite")
                if any(values[index + 3] <= values[index] for index in range(3)):
                    continue
                for index in range(3):
                    minima[index] = min(minima[index], values[index])
                    maxima[index] = max(maxima[index], values[index + 3])
                collision_count += 1
            if collision_count == 0:
                raise ValueError("collision root has no enabled finite geometry")
            bounds = (
                minima[0],
                maxima[0],
                minima[1],
                maxima[1],
                minima[2],
                maxima[2],
            )
            if not all(math.isfinite(value) for value in bounds):
                raise ValueError("collision bounds are incomplete")
            result.append(bounds)
    except Exception as exc:
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_collision_geometry",)
        ) from exc
    return tuple(result)


def _quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if quaternion.shape != (*vector.shape[:-1], 4) or vector.shape[-1] != 3:
        raise RecoveryTelemetryIncompleteError(("runtime_contact_calibration_pose",))
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm <= 1.0e-8).any()):
        raise RecoveryTelemetryIncompleteError(("runtime_contact_calibration_pose",))
    unit = quaternion / norm
    xyz = unit[..., 1:]
    cross = torch.linalg.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (
        unit[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)
    )


def _contact_calibration_touch_state(
    box_state: torch.Tensor,
    body_pose: torch.Tensor,
    body_collision_bounds: Sequence[Sequence[float]],
    box_collision_bounds: Sequence[Sequence[float]],
) -> torch.Tensor:
    num_envs = box_state.shape[0]
    if (
        body_pose.shape != (num_envs, 7)
        or len(body_collision_bounds) != num_envs
        or len(box_collision_bounds) != num_envs
    ):
        raise RecoveryTelemetryIncompleteError(("runtime_contact_collision_geometry",))
    local_offsets = []
    for body_bounds, box_bounds in zip(
        body_collision_bounds, box_collision_bounds, strict=True
    ):
        if len(body_bounds) != 6 or len(box_bounds) != 6:
            raise RecoveryTelemetryIncompleteError(
                ("runtime_contact_collision_geometry",)
            )
        values = tuple(float(value) for value in (*body_bounds, *box_bounds))
        if not all(math.isfinite(value) for value in values):
            raise RecoveryTelemetryIncompleteError(
                ("runtime_contact_collision_geometry",)
            )
        body_x_lo, body_x_hi, body_y_lo, body_y_hi, body_z_lo, body_z_hi = values[:6]
        box_x_lo, box_x_hi, box_y_lo, box_y_hi, box_z_lo, box_z_hi = values[6:]
        if (
            body_x_hi <= body_x_lo
            or body_y_hi <= body_y_lo
            or body_z_hi <= body_z_lo
            or box_x_hi <= box_x_lo
            or box_y_hi <= box_y_lo
            or box_z_hi <= box_z_lo
        ):
            raise RecoveryTelemetryIncompleteError(
                ("runtime_contact_collision_geometry",)
            )
        local_offsets.append(
            (
                0.5 * (body_x_lo + body_x_hi - box_x_lo - box_x_hi),
                0.5 * (body_y_lo + body_y_hi - box_y_lo - box_y_hi),
                body_z_hi + EMPIRICAL_CONTACT_TOUCH_GAP_M - box_z_lo,
            )
        )
    offsets = torch.tensor(
        local_offsets, device=box_state.device, dtype=box_state.dtype
    )
    local_velocity = torch.zeros_like(offsets)
    local_velocity[:, 2] = EMPIRICAL_CONTACT_TOUCH_VELOCITY_M_S
    body_quaternion = body_pose[:, 3:7]
    quaternion_norm = torch.linalg.vector_norm(body_quaternion, dim=-1, keepdim=True)
    normalized_quaternion = body_quaternion / quaternion_norm
    target_state = box_state.detach().clone()
    target_state[:, :3] = body_pose[:, :3] + _quat_rotate_wxyz(
        normalized_quaternion, offsets
    )
    target_state[:, 3:7] = normalized_quaternion
    target_state[:, 7:10] = _quat_rotate_wxyz(normalized_quaternion, local_velocity)
    target_state[:, 10:13] = 0.0
    return target_state


def _write_contact_calibration_phase(
    env: object,
    binding: PairwiseContactBinding,
    phase: Literal["baseline", "target_touch", "target_removed"],
) -> None:
    scene = getattr(env, "scene", None)
    box = _scene_get(scene, "box")
    if box is None:
        box = _scene_get(scene, "Box")
    robot = _scene_get(scene, "robot")
    box_state = getattr(getattr(box, "data", None), "root_state_w", None)
    writer = getattr(box, "write_root_state_to_sim", None)
    if (
        robot is None
        or not isinstance(box_state, torch.Tensor)
        or box_state.ndim != 2
        or box_state.shape[1] < 13
        or not callable(writer)
    ):
        raise RecoveryTelemetryIncompleteError(("runtime_contact_calibration_pose",))
    body_pose = _body_pose_tensor(
        robot, binding.sensor_body_name, num_envs=box_state.shape[0]
    )
    target_state = box_state.detach().clone()
    target_state[:, :3] = body_pose[:, :3]
    target_state[:, 7:13] = 0.0
    if phase == "target_touch":
        sensor = _scene_get(scene, binding.sensor_scene_key)
        body_prim_paths = tuple(
            getattr(getattr(sensor, "body_physx_view", None), "prim_paths", ())
        )
        box_prim_paths = tuple(
            getattr(getattr(box, "root_physx_view", None), "prim_paths", ())
        )
        if (
            len(body_prim_paths) != box_state.shape[0]
            or len(box_prim_paths) != box_state.shape[0]
        ):
            raise RecoveryTelemetryIncompleteError(
                ("runtime_contact_collision_geometry",)
            )
        target_state = _contact_calibration_touch_state(
            box_state,
            body_pose,
            _runtime_collision_local_bounds(body_prim_paths),
            _runtime_collision_local_bounds(box_prim_paths),
        )
    else:
        target_state[:, 2] += 2.0 if phase == "baseline" else 3.0
    env_ids = torch.arange(
        box_state.shape[0],
        device=torch.device(getattr(env, "device", box_state.device)),
        dtype=torch.long,
    )
    writer(target_state, env_ids=env_ids)


def _control_step_cursor(env: object) -> int:
    value = getattr(env, "common_step_counter", None)
    if type(value) is not int or value < 0:
        raise RecoveryTelemetryIncompleteError(("runtime_contact_calibration_step",))
    return value


def _phase_receipt_digest(receipt: ContactCalibrationPhaseReceipt) -> str:
    return _canonical_json_digest(
        {
            "schema_version": receipt.schema_version,
            "task_identity": receipt.task_identity,
            "phase": receipt.phase,
            "sensor_scene_key": receipt.sensor_scene_key,
            "sensor_body_name": receipt.sensor_body_name,
            "source_snapshot_digest": receipt.source_snapshot_digest,
            "phase_state_digest": receipt.phase_state_digest,
            "runtime_identity_digest": receipt.runtime_identity_digest,
            "sensor_identity_digest": receipt.sensor_identity_digest,
            "control_step_before": receipt.control_step_before,
            "control_step_after": receipt.control_step_after,
            "force_shape": receipt.force_shape,
            "force_dtype": receipt.force_dtype,
            "force_device": receipt.force_device,
            "force_byte_order": receipt.force_byte_order,
            "raw_force_size": len(receipt.raw_force_bytes),
            "raw_force_sha256": receipt.raw_force_sha256,
        }
    )


def _issue_phase_receipt(
    *,
    phase: Literal["baseline", "target_touch", "target_removed"],
    binding: PairwiseContactBinding,
    source_snapshot_digest: str,
    phase_state_digest: str,
    runtime_identity_digest: str,
    sensor_identity_digest: str,
    control_step_before: int,
    control_step_after: int,
    force_matrix: torch.Tensor,
) -> ContactCalibrationPhaseReceipt:
    raw_force_bytes = (
        force_matrix.detach().contiguous().cpu().numpy().tobytes(order="C")
    )
    receipt = object.__new__(ContactCalibrationPhaseReceipt)
    values = {
        "schema_version": RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "phase": phase,
        "sensor_scene_key": binding.sensor_scene_key,
        "sensor_body_name": binding.sensor_body_name,
        "source_snapshot_digest": source_snapshot_digest,
        "phase_state_digest": phase_state_digest,
        "runtime_identity_digest": runtime_identity_digest,
        "sensor_identity_digest": sensor_identity_digest,
        "control_step_before": control_step_before,
        "control_step_after": control_step_after,
        "force_shape": tuple(force_matrix.shape),
        "force_dtype": str(force_matrix.dtype),
        "force_device": str(force_matrix.device),
        "force_byte_order": sys.byteorder,
        "raw_force_bytes": raw_force_bytes,
        "raw_force_sha256": hashlib.sha256(raw_force_bytes).hexdigest(),
        "_producer_token": _CONTACT_CALIBRATION_RECEIPT_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    object.__setattr__(receipt, "receipt_digest", _phase_receipt_digest(receipt))
    return receipt


def _tensor_runtime_evidence(value: object) -> Mapping[str, object]:
    if not isinstance(value, torch.Tensor):
        return {"status": "unavailable"}
    raw_bytes = value.detach().contiguous().cpu().numpy().tobytes(order="C")
    return {
        "status": "captured",
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "raw_bytes": raw_bytes,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _phase_force_runtime_evidence(
    receipt: ContactCalibrationPhaseReceipt,
) -> Mapping[str, object]:
    return {
        "status": "captured",
        "phase": receipt.phase,
        "shape": receipt.force_shape,
        "dtype": receipt.force_dtype,
        "device": receipt.force_device,
        "raw_bytes": receipt.raw_force_bytes,
        "raw_sha256": receipt.raw_force_sha256,
        "phase_receipt_digest": receipt.receipt_digest,
    }


def _sensor_receipt_digest(receipt: ContactSensorCalibrationReceipt) -> str:
    return _canonical_json_digest(
        {
            "schema_version": receipt.schema_version,
            "task_identity": receipt.task_identity,
            "sensor_scene_key": receipt.sensor_scene_key,
            "sensor_body_name": receipt.sensor_body_name,
            "target_asset_scene_key": receipt.target_asset_scene_key,
            "target_prim_paths": receipt.target_prim_paths,
            "source_snapshot_digest": receipt.source_snapshot_digest,
            "runtime_identity_digest": receipt.runtime_identity_digest,
            "sensor_identity_digest": receipt.sensor_identity_digest,
            "phase_receipt_digests": tuple(
                phase.receipt_digest for phase in receipt.phases
            ),
        }
    )


def _issue_sensor_receipt(
    *,
    binding: PairwiseContactBinding,
    target_asset_scene_key: str,
    target_prim_paths: tuple[str, ...],
    source_snapshot_digest: str,
    runtime_identity_digest: str,
    sensor_identity_digest: str,
    phases: tuple[ContactCalibrationPhaseReceipt, ...],
) -> ContactSensorCalibrationReceipt:
    receipt = object.__new__(ContactSensorCalibrationReceipt)
    values = {
        "schema_version": RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "sensor_scene_key": binding.sensor_scene_key,
        "sensor_body_name": binding.sensor_body_name,
        "target_asset_scene_key": target_asset_scene_key,
        "target_prim_paths": target_prim_paths,
        "source_snapshot_digest": source_snapshot_digest,
        "runtime_identity_digest": runtime_identity_digest,
        "sensor_identity_digest": sensor_identity_digest,
        "phases": phases,
        "_producer_token": _CONTACT_CALIBRATION_RECEIPT_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    object.__setattr__(receipt, "receipt_digest", _sensor_receipt_digest(receipt))
    return receipt


def _execution_receipt_digest(receipt: ContactCalibrationExecutionReceipt) -> str:
    return _canonical_json_digest(
        {
            "schema_version": receipt.schema_version,
            "task_identity": receipt.task_identity,
            "source_snapshot_digest": receipt.source_snapshot_digest,
            "runtime_identity_digest": receipt.runtime_identity_digest,
            "snapshot_fidelity_tier": receipt.snapshot_fidelity_tier,
            "coordinator_binding_identity": receipt.coordinator_binding_identity,
            "coordinator_binding_digest": receipt.coordinator_binding_digest,
            "sensor_receipt_digests": tuple(
                sensor.receipt_digest for sensor in receipt.sensor_receipts
            ),
        }
    )


def _decode_phase_forces(
    receipt: ContactCalibrationPhaseReceipt,
) -> tuple[tuple[float, ...], ...]:
    format_by_dtype = {"torch.float32": "f", "torch.float64": "d"}
    scalar_format = format_by_dtype.get(receipt.force_dtype)
    if scalar_format is None or receipt.force_byte_order not in {"little", "big"}:
        raise ValueError("unsupported contact receipt force encoding")
    prefix = "<" if receipt.force_byte_order == "little" else ">"
    scalar_size = struct.calcsize(prefix + scalar_format)
    expected_values = math.prod(receipt.force_shape)
    if len(receipt.raw_force_bytes) != expected_values * scalar_size:
        raise ValueError("contact receipt raw force size does not match shape")
    values = tuple(
        float(value[0])
        for value in struct.iter_unpack(prefix + scalar_format, receipt.raw_force_bytes)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("contact receipt force is non-finite")
    return tuple(values[offset : offset + 3] for offset in range(0, len(values), 3))


def _validate_sensor_calibration_receipt(
    receipt: object,
    *,
    binding: PairwiseContactBinding,
    target_asset_scene_key: str,
    target_prim_paths: tuple[str, ...],
    source_snapshot_digest: str,
    runtime_identity_digest: str,
    sensor_identity_digest: str,
    num_envs: int,
) -> ContactSensorCalibrationReceipt:
    capability = f"runtime_contact_mapping_receipt:{binding.sensor_scene_key}"
    try:
        if type(receipt) is not ContactSensorCalibrationReceipt:
            raise ValueError("wrong receipt type")
        expected_phases = ("baseline", "target_touch", "target_removed")
        state_digests = tuple(phase.phase_state_digest for phase in receipt.phases)
        valid = (
            receipt._producer_token is _CONTACT_CALIBRATION_RECEIPT_TOKEN
            and receipt.schema_version == RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION
            and receipt.task_identity == PP_BOX_TASK_IDENTITY
            and receipt.sensor_scene_key == binding.sensor_scene_key
            and receipt.sensor_body_name == binding.sensor_body_name
            and receipt.target_asset_scene_key == target_asset_scene_key
            and receipt.target_prim_paths == target_prim_paths
            and receipt.source_snapshot_digest == source_snapshot_digest
            and receipt.runtime_identity_digest == runtime_identity_digest
            and receipt.sensor_identity_digest == sensor_identity_digest
            and tuple(phase.phase for phase in receipt.phases) == expected_phases
            and len(set(state_digests)) == 3
            and all(
                _SHA256_DIGEST.fullmatch(value) is not None for value in state_digests
            )
            and receipt.receipt_digest == _sensor_receipt_digest(receipt)
        )
        if not valid:
            raise ValueError("receipt identity mismatch")
        decoded = []
        for phase in receipt.phases:
            if (
                type(phase) is not ContactCalibrationPhaseReceipt
                or phase._producer_token is not _CONTACT_CALIBRATION_RECEIPT_TOKEN
                or phase.schema_version
                != RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION
                or phase.task_identity != PP_BOX_TASK_IDENTITY
                or phase.sensor_scene_key != binding.sensor_scene_key
                or phase.sensor_body_name != binding.sensor_body_name
                or phase.source_snapshot_digest != source_snapshot_digest
                or phase.runtime_identity_digest != runtime_identity_digest
                or phase.sensor_identity_digest != sensor_identity_digest
                or phase.control_step_after != phase.control_step_before + 1
                or phase.force_shape != (num_envs, 1, 1, 3)
                or hashlib.sha256(phase.raw_force_bytes).hexdigest()
                != phase.raw_force_sha256
                or phase.receipt_digest != _phase_receipt_digest(phase)
            ):
                raise ValueError("phase receipt mismatch")
            decoded.append(_decode_phase_forces(phase))
        for phase_forces in (decoded[0], decoded[2]):
            if any(
                math.sqrt(sum(value * value for value in force))
                > EMPIRICAL_CONTACT_QUIET_MAX_N
                for force in phase_forces
            ):
                raise ValueError("quiet phase contains contact")
        if len(decoded[1]) != num_envs or any(
            math.sqrt(sum(value * value for value in force))
            < EMPIRICAL_CONTACT_TOUCH_MIN_N
            for force in decoded[1]
        ):
            raise ValueError("touch phase lacks contact")
    except (AttributeError, TypeError, ValueError):
        raise RecoveryTelemetryIncompleteError((capability,)) from None
    return receipt


def _registered_contact_calibration_receipts(
    env: object,
    *,
    bindings: Mapping[str, HandContactBinding],
    runtime_identity_digest: str,
    sensor_identity_digests: Mapping[str, str],
    target_asset_scene_key: str,
    target_prim_paths: tuple[str, ...],
    num_envs: int,
) -> dict[str, ContactSensorCalibrationReceipt]:
    expected_keys = tuple(
        binding.sensor_scene_key
        for side in ("left", "right")
        for binding in bindings[side].sensors
    )
    registered = _CONTACT_CALIBRATION_RECEIPTS.get(id(env))
    if registered is None or registered[0] is not env:
        raise RecoveryTelemetryIncompleteError(
            tuple(f"runtime_contact_mapping_receipt:{key}" for key in expected_keys)
        )
    execution = registered[1]
    executor = _installed_contact_calibration_executor(env)
    try:
        valid_execution = (
            type(execution) is ContactCalibrationExecutionReceipt
            and execution._producer_token is _CONTACT_CALIBRATION_RECEIPT_TOKEN
            and execution.schema_version
            == RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION
            and execution.task_identity == PP_BOX_TASK_IDENTITY
            and execution.runtime_identity_digest == runtime_identity_digest
            and execution.snapshot_fidelity_tier == executor.snapshot_fidelity_tier
            and execution.coordinator_binding_identity
            == executor.coordinator_binding_identity
            and execution.coordinator_binding_digest
            == executor.coordinator_binding_digest
            and _SHA256_DIGEST.fullmatch(execution.source_snapshot_digest) is not None
            and execution.receipt_digest == _execution_receipt_digest(execution)
        )
    except (AttributeError, TypeError, ValueError):
        valid_execution = False
    if not valid_execution:
        raise RecoveryTelemetryIncompleteError(("runtime_contact_mapping_receipts",))
    by_key = {
        receipt.sensor_scene_key: receipt for receipt in execution.sensor_receipts
    }
    if len(by_key) != len(expected_keys) or set(by_key) != set(expected_keys):
        raise RecoveryTelemetryIncompleteError(("runtime_contact_mapping_receipts",))
    result = {}
    bindings_by_key = {
        binding.sensor_scene_key: binding
        for side in ("left", "right")
        for binding in bindings[side].sensors
    }
    for key in expected_keys:
        result[key] = _validate_sensor_calibration_receipt(
            by_key[key],
            binding=bindings_by_key[key],
            target_asset_scene_key=target_asset_scene_key,
            target_prim_paths=target_prim_paths,
            source_snapshot_digest=execution.source_snapshot_digest,
            runtime_identity_digest=runtime_identity_digest,
            sensor_identity_digest=sensor_identity_digests[key],
            num_envs=num_envs,
        )
    return result


def execute_pp_box_contact_calibration(
    env: object,
) -> ContactCalibrationExecutionReceipt:
    """Run the one-time 16-sensor calibration (48 primitive simulator steps)."""

    bindings = _resolve_hand_contact_bindings(env)
    executor = _installed_contact_calibration_executor(env)
    (
        runtime_identity_digest,
        sensor_identity_digests,
        target_asset_scene_key,
        target_prim_paths,
    ) = _runtime_contact_identities(env, bindings)
    coordinator = executor.coordinator
    source_snapshot = coordinator.capture(
        fidelity_tier=executor.snapshot_fidelity_tier,
    )
    source_snapshot_digest = str(coordinator.digest(source_snapshot))
    if _SHA256_DIGEST.fullmatch(source_snapshot_digest) is None:
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_calibration_snapshot",)
        )
    coordinator.preflight(
        source_snapshot,
        snapshot_digest=source_snapshot_digest,
    )
    coordinator.restore(
        source_snapshot,
        snapshot_digest=source_snapshot_digest,
    )
    roundtrip_snapshot = coordinator.capture(
        fidelity_tier=executor.snapshot_fidelity_tier,
    )
    roundtrip_digest = str(coordinator.digest(roundtrip_snapshot))
    if roundtrip_digest != source_snapshot_digest:
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_calibration_snapshot_roundtrip",)
        )
    fixed_action = _calibration_fixed_action(env)
    _CONTACT_CALIBRATION_RECEIPTS.pop(id(env), None)
    sensor_receipts = []
    try:
        for side in ("left", "right"):
            for binding in bindings[side].sensors:
                phases = []
                touch_box_state = None
                touch_body_pose = None
                touch_net_force = None
                for phase in ("baseline", "target_touch", "target_removed"):
                    coordinator.restore(
                        source_snapshot,
                        snapshot_digest=source_snapshot_digest,
                    )
                    _write_contact_calibration_phase(env, binding, phase)
                    if phase == "target_touch":
                        box = _scene_get(getattr(env, "scene", None), "box")
                        if box is None:
                            box = _scene_get(getattr(env, "scene", None), "Box")
                        touch_box_state = getattr(
                            getattr(box, "data", None), "root_state_w", None
                        )
                        if isinstance(touch_box_state, torch.Tensor):
                            touch_box_state = touch_box_state.detach().clone()
                        robot = _scene_get(getattr(env, "scene", None), "robot")
                        if robot is not None:
                            touch_body_pose = (
                                _body_pose_tensor(
                                    robot,
                                    binding.sensor_body_name,
                                    num_envs=int(getattr(env, "num_envs", 0)),
                                )
                                .detach()
                                .clone()
                            )
                    step_before = _control_step_cursor(env)
                    env.step(fixed_action.detach().clone())  # type: ignore[attr-defined]
                    step_after = _control_step_cursor(env)
                    phase_snapshot = coordinator.capture(
                        fidelity_tier=executor.snapshot_fidelity_tier,
                    )
                    phase_state_digest = str(coordinator.digest(phase_snapshot))
                    sensor = _scene_get(
                        getattr(env, "scene", None), binding.sensor_scene_key
                    )
                    force_matrix = getattr(
                        getattr(sensor, "data", None), "force_matrix_w", None
                    )
                    if (
                        not isinstance(force_matrix, torch.Tensor)
                        or force_matrix.shape
                        != (int(getattr(env, "num_envs", 0)), 1, 1, 3)
                        or not force_matrix.is_floating_point()
                        or step_after != step_before + 1
                        or _SHA256_DIGEST.fullmatch(phase_state_digest) is None
                    ):
                        raise RecoveryTelemetryIncompleteError(
                            ("runtime_contact_calibration_step",)
                        )
                    if phase == "target_touch":
                        touch_net_force = getattr(
                            getattr(sensor, "data", None), "net_forces_w", None
                        )
                        if isinstance(touch_net_force, torch.Tensor):
                            touch_net_force = touch_net_force.detach().clone()
                    phases.append(
                        _issue_phase_receipt(
                            phase=phase,
                            binding=binding,
                            source_snapshot_digest=source_snapshot_digest,
                            phase_state_digest=phase_state_digest,
                            runtime_identity_digest=runtime_identity_digest,
                            sensor_identity_digest=sensor_identity_digests[
                                binding.sensor_scene_key
                            ],
                            control_step_before=step_before,
                            control_step_after=step_after,
                            force_matrix=force_matrix,
                        )
                    )
                sensor_receipt = _issue_sensor_receipt(
                    binding=binding,
                    target_asset_scene_key=target_asset_scene_key,
                    target_prim_paths=target_prim_paths,
                    source_snapshot_digest=source_snapshot_digest,
                    runtime_identity_digest=runtime_identity_digest,
                    sensor_identity_digest=sensor_identity_digests[
                        binding.sensor_scene_key
                    ],
                    phases=tuple(phases),
                )
                try:
                    _validate_sensor_calibration_receipt(
                        sensor_receipt,
                        binding=binding,
                        target_asset_scene_key=target_asset_scene_key,
                        target_prim_paths=target_prim_paths,
                        source_snapshot_digest=source_snapshot_digest,
                        runtime_identity_digest=runtime_identity_digest,
                        sensor_identity_digest=sensor_identity_digests[
                            binding.sensor_scene_key
                        ],
                        num_envs=int(getattr(env, "num_envs", 0)),
                    )
                except RecoveryTelemetryIncompleteError as exc:
                    touch = phases[1]
                    try:
                        touch_forces = _decode_phase_forces(touch)
                    except ValueError:
                        touch_forces = ()
                    if not touch_forces or any(
                        math.sqrt(sum(value * value for value in force))
                        < EMPIRICAL_CONTACT_TOUCH_MIN_N
                        for force in touch_forces
                    ):
                        sensor_cfg = getattr(sensor, "cfg", None)
                        contact_view = getattr(sensor, "contact_physx_view", None)
                        raise RecoveryTelemetryIncompleteError(
                            ("runtime_contact_calibration_touch",),
                            runtime_evidence={
                                "schema": (
                                    "pp_box_contact_calibration_failure_evidence_v1"
                                ),
                                "task_identity": PP_BOX_TASK_IDENTITY,
                                "sensor_scene_key": binding.sensor_scene_key,
                                "sensor_body_name": binding.sensor_body_name,
                                "target_asset_scene_key": target_asset_scene_key,
                                "target_prim_paths": target_prim_paths,
                                "source_snapshot_digest": source_snapshot_digest,
                                "runtime_identity_digest": runtime_identity_digest,
                                "sensor_identity_digest": sensor_identity_digests[
                                    binding.sensor_scene_key
                                ],
                                "configured_filter_prim_paths_expr": tuple(
                                    getattr(
                                        sensor_cfg,
                                        "filter_prim_paths_expr",
                                        (),
                                    )
                                ),
                                "contact_filter_count": getattr(
                                    contact_view, "filter_count", None
                                ),
                                "phase_receipts": tuple(phases),
                                "filtered_force": _phase_force_runtime_evidence(touch),
                                "net_force_w": _tensor_runtime_evidence(
                                    touch_net_force
                                ),
                                "box_root_state_w": _tensor_runtime_evidence(
                                    touch_box_state
                                ),
                                "sensor_body_pose_w": _tensor_runtime_evidence(
                                    touch_body_pose
                                ),
                            },
                        ) from exc
                    raise
                sensor_receipts.append(sensor_receipt)
        execution = object.__new__(ContactCalibrationExecutionReceipt)
        values = {
            "schema_version": RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION,
            "task_identity": PP_BOX_TASK_IDENTITY,
            "source_snapshot_digest": source_snapshot_digest,
            "runtime_identity_digest": runtime_identity_digest,
            "snapshot_fidelity_tier": executor.snapshot_fidelity_tier,
            "coordinator_binding_identity": executor.coordinator_binding_identity,
            "coordinator_binding_digest": executor.coordinator_binding_digest,
            "sensor_receipts": tuple(sensor_receipts),
            "_producer_token": _CONTACT_CALIBRATION_RECEIPT_TOKEN,
        }
        for name, value in values.items():
            object.__setattr__(execution, name, value)
        object.__setattr__(
            execution, "receipt_digest", _execution_receipt_digest(execution)
        )
    finally:
        coordinator.restore(
            source_snapshot,
            snapshot_digest=source_snapshot_digest,
        )
    _CONTACT_CALIBRATION_RECEIPTS[id(env)] = (env, execution)
    env.recovery_contact_mapping_receipt = execution
    env.recovery_runtime_identity_digest = runtime_identity_digest
    return execution


def _resolve_hand_contact_bindings(env: object) -> dict[str, HandContactBinding]:
    cfg = getattr(env, "cfg", None)
    raw = getattr(cfg, "recovery_contact_bindings", None)
    if raw is None:
        raw = getattr(env, "recovery_contact_bindings", None)
    missing = []
    if not isinstance(raw, Mapping):
        raise RecoveryTelemetryIncompleteError(
            ("left_box_pairwise_contact", "right_box_pairwise_contact")
        )

    bindings: dict[str, HandContactBinding] = {}
    for side in ("left", "right"):
        value = raw.get(side)
        if value is None:
            missing.append(f"{side}_box_pairwise_contact")
            continue
        try:
            binding = HandContactBinding.from_value(value)
            binding.validate(expected_side=side)
        except (TypeError, ValueError):
            missing.append(f"{side}_box_pairwise_contact")
            continue
        bindings[side] = binding
    if missing:
        raise RecoveryTelemetryIncompleteError(missing)
    return bindings


def validate_runtime_hand_contact_sensors(
    env: object,
) -> tuple[RuntimeContactSensorReport, ...]:
    """Materialize and validate all 16 exact hand-to-Box contact sensors."""

    bindings = _resolve_hand_contact_bindings(env)
    num_envs = int(getattr(env, "num_envs", 0))
    scene = getattr(env, "scene", None)
    if num_envs <= 0 or scene is None:
        raise RecoveryTelemetryIncompleteError(("scene",))
    try:
        expected_device = torch.device(env.device)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise RecoveryTelemetryIncompleteError(
            ("runtime_contact_tensor_device",)
        ) from exc

    (
        candidate_asset_scene_key,
        _candidate_asset,
        candidate_asset_prim_path,
        candidate_asset_prim_paths,
        candidate_namespaces,
    ) = _resolve_candidate_filter_asset(scene, num_envs=num_envs)
    materialized = []
    force_matrices = []
    first_dtype: torch.dtype | None = None
    for side in ("left", "right"):
        for binding in bindings[side].sensors:
            capability = f"runtime_contact_sensor:{binding.sensor_scene_key}"
            sensor = _scene_get(scene, binding.sensor_scene_key)
            if sensor is None:
                raise RecoveryTelemetryIncompleteError((capability,))
            try:
                # ContactSensor.data performs the lazy materialization/update.
                force_matrix = getattr(sensor.data, "force_matrix_w", None)
                body_names = tuple(str(name) for name in sensor.body_names)
                num_bodies = int(sensor.num_bodies)
                sensor_cfg = sensor.cfg
                sensor_prim_path = str(sensor_cfg.prim_path)
                filter_prim_paths = tuple(
                    str(path) for path in sensor_cfg.filter_prim_paths_expr
                )
                filter_count = int(sensor.contact_physx_view.filter_count)
                resolved_sensor_prim_paths, sensor_namespaces = _materialized_env_paths(
                    sensor.body_physx_view.prim_paths,
                    expected_expression=sensor_prim_path,
                    expected_leaf_name=binding.sensor_body_name,
                    num_envs=num_envs,
                    capability=capability,
                )
            except RecoveryTelemetryIncompleteError:
                raise
            except Exception as exc:
                raise RecoveryTelemetryIncompleteError((capability,)) from exc

            valid_tensor = (
                isinstance(force_matrix, torch.Tensor)
                and force_matrix.shape == (num_envs, 1, 1, 3)
                and force_matrix.is_floating_point()
                and force_matrix.device == expected_device
            )
            if not valid_tensor:
                raise RecoveryTelemetryIncompleteError((capability,))
            assert isinstance(force_matrix, torch.Tensor)
            if first_dtype is None:
                first_dtype = force_matrix.dtype
            elif force_matrix.dtype != first_dtype:
                raise RecoveryTelemetryIncompleteError((capability,))

            if (
                num_bodies != 1
                or filter_count != 1
                or body_names != (binding.sensor_body_name,)
                or not sensor_prim_path
                or len(filter_prim_paths) != 1
                or sensor_namespaces != candidate_namespaces
                or _canonical_env_relative_path(sensor_prim_path)
                != f"/Robot/{binding.sensor_body_name}"
                or _canonical_env_relative_path(filter_prim_paths[0])
                != _canonical_env_relative_path(candidate_asset_prim_path)
                or binding.filtered_body_name
                != candidate_asset_prim_path.rsplit("/", 1)[-1]
            ):
                raise RecoveryTelemetryIncompleteError((capability,))
            force_matrices.append(force_matrix[:, 0, 0, :])
            materialized.append(
                (
                    side,
                    binding,
                    sensor_prim_path,
                    resolved_sensor_prim_paths,
                    body_names,
                    filter_prim_paths,
                    num_bodies,
                    filter_count,
                    force_matrix,
                )
            )

    finite_by_sensor = (
        torch.isfinite(torch.stack(force_matrices, dim=1))
        .all(dim=0)
        .all(dim=-1)
        .detach()
        .cpu()
        .tolist()
    )
    for is_finite, values in zip(finite_by_sensor, materialized, strict=True):
        if not bool(is_finite):
            binding = values[1]
            raise RecoveryTelemetryIncompleteError(
                (f"runtime_contact_sensor:{binding.sensor_scene_key}",)
            )

    (
        runtime_identity_digest,
        sensor_identity_digests,
        identity_asset_scene_key,
        identity_target_prim_paths,
    ) = _runtime_contact_identities(env, bindings)
    if (
        identity_asset_scene_key != candidate_asset_scene_key
        or identity_target_prim_paths != candidate_asset_prim_paths
    ):
        raise RecoveryTelemetryIncompleteError(("runtime_contact_filter_asset",))
    mapping_receipts = _registered_contact_calibration_receipts(
        env,
        bindings=bindings,
        runtime_identity_digest=runtime_identity_digest,
        sensor_identity_digests=sensor_identity_digests,
        target_asset_scene_key=candidate_asset_scene_key,
        target_prim_paths=candidate_asset_prim_paths,
        num_envs=num_envs,
    )

    reports = []
    for values in materialized:
        (
            side,
            binding,
            sensor_prim_path,
            resolved_sensor_prim_paths,
            body_names,
            filter_prim_paths,
            num_bodies,
            filter_count,
            force_matrix,
        ) = values
        receipt = mapping_receipts[binding.sensor_scene_key]
        reports.append(
            RuntimeContactSensorReport(
                schema_version=RUNTIME_CONTACT_SENSOR_REPORT_SCHEMA_VERSION,
                side=side,
                sensor_scene_key=binding.sensor_scene_key,
                sensor_prim_path_expression=sensor_prim_path,
                resolved_sensor_prim_paths=resolved_sensor_prim_paths,
                resolved_sensor_body_names=body_names,
                configured_filter_prim_path_expressions=filter_prim_paths,
                candidate_filter_asset_scene_key=candidate_asset_scene_key,
                candidate_filter_asset_prim_path_expression=candidate_asset_prim_path,
                candidate_filter_asset_prim_paths=candidate_asset_prim_paths,
                proven_filter_prim_paths=receipt.target_prim_paths,
                proven_filter_body_name=binding.filtered_body_name,
                filter_mapping_proof_schema_version=receipt.schema_version,
                filter_mapping_proof_source="controlled_three_phase_executor",
                filter_mapping_proof_digest=receipt.receipt_digest,
                num_bodies=num_bodies,
                filter_count=filter_count,
                force_matrix_shape=tuple(force_matrix.shape),
                force_matrix_dtype=str(force_matrix.dtype),
                force_matrix_device=str(force_matrix.device),
                force_matrix_finite=True,
            )
        )
    return tuple(reports)


def _resolve_terminal_contexts(
    env: object,
    supplied: object | None,
    *,
    num_envs: int,
) -> tuple[DriverTerminalContext, ...]:
    raw = supplied
    if raw is None:
        raw = getattr(env, "recovery_terminal_contexts", None)
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes, bytearray))
        or len(raw) != num_envs
    ):
        raise RecoveryTelemetryIncompleteError(("authoritative_terminal_context",))

    try:
        contexts = tuple(DriverTerminalContext.from_value(value) for value in raw)
        for context in contexts:
            context.validate()
    except (TypeError, ValueError) as exc:
        raise RecoveryTelemetryIncompleteError(
            ("authoritative_terminal_context",)
        ) from exc
    return contexts


def _resolve_telemetry_thresholds(env: object) -> tuple[float, float]:
    raw = getattr(getattr(env, "cfg", None), "recovery_telemetry_thresholds", None)
    if not isinstance(raw, Mapping) or not {
        "contact_force_n",
        "max_ee_box_distance_m",
    }.issubset(raw):
        raise RecoveryTelemetryIncompleteError(("recovery_telemetry_thresholds",))
    try:
        contact_threshold = float(raw["contact_force_n"])
        ee_distance_threshold = float(raw["max_ee_box_distance_m"])
    except (TypeError, ValueError) as exc:
        raise RecoveryTelemetryIncompleteError(
            ("recovery_telemetry_thresholds",)
        ) from exc
    if (
        not math.isfinite(contact_threshold)
        or contact_threshold <= 0.0
        or not math.isfinite(ee_distance_threshold)
        or ee_distance_threshold <= 0.0
    ):
        raise RecoveryTelemetryIncompleteError(("recovery_telemetry_thresholds",))
    return contact_threshold, ee_distance_threshold


def _box_state_tensors(
    box: object,
    *,
    num_envs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = getattr(box, "data", None)
    root_state = getattr(data, "root_state_w", None)
    center = getattr(data, "root_com_pos_w", None)
    if center is None:
        center = getattr(data, "root_pos_w", None)
    if center is None and isinstance(root_state, torch.Tensor):
        center = root_state[:, 0:3]
    linear_velocity = getattr(data, "root_lin_vel_w", None)
    angular_velocity = getattr(data, "root_ang_vel_w", None)
    if (
        isinstance(root_state, torch.Tensor)
        and root_state.ndim == 2
        and root_state.shape[1] >= 13
    ):
        if linear_velocity is None:
            linear_velocity = root_state[:, 7:10]
        if angular_velocity is None:
            angular_velocity = root_state[:, 10:13]
    tensors = (center, linear_velocity, angular_velocity)
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape != (num_envs, 3)
        for value in tensors
    ):
        raise RecoveryTelemetryIncompleteError(("box_dynamics",))
    return tensors  # type: ignore[return-value]


def _body_pose_tensor(
    robot: object,
    body_name: str,
    *,
    num_envs: int,
) -> torch.Tensor:
    data = getattr(robot, "data", None)
    body_names = list(getattr(data, "body_names", ()) or ())
    matches = [index for index, name in enumerate(body_names) if str(name) == body_name]
    if len(matches) != 1:
        raise RecoveryTelemetryIncompleteError((f"ee_body:{body_name}",))
    body_index = matches[0]

    body_state = getattr(data, "body_state_w", None)
    if (
        isinstance(body_state, torch.Tensor)
        and body_state.ndim == 3
        and body_state.shape[0] == num_envs
        and body_index < body_state.shape[1]
        and body_state.shape[2] >= 7
    ):
        return body_state[:, body_index, 0:7]

    body_pos = getattr(data, "body_pos_w", None)
    body_quat = getattr(data, "body_quat_w", None)
    if (
        not isinstance(body_pos, torch.Tensor)
        or not isinstance(body_quat, torch.Tensor)
        or body_pos.shape != (num_envs, len(body_names), 3)
        or body_quat.shape != (num_envs, len(body_names), 4)
    ):
        raise RecoveryTelemetryIncompleteError((f"ee_pose:{body_name}",))
    return torch.cat((body_pos[:, body_index, :], body_quat[:, body_index, :]), dim=-1)


def _batched_pairwise_force_rows(
    scene: object,
    *,
    bindings: Mapping[str, HandContactBinding],
    num_envs: int,
    reports_by_scene_key: Mapping[str, RuntimeContactSensorReport],
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    force_rows = []
    for side in ("left", "right"):
        for binding in bindings[side].sensors:
            sensor = _scene_get(scene, binding.sensor_scene_key)
            force_matrix = getattr(
                getattr(sensor, "data", None), "force_matrix_w", None
            )
            report = reports_by_scene_key[binding.sensor_scene_key]
            if (
                not isinstance(force_matrix, torch.Tensor)
                or force_matrix.shape != (num_envs, 1, 1, 3)
                or not force_matrix.is_floating_point()
                or getattr(sensor, "num_bodies", None) != 1
                or tuple(getattr(sensor, "body_names", ()) or ())
                != (binding.sensor_body_name,)
                or report.sensor_scene_key != binding.sensor_scene_key
                or report.resolved_sensor_body_names != (binding.sensor_body_name,)
                or report.proven_filter_body_name != binding.filtered_body_name
                or not report.proven_filter_prim_paths
                or report.filter_mapping_proof_source
                != "controlled_three_phase_executor"
                or str(force_matrix.dtype) != report.force_matrix_dtype
                or str(force_matrix.device) != report.force_matrix_device
            ):
                raise RecoveryTelemetryIncompleteError(
                    (f"{side}_box_pairwise_contact",)
                )
            force_rows.append(force_matrix[:, 0, 0, :])
    materialized = torch.stack(force_rows, dim=1).detach().cpu()
    if not bool(torch.isfinite(materialized).all().item()):
        raise RecoveryTelemetryIncompleteError(
            ("left_box_pairwise_contact", "right_box_pairwise_contact")
        )
    return tuple(
        tuple((float(force[0]), float(force[1]), float(force[2])) for force in env_row)
        for env_row in materialized.tolist()
    )


def _root_quaternion_tensor(robot: object, *, num_envs: int) -> torch.Tensor:
    root_state = getattr(getattr(robot, "data", None), "root_state_w", None)
    if (
        not isinstance(root_state, torch.Tensor)
        or root_state.ndim != 2
        or root_state.shape[0] != num_envs
        or root_state.shape[1] < 7
    ):
        raise RecoveryTelemetryIncompleteError(("root_orientation",))
    return root_state[:, 3:7]


def _critical_body_contact_by_env(
    scene: object,
    robot: object,
    *,
    num_envs: int,
    critical_body_indices: tuple[int, ...],
    critical_body_names: tuple[str, ...],
    force_threshold_n: float,
) -> list[bool | None]:
    data = getattr(robot, "data", None)
    body_names = list(getattr(data, "body_names", ()) or ())
    if (
        not critical_body_indices
        or len(critical_body_indices) != len(critical_body_names)
        or any(index < 0 or index >= len(body_names) for index in critical_body_indices)
        or tuple(str(body_names[index]) for index in critical_body_indices)
        != critical_body_names
    ):
        raise RecoveryTelemetryIncompleteError(("critical_body_identity",))
    threshold = float(force_threshold_n)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise RecoveryTelemetryIncompleteError(("fall_detector_config",))

    direct_forces = getattr(data, "body_net_contact_force_w", None)
    if direct_forces is None:
        direct_forces = getattr(data, "body_net_contact_forces_w", None)
    if direct_forces is not None:
        if not isinstance(direct_forces, torch.Tensor) or direct_forces.shape != (
            num_envs,
            len(body_names),
            3,
        ):
            return [None] * num_envs
        contact_forces = direct_forces[:, critical_body_indices, :]
    else:
        contact_sensor = _scene_get(scene, "contact_forces")
        sensor_body_names = list(getattr(contact_sensor, "body_names", ()) or ())
        sensor_forces = getattr(
            getattr(contact_sensor, "data", None), "net_forces_w", None
        )
        if (
            not isinstance(sensor_forces, torch.Tensor)
            or sensor_forces.shape != (num_envs, len(sensor_body_names), 3)
            or len(sensor_body_names) != len(set(sensor_body_names))
            or getattr(contact_sensor, "num_bodies", None) != len(sensor_body_names)
        ):
            return [None] * num_envs
        sensor_name_to_index = {
            str(name): index for index, name in enumerate(sensor_body_names)
        }
        critical_sensor_indices = [
            sensor_name_to_index.get(str(body_names[index]))
            for index in critical_body_indices
        ]
        if any(index is None for index in critical_sensor_indices):
            return [None] * num_envs
        contact_forces = sensor_forces[:, critical_sensor_indices, :]

    finite_by_env = torch.isfinite(contact_forces).all(dim=-1).all(dim=-1)
    contact_by_env = (
        torch.linalg.vector_norm(contact_forces, dim=-1) >= threshold
    ).any(dim=1)
    materialized = (
        torch.stack((finite_by_env, contact_by_env), dim=1).detach().cpu().tolist()
    )
    if any(not bool(finite) for finite, _contact in materialized):
        raise RecoveryTelemetryIncompleteError(("critical_body_contact",))
    return [bool(contact) for _finite, contact in materialized]


def _normalize_evaluator_fall_detector(
    env: object,
    value: object,
) -> EvaluatorFallDetectorConfig:
    if value is None:
        return EvaluatorFallDetectorConfig(
            enabled=False,
            soft_up_alignment=SOFT_FALL_UP_ALIGNMENT,
            hard_up_alignment=HARD_FALL_UP_ALIGNMENT,
            contact_force_threshold_n=CRITICAL_BODY_CONTACT_THRESHOLD_N,
            confirm_steps=5,
            critical_body_indices=(),
            critical_body_names=(),
        )
    if not isinstance(value, Mapping):
        raise RecoveryTelemetryIncompleteError(("fall_detector_config",))
    try:
        soft = float(value["soft_up_alignment"])
        hard = float(value["hard_up_alignment"])
        force_threshold = float(value["contact_force_threshold"])
        confirm_steps = value["confirm_steps"]
        raw_indices = value["critical_body_indices"]
        raw_names = value["critical_body_names"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryTelemetryIncompleteError(("fall_detector_config",)) from exc
    if (
        type(confirm_steps) is not int
        or confirm_steps <= 0
        or not math.isfinite(soft)
        or not math.isfinite(hard)
        or not -1.0 <= hard < soft <= 1.0
        or not math.isfinite(force_threshold)
        or force_threshold <= 0.0
        or not isinstance(raw_indices, Sequence)
        or isinstance(raw_indices, (str, bytes, bytearray))
        or not isinstance(raw_names, Sequence)
        or isinstance(raw_names, (str, bytes, bytearray))
    ):
        raise RecoveryTelemetryIncompleteError(("fall_detector_config",))
    indices = tuple(raw_indices)
    names = tuple(str(name) for name in raw_names)
    if (
        not indices
        or len(indices) != len(names)
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
        or len(set(names)) != len(names)
    ):
        raise RecoveryTelemetryIncompleteError(("fall_detector_config",))

    robot = _scene_get(getattr(env, "scene", None), "robot")
    body_names = tuple(
        str(name) for name in getattr(getattr(robot, "data", None), "body_names", ())
    )
    expected_indices = []
    expected_names = []
    for index, name in enumerate(body_names):
        lowered = name.lower()
        if any(token in lowered for token in _FALL_SUPPORTED_BODY_TOKENS):
            continue
        if any(token in lowered for token in _FALL_CRITICAL_BODY_TOKENS):
            expected_indices.append(index)
            expected_names.append(name)
    if indices != tuple(expected_indices) or names != tuple(expected_names):
        raise RecoveryTelemetryIncompleteError(("fall_detector_body_identity",))
    return EvaluatorFallDetectorConfig(
        enabled=True,
        soft_up_alignment=soft,
        hard_up_alignment=hard,
        contact_force_threshold_n=force_threshold,
        confirm_steps=confirm_steps,
        critical_body_indices=indices,
        critical_body_names=names,
    )


def _fall_detector_config_digest(config: EvaluatorFallDetectorConfig) -> str:
    return _canonical_json_digest(
        {
            "enabled": config.enabled,
            "soft_up_alignment": config.soft_up_alignment,
            "hard_up_alignment": config.hard_up_alignment,
            "contact_force_threshold_n": config.contact_force_threshold_n,
            "confirm_steps": config.confirm_steps,
            "critical_body_indices": config.critical_body_indices,
            "critical_body_names": config.critical_body_names,
        }
    )


def live_fall_evidence_digest(evidence: LiveFallProducerEvidence) -> str:
    """Recompute the canonical digest over all live root/contact observations."""

    if type(evidence) is not LiveFallProducerEvidence:
        raise TypeError("live fall digest requires LiveFallProducerEvidence")
    payload = {
        "schema_version": evidence.schema_version,
        "task_identity": evidence.task_identity,
        "runtime_identity_digest": evidence.runtime_identity_digest,
        "detector_config_digest": evidence.detector_config_digest,
        "lanes": [
            {
                "env_index": lane.env_index,
                "control_step_count": lane.control_step_count,
                "root_quat_wxyz": lane.root_quat_wxyz,
                "root_up_alignment": lane.root_up_alignment,
                "critical_body_contact": lane.critical_body_contact,
                "fall_candidate": lane.fall_candidate,
                "detector_enabled": lane.detector_enabled,
                "soft_up_alignment": lane.soft_up_alignment,
                "hard_up_alignment": lane.hard_up_alignment,
            }
            for lane in evidence.lanes
        ],
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def _produce_live_fall_evidence(
    env: object,
    *,
    control_step_counts: Sequence[int],
    detector_config: EvaluatorFallDetectorConfig,
) -> LiveFallProducerEvidence:
    """Read the sole live root/contact fall source for one primitive step."""

    _require_pp_box_task_identity(env)
    num_envs = int(getattr(env, "num_envs", 0))
    scene = getattr(env, "scene", None)
    if num_envs <= 0 or scene is None:
        raise RecoveryTelemetryIncompleteError(("scene",))
    if (
        not isinstance(control_step_counts, Sequence)
        or isinstance(control_step_counts, (str, bytes, bytearray))
        or len(control_step_counts) != num_envs
        or any(type(step) is not int or step < 0 for step in control_step_counts)
    ):
        raise RecoveryTelemetryIncompleteError(("live_fall_control_step",))
    bindings = _resolve_hand_contact_bindings(env)
    runtime_identity_digest = _runtime_contact_identities(env, bindings)[0]
    robot = _scene_get(scene, "robot")
    if robot is None:
        raise RecoveryTelemetryIncompleteError(("robot_state",))
    root_quaternions = _root_quaternion_tensor(robot, num_envs=num_envs)
    if detector_config.enabled:
        critical_contacts = _critical_body_contact_by_env(
            scene,
            robot,
            num_envs=num_envs,
            critical_body_indices=detector_config.critical_body_indices,
            critical_body_names=detector_config.critical_body_names,
            force_threshold_n=detector_config.contact_force_threshold_n,
        )
    else:
        critical_contacts = [None] * num_envs
    lanes = []
    for env_index in range(num_envs):
        root_quaternion = _as_float_tuple(
            root_quaternions[env_index],
            4,
            name="live fall root quaternion",
        )
        root_up_alignment = compute_root_up_alignment(root_quaternion)
        critical_contact = critical_contacts[env_index]
        lanes.append(
            LiveFallLaneEvidence(
                env_index=env_index,
                control_step_count=control_step_counts[env_index],
                root_quat_wxyz=(
                    root_quaternion[0],
                    root_quaternion[1],
                    root_quaternion[2],
                    root_quaternion[3],
                ),
                root_up_alignment=root_up_alignment,
                critical_body_contact=critical_contact,
                fall_candidate=classify_fall(
                    root_up_alignment,
                    critical_body_contact=critical_contact,
                    detector_enabled=detector_config.enabled,
                    soft_up_alignment=detector_config.soft_up_alignment,
                    hard_up_alignment=detector_config.hard_up_alignment,
                ),
                detector_enabled=detector_config.enabled,
                soft_up_alignment=detector_config.soft_up_alignment,
                hard_up_alignment=detector_config.hard_up_alignment,
            )
        )
    evidence = object.__new__(LiveFallProducerEvidence)
    values = {
        "schema_version": LIVE_FALL_EVIDENCE_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "runtime_identity_digest": runtime_identity_digest,
        "detector_config_digest": _fall_detector_config_digest(detector_config),
        "lanes": tuple(lanes),
        "_producer_token": _LIVE_FALL_PRODUCER_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(evidence, name, value)
    object.__setattr__(
        evidence,
        "evidence_digest",
        live_fall_evidence_digest(evidence),
    )
    return evidence


def evaluator_terminal_evidence_digest(evidence: EvaluatorTerminalEvidence) -> str:
    """Recompute the canonical evaluator-step evidence digest."""

    if type(evidence) is not EvaluatorTerminalEvidence:
        raise TypeError("terminal digest requires EvaluatorTerminalEvidence")
    return _canonical_json_digest(
        {
            "schema_version": evidence.schema_version,
            "task_identity": evidence.task_identity,
            "runtime_identity_digest": evidence.runtime_identity_digest,
            "detector_config_digest": evidence.detector_config_digest,
            "step_idx": evidence.step_idx,
            "max_steps": evidence.max_steps,
            "previous_evidence_digest": evidence.previous_evidence_digest,
            "contexts": [
                {
                    "control_step_count": context.control_step_count,
                    "max_control_steps": context.max_control_steps,
                    "fall_streak": context.fall_streak,
                    "fall_confirm_steps": context.fall_confirm_steps,
                    "time_limit": context.time_limit,
                    "fall_confirmed": context.fall_confirmed,
                }
                for context in evidence.contexts
            ],
            "live_fall_evidence_digest": evidence.live_fall_evidence.evidence_digest,
        }
    )


def _registered_evaluator_terminal_evidence(
    env: object,
) -> EvaluatorTerminalEvidence | None:
    registered = _EVALUATOR_TERMINAL_EVIDENCE.get(id(env))
    if registered is None or registered[0] is not env:
        return None
    evidence = registered[1]
    if type(evidence) is not EvaluatorTerminalEvidence:
        return None
    return evidence


def produce_evaluator_terminal_evidence(
    env: object,
    *,
    step_idx: int,
    max_steps: int,
    fall_detector: object,
) -> EvaluatorTerminalEvidence:
    """Produce the sole fall/timeout context from the evaluator's live cadence."""

    _require_pp_box_task_identity(env)
    if (
        type(step_idx) is not int
        or step_idx <= 0
        or type(max_steps) is not int
        or max_steps <= 0
    ):
        raise RecoveryTelemetryIncompleteError(("evaluator_step_context",))
    detector_config = _normalize_evaluator_fall_detector(env, fall_detector)
    detector_digest = _fall_detector_config_digest(detector_config)
    previous = _registered_evaluator_terminal_evidence(env)
    if step_idx == 1:
        previous = None
    elif (
        previous is None
        or previous._producer_token is not _EVALUATOR_TERMINAL_EVIDENCE_TOKEN
        or previous.step_idx != step_idx - 1
        or previous.max_steps != max_steps
        or previous.detector_config_digest != detector_digest
        or previous.evidence_digest != evaluator_terminal_evidence_digest(previous)
    ):
        raise RecoveryTelemetryIncompleteError(("evaluator_terminal_sequence",))

    num_envs = int(getattr(env, "num_envs", 0))
    live_evidence = _produce_live_fall_evidence(
        env,
        control_step_counts=(step_idx,) * num_envs,
        detector_config=detector_config,
    )
    contexts = []
    for env_index, lane in enumerate(live_evidence.lanes):
        previous_streak = (
            0 if previous is None else previous.contexts[env_index].fall_streak
        )
        fall_streak = previous_streak + 1 if lane.fall_candidate else 0
        contexts.append(
            DriverTerminalContext(
                control_step_count=step_idx,
                max_control_steps=max_steps,
                fall_streak=fall_streak,
                fall_confirm_steps=detector_config.confirm_steps,
                time_limit=step_idx >= max_steps,
                fall_confirmed=(
                    detector_config.enabled
                    and fall_streak >= detector_config.confirm_steps
                ),
            )
        )

    evidence = object.__new__(EvaluatorTerminalEvidence)
    values = {
        "schema_version": EVALUATOR_TERMINAL_EVIDENCE_SCHEMA_VERSION,
        "task_identity": PP_BOX_TASK_IDENTITY,
        "runtime_identity_digest": live_evidence.runtime_identity_digest,
        "detector_config": detector_config,
        "detector_config_digest": detector_digest,
        "step_idx": step_idx,
        "max_steps": max_steps,
        "previous_evidence_digest": ""
        if previous is None
        else previous.evidence_digest,
        "contexts": tuple(contexts),
        "live_fall_evidence": live_evidence,
        "_producer_token": _EVALUATOR_TERMINAL_EVIDENCE_TOKEN,
    }
    for name, value in values.items():
        object.__setattr__(evidence, name, value)
    object.__setattr__(
        evidence,
        "evidence_digest",
        evaluator_terminal_evidence_digest(evidence),
    )
    _EVALUATOR_TERMINAL_EVIDENCE[id(env)] = (env, evidence)
    return evidence


def _validate_live_fall_evidence(
    env: object,
    evidence: object,
    terminal_contexts: tuple[DriverTerminalContext, ...],
    *,
    num_envs: int,
) -> tuple[LiveFallLaneEvidence, ...]:
    bindings = _resolve_hand_contact_bindings(env)
    runtime_identity_digest = _runtime_contact_identities(env, bindings)[0]
    if (
        type(evidence) is not LiveFallProducerEvidence
        or evidence.schema_version != LIVE_FALL_EVIDENCE_SCHEMA_VERSION
        or evidence.task_identity != PP_BOX_TASK_IDENTITY
        or evidence.runtime_identity_digest != runtime_identity_digest
        or evidence._producer_token is not _LIVE_FALL_PRODUCER_TOKEN
        or not isinstance(evidence.lanes, tuple)
        or len(evidence.lanes) != num_envs
    ):
        raise RecoveryTelemetryIncompleteError(("live_fall_evidence",))
    try:
        expected_digest = live_fall_evidence_digest(evidence)
    except (TypeError, ValueError) as exc:
        raise RecoveryTelemetryIncompleteError(("live_fall_evidence",)) from exc
    if evidence.evidence_digest != expected_digest:
        raise RecoveryTelemetryIncompleteError(("live_fall_evidence",))
    lanes = []
    for env_index, context in enumerate(terminal_contexts):
        lane = _validate_live_fall_lane(
            evidence.lanes[env_index],
            env_index=env_index,
            control_step_count=context.control_step_count,
        )
        if lane.fall_candidate != (context.fall_streak > 0):
            raise RecoveryTelemetryIncompleteError(
                ("authoritative_terminal_context", "live_fall_evidence")
            )
        lanes.append(lane)
    return tuple(lanes)


def _validate_evaluator_terminal_evidence(
    env: object,
    evidence: object,
    *,
    num_envs: int,
) -> tuple[tuple[DriverTerminalContext, ...], tuple[LiveFallLaneEvidence, ...]]:
    registered = _registered_evaluator_terminal_evidence(env)
    if (
        type(evidence) is not EvaluatorTerminalEvidence
        or registered is not evidence
        or evidence._producer_token is not _EVALUATOR_TERMINAL_EVIDENCE_TOKEN
        or evidence.schema_version != EVALUATOR_TERMINAL_EVIDENCE_SCHEMA_VERSION
        or evidence.task_identity != PP_BOX_TASK_IDENTITY
        or type(evidence.step_idx) is not int
        or evidence.step_idx <= 0
        or type(evidence.max_steps) is not int
        or evidence.max_steps <= 0
        or not isinstance(evidence.contexts, tuple)
        or len(evidence.contexts) != num_envs
        or evidence.evidence_digest != evaluator_terminal_evidence_digest(evidence)
        or evidence.detector_config_digest
        != _fall_detector_config_digest(evidence.detector_config)
        or evidence.live_fall_evidence.detector_config_digest
        != evidence.detector_config_digest
        or evidence.runtime_identity_digest
        != evidence.live_fall_evidence.runtime_identity_digest
    ):
        raise RecoveryTelemetryIncompleteError(("evaluator_terminal_evidence",))
    contexts = tuple(evidence.contexts)
    try:
        for context in contexts:
            context.validate()
            if (
                context.control_step_count != evidence.step_idx
                or context.max_control_steps != evidence.max_steps
                or context.fall_confirm_steps != evidence.detector_config.confirm_steps
            ):
                raise ValueError("terminal context does not match evaluator evidence")
    except (TypeError, ValueError) as exc:
        raise RecoveryTelemetryIncompleteError(
            ("evaluator_terminal_evidence",)
        ) from exc
    lanes = _validate_live_fall_evidence(
        env,
        evidence.live_fall_evidence,
        contexts,
        num_envs=num_envs,
    )
    return contexts, lanes


def extract_privileged_telemetry(
    env: object,
    *,
    support_resolver: Any | None = None,
    terminal_evidence: EvaluatorTerminalEvidence | None = None,
    actor_observation: object | None = None,
) -> tuple[PrivilegedRecoveryTelemetry, ...]:
    """Extract privileged state without registering it as a policy observation."""

    task_identity = _require_pp_box_task_identity(env)
    if actor_observation is None:
        raise RecoveryTelemetryIncompleteError(("actor_observation",))
    bindings = _resolve_hand_contact_bindings(env)
    num_envs = int(getattr(env, "num_envs", 0))
    scene = getattr(env, "scene", None)
    if num_envs <= 0 or scene is None:
        raise RecoveryTelemetryIncompleteError(("scene",))
    try:
        contact_reports = validate_runtime_hand_contact_sensors(env)
    except RecoveryTelemetryIncompleteError as exc:
        extraction_capabilities = []
        for capability in exc.missing_capabilities:
            if ":left_" in capability:
                extraction_capabilities.append("left_box_pairwise_contact")
            elif ":right_" in capability:
                extraction_capabilities.append("right_box_pairwise_contact")
            else:
                extraction_capabilities.append(capability)
        raise RecoveryTelemetryIncompleteError(extraction_capabilities) from exc
    reports_by_scene_key = {
        report.sensor_scene_key: report for report in contact_reports
    }
    expected_contact_scene_keys = {
        sensor.sensor_scene_key
        for side in ("left", "right")
        for sensor in bindings[side].sensors
    }
    if (
        len(contact_reports) != 16
        or len(reports_by_scene_key) != 16
        or set(reports_by_scene_key) != expected_contact_scene_keys
    ):
        raise RecoveryTelemetryIncompleteError(("runtime_contact_sensor_set",))
    resolved_terminal_contexts, fall_evidence_by_env = (
        _validate_evaluator_terminal_evidence(
            env,
            terminal_evidence,
            num_envs=num_envs,
        )
    )
    contact_threshold, ee_distance_threshold = _resolve_telemetry_thresholds(env)

    box = _scene_get(scene, "box")
    if box is None:
        box = _scene_get(scene, "Box")
    robot = _scene_get(scene, "robot")
    if box is None or robot is None:
        missing = []
        if box is None:
            missing.append("box_dynamics")
        if robot is None:
            missing.append("robot_state")
        raise RecoveryTelemetryIncompleteError(missing)

    box_centers, box_linear_velocity, box_angular_velocity = _box_state_tensors(
        box,
        num_envs=num_envs,
    )
    left_poses = _body_pose_tensor(
        robot,
        bindings["left"].ee_body_name,
        num_envs=num_envs,
    )
    right_poses = _body_pose_tensor(
        robot,
        bindings["right"].ee_body_name,
        num_envs=num_envs,
    )
    force_rows = _batched_pairwise_force_rows(
        scene,
        bindings=bindings,
        num_envs=num_envs,
        reports_by_scene_key=reports_by_scene_key,
    )
    if support_resolver is None:
        try:
            from .rewards import compute_box_support_evidence
        except ImportError as exc:
            raise RecoveryTelemetryIncompleteError(("shelf_support_geometry",)) from exc
        support_resolver = compute_box_support_evidence
    support_evidence = support_resolver(env)
    if not isinstance(support_evidence, Sequence) or len(support_evidence) != num_envs:
        raise RecoveryTelemetryIncompleteError(("shelf_support_geometry",))

    states = []
    for env_index in range(num_envs):
        hand_contacts = {}
        sensor_offset = 0
        for side in ("left", "right"):
            side_bindings = bindings[side].sensors
            link_evidence = tuple(
                pairwise_contact_evidence(
                    force_rows[env_index][sensor_offset + sensor_index],
                    sensor_body=sensor_binding.sensor_body_name,
                    filtered_body=sensor_binding.filtered_body_name,
                    threshold_n=contact_threshold,
                )
                for sensor_index, sensor_binding in enumerate(
                    side_bindings,
                )
            )
            hand_contacts[side] = aggregate_hand_contact_evidence(side, link_evidence)
            sensor_offset += len(side_bindings)
        states.append(
            build_privileged_telemetry(
                task_identity=task_identity,
                env_index=env_index,
                box_center_w=box_centers[env_index],
                box_linear_velocity_w=box_linear_velocity[env_index],
                box_angular_velocity_w=box_angular_velocity[env_index],
                support=support_evidence[env_index],
                left_ee_pose_w=left_poses[env_index],
                right_ee_pose_w=right_poses[env_index],
                left_contact=hand_contacts["left"],
                right_contact=hand_contacts["right"],
                live_fall_evidence=fall_evidence_by_env[env_index],
                terminal_context=resolved_terminal_contexts[env_index],
                max_ee_box_distance_m=ee_distance_threshold,
            )
        )
    result = tuple(states)
    for state in result:
        assert_actor_observation_isolated(actor_observation, state, env=env)
    return result


__all__ = [
    "DEFAULT_CONTACT_FORCE_THRESHOLD_N",
    "DEFAULT_MAX_EE_BOX_DISTANCE_M",
    "EMPIRICAL_CONTACT_QUIET_MAX_N",
    "EMPIRICAL_CONTACT_TOUCH_MIN_N",
    "EVALUATOR_TERMINAL_EVIDENCE_SCHEMA_VERSION",
    "HARD_FALL_UP_ALIGNMENT",
    "LIVE_FALL_EVIDENCE_SCHEMA_VERSION",
    "PP_BOX_TASK_IDENTITY",
    "RECOVERY_TELEMETRY_SCHEMA_VERSION",
    "RESIDUAL_ACTOR_OBSERVATION_SCHEMA_VERSION",
    "RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION",
    "RUNTIME_CONTACT_SENSOR_REPORT_SCHEMA_VERSION",
    "SOFT_FALL_UP_ALIGNMENT",
    "BimanualGraspEvidence",
    "ContactCalibrationExecutionReceipt",
    "ContactCalibrationPhaseReceipt",
    "ContactSensorCalibrationReceipt",
    "DriverTerminalContext",
    "EvaluatorFallDetectorConfig",
    "EvaluatorTerminalEvidence",
    "HandContactBinding",
    "HandContactEvidence",
    "LiveFallLaneEvidence",
    "LiveFallProducerEvidence",
    "PairwiseContactBinding",
    "PairwiseContactEvidence",
    "PrivilegedObservationLeakError",
    "PrivilegedRecoveryTelemetry",
    "RecoveryTelemetryIncompleteError",
    "ResidualActorObservation",
    "RuntimeContactSensorReport",
    "aggregate_hand_contact_evidence",
    "assert_actor_observation_isolated",
    "build_privileged_telemetry",
    "classify_bimanual_grasp",
    "classify_fall",
    "classify_terminal",
    "compute_root_up_alignment",
    "default_hand_contact_bindings",
    "execute_pp_box_contact_calibration",
    "extract_privileged_telemetry",
    "has_pairwise_bimanual_contact",
    "install_pp_box_contact_calibration_executor",
    "issue_residual_actor_observation",
    "live_fall_evidence_digest",
    "pairwise_contact_evidence",
    "produce_evaluator_terminal_evidence",
    "validate_runtime_hand_contact_sensors",
]
