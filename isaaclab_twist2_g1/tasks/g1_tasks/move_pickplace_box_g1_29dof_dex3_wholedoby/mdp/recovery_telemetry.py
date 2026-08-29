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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Literal

import torch

RECOVERY_TELEMETRY_SCHEMA_VERSION = 1
RUNTIME_CONTACT_MAPPING_PROOF_SCHEMA_VERSION = 1
RUNTIME_CONTACT_SENSOR_REPORT_SCHEMA_VERSION = 2
PP_BOX_TASK_IDENTITY = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"

DEFAULT_CONTACT_FORCE_THRESHOLD_N = 1.0
DEFAULT_MAX_EE_BOX_DISTANCE_M = 0.3
SOFT_FALL_UP_ALIGNMENT = math.cos(math.radians(60.0))
HARD_FALL_UP_ALIGNMENT = math.cos(math.radians(75.0))
CRITICAL_BODY_CONTACT_THRESHOLD_N = 50.0
EMPIRICAL_CONTACT_QUIET_MAX_N = 0.05
EMPIRICAL_CONTACT_TOUCH_MIN_N = 1.0

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


class RecoveryTelemetryIncompleteError(RuntimeError):
    """Raised when required simulator truth is absent or structurally invalid."""

    def __init__(self, missing_capabilities: Sequence[str]) -> None:
        self.missing_capabilities = tuple(sorted(set(missing_capabilities)))
        detail = ", ".join(self.missing_capabilities) or "unknown"
        super().__init__(f"privileged recovery telemetry is incomplete: {detail}")


class PrivilegedObservationLeakError(RuntimeError):
    """Raised when privileged telemetry appears inside an actor observation."""

    def __init__(self, leak_paths: Sequence[str]) -> None:
        self.leak_paths = tuple(sorted(set(leak_paths)))
        super().__init__("privileged actor-observation leak: " + ", ".join(self.leak_paths))


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
    def from_value(cls, value: object) -> "DriverTerminalContext":
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
                raise ValueError(f"missing terminal context field: {exc.args[0]}") from exc
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
class PairwiseContactBinding:
    """Runtime identity for one-body-to-one-filter pairwise contact evidence."""

    sensor_scene_key: str
    sensor_body_name: str
    filtered_body_name: str

    @classmethod
    def from_value(cls, value: object) -> "PairwiseContactBinding":
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
                raise ValueError(f"missing contact binding field: {exc.args[0]}") from exc
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
    def from_value(cls, value: object) -> "HandContactBinding":
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
                raise ValueError(f"missing hand contact binding field: {exc.args[0]}") from exc
        raise TypeError(f"unsupported hand contact binding type: {type(value).__name__}")

    def validate(self, *, expected_side: Literal["left", "right"] | None = None) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError(f"unknown hand side: {self.side!r}")
        if expected_side is not None and self.side != expected_side:
            raise ValueError(f"expected {expected_side} hand binding, got {self.side}")
        for sensor in self.sensors:
            sensor.validate()
        expected = _default_hand_contact_binding(self.side)
        if self != expected:
            raise ValueError(f"{self.side} hand binding does not match the verified leaf catalog")


@dataclass(frozen=True)
class EmpiricalContactMappingProof:
    """Immutable baseline/touch/removed evidence for one pairwise filter."""

    schema_version: int
    task_identity: str
    sensor_scene_key: str
    sensor_body_name: str
    target_asset_scene_key: str
    target_prim_paths: tuple[str, ...]
    baseline_force_w: tuple[tuple[float, float, float], ...]
    target_touch_force_w: tuple[tuple[float, float, float], ...]
    target_removed_force_w: tuple[tuple[float, float, float], ...]
    baseline_state_digest: str
    target_touch_state_digest: str
    target_removed_state_digest: str
    runtime_identity_digest: str
    proof_digest: str

    @classmethod
    def from_value(cls, value: object) -> EmpiricalContactMappingProof:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"unsupported contact mapping proof: {type(value).__name__}")
        try:
            return cls(
                schema_version=value["schema_version"],
                task_identity=str(value["task_identity"]),
                sensor_scene_key=str(value["sensor_scene_key"]),
                sensor_body_name=str(value["sensor_body_name"]),
                target_asset_scene_key=str(value["target_asset_scene_key"]),
                target_prim_paths=tuple(str(path) for path in value["target_prim_paths"]),
                baseline_force_w=_empirical_force_rows(
                    value["baseline_force_w"],
                    name="baseline force",
                ),
                target_touch_force_w=_empirical_force_rows(
                    value["target_touch_force_w"],
                    name="target-touch force",
                ),
                target_removed_force_w=_empirical_force_rows(
                    value["target_removed_force_w"],
                    name="target-removed force",
                ),
                baseline_state_digest=str(value["baseline_state_digest"]),
                target_touch_state_digest=str(value["target_touch_state_digest"]),
                target_removed_state_digest=str(value["target_removed_state_digest"]),
                runtime_identity_digest=str(value["runtime_identity_digest"]),
                proof_digest=str(value["proof_digest"]),
            )
        except KeyError as exc:
            raise ValueError(f"missing contact mapping proof field: {exc.args[0]}") from exc


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

    return {
        side: _default_hand_contact_binding(side)
        for side in ("left", "right")
    }


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
    z_mismatch_m: float
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
        raise ValueError(f"{name} must be a length-{length} vector")
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    result = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _empirical_force_rows(
    value: object,
    *,
    name: str,
) -> tuple[tuple[float, float, float], ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(f"{name} must have shape [num_envs, 3]")
        rows = value.detach().cpu()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = value
    else:
        raise TypeError(f"{name} must have shape [num_envs, 3]")
    result = tuple(
        _as_float_tuple(row, 3, name=f"{name} row")
        for row in rows
    )
    if not result:
        raise ValueError(f"{name} must contain at least one environment")
    return tuple((row[0], row[1], row[2]) for row in result)


def empirical_contact_mapping_proof_digest(
    proof: EmpiricalContactMappingProof | Mapping[str, object],
) -> str:
    """Recompute the typed proof digest, excluding only the digest field itself."""

    value = EmpiricalContactMappingProof.from_value(proof)
    payload = {
        "schema_version": value.schema_version,
        "task_identity": value.task_identity,
        "sensor_scene_key": value.sensor_scene_key,
        "sensor_body_name": value.sensor_body_name,
        "target_asset_scene_key": value.target_asset_scene_key,
        "target_prim_paths": value.target_prim_paths,
        "baseline_force_w": value.baseline_force_w,
        "target_touch_force_w": value.target_touch_force_w,
        "target_removed_force_w": value.target_removed_force_w,
        "baseline_state_digest": value.baseline_state_digest,
        "target_touch_state_digest": value.target_touch_state_digest,
        "target_removed_state_digest": value.target_removed_state_digest,
        "runtime_identity_digest": value.runtime_identity_digest,
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def build_empirical_contact_mapping_proof(
    *,
    sensor_scene_key: str,
    sensor_body_name: str,
    target_asset_scene_key: str,
    target_prim_paths: Sequence[str],
    baseline_force_w: object,
    target_touch_force_w: object,
    target_removed_force_w: object,
    baseline_state_digest: str,
    target_touch_state_digest: str,
    target_removed_state_digest: str,
    runtime_identity_digest: str,
) -> EmpiricalContactMappingProof:
    """Freeze raw simulator measurements into a content-addressed proof payload."""

    proof = EmpiricalContactMappingProof(
        schema_version=RUNTIME_CONTACT_MAPPING_PROOF_SCHEMA_VERSION,
        task_identity=PP_BOX_TASK_IDENTITY,
        sensor_scene_key=str(sensor_scene_key),
        sensor_body_name=str(sensor_body_name),
        target_asset_scene_key=str(target_asset_scene_key),
        target_prim_paths=tuple(str(path) for path in target_prim_paths),
        baseline_force_w=_empirical_force_rows(
            baseline_force_w,
            name="baseline force",
        ),
        target_touch_force_w=_empirical_force_rows(
            target_touch_force_w,
            name="target-touch force",
        ),
        target_removed_force_w=_empirical_force_rows(
            target_removed_force_w,
            name="target-removed force",
        ),
        baseline_state_digest=str(baseline_state_digest),
        target_touch_state_digest=str(target_touch_state_digest),
        target_removed_state_digest=str(target_removed_state_digest),
        runtime_identity_digest=str(runtime_identity_digest),
        proof_digest="",
    )
    return replace(
        proof,
        proof_digest=empirical_contact_mapping_proof_digest(proof),
    )


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
        raise ValueError(f"{side} hand contact evidence contains a body from another side")
    filtered_bodies = {link.filtered_body for link in link_tuple}
    if filtered_bodies != {"Box"}:
        raise ValueError("hand contact evidence must be filtered only to Box")

    resultant_force = tuple(
        sum(link.force_w[axis] for link in link_tuple)
        for axis in range(3)
    )
    contacting_bodies = tuple(link.sensor_body for link in link_tuple if link.in_contact)
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
) -> bool:
    """Mirror the evaluator's instantaneous hard/soft tilt predicate."""

    alignment = float(root_up_alignment)
    if not math.isfinite(alignment) or alignment < -1.0 or alignment > 1.0:
        raise ValueError("root up-alignment must be finite and within [-1, 1]")
    if alignment < HARD_FALL_UP_ALIGNMENT:
        return True
    if alignment >= SOFT_FALL_UP_ALIGNMENT:
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


def build_privileged_telemetry(
    *,
    env_index: int,
    box_center_w: object,
    box_linear_velocity_w: object,
    box_angular_velocity_w: object,
    support: object,
    left_ee_pose_w: object,
    right_ee_pose_w: object,
    left_contact: HandContactEvidence,
    right_contact: HandContactEvidence,
    root_quat_wxyz: object,
    critical_body_contact: bool | None,
    terminal_context: DriverTerminalContext,
    max_ee_box_distance_m: float = DEFAULT_MAX_EE_BOX_DISTANCE_M,
) -> PrivilegedRecoveryTelemetry:
    if not isinstance(left_contact, HandContactEvidence) or left_contact.side != "left":
        raise ValueError("privileged telemetry requires aggregated left hand contact")
    if not isinstance(right_contact, HandContactEvidence) or right_contact.side != "right":
        raise ValueError("privileged telemetry requires aggregated right hand contact")
    if not isinstance(terminal_context, DriverTerminalContext):
        raise ValueError("privileged telemetry requires authoritative driver terminal context")
    terminal_context.validate()
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
        getattr(support, "support_bounds_w"),
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
    root_up_alignment = compute_root_up_alignment(root_quat_wxyz)
    fall_candidate = classify_fall(
        root_up_alignment,
        critical_body_contact=critical_body_contact,
    )
    if fall_candidate != (terminal_context.fall_streak > 0):
        raise RecoveryTelemetryIncompleteError(("authoritative_terminal_context",))
    fall = terminal_context.fall_confirmed
    success = bool(getattr(support, "placed"))
    timeout = terminal_context.time_limit

    return PrivilegedRecoveryTelemetry(
        schema_version=RECOVERY_TELEMETRY_SCHEMA_VERSION,
        task_identity=PP_BOX_TASK_IDENTITY,
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
        support_surface_z_m=float(getattr(support, "support_top_z_m")),
        target_support_surface_z_m=float(getattr(support, "target_support_top_z_m")),
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
        xy_mismatch_m=float(getattr(support, "xy_mismatch_m")),
        z_mismatch_m=float(getattr(support, "z_mismatch_m")),
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
        terminal_reason=classify_terminal(success=success, fall=fall, time_limit=timeout),
    )


def _collect_privileged_object_ids(value: object) -> set[int]:
    object_ids: set[int] = set()
    visited: set[int] = set()

    def visit(item: object) -> None:
        if item is None or isinstance(item, (bool, int, float, complex, str, bytes, bytearray)):
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
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for nested in item:
                visit(nested)

    visit(value)
    return object_ids


def assert_actor_observation_isolated(
    actor_observation: object,
    privileged_state: PrivilegedRecoveryTelemetry,
) -> None:
    """Reject privileged keys or shared privileged objects at any nesting depth."""

    privileged_object_ids = _collect_privileged_object_ids(privileged_state)
    leak_paths: set[str] = set()
    visited: set[int] = set()

    def visit(item: object, path: str) -> None:
        if item is None or isinstance(item, (bool, int, float, complex, str, bytes, bytearray)):
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
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                field_path = f"{path}.{field.name}"
                if field.name in _PRIVILEGED_FIELD_NAMES:
                    leak_paths.add(field_path)
                visit(getattr(item, field.name), field_path)

    visit(actor_observation, "$")
    if leak_paths:
        raise PrivilegedObservationLeakError(tuple(leak_paths))


def _scene_get(scene: object, key: str) -> object | None:
    if isinstance(scene, Mapping):
        return scene.get(key)
    try:
        return scene[key]  # type: ignore[index]
    except Exception:
        return getattr(scene, key, None)


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


def _resolve_contact_mapping_proofs(
    env: object,
    *,
    expected_scene_keys: tuple[str, ...],
) -> tuple[Mapping[str, object], str]:
    raw_proofs = getattr(env, "recovery_contact_mapping_proofs", None)
    expected = set(expected_scene_keys)
    if not isinstance(raw_proofs, Mapping):
        raise RecoveryTelemetryIncompleteError(
            tuple(f"runtime_contact_mapping_proof:{key}" for key in expected_scene_keys)
        )
    provided = {str(key) for key in raw_proofs}
    missing = tuple(
        f"runtime_contact_mapping_proof:{key}"
        for key in expected_scene_keys
        if key not in raw_proofs
    )
    if missing:
        raise RecoveryTelemetryIncompleteError(missing)
    if provided != expected:
        raise RecoveryTelemetryIncompleteError(("runtime_contact_mapping_proofs",))

    runtime_identity_digest = str(
        getattr(env, "recovery_runtime_identity_digest", "")
    )
    if _SHA256_DIGEST.fullmatch(runtime_identity_digest) is None:
        raise RecoveryTelemetryIncompleteError(("runtime_contact_proof_runtime_identity",))
    return raw_proofs, runtime_identity_digest


def _force_magnitude(force: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in force))


def _validated_contact_mapping_proof(
    raw_proof: object,
    *,
    binding: PairwiseContactBinding,
    candidate_asset_scene_key: str,
    candidate_prim_paths: tuple[str, ...],
    runtime_identity_digest: str,
    num_envs: int,
    capability: str,
) -> EmpiricalContactMappingProof:
    try:
        proof = EmpiricalContactMappingProof.from_value(raw_proof)
        force_groups = (
            proof.baseline_force_w,
            proof.target_touch_force_w,
            proof.target_removed_force_w,
        )
        normalized_groups = tuple(
            _empirical_force_rows(group, name="contact mapping proof force")
            for group in force_groups
        )
        state_digests = (
            proof.baseline_state_digest,
            proof.target_touch_state_digest,
            proof.target_removed_state_digest,
        )
        quiet_groups = (proof.baseline_force_w, proof.target_removed_force_w)
        valid = (
            type(proof.schema_version) is int
            and proof.schema_version == RUNTIME_CONTACT_MAPPING_PROOF_SCHEMA_VERSION
            and proof.task_identity == PP_BOX_TASK_IDENTITY
            and proof.sensor_scene_key == binding.sensor_scene_key
            and proof.sensor_body_name == binding.sensor_body_name
            and proof.target_asset_scene_key == candidate_asset_scene_key
            and proof.target_prim_paths == candidate_prim_paths
            and normalized_groups == force_groups
            and all(len(group) == num_envs for group in force_groups)
            and all(_SHA256_DIGEST.fullmatch(value) is not None for value in state_digests)
            and len(set(state_digests)) == len(state_digests)
            and proof.runtime_identity_digest == runtime_identity_digest
            and proof.proof_digest == empirical_contact_mapping_proof_digest(proof)
            and all(
                _force_magnitude(force) <= EMPIRICAL_CONTACT_QUIET_MAX_N
                for group in quiet_groups
                for force in group
            )
            and all(
                _force_magnitude(force) >= EMPIRICAL_CONTACT_TOUCH_MIN_N
                for force in proof.target_touch_force_w
            )
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise RecoveryTelemetryIncompleteError((capability,))
    return proof


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

    expected_scene_keys = tuple(
        binding.sensor_scene_key
        for side in ("left", "right")
        for binding in bindings[side].sensors
    )
    (
        candidate_asset_scene_key,
        _candidate_asset,
        candidate_asset_prim_path,
        candidate_asset_prim_paths,
        candidate_namespaces,
    ) = _resolve_candidate_filter_asset(scene, num_envs=num_envs)
    mapping_proofs, runtime_identity_digest = _resolve_contact_mapping_proofs(
        env,
        expected_scene_keys=expected_scene_keys,
    )

    reports = []
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
            finite = bool(torch.isfinite(force_matrix).all().item())
            if not finite:
                raise RecoveryTelemetryIncompleteError((capability,))
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

            proof_capability = f"runtime_contact_mapping_proof:{binding.sensor_scene_key}"
            proof = _validated_contact_mapping_proof(
                mapping_proofs[binding.sensor_scene_key],
                binding=binding,
                candidate_asset_scene_key=candidate_asset_scene_key,
                candidate_prim_paths=candidate_asset_prim_paths,
                runtime_identity_digest=runtime_identity_digest,
                num_envs=num_envs,
                capability=proof_capability,
            )
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
                    proven_filter_prim_paths=proof.target_prim_paths,
                    proven_filter_body_name=binding.filtered_body_name,
                    filter_mapping_proof_schema_version=proof.schema_version,
                    filter_mapping_proof_source="empirical_baseline_touch_removed",
                    filter_mapping_proof_digest=proof.proof_digest,
                    num_bodies=num_bodies,
                    filter_count=filter_count,
                    force_matrix_shape=tuple(force_matrix.shape),
                    force_matrix_dtype=str(force_matrix.dtype),
                    force_matrix_device=str(force_matrix.device),
                    force_matrix_finite=finite,
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
        raise RecoveryTelemetryIncompleteError(("authoritative_terminal_context",)) from exc
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
        raise RecoveryTelemetryIncompleteError(("recovery_telemetry_thresholds",)) from exc
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
    if isinstance(root_state, torch.Tensor) and root_state.ndim == 2 and root_state.shape[1] >= 13:
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


def _pairwise_force_tensor(
    scene: object,
    binding: PairwiseContactBinding,
    *,
    side: str,
    num_envs: int,
    validated_report: RuntimeContactSensorReport,
) -> torch.Tensor:
    sensor = _scene_get(scene, binding.sensor_scene_key)
    data = getattr(sensor, "data", None)
    force_matrix = getattr(data, "force_matrix_w", None)
    body_names = list(getattr(sensor, "body_names", ()) or ())

    if (
        not isinstance(force_matrix, torch.Tensor)
        or force_matrix.shape != (num_envs, 1, 1, 3)
        or not force_matrix.is_floating_point()
        or not bool(torch.isfinite(force_matrix).all().item())
        or getattr(sensor, "num_bodies", None) != 1
        or body_names != [binding.sensor_body_name]
        or validated_report.sensor_scene_key != binding.sensor_scene_key
        or validated_report.resolved_sensor_body_names != (binding.sensor_body_name,)
        or validated_report.proven_filter_body_name != binding.filtered_body_name
        or not validated_report.proven_filter_prim_paths
        or validated_report.filter_mapping_proof_source
        != "empirical_baseline_touch_removed"
        or str(force_matrix.dtype) != validated_report.force_matrix_dtype
        or str(force_matrix.device) != validated_report.force_matrix_device
    ):
        raise RecoveryTelemetryIncompleteError((f"{side}_box_pairwise_contact",))
    return force_matrix[:, 0, 0, :]


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
) -> list[bool | None]:
    data = getattr(robot, "data", None)
    body_names = list(getattr(data, "body_names", ()) or ())
    indices = []
    for index, name in enumerate(body_names):
        lowered = str(name).lower()
        if any(token in lowered for token in _FALL_SUPPORTED_BODY_TOKENS):
            continue
        if any(token in lowered for token in _FALL_CRITICAL_BODY_TOKENS):
            indices.append(index)

    if not indices:
        return [None] * num_envs

    direct_forces = getattr(data, "body_net_contact_force_w", None)
    if direct_forces is None:
        direct_forces = getattr(data, "body_net_contact_forces_w", None)
    if direct_forces is not None:
        if (
            not isinstance(direct_forces, torch.Tensor)
            or direct_forces.shape != (num_envs, len(body_names), 3)
        ):
            return [None] * num_envs
        contact_forces = direct_forces[:, indices, :]
    else:
        contact_sensor = _scene_get(scene, "contact_forces")
        sensor_body_names = list(getattr(contact_sensor, "body_names", ()) or ())
        sensor_forces = getattr(getattr(contact_sensor, "data", None), "net_forces_w", None)
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
            sensor_name_to_index.get(str(body_names[index])) for index in indices
        ]
        if any(index is None for index in critical_sensor_indices):
            return [None] * num_envs
        contact_forces = sensor_forces[:, critical_sensor_indices, :]

    magnitudes = torch.linalg.vector_norm(contact_forces, dim=-1)
    return [
        bool(value)
        for value in (magnitudes >= CRITICAL_BODY_CONTACT_THRESHOLD_N).any(dim=1).tolist()
    ]


def compute_live_fall_candidates(env: object) -> tuple[bool, ...]:
    """Return instantaneous fall predicates without confirmation/streak state."""

    num_envs = int(getattr(env, "num_envs", 0))
    scene = getattr(env, "scene", None)
    if num_envs <= 0 or scene is None:
        raise RecoveryTelemetryIncompleteError(("scene",))
    robot = _scene_get(scene, "robot")
    if robot is None:
        raise RecoveryTelemetryIncompleteError(("robot_state",))
    root_quaternions = _root_quaternion_tensor(robot, num_envs=num_envs)
    critical_contacts = _critical_body_contact_by_env(scene, robot, num_envs=num_envs)
    return tuple(
        classify_fall(
            compute_root_up_alignment(root_quaternions[env_index]),
            critical_body_contact=critical_contacts[env_index],
        )
        for env_index in range(num_envs)
    )


def extract_privileged_telemetry(
    env: object,
    *,
    support_resolver: Any | None = None,
    terminal_contexts: object | None = None,
) -> tuple[PrivilegedRecoveryTelemetry, ...]:
    """Extract privileged state without registering it as a policy observation."""

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
    resolved_terminal_contexts = _resolve_terminal_contexts(
        env,
        terminal_contexts,
        num_envs=num_envs,
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
    force_tensors: dict[str, tuple[torch.Tensor, ...]] = {}
    for side in ("left", "right"):
        force_tensors[side] = tuple(
            _pairwise_force_tensor(
                scene,
                sensor_binding,
                side=side,
                num_envs=num_envs,
                validated_report=reports_by_scene_key[sensor_binding.sensor_scene_key],
            )
            for sensor_binding in bindings[side].sensors
        )
    root_quaternions = _root_quaternion_tensor(robot, num_envs=num_envs)
    critical_contacts = _critical_body_contact_by_env(scene, robot, num_envs=num_envs)

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
        for side in ("left", "right"):
            link_evidence = tuple(
                pairwise_contact_evidence(
                    force_tensor[env_index],
                    sensor_body=sensor_binding.sensor_body_name,
                    filtered_body=sensor_binding.filtered_body_name,
                    threshold_n=contact_threshold,
                )
                for sensor_binding, force_tensor in zip(
                    bindings[side].sensors,
                    force_tensors[side],
                    strict=True,
                )
            )
            hand_contacts[side] = aggregate_hand_contact_evidence(side, link_evidence)
        states.append(
            build_privileged_telemetry(
                env_index=env_index,
                box_center_w=box_centers[env_index],
                box_linear_velocity_w=box_linear_velocity[env_index],
                box_angular_velocity_w=box_angular_velocity[env_index],
                support=support_evidence[env_index],
                left_ee_pose_w=left_poses[env_index],
                right_ee_pose_w=right_poses[env_index],
                left_contact=hand_contacts["left"],
                right_contact=hand_contacts["right"],
                root_quat_wxyz=root_quaternions[env_index],
                critical_body_contact=critical_contacts[env_index],
                terminal_context=resolved_terminal_contexts[env_index],
                max_ee_box_distance_m=ee_distance_threshold,
            )
        )
    return tuple(states)


__all__ = [
    "DEFAULT_CONTACT_FORCE_THRESHOLD_N",
    "DEFAULT_MAX_EE_BOX_DISTANCE_M",
    "EMPIRICAL_CONTACT_QUIET_MAX_N",
    "EMPIRICAL_CONTACT_TOUCH_MIN_N",
    "HARD_FALL_UP_ALIGNMENT",
    "PP_BOX_TASK_IDENTITY",
    "RECOVERY_TELEMETRY_SCHEMA_VERSION",
    "RUNTIME_CONTACT_MAPPING_PROOF_SCHEMA_VERSION",
    "RUNTIME_CONTACT_SENSOR_REPORT_SCHEMA_VERSION",
    "SOFT_FALL_UP_ALIGNMENT",
    "BimanualGraspEvidence",
    "DriverTerminalContext",
    "EmpiricalContactMappingProof",
    "HandContactBinding",
    "HandContactEvidence",
    "PairwiseContactBinding",
    "PairwiseContactEvidence",
    "PrivilegedObservationLeakError",
    "PrivilegedRecoveryTelemetry",
    "RecoveryTelemetryIncompleteError",
    "RuntimeContactSensorReport",
    "aggregate_hand_contact_evidence",
    "assert_actor_observation_isolated",
    "build_empirical_contact_mapping_proof",
    "build_privileged_telemetry",
    "classify_bimanual_grasp",
    "classify_fall",
    "classify_terminal",
    "compute_live_fall_candidates",
    "compute_root_up_alignment",
    "default_hand_contact_bindings",
    "empirical_contact_mapping_proof_digest",
    "extract_privileged_telemetry",
    "has_pairwise_bimanual_contact",
    "pairwise_contact_evidence",
    "validate_runtime_hand_contact_sensors",
]
