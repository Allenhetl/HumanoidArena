"""Live HOI_pp_box attempt provenance for ReCoVLA recovery activation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from . import recovery_failures, rewards
from .recovery_telemetry import PP_BOX_TASK_IDENTITY, PrivilegedRecoveryTelemetry

RECOVERY_RUNTIME_SCHEMA_VERSION = 1
_SNAPSHOT_SCHEMA = "ha_pp_box_live_recovery_runtime_v1"
_THRESHOLD_KEYS = {
    "ground_surface_z_m",
    "ground_support_tolerance_m",
    "linear_stable_speed_mps",
    "angular_stable_speed_radps",
    "progress_epsilon_m",
    "stall_confirm_steps",
    "stable_confirm_steps",
    "place_attempt_distance_m",
    "axis_alignment_tolerance_deg",
}


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _finite_float(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class RecoveryRuntimeThresholds:
    ground_surface_z_m: float
    ground_support_tolerance_m: float
    linear_stable_speed_mps: float
    angular_stable_speed_radps: float
    progress_epsilon_m: float
    stall_confirm_steps: int
    stable_confirm_steps: int
    place_attempt_distance_m: float
    axis_alignment_tolerance_deg: float

    @classmethod
    def from_value(cls, value: object) -> RecoveryRuntimeThresholds:
        if not isinstance(value, Mapping) or set(value) != _THRESHOLD_KEYS:
            raise ValueError(
                "recovery_runtime_thresholds must contain the exact task schema"
            )
        angle = _finite_float(
            value["axis_alignment_tolerance_deg"],
            name="axis alignment tolerance",
            positive=True,
        )
        if angle >= 45.0:
            raise ValueError("axis alignment tolerance must be below 45 degrees")
        return cls(
            ground_surface_z_m=_finite_float(
                value["ground_surface_z_m"], name="ground surface z"
            ),
            ground_support_tolerance_m=_finite_float(
                value["ground_support_tolerance_m"],
                name="ground support tolerance",
                positive=True,
            ),
            linear_stable_speed_mps=_finite_float(
                value["linear_stable_speed_mps"],
                name="linear stable speed",
                positive=True,
            ),
            angular_stable_speed_radps=_finite_float(
                value["angular_stable_speed_radps"],
                name="angular stable speed",
                positive=True,
            ),
            progress_epsilon_m=_finite_float(
                value["progress_epsilon_m"],
                name="progress epsilon",
                positive=True,
            ),
            stall_confirm_steps=_positive_int(
                value["stall_confirm_steps"], name="stall confirm steps"
            ),
            stable_confirm_steps=_positive_int(
                value["stable_confirm_steps"], name="stable confirm steps"
            ),
            place_attempt_distance_m=_finite_float(
                value["place_attempt_distance_m"],
                name="place attempt distance",
                positive=True,
            ),
            axis_alignment_tolerance_deg=angle,
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(asdict(self))


def _scene_box(env: object) -> object:
    scene = getattr(env, "scene", None)
    for key in ("box", "Box"):
        try:
            return scene[key]
        except (KeyError, TypeError):
            continue
    raise RuntimeError("HOI_pp_box live runtime cannot resolve the Box asset")


def _box_pose_w(env: object) -> tuple[tuple[float, float, float], tuple[float, ...]]:
    data = getattr(_scene_box(env), "data", None)
    state = getattr(data, "root_state_w", None)
    if not isinstance(state, torch.Tensor) or tuple(state.shape) != (1, 13):
        raise RuntimeError("HOI_pp_box Box root_state_w must be tensor [1,13]")
    center = getattr(data, "root_com_pos_w", None)
    if center is None:
        center = getattr(data, "root_pos_w", None)
    if center is None:
        center = state[:, :3]
    if not isinstance(center, torch.Tensor) or tuple(center.shape) != (1, 3):
        raise RuntimeError("HOI_pp_box Box center must be tensor [1,3]")
    pose_row = (
        torch.cat((center, state[:, 3:7]), dim=1)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .numpy()[0]
    )
    if not np.isfinite(pose_row).all():
        raise ValueError("HOI_pp_box Box pose must be finite")
    return tuple(float(value) for value in pose_row[:3]), tuple(
        float(value) for value in pose_row[3:7]
    )


def _axis_aligned(quaternion_wxyz: tuple[float, ...], tolerance_deg: float) -> bool:
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("HOI_pp_box Box quaternion must be finite and non-zero")
    w, x, y, z = quat / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    absolute = np.abs(rotation)
    dominant = np.argmax(absolute, axis=1)
    threshold = math.cos(math.radians(tolerance_deg))
    return bool(
        len({int(value) for value in dominant}) == 3
        and np.all(absolute[np.arange(3), dominant] >= threshold)
    )


class PPBoxLiveRecoveryRuntime:
    """Track task attempts while deriving every physical predicate from live state."""

    def __init__(self, env: Any, provider: Any) -> None:
        cfg = getattr(env, "cfg", None)
        if getattr(cfg, "env_name", None) != PP_BOX_TASK_IDENTITY:
            raise ValueError("recovery runtime cfg.env_name is not HOI_pp_box")
        if getattr(env, "num_envs", None) != 1:
            raise ValueError("recovery runtime requires one exclusive environment lane")
        seed = getattr(cfg, "seed", None)
        if type(seed) is not int or seed < 0:
            raise ValueError("recovery runtime cfg.seed must be a non-negative integer")
        self.env = env
        self.provider = provider
        self.thresholds = RecoveryRuntimeThresholds.from_value(
            getattr(cfg, "recovery_runtime_thresholds", None)
        )
        self.failure_seed = seed
        self.clear()

    def identity(self) -> dict[str, object]:
        return {
            "schema": "ha_pp_box_live_recovery_runtime_identity_v1",
            "task_identity": PP_BOX_TASK_IDENTITY,
            "thresholds_sha256": self.thresholds.digest,
            "exclusive_num_envs": 1,
        }

    def clear(self) -> None:
        self.last_control_step: int | None = None
        self.provider_source_control_step: int | None = None
        self.attempt_count = 0
        self.pickup_attempted = False
        self.place_attempted = False
        self.release_attempted = False
        self.ever_grasped = False
        self.previous_both_closed: bool | None = None
        self.best_progress_m: float | None = None
        self.last_progress_step = 0
        self.no_progress_steps = 0
        self.stable_steps = 0

    def _provider_action(self) -> tuple[np.ndarray, int]:
        raw = getattr(self.provider, "_latest_executed_canonical_action", None)
        action = np.asarray(raw) if raw is not None else np.empty(0)
        if (
            action.dtype != np.float32
            or action.shape != (40,)
            or not np.isfinite(action).all()
        ):
            raise ValueError(
                "provider executed semantic40 must be finite float32 shape (40,)"
            )
        source_step = getattr(
            self.provider, "_latest_executed_source_control_step", None
        )
        if type(source_step) is not int or source_step < 0:
            raise ValueError("provider executed source control step is invalid")
        return np.ascontiguousarray(action), source_step

    def _physical_facts(
        self, telemetry: PrivilegedRecoveryTelemetry
    ) -> tuple[bool, bool, bool]:
        position, quaternion = _box_pose_w(self.env)
        if any(
            not math.isclose(
                position[index], telemetry.box_center_w[index], abs_tol=1e-5
            )
            for index in range(3)
        ):
            raise RuntimeError("live Box center differs from privileged telemetry")
        box_bottom = position[2] - rewards.BOX_HALF_EXTENTS_M[2]
        ground_supported = bool(
            abs(box_bottom - self.thresholds.ground_surface_z_m)
            <= self.thresholds.ground_support_tolerance_m
            and self.stable_steps >= self.thresholds.stable_confirm_steps
        )
        x_lo, x_hi, y_lo, y_hi = telemetry.shelf_bounds_w
        half_x, half_y = rewards.BOX_HALF_EXTENTS_M[:2]
        target_disjoint = bool(
            position[0] + half_x < x_lo
            or position[0] - half_x > x_hi
            or position[1] + half_y < y_lo
            or position[1] - half_y > y_hi
        )
        return (
            ground_supported,
            target_disjoint,
            _axis_aligned(quaternion, self.thresholds.axis_alignment_tolerance_deg),
        )

    def observe(
        self, telemetry: PrivilegedRecoveryTelemetry
    ) -> recovery_failures.FailurePredicateContext:
        if not isinstance(telemetry, PrivilegedRecoveryTelemetry):
            raise TypeError("recovery runtime requires PrivilegedRecoveryTelemetry")
        if telemetry.task_identity != PP_BOX_TASK_IDENTITY or telemetry.env_index != 0:
            raise ValueError("recovery runtime telemetry identity differs")
        step = telemetry.control_step_count
        if type(step) is not int or step < 1:
            raise ValueError("recovery runtime control step must be positive")
        if self.last_control_step is not None and step != self.last_control_step + 1:
            raise ValueError(
                "recovery runtime observations must be primitive-step contiguous"
            )
        action, source_step = self._provider_action()
        both_closed = bool(action[38] == 1.0 and action[39] == 1.0)
        both_open = bool(action[38] == 0.0 and action[39] == 0.0)
        if not ((action[38] in (0.0, 1.0)) and (action[39] in (0.0, 1.0))):
            raise ValueError("provider executed hand channels must be canonical binary")
        if both_closed and self.previous_both_closed is not True:
            self.attempt_count += 1
            self.pickup_attempted = True
        self.ever_grasped = self.ever_grasped or telemetry.grasp
        self.place_attempted = self.place_attempted or bool(
            self.ever_grasped
            and telemetry.placement_distance_m
            <= self.thresholds.place_attempt_distance_m
        )
        self.release_attempted = self.release_attempted or bool(
            self.ever_grasped and self.previous_both_closed is True and both_open
        )

        progress = (
            telemetry.placement_distance_m
            if self.ever_grasped
            else telemetry.grasp_evidence.max_ee_box_distance_m
        )
        if (
            self.best_progress_m is None
            or progress < self.best_progress_m - self.thresholds.progress_epsilon_m
        ):
            self.best_progress_m = float(progress)
            self.last_progress_step = step
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        linear_speed = math.sqrt(
            sum(value * value for value in telemetry.box_linear_velocity_w)
        )
        angular_speed = math.sqrt(
            sum(value * value for value in telemetry.box_angular_velocity_w)
        )
        if (
            linear_speed <= self.thresholds.linear_stable_speed_mps
            and angular_speed <= self.thresholds.angular_stable_speed_radps
        ):
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        self.last_control_step = step
        self.provider_source_control_step = source_step
        self.previous_both_closed = both_closed

        trigger_kind = (
            "post-release"
            if self.release_attempted
            else "place-attempt"
            if self.place_attempted
            else "pickup-attempt"
        )
        attempt = recovery_failures.RecoveryAttemptEvidence(
            schema_version=recovery_failures.RECOVERY_ATTEMPT_SCHEMA_VERSION,
            trigger_kind=trigger_kind,
            anchor_id=None,
            anchor_digest=None,
            failure_seed=self.failure_seed,
            attempt_count=self.attempt_count,
            pickup_attempted=self.pickup_attempted,
            place_attempted=self.place_attempted,
            release_attempted=self.release_attempted,
            last_progress_step=self.last_progress_step,
            no_progress_steps=self.no_progress_steps,
            stall_confirm_steps=self.thresholds.stall_confirm_steps,
            stable_steps=self.stable_steps,
            stable_confirm_steps=self.thresholds.stable_confirm_steps,
            injected_category=None,
            transform_digest=None,
        )
        ground, disjoint, aligned = self._physical_facts(telemetry)
        return recovery_failures.FailurePredicateContext(
            telemetry=telemetry,
            attempt=attempt,
            ground_supported=ground,
            target_disjoint=disjoint,
            box_axis_aligned=aligned,
        )

    def capture_state(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": _SNAPSHOT_SCHEMA,
            "task_identity": PP_BOX_TASK_IDENTITY,
            "thresholds_digest": self.thresholds.digest,
            "failure_seed": self.failure_seed,
            "last_control_step": self.last_control_step,
            "provider_source_control_step": self.provider_source_control_step,
            "attempt_count": self.attempt_count,
            "pickup_attempted": self.pickup_attempted,
            "place_attempted": self.place_attempted,
            "release_attempted": self.release_attempted,
            "ever_grasped": self.ever_grasped,
            "previous_both_closed": self.previous_both_closed,
            "best_progress_m": self.best_progress_m,
            "last_progress_step": self.last_progress_step,
            "no_progress_steps": self.no_progress_steps,
            "stable_steps": self.stable_steps,
        }
        return {**payload, "state_digest": _canonical_digest(payload)}

    def restore_state(self, state: Mapping[str, object]) -> None:
        expected = set(self.capture_state())
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("recovery runtime snapshot schema differs")
        payload = {key: state[key] for key in state if key != "state_digest"}
        if (
            state["schema"] != _SNAPSHOT_SCHEMA
            or state["task_identity"] != PP_BOX_TASK_IDENTITY
            or state["thresholds_digest"] != self.thresholds.digest
            or state["failure_seed"] != self.failure_seed
            or state["state_digest"] != _canonical_digest(payload)
        ):
            raise ValueError("recovery runtime snapshot identity or digest differs")
        integer_fields = (
            "attempt_count",
            "last_progress_step",
            "no_progress_steps",
            "stable_steps",
        )
        if any(
            type(state[name]) is not int or state[name] < 0 for name in integer_fields
        ):
            raise ValueError("recovery runtime snapshot counters are invalid")
        for name in (
            "last_control_step",
            "provider_source_control_step",
        ):
            value = state[name]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("recovery runtime snapshot step is invalid")
        for name in (
            "pickup_attempted",
            "place_attempted",
            "release_attempted",
            "ever_grasped",
        ):
            if type(state[name]) is not bool:
                raise ValueError("recovery runtime snapshot flags are invalid")
        previous = state["previous_both_closed"]
        if previous is not None and type(previous) is not bool:
            raise ValueError("recovery runtime previous hand state is invalid")
        best = state["best_progress_m"]
        if best is not None and (
            isinstance(best, bool)
            or not isinstance(best, (int, float))
            or not math.isfinite(float(best))
            or float(best) < 0.0
        ):
            raise ValueError("recovery runtime best progress is invalid")

        self.last_control_step = state["last_control_step"]  # type: ignore[assignment]
        self.provider_source_control_step = state[  # type: ignore[assignment]
            "provider_source_control_step"
        ]
        self.attempt_count = int(state["attempt_count"])
        self.pickup_attempted = bool(state["pickup_attempted"])
        self.place_attempted = bool(state["place_attempted"])
        self.release_attempted = bool(state["release_attempted"])
        self.ever_grasped = bool(state["ever_grasped"])
        self.previous_both_closed = previous  # type: ignore[assignment]
        self.best_progress_m = None if best is None else float(best)
        self.last_progress_step = int(state["last_progress_step"])
        self.no_progress_steps = int(state["no_progress_steps"])
        self.stable_steps = int(state["stable_steps"])


__all__ = [
    "RECOVERY_RUNTIME_SCHEMA_VERSION",
    "PPBoxLiveRecoveryRuntime",
    "RecoveryRuntimeThresholds",
]
