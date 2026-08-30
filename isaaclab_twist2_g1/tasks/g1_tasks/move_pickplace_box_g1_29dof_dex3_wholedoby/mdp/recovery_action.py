"""ReCoVLA semantic40 residual ownership for HOI_pp_box."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from action_provider.execution_identity import (
    HAND_BINARY_THRESHOLD,
    canonicalize_provider_semantic40,
    validate_finite_float32_vector,
    validate_provider_semantic40,
)

RECOVLA_GR00T_ARMS14_NAME = "ReCoVLA-GR00T-arms14"
RECOVLA_GR00T_ARMS14_OWNED_INDICES = (
    20,
    21,
    24,
    25,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
)
RECOVLA_GR00T_ARMS14_HAND_INDICES = (38, 39)
RECOVLA_GR00T_ARMS14_SCALE = (0.25,) * len(RECOVLA_GR00T_ARMS14_OWNED_INDICES)


@dataclass(frozen=True)
class RecoveryActionContract:
    """Frozen external action contract consumed by the RLinf adapter."""

    name: str
    canonical_action_dim: int
    reference_horizon: int
    committed_horizon: int
    residual_cadence: str
    residual_owned_indices: tuple[int, ...]
    hand_indices: tuple[int, int]
    residual_scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.name != RECOVLA_GR00T_ARMS14_NAME:
            raise ValueError("recovery action contract name is not arms14")
        if self.canonical_action_dim != 40:
            raise ValueError("arms14 requires canonical semantic40 actions")
        if self.reference_horizon != 40 or self.committed_horizon != 40:
            raise ValueError("arms14 requires HumanoidArena Base H=40 and committed C=40")
        if self.residual_cadence != "primitive-control-step":
            raise ValueError("arms14 residual cadence must be primitive-control-step")
        if tuple(self.residual_owned_indices) != RECOVLA_GR00T_ARMS14_OWNED_INDICES:
            raise ValueError("arms14 residual owned indices do not match the fixed contract")
        if tuple(self.hand_indices) != RECOVLA_GR00T_ARMS14_HAND_INDICES:
            raise ValueError("arms14 hand indices must remain Base-owned")
        scales = tuple(self.residual_scale)
        if len(scales) != len(RECOVLA_GR00T_ARMS14_OWNED_INDICES):
            raise ValueError("arms14 residual scale must explicitly cover all 14 dimensions")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in scales
        ):
            raise ValueError("arms14 residual scale values must be finite and positive")
        object.__setattr__(self, "residual_owned_indices", tuple(self.residual_owned_indices))
        object.__setattr__(self, "hand_indices", tuple(self.hand_indices))
        object.__setattr__(self, "residual_scale", tuple(float(value) for value in scales))

    @property
    def base_owned_indices(self) -> tuple[int, ...]:
        residual_owned = set(self.residual_owned_indices)
        return tuple(
            index for index in range(self.canonical_action_dim) if index not in residual_owned
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "canonical_action_dim": self.canonical_action_dim,
            "reference_horizon": self.reference_horizon,
            "committed_horizon": self.committed_horizon,
            "residual_cadence": self.residual_cadence,
            "residual_owned_indices": self.residual_owned_indices,
            "hand_indices": self.hand_indices,
            "residual_scale": self.residual_scale,
        }


@dataclass(frozen=True)
class RecoveryActionComposition:
    """One audited provider-canonical primitive action."""

    canonical_reference40: np.ndarray
    scaled_residual40: np.ndarray
    executed_action: np.ndarray
    owned_mutation_count: int
    non_owned_mutation_count: int
    hand_mutation_count: int


_ACTION_CONTRACT = RecoveryActionContract(
    name=RECOVLA_GR00T_ARMS14_NAME,
    canonical_action_dim=40,
    reference_horizon=40,
    committed_horizon=40,
    residual_cadence="primitive-control-step",
    residual_owned_indices=RECOVLA_GR00T_ARMS14_OWNED_INDICES,
    hand_indices=RECOVLA_GR00T_ARMS14_HAND_INDICES,
    residual_scale=RECOVLA_GR00T_ARMS14_SCALE,
)


def recovla_gr00t_arms14_action_contract() -> RecoveryActionContract:
    return _ACTION_CONTRACT


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    result.setflags(write=False)
    return result


def compose_recovla_gr00t_arms14_action(
    reference40: object,
    residual14: object,
    *,
    contract: RecoveryActionContract = _ACTION_CONTRACT,
) -> RecoveryActionComposition:
    """Expand arms14, add it to Base, then use the provider's canonicalizer."""

    if contract != _ACTION_CONTRACT:
        raise ValueError("composition requires the fixed ReCoVLA-GR00T-arms14 contract")
    reference = validate_provider_semantic40(reference40, field="Base reference40")
    residual = validate_finite_float32_vector(
        residual14,
        expected_dim=len(contract.residual_owned_indices),
        field="arms14 primitive residual",
    )
    left_closed = bool(reference[contract.hand_indices[0]] >= HAND_BINARY_THRESHOLD)
    right_closed = bool(reference[contract.hand_indices[1]] >= HAND_BINARY_THRESHOLD)
    canonical_reference = canonicalize_provider_semantic40(
        reference,
        left_closed=left_closed,
        right_closed=right_closed,
        field="Base provider-canonical reference40",
    )

    scaled_residual40 = np.zeros(contract.canonical_action_dim, dtype=np.float32)
    owned = np.asarray(contract.residual_owned_indices, dtype=np.int64)
    scales = np.asarray(contract.residual_scale, dtype=np.float32)
    scaled_residual40[owned] = residual * scales
    candidate = canonical_reference + scaled_residual40
    executed = canonicalize_provider_semantic40(
        candidate,
        left_closed=left_closed,
        right_closed=right_closed,
        field="ReCoVLA-GR00T-arms14 provider candidate",
    )

    changed = executed.view(np.uint32) != canonical_reference.view(np.uint32)
    base_owned = np.asarray(contract.base_owned_indices, dtype=np.int64)
    hands = np.asarray(contract.hand_indices, dtype=np.int64)
    return RecoveryActionComposition(
        canonical_reference40=_readonly(canonical_reference),
        scaled_residual40=_readonly(scaled_residual40),
        executed_action=_readonly(executed),
        owned_mutation_count=int(np.count_nonzero(changed[owned])),
        non_owned_mutation_count=int(np.count_nonzero(changed[base_owned])),
        hand_mutation_count=int(np.count_nonzero(changed[hands])),
    )


__all__ = [
    "RECOVLA_GR00T_ARMS14_HAND_INDICES",
    "RECOVLA_GR00T_ARMS14_NAME",
    "RECOVLA_GR00T_ARMS14_OWNED_INDICES",
    "RECOVLA_GR00T_ARMS14_SCALE",
    "RecoveryActionComposition",
    "RecoveryActionContract",
    "compose_recovla_gr00t_arms14_action",
    "recovla_gr00t_arms14_action_contract",
]
