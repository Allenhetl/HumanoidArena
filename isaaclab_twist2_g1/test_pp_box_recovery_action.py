from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

THIS_DIR = Path(__file__).resolve().parent
RECOVERY_ACTION_PATH = (
    THIS_DIR
    / "tasks"
    / "g1_tasks"
    / "move_pickplace_box_g1_29dof_dex3_wholedoby"
    / "mdp"
    / "recovery_action.py"
)


def _load_recovery_action():
    spec = importlib.util.spec_from_file_location(
        "pp_box_recovery_action", RECOVERY_ACTION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _provider_canonical_reference() -> np.ndarray:
    reference = np.linspace(-0.8, 0.8, 40, dtype=np.float32)
    reference[38:40] = np.asarray([0.0, 1.0], dtype=np.float32)
    return reference


def test_recovla_gr00t_arms14_contract_is_exact_and_explicit() -> None:
    action = _load_recovery_action()

    contract = action.recovla_gr00t_arms14_action_contract()

    assert contract.name == "ReCoVLA-GR00T-arms14"
    assert contract.canonical_action_dim == 40
    assert contract.reference_horizon == 16
    assert contract.committed_horizon == 10
    assert contract.residual_cadence == "primitive-control-step"
    assert contract.residual_owned_indices == (
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
    assert contract.hand_indices == (38, 39)
    assert len(contract.residual_scale) == 14
    assert contract.residual_scale == (0.25,) * 14
    assert set(contract.base_owned_indices) == set(range(40)) - set(
        contract.residual_owned_indices
    )


def test_zero_residual_is_byte_identical_to_provider_canonical_base() -> None:
    action = _load_recovery_action()
    reference = _provider_canonical_reference()

    result = action.compose_recovla_gr00t_arms14_action(
        reference,
        np.zeros(14, dtype=np.float32),
    )

    assert result.executed_action.tobytes() == reference.tobytes()
    assert result.non_owned_mutation_count == 0
    assert result.hand_mutation_count == 0
    assert result.owned_mutation_count == 0


def test_only_owned_indices_receive_scaled_primitive_residual() -> None:
    action = _load_recovery_action()
    reference = _provider_canonical_reference()
    residual = np.linspace(-1.0, 1.0, 14, dtype=np.float32)

    result = action.compose_recovla_gr00t_arms14_action(reference, residual)
    expected = reference.copy()
    owned = np.asarray(action.RECOVLA_GR00T_ARMS14_OWNED_INDICES, dtype=np.int64)
    expected[owned] += residual * np.float32(0.25)

    np.testing.assert_array_equal(result.executed_action, expected)
    np.testing.assert_array_equal(result.scaled_residual40[owned], residual * 0.25)
    np.testing.assert_array_equal(
        np.delete(result.scaled_residual40, owned),
        np.zeros(26, dtype=np.float32),
    )
    assert result.non_owned_mutation_count == 0
    assert result.hand_mutation_count == 0
    assert result.owned_mutation_count == 14
    assert result.executed_action[38:40].tobytes() == reference[38:40].tobytes()


def test_provider_canonicalization_owns_hand_projection() -> None:
    action = _load_recovery_action()
    raw_reference = _provider_canonical_reference()
    raw_reference[38:40] = np.asarray([0.25, 0.75], dtype=np.float32)

    result = action.compose_recovla_gr00t_arms14_action(
        raw_reference,
        np.zeros(14, dtype=np.float32),
    )

    np.testing.assert_array_equal(result.executed_action[38:40], [0.0, 1.0])
    assert result.non_owned_mutation_count == 0
    assert result.hand_mutation_count == 0


@pytest.mark.parametrize(
    ("reference", "residual"),
    [
        (np.zeros(39, dtype=np.float32), np.zeros(14, dtype=np.float32)),
        (np.zeros(40, dtype=np.float32), np.zeros(40, dtype=np.float32)),
        (np.zeros(40, dtype=np.float32), np.zeros(13, dtype=np.float32)),
        (
            np.concatenate([np.zeros(39, dtype=np.float32), [np.nan]]),
            np.zeros(14, dtype=np.float32),
        ),
        (
            np.zeros(40, dtype=np.float32),
            np.concatenate([np.zeros(13, dtype=np.float32), [np.inf]]),
        ),
    ],
)
def test_composition_rejects_wrong_or_nonfinite_action_shapes(
    reference: np.ndarray,
    residual: np.ndarray,
) -> None:
    action = _load_recovery_action()

    with pytest.raises(ValueError):
        action.compose_recovla_gr00t_arms14_action(reference, residual)


def test_contract_rejects_implicit_or_misindexed_scale() -> None:
    action = _load_recovery_action()
    contract = action.recovla_gr00t_arms14_action_contract()

    with pytest.raises(ValueError, match="residual scale"):
        action.RecoveryActionContract(
            **{
                **contract.as_dict(),
                "residual_scale": (0.25,) * 13,
            }
        )
    with pytest.raises(ValueError, match="owned indices"):
        action.RecoveryActionContract(
            **{
                **contract.as_dict(),
                "residual_owned_indices": contract.residual_owned_indices[:-1]
                + (38,),
            }
        )
