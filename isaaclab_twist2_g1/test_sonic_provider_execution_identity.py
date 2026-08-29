from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from action_provider.execution_identity import (
    HAND_BINARY_THRESHOLD,
    canonicalize_provider_semantic40,
    parse_sonic_output_delay_steps,
    resolve_binary_hand_states,
    validate_hand_binary_threshold,
    validate_provider_semantic40,
    validate_sonic_body_action,
    validate_source_control_step,
)

THIS_DIR = Path(__file__).resolve().parent
PROVIDER_PATH = THIS_DIR / "action_provider" / "action_provider_sonic.py"


def _semantic40() -> np.ndarray:
    action = np.linspace(-1.0, 1.0, 40, dtype=np.float32)
    action[38:40] = np.asarray([0.25, 0.75], dtype=np.float32)
    return action


def _method_source(name: str) -> str:
    source = PROVIDER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    provider = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SonicActionProvider"
    )
    method = next(
        node
        for node in provider.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    return ast.get_source_segment(source, method) or ""


def test_provider_semantic40_binary_projection_is_byte_stable_off_hands() -> None:
    candidate = _semantic40()

    executed = canonicalize_provider_semantic40(
        candidate,
        left_closed=False,
        right_closed=True,
        field="test semantic40",
    )

    np.testing.assert_array_equal(executed[:38], candidate[:38])
    assert executed[:38].tobytes() == candidate[:38].tobytes()
    np.testing.assert_array_equal(executed[38:40], [0.0, 1.0])
    assert executed.dtype == np.float32
    assert executed.flags.c_contiguous

    already_canonical = candidate.copy()
    already_canonical[38:40] = np.asarray([0.0, 1.0], dtype=np.float32)
    replay_identity = canonicalize_provider_semantic40(
        already_canonical,
        left_closed=False,
        right_closed=True,
        field="submitted provider-canonical semantic40",
    )
    assert replay_identity.tobytes() == already_canonical.tobytes()


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(39, dtype=np.float32),
        np.zeros((1, 40), dtype=np.float32),
        np.concatenate([np.zeros(39, dtype=np.float32), [np.nan]]),
        np.concatenate([np.zeros(39, dtype=np.float32), [np.inf]]),
    ],
)
def test_provider_semantic40_rejects_bad_shape_and_nonfinite(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_provider_semantic40(value, field="test semantic40")


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(28, dtype=np.float32),
        np.zeros((1, 29), dtype=np.float32),
        np.concatenate([np.zeros(28, dtype=np.float32), [np.nan]]),
    ],
)
def test_sonic_body_action_rejects_bad_shape_and_nonfinite(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_sonic_body_action(value, field="test body action")


def test_hand_projection_validates_values_and_threshold() -> None:
    assert validate_hand_binary_threshold(
        HAND_BINARY_THRESHOLD,
        require_provider_identity=True,
    ) == HAND_BINARY_THRESHOLD
    assert resolve_binary_hand_states(
        [0.499, 0.5],
        threshold=HAND_BINARY_THRESHOLD,
        field="test hands",
    ) == (False, True)

    for bad in (float("nan"), float("inf"), -0.1, 1.1, True):
        with pytest.raises((TypeError, ValueError)):
            validate_hand_binary_threshold(
                bad,
                require_provider_identity=True,
            )
    with pytest.raises(ValueError, match="requires lerobot_gripper_threshold"):
        validate_hand_binary_threshold(0.49, require_provider_identity=True)
    with pytest.raises(ValueError, match="finite values"):
        resolve_binary_hand_states(
            [0.0, np.nan],
            threshold=HAND_BINARY_THRESHOLD,
            field="test hands",
        )


def test_vla_output_delay_identity_is_strict_and_fail_closed() -> None:
    assert parse_sonic_output_delay_steps("0", require_zero=True) == 0
    assert parse_sonic_output_delay_steps(3, require_zero=False) == 3

    for bad in (1, -1, "1.0", "bad", None, True):
        with pytest.raises((TypeError, ValueError)):
            parse_sonic_output_delay_steps(bad, require_zero=True)


def test_source_control_step_requires_nonnegative_integer() -> None:
    assert validate_source_control_step(np.int64(7), field="step") == 7
    for bad in (-1, 1.0, "1", True):
        with pytest.raises((TypeError, ValueError)):
            validate_source_control_step(bad, field="step")


def test_source_control_step_binds_consumed_queue_row_to_execution_bundle() -> None:
    apply_pose = _method_source("_apply_pose_data")
    assert "self._latest_consumed_control_step = int(self._frame_count)" in apply_pose

    bundle = _method_source("_build_action_execution_bundle")
    assert '"source_control_step": int(self._latest_consumed_control_step)' in bundle

    publish = _method_source("_publish_executed_action_telemetry")
    assert "validate_source_control_step" in publish
    assert "self._latest_executed_source_control_step = source_control_step" in publish


def test_provider_records_only_after_pop_projection_and_hand_materialization() -> None:
    apply_semantic = _method_source("_apply_lerobot_semantic_action")
    assert apply_semantic.index("_apply_pose_data") < apply_semantic.index(
        "resolve_binary_hand_states"
    )
    assert "did not produce" in apply_semantic
    assert "a new provider control frame" in apply_semantic
    assert apply_semantic.index("resolve_binary_hand_states") < apply_semantic.index(
        "_apply_hand_binary_targets"
    )
    assert apply_semantic.index("_apply_hand_binary_targets") < apply_semantic.index(
        "canonicalize_provider_semantic40"
    )

    get_action = _method_source("get_action")
    assert get_action.index("_run_gear_sonic_from_vla") < get_action.index(
        "_apply_sonic_output_delay"
    )
    assert get_action.index("_apply_sonic_output_delay") < get_action.index(
        "_apply_hand_targets"
    )
    assert get_action.index("_apply_hand_targets") < get_action.index("env.sim.step")
    assert get_action.index("env.sim.step") < get_action.index(
        "_publish_executed_action_telemetry"
    )
    assert get_action.index("_publish_executed_action_telemetry") < get_action.index(
        "recording_manager.add_frame"
    )

    apply_delay = _method_source("_apply_sonic_output_delay")
    assert "_latest_executed_canonical_action" not in apply_delay
    assert "validate_sonic_body_action" in apply_delay


def test_provider_reset_invalidates_execution_telemetry_and_queue() -> None:
    reset = _method_source("on_env_reset")
    assert "self._lerobot_action_chunk_queue.clear()" in reset
    assert "self._latest_vla_action = None" in reset
    assert "self._invalidate_executed_action_telemetry()" in reset

    invalidate = _method_source("_invalidate_executed_action_telemetry")
    assert "self._latest_executed_canonical_action = _default_vla_action()" in invalidate
    assert "self._latest_executed_source_control_step = -1" in invalidate


def test_vla_inference_failure_cannot_fall_back_to_a_fake_executed_action() -> None:
    run_sonic = _method_source("_run_gear_sonic")
    assert "if self._use_lerobot_vla:" in run_sonic
    assert "execution telemetry remains invalid" in run_sonic

    get_action = _method_source("get_action")
    assert get_action.index("_invalidate_executed_action_telemetry") < get_action.index(
        "_run_gear_sonic_from_vla"
    )


def test_vla_hand_execution_bypasses_external_dds_override() -> None:
    apply_hands = _method_source("_apply_hand_targets")
    assert "if not self._use_lerobot_vla and self._dex3_dds is not None:" in apply_hands
