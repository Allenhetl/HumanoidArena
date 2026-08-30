from __future__ import annotations

import ast
import importlib.util
import math
import random
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
MDP_DIR = (
    THIS_DIR
    / "tasks"
    / "g1_tasks"
    / "move_pickplace_box_g1_29dof_dex3_wholedoby"
    / "mdp"
)
PACKAGE_NAME = "pp_box_recovery_failure_tests"
MDP_INIT_PATH = MDP_DIR / "__init__.py"
FAILURES_PATH = MDP_DIR / "recovery_failures.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def modules():
    for name in tuple(sys.modules):
        if name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."):
            del sys.modules[name]
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(MDP_DIR)]
    sys.modules[PACKAGE_NAME] = package
    state = _load_module(
        f"{PACKAGE_NAME}.recovery_state", MDP_DIR / "recovery_state.py"
    )
    telemetry = _load_module(
        f"{PACKAGE_NAME}.recovery_telemetry",
        MDP_DIR / "recovery_telemetry.py",
    )
    failures = _load_module(
        f"{PACKAGE_NAME}.recovery_failures",
        MDP_DIR / "recovery_failures.py",
    )
    return state, telemetry, failures


DECLARED_CATEGORIES = {
    "dropped",
    "failed-grasp",
    "misaligned",
    "near-shelf-misplaced",
}


def _descriptor_payload(failures, category: str = "dropped") -> dict[str, object]:
    entry = failures.declared_failure_catalog()[category]
    return {
        "schema_version": failures.RECOVERY_FAILURE_DESCRIPTOR_SCHEMA_VERSION,
        "task_identity": failures.PP_BOX_TASK_IDENTITY,
        "category": category,
        "stage": entry.initial_stage,
        "entities": ("box", "shelf_target", "bimanual_ee"),
        "confidence": 1.0,
        "reward_mask": {"distance": True, "grasp": True, "placement": True},
        "failure_seed": 17,
        "snapshot_digest": "a" * 64,
    }


def _attempt_payload(failures, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": failures.RECOVERY_ATTEMPT_SCHEMA_VERSION,
        "trigger_kind": "pickup-attempt",
        "anchor_id": "anchor-17",
        "anchor_digest": "b" * 64,
        "failure_seed": 17,
        "phase": "acquire",
        "phase_enter_step": 20,
        "attempt_count": 1,
        "pickup_attempted": True,
        "place_attempted": False,
        "release_attempted": False,
        "last_progress_step": 22,
        "no_progress_steps": 8,
        "stall_confirm_steps": 8,
        "stable_steps": 5,
        "stable_confirm_steps": 5,
        "injected_category": "failed-grasp",
        "transform_digest": "c" * 64,
    }
    payload.update(overrides)
    return payload


def _hand_contact(telemetry, side: str, active: bool):
    link = telemetry.pairwise_contact_evidence(
        (0.0, 0.0, 2.0 if active else 0.0),
        sensor_body=f"{side}_hand_palm_link",
        filtered_body="Box",
        threshold_n=1.0,
    )
    return telemetry.aggregate_hand_contact_evidence(side, (link,))


def _telemetry_state(
    telemetry,
    *,
    env_index: int = 0,
    grasp: bool = False,
    pose_valid: bool = True,
    ee_distance_m: float = 0.15,
    xy_mismatch_m: float = 0.3,
    z_mismatch_m: float = 0.0,
    terminal_reason: str = "running",
    fall_candidate: bool = False,
):
    success = terminal_reason == "success"
    if success:
        xy_mismatch_m = 0.0
        z_mismatch_m = 0.0
    support = SimpleNamespace(
        support_bounds_w=(-1.0, 1.0, -0.5, 0.5),
        support_top_z_m=0.4,
        target_support_top_z_m=0.4,
        xy_mismatch_m=xy_mismatch_m,
        z_mismatch_m=z_mismatch_m,
        placed=success,
    )
    box_center = (0.0, 0.0, 0.505)
    left_pose = (
        ee_distance_m if pose_valid else 1.0,
        0.0,
        0.505,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    right_pose = (
        -ee_distance_m if pose_valid else -1.0,
        0.0,
        0.505,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    is_fall = terminal_reason == "fall"
    fall_streak = 5 if is_fall else (1 if fall_candidate else 0)
    control_step = 2000 if terminal_reason == "time_limit" else 20
    root_quat = (
        (0.0, 1.0, 0.0, 0.0) if (is_fall or fall_candidate) else (1.0, 0.0, 0.0, 0.0)
    )
    root_up_alignment = telemetry.compute_root_up_alignment(root_quat)
    live_fall_evidence = telemetry.LiveFallLaneEvidence(
        env_index=env_index,
        control_step_count=control_step,
        root_quat_wxyz=root_quat,
        root_up_alignment=root_up_alignment,
        critical_body_contact=False,
        fall_candidate=telemetry.classify_fall(
            root_up_alignment,
            critical_body_contact=False,
        ),
    )
    return telemetry.build_privileged_telemetry(
        task_identity=telemetry.PP_BOX_TASK_IDENTITY,
        env_index=env_index,
        box_center_w=box_center,
        box_linear_velocity_w=(0.0, 0.0, 0.0),
        box_angular_velocity_w=(0.0, 0.0, 0.0),
        support=support,
        left_ee_pose_w=left_pose,
        right_ee_pose_w=right_pose,
        left_contact=_hand_contact(telemetry, "left", grasp),
        right_contact=_hand_contact(telemetry, "right", grasp),
        live_fall_evidence=live_fall_evidence,
        terminal_context=telemetry.DriverTerminalContext(
            control_step_count=control_step,
            max_control_steps=2000,
            fall_streak=fall_streak,
            fall_confirm_steps=5,
            time_limit=terminal_reason == "time_limit",
            fall_confirmed=is_fall,
        ),
        max_ee_box_distance_m=0.3,
    )


def _predicate_context(
    failures,
    telemetry,
    *,
    attempt_overrides: dict[str, object] | None = None,
    ground_supported: bool = False,
    target_disjoint: bool = True,
    box_axis_aligned: bool = True,
    **telemetry_overrides: object,
):
    return failures.FailurePredicateContext(
        telemetry=_telemetry_state(telemetry, **telemetry_overrides),
        attempt=failures.RecoveryAttemptEvidence.from_mapping(
            _attempt_payload(failures, **(attempt_overrides or {}))
        ),
        ground_supported=ground_supported,
        target_disjoint=target_disjoint,
        box_axis_aligned=box_axis_aligned,
    )


def _live_geometry(failures, *, ground: bool = False):
    return failures.LiveSupportGeometry(
        schema_version=failures.RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION,
        geometry_id="ground-live-0" if ground else "shelf-live-0",
        geometry_digest=("d" if ground else "e") * 64,
        bounds_w=(-1.5, 1.5, -1.0, 1.0) if ground else (-1.0, 1.0, -0.5, 0.5),
        top_z_m=0.0 if ground else 0.4,
        source="live-stage",
        target_disjoint=ground,
    )


def _anchor(failures, category: str):
    kind = (
        "verified-pickup-anchor"
        if category == "failed-grasp"
        else "verified-grasp-preserving-anchor"
    )
    return failures.VerifiedFailureAnchor(
        schema_version=failures.RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION,
        category=category,
        anchor_id=f"{category}-anchor-0",
        anchor_digest=("f" if category == "failed-grasp" else "1") * 64,
        kind=kind,
        state_transform={
            "box_position_w": (0.0, 0.0, 0.505),
            "box_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "box_linear_velocity_w": (0.0, 0.0, 0.0),
            "box_angular_velocity_w": (0.0, 0.0, 0.0),
        },
        predicate_verified=True,
    )


def _runtime_evidence(
    failures,
    category: str,
    *,
    capabilities: frozenset[str] | None = None,
    include_category_resource: bool = True,
):
    return failures.FailureRuntimeCapabilityEvidence(
        schema_version=failures.RECOVERY_RUNTIME_CAPABILITY_SCHEMA_VERSION,
        task_identity=failures.PP_BOX_TASK_IDENTITY,
        category=category,
        validated_capabilities=(
            failures.required_runtime_capabilities(category)
            if capabilities is None
            else capabilities
        ),
        evidence_id=f"runtime-{category}-0",
        target_shelf=_live_geometry(failures),
        ground_support=(
            _live_geometry(failures, ground=True)
            if category == "dropped" and include_category_resource
            else None
        ),
        verified_anchor=(
            _anchor(failures, category)
            if category in {"failed-grasp", "misaligned"} and include_category_resource
            else None
        ),
    )


def _replay_context(failures, telemetry, plan):
    attempt_overrides: dict[str, object] = {
        "failure_seed": plan.failure_seed,
        "injected_category": plan.category,
        "transform_digest": plan.transform_digest,
    }
    telemetry_overrides: dict[str, object] = {}
    context_overrides: dict[str, object] = {}
    if plan.category == "dropped":
        attempt_overrides.update(
            trigger_kind="post-release",
            pickup_attempted=True,
            release_attempted=True,
        )
        telemetry_overrides.update(grasp=False, pose_valid=False, xy_mismatch_m=0.3)
        context_overrides.update(ground_supported=True, target_disjoint=True)
    elif plan.category == "failed-grasp":
        attempt_overrides.update(
            trigger_kind="pickup-attempt",
            pickup_attempted=True,
        )
        telemetry_overrides.update(grasp=False, pose_valid=True, xy_mismatch_m=0.3)
    elif plan.category == "misaligned":
        attempt_overrides.update(
            trigger_kind="place-attempt",
            pickup_attempted=True,
            place_attempted=True,
        )
        telemetry_overrides.update(grasp=True, pose_valid=True, xy_mismatch_m=0.2)
    else:
        attempt_overrides.update(
            trigger_kind="post-release",
            pickup_attempted=True,
            place_attempted=True,
            release_attempted=True,
        )
        telemetry_overrides.update(grasp=False, pose_valid=False, xy_mismatch_m=0.04)
    return _predicate_context(
        failures,
        telemetry,
        attempt_overrides=attempt_overrides,
        **telemetry_overrides,
        **context_overrides,
    )


def _recovery_snapshot(state):
    return state.RecoveryStateSnapshot(
        schema_version=state.RECOVERY_STATE_SCHEMA_VERSION,
        task_identity=state.PP_BOX_TASK_IDENTITY,
        capabilities=state.RecoveryStateCapabilities(
            schema_version=state.RECOVERY_STATE_SCHEMA_VERSION,
            available={"task_state": False},
        ),
        scene_state={
            "box": {"root_state": np.arange(13, dtype=np.float32).reshape(1, 13)},
            "robot": {"joint_pos": torch.tensor([[0.1, 0.2]], dtype=torch.float32)},
        },
        task_counters={"episode_length_buf": torch.tensor([7], dtype=torch.int64)},
        task_state=None,
        rng_state=state.RecoveryRngState(
            python=random.Random(101).getstate(),
            numpy=np.random.RandomState(202).get_state(),
            torch_cpu=torch.Generator().manual_seed(303).get_state(),
            torch_cuda=None,
            task_local=None,
            wrapper=None,
        ),
        runtime_state={},
    )


def test_declared_catalog_is_four_candidates_but_effective_catalog_defaults_empty(
    modules,
) -> None:
    _state, _telemetry, failures = modules

    declared = failures.declared_failure_catalog()

    assert set(declared) == DECLARED_CATEGORIES
    assert failures.effective_failure_catalog() == {}
    assert all(entry.declared for entry in declared.values())
    assert all(entry.recoverable for entry in declared.values())


def test_descriptor_parser_requires_exact_versioned_schema_and_catalog_values(
    modules,
) -> None:
    _state, _telemetry, failures = modules
    payload = _descriptor_payload(failures)

    descriptor = failures.RecoveryFailureDescriptor.from_mapping(payload)

    assert descriptor.category == "dropped"
    assert descriptor.failure_seed == 17
    assert descriptor.snapshot_digest == "a" * 64

    for mutation, message in (
        ({**payload, "extra": True}, "descriptor schema"),
        ({**payload, "schema_version": 999}, "schema version"),
        ({**payload, "category": "fall"}, "category"),
        ({**payload, "stage": "place"}, "stage"),
        ({**payload, "reward_mask": {"distance": True}}, "reward mask"),
    ):
        with pytest.raises(failures.RecoveryFailureSchemaError, match=message):
            failures.RecoveryFailureDescriptor.from_mapping(mutation)


def test_attempt_evidence_is_versioned_strict_and_counter_consistent(modules) -> None:
    _state, _telemetry, failures = modules
    payload = _attempt_payload(failures)

    evidence = failures.RecoveryAttemptEvidence.from_mapping(payload)

    assert evidence.stalled is True
    assert evidence.stable is True
    assert evidence.pickup_attempted is True

    for mutation, message in (
        ({**payload, "unknown": 1}, "attempt evidence schema"),
        ({**payload, "schema_version": 999}, "schema version"),
        ({**payload, "no_progress_steps": -1}, "non-negative"),
        ({**payload, "phase_enter_step": 23}, "progress step"),
        ({**payload, "injected_category": "fall"}, "category"),
    ):
        with pytest.raises(failures.RecoveryFailureSchemaError, match=message):
            failures.RecoveryAttemptEvidence.from_mapping(mutation)


@pytest.mark.parametrize(
    ("category", "telemetry_overrides", "attempt_overrides", "context_overrides"),
    [
        (
            "dropped",
            {"grasp": False, "pose_valid": False, "xy_mismatch_m": 0.3},
            {
                "trigger_kind": "post-release",
                "pickup_attempted": True,
                "release_attempted": True,
                "injected_category": "dropped",
            },
            {"ground_supported": True, "target_disjoint": True},
        ),
        (
            "failed-grasp",
            {"grasp": False, "pose_valid": True, "xy_mismatch_m": 0.3},
            {
                "trigger_kind": "pickup-attempt",
                "pickup_attempted": True,
                "injected_category": "failed-grasp",
            },
            {},
        ),
        (
            "misaligned",
            {"grasp": True, "pose_valid": True, "xy_mismatch_m": 0.2},
            {
                "trigger_kind": "place-attempt",
                "pickup_attempted": True,
                "place_attempted": True,
                "injected_category": "misaligned",
            },
            {},
        ),
        (
            "near-shelf-misplaced",
            {"grasp": False, "pose_valid": False, "xy_mismatch_m": 0.04},
            {
                "trigger_kind": "post-release",
                "pickup_attempted": True,
                "place_attempted": True,
                "release_attempted": True,
                "injected_category": "near-shelf-misplaced",
            },
            {},
        ),
    ],
)
def test_recoverable_failure_predicates_are_source_decidable_and_mutually_exclusive(
    modules,
    category: str,
    telemetry_overrides: dict[str, object],
    attempt_overrides: dict[str, object],
    context_overrides: dict[str, object],
) -> None:
    _state, telemetry, failures = modules
    context = _predicate_context(
        failures,
        telemetry,
        attempt_overrides=attempt_overrides,
        **telemetry_overrides,
        **context_overrides,
    )

    matches = failures.evaluate_failure_predicates(context)

    assert matches == {
        candidate: candidate == category for candidate in DECLARED_CATEGORIES
    }
    assert failures.classify_recoverable_failure(context) == category


def test_conflicting_ground_and_near_shelf_truth_fails_closed(modules) -> None:
    _state, telemetry, failures = modules
    context = _predicate_context(
        failures,
        telemetry,
        attempt_overrides={
            "trigger_kind": "post-release",
            "release_attempted": True,
            "injected_category": "near-shelf-misplaced",
        },
        grasp=False,
        pose_valid=False,
        xy_mismatch_m=0.04,
        ground_supported=True,
        target_disjoint=True,
    )

    with pytest.raises(
        failures.RecoveryFailurePredicateConflictError, match="conflict"
    ):
        failures.classify_recoverable_failure(context)


@pytest.mark.parametrize(
    ("terminal_reason", "fall_candidate"),
    [
        ("success", False),
        ("fall", False),
        ("time_limit", False),
        ("running", True),
    ],
)
def test_terminal_and_fall_candidate_states_are_never_recovery_experts(
    modules,
    terminal_reason: str,
    fall_candidate: bool,
) -> None:
    _state, telemetry, failures = modules
    context = _predicate_context(
        failures,
        telemetry,
        attempt_overrides={
            "trigger_kind": "post-release",
            "release_attempted": True,
            "injected_category": "dropped",
        },
        ground_supported=True,
        xy_mismatch_m=0.04,
        terminal_reason=terminal_reason,
        fall_candidate=fall_candidate,
    )

    assert not any(failures.evaluate_failure_predicates(context).values())
    assert failures.classify_recoverable_failure(context) is None


def test_recovery_success_is_exactly_the_task_success_terminal(modules) -> None:
    _state, telemetry, failures = modules
    success = _telemetry_state(telemetry, terminal_reason="success")
    running = _telemetry_state(telemetry, terminal_reason="running")

    assert failures.recovery_succeeded(success) is True
    assert failures.recovery_succeeded(running) is False

    contradictory = replace(running, placement=True, success=True)
    with pytest.raises(failures.RecoveryFailureSchemaError, match="terminal"):
        failures.recovery_succeeded(contradictory)


def test_recovery_reward_api_has_no_persistent_stage_or_fsm_surface(modules) -> None:
    _state, _telemetry, failures = modules
    tree = ast.parse(FAILURES_PATH.read_text(encoding="utf-8"))
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    forbidden = {
        "RecoveryStageSpec",
        "recovery_stage_fsm",
        "stage_gates",
        "current_stage",
        "fsm_stage",
    }

    assert forbidden.isdisjoint(identifiers)
    assert all(not hasattr(failures, name) for name in forbidden)


def test_reward_bindings_are_task_truth_for_distance_grasp_and_placement_only(
    modules,
) -> None:
    _state, _telemetry, failures = modules

    bindings = failures.recovery_reward_bindings()

    assert tuple(binding.term for binding in bindings) == (
        "distance",
        "grasp",
        "placement",
    )
    assert tuple(binding.stage for binding in bindings) == (
        "approach",
        "acquire",
        "place",
    )
    assert bindings[0].entity_roles == {"source": "bimanual_ee", "target": "box"}
    assert bindings[0].telemetry_binding == "max_bimanual_ee_box_distance_m"
    assert bindings[1].entity_roles == {"end_effector": "bimanual_ee", "object": "box"}
    assert bindings[1].telemetry_binding == "bimanual_grasp"
    assert bindings[2].entity_roles == {"object": "box", "target": "shelf_target"}
    assert bindings[2].telemetry_binding == "hypot_xy_z_mismatch"
    assert "articulation" not in {binding.term for binding in bindings}
    assert all(
        not hasattr(entry, "stage_fsm") and entry.reward_bindings == bindings
        for entry in failures.declared_failure_catalog().values()
    )


def test_recovery_activation_freezes_d_init_across_primitive_steps(
    modules,
) -> None:
    _state, telemetry, failures = modules
    initial = _telemetry_state(
        telemetry,
        grasp=False,
        pose_valid=True,
        ee_distance_m=0.4,
        xy_mismatch_m=0.3,
        z_mismatch_m=0.4,
    )
    activation = failures.begin_recovery_activation(
        activation_id="recovery-17",
        telemetry_by_env={0: initial},
    )
    bindings = {
        binding.term: binding for binding in failures.recovery_reward_bindings()
    }
    scopes = {
        term: failures.recovery_reward_component_scope(binding)
        for term, binding in bindings.items()
    }
    assert scopes == {
        "distance": ('["distance",[["source","bimanual_ee"],["target","box"]]]'),
        "grasp": ('["grasp",[["end_effector","bimanual_ee"],["object","box"]]]'),
        "placement": ('["placement",[["object","box"],["target","shelf_target"]]]'),
    }
    for current_ee, current_xy, current_z in (
        (0.3, 0.24, 0.18),
        (0.2, 0.12, 0.05),
        (0.1, 0.04, 0.03),
    ):
        resolved = failures.resolve_recovery_reward_telemetry(
            _telemetry_state(
                telemetry,
                grasp=current_ee < 0.25,
                ee_distance_m=current_ee,
                xy_mismatch_m=current_xy,
                z_mismatch_m=current_z,
            ),
            activation_context=activation,
            env_index=0,
        )
        distance = resolved["components"][scopes["distance"]]
        placement = resolved["components"][scopes["placement"]]
        assert distance["distance"] == pytest.approx(current_ee)
        assert distance["d_init"] == pytest.approx(0.4)
        assert placement["distance"] == pytest.approx(
            math.hypot(current_xy, current_z)
        )
        assert placement["d_init"] == pytest.approx(0.5)
        assert len(resolved["components"]) == 3
        assert "articulation" not in repr(resolved).lower()
        assert "rgb" not in repr(resolved).lower()
        assert "vlm" not in repr(resolved).lower()


def test_repeated_begin_keeps_baseline_but_new_activation_refreshes_it(
    modules,
) -> None:
    _state, telemetry, failures = modules
    first = failures.begin_recovery_activation(
        activation_id="recovery-17",
        telemetry_by_env={
            0: _telemetry_state(
                telemetry,
                ee_distance_m=0.4,
                xy_mismatch_m=0.3,
                z_mismatch_m=0.4,
            )
        },
    )
    repeated = failures.begin_recovery_activation(
        activation_id="recovery-17",
        telemetry_by_env={
            0: _telemetry_state(
                telemetry,
                ee_distance_m=0.1,
                xy_mismatch_m=0.06,
                z_mismatch_m=0.08,
            )
        },
        active_context=first,
    )
    second = failures.begin_recovery_activation(
        activation_id="recovery-18",
        telemetry_by_env={
            0: _telemetry_state(
                telemetry,
                ee_distance_m=0.1,
                xy_mismatch_m=0.06,
                z_mismatch_m=0.08,
            )
        },
        active_context=first,
    )

    assert repeated is first
    assert first.baselines[0].distance_d_init_m == pytest.approx(0.4)
    assert first.baselines[0].placement_d_init_m == pytest.approx(0.5)
    assert second.activation_id == "recovery-18"
    assert second.baselines[0].distance_d_init_m == pytest.approx(0.1)
    assert second.baselines[0].placement_d_init_m == pytest.approx(0.1)
    with pytest.raises(TypeError):
        failures.RecoveryActivationContext(
            schema_version=failures.RECOVERY_ACTIVATION_SCHEMA_VERSION,
            task_identity=failures.PP_BOX_TASK_IDENTITY,
            activation_id="recovery-forged",
            baselines=first.baselines,
        )
    with pytest.raises(TypeError):
        replace(first, activation_id="recovery-forged")


def test_activation_baselines_are_lane_bound_and_context_is_mandatory(modules) -> None:
    _state, telemetry, failures = modules
    activation = failures.begin_recovery_activation(
        activation_id="recovery-multi-env",
        telemetry_by_env={
            0: _telemetry_state(
                telemetry,
                env_index=0,
                ee_distance_m=0.4,
                xy_mismatch_m=0.3,
                z_mismatch_m=0.4,
            ),
            1: _telemetry_state(
                telemetry,
                env_index=1,
                ee_distance_m=0.2,
                xy_mismatch_m=0.12,
                z_mismatch_m=0.05,
            ),
        },
    )

    assert activation.baselines[0].distance_d_init_m == pytest.approx(0.4)
    assert activation.baselines[0].placement_d_init_m == pytest.approx(0.5)
    assert activation.baselines[1].distance_d_init_m == pytest.approx(0.2)
    assert activation.baselines[1].placement_d_init_m == pytest.approx(0.13)
    with pytest.raises(failures.RecoveryFailureSchemaError, match="activation context"):
        failures.resolve_recovery_reward_telemetry(
            _telemetry_state(telemetry),
            activation_context=None,
            env_index=0,
        )
    with pytest.raises(failures.RecoveryFailureSchemaError, match="env index"):
        failures.resolve_recovery_reward_telemetry(
            _telemetry_state(telemetry, env_index=1),
            activation_context=activation,
            env_index=0,
        )


def test_reward_resolver_emits_stateless_gate_truth(modules) -> None:
    _state, telemetry, failures = modules
    initial = _telemetry_state(telemetry, grasp=False, pose_valid=True)
    activation = failures.begin_recovery_activation(
        activation_id="recovery-gates",
        telemetry_by_env={0: initial},
    )

    def resolve(**overrides: object):
        return failures.resolve_recovery_reward_telemetry(
            _telemetry_state(telemetry, **overrides),
            activation_context=activation,
            env_index=0,
        )["gate_truth"]

    pose_without_grasp = resolve(grasp=False, pose_valid=True)
    grasped = resolve(grasp=True, pose_valid=True)
    no_pose = resolve(grasp=False, pose_valid=False)
    falling = resolve(grasp=False, pose_valid=True, fall_candidate=True)

    assert pose_without_grasp["running_and_not_grasp"] is True
    assert pose_without_grasp["running_and_not_grasp_and_bimanual_pose"] is True
    assert pose_without_grasp["grasp"] is False
    assert grasped["grasp"] is True
    assert grasped["running_and_grasp"] is True
    assert no_pose["lost_bimanual_pose"] is True
    assert no_pose["lost_grasp_without_bimanual_pose"] is True
    assert falling["running_and_not_grasp"] is False
    assert falling["running_and_not_grasp_and_bimanual_pose"] is False
    assert falling["running_and_grasp"] is False


def test_recovery_geometry_constants_are_the_public_reward_source_of_truth(
    modules,
) -> None:
    _state, _telemetry, failures = modules

    assert failures.BOX_HALF_EXTENTS_M is failures.rewards.BOX_HALF_EXTENTS_M
    assert failures.BOX_HALF_EXTENT_M == failures.rewards.BOX_HALF_EXTENTS_M[0]
    assert (
        failures.PLACEMENT_Z_TOLERANCE_M
        == failures.rewards.BOX_BOTTOM_SURFACE_TOLERANCE_M
    )


@pytest.mark.parametrize("category", sorted(DECLARED_CATEGORIES))
def test_each_category_requires_all_declared_runtime_capabilities(
    modules, category: str
) -> None:
    _state, _telemetry, failures = modules
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, category)
    )
    required = failures.required_runtime_capabilities(category)
    missing_name = min(required)
    evidence = _runtime_evidence(
        failures,
        category,
        capabilities=frozenset(required - {missing_name}),
    )

    with pytest.raises(failures.RecoveryFailureCapabilityError) as exc_info:
        failures.build_failure_injection_plan(descriptor, evidence)

    assert exc_info.value.category == category
    assert exc_info.value.missing_capabilities == (missing_name,)


@pytest.mark.parametrize("category", ["dropped", "failed-grasp", "misaligned"])
def test_category_specific_geometry_or_anchor_is_mandatory(
    modules, category: str
) -> None:
    _state, _telemetry, failures = modules
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, category)
    )
    evidence = _runtime_evidence(
        failures,
        category,
        include_category_resource=False,
    )

    with pytest.raises(failures.RecoveryFailureCapabilityError) as exc_info:
        failures.build_failure_injection_plan(descriptor, evidence)

    expected = {
        "dropped": "live_ground_support",
        "failed-grasp": "verified_pickup_anchor",
        "misaligned": "verified_grasp_preserving_anchor",
    }[category]
    assert expected in exc_info.value.missing_capabilities


def test_category_local_seed_and_plan_digest_are_reproducible_without_resampling(
    modules,
) -> None:
    _state, _telemetry, failures = modules
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, "near-shelf-misplaced")
    )
    evidence = _runtime_evidence(failures, "near-shelf-misplaced")

    first = failures.build_failure_injection_plan(descriptor, evidence)
    second = failures.build_failure_injection_plan(descriptor, evidence)
    other_seed = failures.build_failure_injection_plan(
        replace(descriptor, failure_seed=18),
        evidence,
    )

    assert first == second
    assert first.category_seed == second.category_seed
    assert first.transform_digest == second.transform_digest
    assert first != other_seed
    assert first.category_seed != other_seed.category_seed
    assert first.transform_digest != other_seed.transform_digest
    assert failures.derive_category_seed(
        "dropped", 17, "a" * 64
    ) != failures.derive_category_seed("near-shelf-misplaced", 17, "a" * 64)


def test_near_shelf_plan_uses_live_inner_bounds_and_axis_aligned_zero_dynamics(
    modules,
) -> None:
    _state, _telemetry, failures = modules
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, "near-shelf-misplaced")
    )
    evidence = _runtime_evidence(failures, "near-shelf-misplaced")

    plan = failures.build_failure_injection_plan(descriptor, evidence)

    x, y, z = plan.state_transform["box_position_w"]
    x_lo, x_hi, y_lo, y_hi = evidence.target_shelf.bounds_w
    half = failures.BOX_HALF_EXTENT_M
    dx = max(x_lo + half - x, 0.0, x - (x_hi - half))
    dy = max(y_lo + half - y, 0.0, y - (y_hi - half))
    assert 0.0 < (dx * dx + dy * dy) ** 0.5 < half
    assert z == pytest.approx(evidence.target_shelf.top_z_m + half)
    assert plan.state_transform["box_orientation_wxyz"] == (1.0, 0.0, 0.0, 0.0)
    assert plan.state_transform["box_linear_velocity_w"] == (0.0, 0.0, 0.0)
    assert plan.state_transform["box_angular_velocity_w"] == (0.0, 0.0, 0.0)


def test_dropped_plan_requires_live_disjoint_ground_and_zeroes_box_dynamics(
    modules,
) -> None:
    _state, _telemetry, failures = modules
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, "dropped")
    )
    evidence = _runtime_evidence(failures, "dropped")

    plan = failures.build_failure_injection_plan(descriptor, evidence)

    x, y, z = plan.state_transform["box_position_w"]
    x_lo, x_hi, y_lo, y_hi = evidence.ground_support.bounds_w
    half = failures.BOX_HALF_EXTENT_M
    assert x_lo + half <= x <= x_hi - half
    assert y_lo + half <= y <= y_hi - half
    assert z == pytest.approx(evidence.ground_support.top_z_m + half)
    assert plan.transform_kind == "live-ground-box-root"
    assert plan.state_transform["box_linear_velocity_w"] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("category", "kind"),
    [
        ("failed-grasp", "verified-pickup-anchor"),
        ("misaligned", "verified-grasp-preserving-anchor"),
    ],
)
def test_anchor_categories_reuse_only_the_verified_anchor_transform(
    modules,
    category: str,
    kind: str,
) -> None:
    _state, _telemetry, failures = modules
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, category)
    )
    evidence = _runtime_evidence(failures, category)

    plan = failures.build_failure_injection_plan(descriptor, evidence)

    assert plan.transform_kind == kind
    assert plan.anchor_id == evidence.verified_anchor.anchor_id
    assert plan.anchor_digest == evidence.verified_anchor.anchor_digest
    assert plan.state_transform == evidence.verified_anchor.state_transform


def test_runtime_evidence_digest_is_content_derived(
    modules,
) -> None:
    _state, _telemetry, failures = modules
    category = "misaligned"
    evidence = _runtime_evidence(failures, category)
    identical = _runtime_evidence(failures, category)

    assert evidence.evidence_digest == identical.evidence_digest

    shifted_shelf = replace(
        evidence.target_shelf,
        bounds_w=(-0.9, 1.1, -0.5, 0.5),
    )
    shifted_anchor = replace(
        evidence.verified_anchor,
        state_transform={
            **dict(evidence.verified_anchor.state_transform),
            "box_position_w": (0.01, 0.0, 0.505),
        },
    )
    reduced_capabilities = frozenset(
        evidence.validated_capabilities - {"verified_grasp_preserving_anchor"}
    )
    mutations = (
        replace(evidence, target_shelf=shifted_shelf),
        replace(evidence, verified_anchor=shifted_anchor),
        replace(evidence, validated_capabilities=reduced_capabilities),
    )

    for mutated in mutations:
        assert mutated.evidence_digest != evidence.evidence_digest


class _RawSceneTrapEnv:
    @property
    def scene(self):
        raise AssertionError("failure injector must not mutate raw scene assets")


class _InjectionHooks:
    def __init__(self, events: list[str], readback) -> None:
        self.events = events
        self._readback = readback
        self.write_count = 0
        self.settle_count = 0
        self.read_count = 0

    def write_recovery_failure(self, plan) -> None:
        self.write_count += 1
        self.events.append("write")

    def settle_recovery_failure(self, plan) -> None:
        self.settle_count += 1
        self.events.append("settle")

    def read_recovery_failure(self, plan):
        self.read_count += 1
        self.events.append("readback")
        return self._readback(plan)


class _StaticDigestAttestedSnapshotEndpoint:
    def __init__(self, state, events: list[str], *, restore_attestation=None) -> None:
        self.state = state
        self.events = events
        self.restore_attestation = restore_attestation
        self.digest_calls = []
        self.restore_calls = []

    def recovery_failure_snapshot_digest(self, snapshot):
        digest = self.state.recovery_state_digest(snapshot)
        self.digest_calls.append(digest)
        return digest

    def restore_recovery_failure_snapshot(self, snapshot, *, snapshot_digest):
        actual = self.state.recovery_state_digest(snapshot)
        self.restore_calls.append((snapshot_digest, actual))
        self.events.append("restore")
        return actual if self.restore_attestation is None else self.restore_attestation


class _QualificationContinuationRunner:
    def __init__(
        self,
        modules,
        events: list[str],
        *,
        drift_second_execution: bool = False,
        skip_primitive_step: bool = False,
        applied_action_offset: float = 0.0,
    ) -> None:
        self.state, self.telemetry, self.failures = modules
        self.events = events
        self.step_count = 0
        self.drift_second_execution = drift_second_execution
        self.skip_primitive_step = skip_primitive_step
        self.applied_action_offset = applied_action_offset

    def execute_qualification_primitive_step(self, fixed_action40):
        self.step_count += 1
        self.events.append("primitive_step")
        applied_action40 = list(fixed_action40)
        applied_action40[20] += self.applied_action_offset
        return tuple(applied_action40)

    def run_failure_continuation(self, plan, fixed_action40, primitive_step):
        applied_action40 = (
            fixed_action40 if self.skip_primitive_step else primitive_step()
        )
        continuation_readback = _replay_context(self.failures, self.telemetry, plan)
        self.events.append("continuation_readback")

        def rng_state(seed: int):
            return self.state.RecoveryRngState(
                python=random.Random(seed).getstate(),
                numpy=np.random.RandomState(seed + 1).get_state(),
                torch_cpu=torch.Generator().manual_seed(seed + 2).get_state(),
                torch_cuda=None,
                task_local=None,
                wrapper=None,
            )

        def contacts(offset: float):
            return tuple(
                {
                    "sensor_index": index,
                    "sensor_scene_key": f"hand_box_contact_{index}",
                    "force_matrix_w": torch.tensor(
                        [[offset + float(index), 0.0, 1.0]], dtype=torch.float32
                    ),
                }
                for index in range(16)
            )

        return self.failures.FailureContinuationRaw(
            schema_version=self.failures.RECOVERY_RAW_RECEIPT_SCHEMA_VERSION,
            env_index=0,
            runtime_identity_digest="9" * 64,
            fixed_action40=fixed_action40,
            applied_action40=applied_action40,
            observation_before={
                "policy": torch.arange(6, dtype=torch.float32).reshape(1, 6)
            },
            observation_after={
                "policy": torch.arange(6, dtype=torch.float32).reshape(1, 6)
                + 1.0
                + float(self.drift_second_execution and self.step_count == 2)
            },
            reward=1.25,
            reward_terms={
                "distance": 0.1,
                "grasp": 0.2,
                "placement": 0.3,
                "articulation": 0.0,
            },
            terminated=False,
            truncated=False,
            terminal_context=self.telemetry.DriverTerminalContext(
                control_step_count=21,
                max_control_steps=2000,
                fall_streak=0,
                fall_confirm_steps=5,
                time_limit=False,
                fall_confirmed=False,
            ),
            task_state_before={
                "episode_length_buf": torch.tensor([20]),
                "hand_state": torch.tensor([[0.0, 1.0]]),
            },
            task_state_after={
                "episode_length_buf": torch.tensor([21]),
                "hand_state": torch.tensor([[0.0, 1.0]]),
            },
            rng_before=rng_state(301),
            rng_after=rng_state(302),
            contact16_before=contacts(0.0),
            contact16_after=contacts(0.5),
            live_fall_evidence_before={"producer_step": 20, "fall": False},
            live_fall_evidence_after={"producer_step": 21, "fall": False},
            continuation_readback=continuation_readback,
        )


def _execute_qualification(
    modules,
    _monkeypatch,
    *,
    category="near-shelf-misplaced",
    drift_second_execution: bool = False,
    skip_primitive_step: bool = False,
    applied_action_offset: float = 0.0,
    descriptor_confidence: float = 1.0,
    descriptor_reward_mask: dict[str, bool] | None = None,
    events: list[str] | None = None,
):
    state, telemetry, failures = modules
    snapshot = _recovery_snapshot(state)
    descriptor = replace(
        failures.RecoveryFailureDescriptor.from_mapping(
            _descriptor_payload(failures, category)
        ),
        snapshot_digest=state.recovery_state_digest(snapshot),
        confidence=descriptor_confidence,
        reward_mask=(
            {"distance": True, "grasp": True, "placement": True}
            if descriptor_reward_mask is None
            else descriptor_reward_mask
        ),
    )
    evidence = _runtime_evidence(failures, category)
    events = [] if events is None else events
    snapshot_endpoint = _StaticDigestAttestedSnapshotEndpoint(state, events)

    def readback(plan):
        return _replay_context(failures, telemetry, plan)

    hooks = _InjectionHooks(events, readback)
    runner = _QualificationContinuationRunner(
        modules,
        events,
        drift_second_execution=drift_second_execution,
        skip_primitive_step=skip_primitive_step,
        applied_action_offset=applied_action_offset,
    )
    runner.snapshot_endpoint = snapshot_endpoint
    fixed_action40 = tuple(float(index) / 100.0 for index in range(40))
    qualification = failures.qualify_failure_catalog_entry(
        env=_RawSceneTrapEnv(),
        snapshot=snapshot,
        descriptor=descriptor,
        runtime_evidence=evidence,
        snapshot_endpoint=snapshot_endpoint,
        writer=hooks,
        settler=hooks,
        reader=hooks,
        continuation_runner=runner,
        qualification_step_executor=runner,
        fixed_action40=fixed_action40,
    )
    return qualification, evidence, events, hooks, runner


def test_qualification_rejects_zero_step_self_report(
    modules,
    monkeypatch,
) -> None:
    _state, _telemetry, failures = modules

    with pytest.raises(
        failures.RecoveryFailureVerificationError,
        match="primitive step",
    ):
        _execute_qualification(
            modules,
            monkeypatch,
            skip_primitive_step=True,
        )


def test_qualification_rejects_applied_action_that_differs_from_fixed_action(
    modules,
    monkeypatch,
) -> None:
    _state, _telemetry, failures = modules

    with pytest.raises(
        failures.RecoveryFailureSchemaError,
        match="applied action40",
    ):
        _execute_qualification(
            modules,
            monkeypatch,
            applied_action_offset=0.5,
        )


def test_qualification_step_protocol_excludes_residual_only_executor(modules) -> None:
    _state, _telemetry, failures = modules

    class _ResidualOnly:
        def execute_residual_primitive_step(self, base_action40, applied_action40):
            return applied_action40

    assert not isinstance(
        _ResidualOnly(),
        failures.RecoveryFailureQualificationStepExecutor,
    )


def test_qualification_binds_descriptor_confidence_and_reward_mask(
    modules,
    monkeypatch,
) -> None:
    _state, _telemetry, failures = modules
    reward_mask = {"distance": False, "grasp": True, "placement": False}
    qualification, _evidence, _events, _hooks, _runner = _execute_qualification(
        modules,
        monkeypatch,
        descriptor_confidence=0.25,
        descriptor_reward_mask=reward_mask,
    )
    plan = qualification.receipts[0].plan

    assert plan.descriptor_confidence == pytest.approx(0.25)
    assert dict(plan.descriptor_reward_mask) == reward_mask
    assert set(
        failures.effective_failure_catalog(
            {qualification.category: qualification}
        )
    ) == {qualification.category}

    object.__setattr__(plan, "descriptor_confidence", 1.0)
    assert failures.effective_failure_catalog(
        {qualification.category: qualification}
    ) == {}

    qualification, _evidence, _events, _hooks, _runner = _execute_qualification(
        modules,
        monkeypatch,
        descriptor_confidence=0.25,
        descriptor_reward_mask=reward_mask,
    )
    object.__setattr__(
        qualification.receipts[0].plan,
        "descriptor_reward_mask",
        {"distance": True, "grasp": True, "placement": True},
    )
    assert failures.effective_failure_catalog(
        {qualification.category: qualification}
    ) == {}


def test_effective_catalog_requires_factory_executed_canonical_receipts(
    modules,
    monkeypatch,
) -> None:
    _state, _telemetry, failures = modules
    qualification, evidence, events, hooks, runner = _execute_qualification(
        modules, monkeypatch
    )
    endpoint = runner.snapshot_endpoint

    assert not hasattr(evidence, "replay_records")
    assert len(qualification.receipts) == 2
    assert tuple(receipt.repeat_index for receipt in qualification.receipts) == (0, 1)
    assert events == [
        "restore",
        "write",
        "settle",
        "readback",
        "primitive_step",
        "continuation_readback",
    ] * 2
    assert (hooks.write_count, hooks.settle_count, hooks.read_count) == (2, 2, 2)
    assert runner.step_count == 2
    source_snapshot_digest = qualification.receipts[0].source_snapshot_digest
    assert endpoint.digest_calls == [source_snapshot_digest] * 2
    assert endpoint.restore_calls == [
        (source_snapshot_digest, source_snapshot_digest)
    ] * 2
    expected_trace = (
        "restore",
        "write",
        "settle",
        "readback",
        "primitive_step",
        "readback",
    )
    assert all(receipt.operation_trace == expected_trace for receipt in qualification.receipts)
    assert all(len(receipt.continuation.fixed_action40) == 40 for receipt in qualification.receipts)
    assert all(len(receipt.continuation.contact16_before) == 16 for receipt in qualification.receipts)
    assert all(len(receipt.continuation.contact16_after) == 16 for receipt in qualification.receipts)
    assert set(
        failures.effective_failure_catalog(
            {qualification.category: qualification}
        )
    ) == {qualification.category}
    with pytest.raises(failures.RecoveryFailureSchemaError, match="qualification"):
        failures.effective_failure_catalog({evidence.category: evidence})
    with pytest.raises(TypeError):
        failures.RawInjectorExecutionReceipt(
            schema_version=failures.RECOVERY_RAW_RECEIPT_SCHEMA_VERSION,
            repeat_index=0,
            task_identity=failures.PP_BOX_TASK_IDENTITY,
            env_index=0,
            runtime_identity_digest="9" * 64,
            source_snapshot_digest="a" * 64,
            plan=qualification.receipts[0].plan,
            injection_readback=qualification.receipts[0].injection_readback,
            continuation=qualification.receipts[0].continuation,
        )
    with pytest.raises(TypeError):
        failures.FailureCatalogQualification(
            schema_version=failures.RECOVERY_CATALOG_QUALIFICATION_SCHEMA_VERSION,
            task_identity=failures.PP_BOX_TASK_IDENTITY,
            category=qualification.category,
            runtime_evidence=evidence,
            receipts=qualification.receipts,
        )


@pytest.mark.parametrize("category", sorted(DECLARED_CATEGORIES))
def test_factory_qualification_activates_each_declared_category(
    modules,
    monkeypatch,
    category: str,
) -> None:
    _state, _telemetry, failures = modules
    qualification, _evidence, _events, _hooks, _runner = _execute_qualification(
        modules,
        monkeypatch,
        category=category,
    )

    assert set(failures.effective_failure_catalog({category: qualification})) == {
        category
    }


def test_factory_rejects_nonidentical_executions_only_after_both_real_runs(
    modules,
    monkeypatch,
) -> None:
    _state, _telemetry, failures = modules
    events: list[str] = []

    with pytest.raises(
        failures.RecoveryFailureVerificationError,
        match="not identical",
    ):
        _execute_qualification(
            modules,
            monkeypatch,
            drift_second_execution=True,
            events=events,
        )

    assert events == [
        "restore",
        "write",
        "settle",
        "readback",
        "primitive_step",
        "continuation_readback",
    ] * 2


@pytest.mark.parametrize(
    "payload_name",
    [
        "fixed_action40",
        "applied_action40",
        "observation_before",
        "observation_after",
        "reward",
        "reward_terms",
        "terminated",
        "truncated",
        "terminal_context",
        "task_state_before",
        "task_state_after",
        "rng_before",
        "rng_after",
        "contact16_before",
        "contact16_after",
        "live_fall_evidence_before",
        "live_fall_evidence_after",
        "continuation_readback",
    ],
)
def test_effective_catalog_rejects_any_raw_continuation_mutation(
    modules,
    monkeypatch,
    payload_name: str,
) -> None:
    state, telemetry, failures = modules
    qualification, _evidence, _events, _hooks, _runner = _execute_qualification(
        modules, monkeypatch
    )
    continuation = qualification.receipts[1].continuation
    if payload_name in {"fixed_action40", "applied_action40"}:
        value = list(getattr(continuation, payload_name))
        value[20] += 0.01
        mutated = tuple(value)
    elif payload_name in {"observation_before", "observation_after"}:
        mutated = {"policy": torch.full((1, 6), 99.0)}
    elif payload_name == "reward":
        mutated = 99.0
    elif payload_name == "reward_terms":
        mutated = {
            "distance": 9.0,
            "grasp": 0.2,
            "placement": 0.3,
            "articulation": 0.0,
        }
    elif payload_name in {"terminated", "truncated"}:
        mutated = True
    elif payload_name == "terminal_context":
        mutated = telemetry.DriverTerminalContext(
            control_step_count=22,
            max_control_steps=2000,
            fall_streak=0,
            fall_confirm_steps=5,
            time_limit=False,
            fall_confirmed=False,
        )
    elif payload_name in {"task_state_before", "task_state_after"}:
        mutated = {"episode_length_buf": torch.tensor([999]), "hand_state": torch.zeros(1, 2)}
    elif payload_name in {"rng_before", "rng_after"}:
        mutated = state.RecoveryRngState(
            python=random.Random(999).getstate(),
            numpy=np.random.RandomState(999).get_state(),
            torch_cpu=torch.Generator().manual_seed(999).get_state(),
            torch_cuda=None,
            task_local=None,
            wrapper=None,
        )
    elif payload_name in {"contact16_before", "contact16_after"}:
        mutated_rows = list(getattr(continuation, payload_name))
        mutated_rows[7] = {
            "sensor_index": 7,
            "sensor_scene_key": "hand_box_contact_7",
            "force_matrix_w": torch.tensor([[999.0, 0.0, 0.0]]),
        }
        mutated = tuple(mutated_rows)
    elif payload_name in {"live_fall_evidence_before", "live_fall_evidence_after"}:
        mutated = {"producer_step": 999, "fall": False}
    else:
        base = _replay_context(
            failures,
            telemetry,
            qualification.receipts[1].plan,
        )
        mutated = replace(base, telemetry=replace(base.telemetry, xy_mismatch_m=0.3))
    object.__setattr__(continuation, payload_name, mutated)

    assert (
        failures.effective_failure_catalog(
            {qualification.category: qualification}
        )
        == {}
    )


@pytest.mark.parametrize(
    "payload_name",
    [
        "operation_trace",
        "source_snapshot_digest",
        "injection_readback",
        "injection_readback_digest",
        "continuation_digest",
        "execution_digest",
        "receipt_digest",
    ],
)
def test_effective_catalog_rejects_receipt_or_cached_digest_forgery(
    modules,
    monkeypatch,
    payload_name: str,
) -> None:
    _state, _telemetry, failures = modules
    qualification, _evidence, _events, _hooks, _runner = _execute_qualification(
        modules,
        monkeypatch,
    )
    receipt = qualification.receipts[1]
    if payload_name == "operation_trace":
        mutated = ("restore", "write")
    elif payload_name == "source_snapshot_digest":
        mutated = "f" * 64
    elif payload_name == "injection_readback":
        base = receipt.injection_readback
        mutated = replace(base, telemetry=replace(base.telemetry, xy_mismatch_m=0.3))
    else:
        mutated = "f" * 64
    object.__setattr__(receipt, payload_name, mutated)

    assert failures.effective_failure_catalog({qualification.category: qualification}) == {}


@pytest.mark.parametrize("payload_name", ["qualification_digest", "evidence_digest"])
def test_effective_catalog_rejects_qualification_or_evidence_digest_forgery(
    modules,
    monkeypatch,
    payload_name: str,
) -> None:
    _state, _telemetry, failures = modules
    qualification, evidence, _events, _hooks, _runner = _execute_qualification(
        modules,
        monkeypatch,
    )
    target = qualification if payload_name == "qualification_digest" else evidence
    object.__setattr__(target, payload_name, "f" * 64)

    assert failures.effective_failure_catalog({qualification.category: qualification}) == {}


def test_snapshot_digest_mismatch_fails_before_restore_rng_or_hooks(
    modules,
) -> None:
    state, _telemetry, failures = modules
    snapshot = _recovery_snapshot(state)
    actual_digest = state.recovery_state_digest(snapshot)
    assert actual_digest != "a" * 64
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, "near-shelf-misplaced")
    )
    evidence = _runtime_evidence(failures, "near-shelf-misplaced")
    events: list[str] = []
    endpoint = _StaticDigestAttestedSnapshotEndpoint(state, events)
    hooks = _InjectionHooks(
        events,
        lambda plan: pytest.fail("readback must not run for a mismatched snapshot"),
    )
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(failures.RecoveryFailureSnapshotDigestError) as exc_info:
        failures.inject_recovery_failure(
            _RawSceneTrapEnv(),
            snapshot,
            descriptor,
            evidence,
            snapshot_endpoint=endpoint,
            writer=hooks,
            settler=hooks,
            reader=hooks,
        )

    assert exc_info.value.expected_digest == "a" * 64
    assert exc_info.value.actual_digest == actual_digest
    assert events == []
    assert (hooks.write_count, hooks.settle_count, hooks.read_count) == (0, 0, 0)
    assert random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)


def test_injection_uses_digest_attested_snapshot_endpoint(modules, monkeypatch) -> None:
    state, telemetry, failures = modules
    snapshot = _recovery_snapshot(state)
    snapshot_digest = state.recovery_state_digest(snapshot)
    descriptor = replace(
        failures.RecoveryFailureDescriptor.from_mapping(
            _descriptor_payload(failures, "near-shelf-misplaced")
        ),
        snapshot_digest=snapshot_digest,
    )
    evidence = _runtime_evidence(failures, "near-shelf-misplaced")
    events: list[str] = []
    endpoint = _StaticDigestAttestedSnapshotEndpoint(state, events)
    monkeypatch.setattr(
        failures.recovery_state,
        "restore_recovery_state",
        lambda *args, **kwargs: pytest.fail("local HA restore must not be used"),
    )
    hooks = _InjectionHooks(
        events,
        lambda plan: _replay_context(failures, telemetry, plan),
    )

    result = failures.inject_recovery_failure(
        _RawSceneTrapEnv(),
        snapshot,
        descriptor,
        evidence,
        snapshot_endpoint=endpoint,
        writer=hooks,
        settler=hooks,
        reader=hooks,
    )

    assert result.readback_passed is True
    assert endpoint.digest_calls == [snapshot_digest]
    assert endpoint.restore_calls == [(snapshot_digest, snapshot_digest)]
    assert events == ["restore", "write", "settle", "readback"]


def test_restore_attestation_mismatch_stops_before_injection_or_step(modules) -> None:
    state, _telemetry, failures = modules
    snapshot = _recovery_snapshot(state)
    snapshot_digest = state.recovery_state_digest(snapshot)
    descriptor = replace(
        failures.RecoveryFailureDescriptor.from_mapping(
            _descriptor_payload(failures, "near-shelf-misplaced")
        ),
        snapshot_digest=snapshot_digest,
    )
    evidence = _runtime_evidence(failures, "near-shelf-misplaced")
    events: list[str] = []
    endpoint = _StaticDigestAttestedSnapshotEndpoint(
        state,
        events,
        restore_attestation="f" * 64,
    )
    hooks = _InjectionHooks(
        events,
        lambda plan: pytest.fail("readback must not run after attestation mismatch"),
    )

    with pytest.raises(failures.RecoveryFailureSnapshotDigestError) as exc_info:
        failures.inject_recovery_failure(
            _RawSceneTrapEnv(),
            snapshot,
            descriptor,
            evidence,
            snapshot_endpoint=endpoint,
            writer=hooks,
            settler=hooks,
            reader=hooks,
        )

    assert exc_info.value.expected_digest == snapshot_digest
    assert exc_info.value.actual_digest == "f" * 64
    assert endpoint.restore_calls == [(snapshot_digest, snapshot_digest)]
    assert events == ["restore"]
    assert (hooks.write_count, hooks.settle_count, hooks.read_count) == (0, 0, 0)


def test_injection_delegates_restore_before_explicit_write_settle_readback_without_raw_mutation(
    modules,
) -> None:
    state, telemetry, failures = modules
    snapshot = _recovery_snapshot(state)
    descriptor = replace(
        failures.RecoveryFailureDescriptor.from_mapping(
            _descriptor_payload(failures, "near-shelf-misplaced")
        ),
        snapshot_digest=state.recovery_state_digest(snapshot),
    )
    evidence = _runtime_evidence(failures, "near-shelf-misplaced")
    events: list[str] = []
    endpoint = _StaticDigestAttestedSnapshotEndpoint(state, events)

    def readback(plan):
        return _predicate_context(
            failures,
            telemetry,
            attempt_overrides={
                "trigger_kind": "post-release",
                "pickup_attempted": True,
                "place_attempted": True,
                "release_attempted": True,
                "failure_seed": plan.failure_seed,
                "injected_category": plan.category,
                "transform_digest": plan.transform_digest,
            },
            grasp=False,
            pose_valid=False,
            xy_mismatch_m=0.04,
        )

    hooks = _InjectionHooks(events, readback)
    result = failures.inject_recovery_failure(
        _RawSceneTrapEnv(),
        snapshot,
        descriptor,
        evidence,
        snapshot_endpoint=endpoint,
        writer=hooks,
        settler=hooks,
        reader=hooks,
    )

    assert result.category == "near-shelf-misplaced"
    assert result.readback_passed is True
    assert events == ["restore", "write", "settle", "readback"]
    assert (hooks.write_count, hooks.settle_count, hooks.read_count) == (1, 1, 1)


def test_failed_readback_raises_once_without_retry_or_seed_change(
    modules,
) -> None:
    state, telemetry, failures = modules
    snapshot = _recovery_snapshot(state)
    descriptor = replace(
        failures.RecoveryFailureDescriptor.from_mapping(
            _descriptor_payload(failures, "dropped")
        ),
        snapshot_digest=state.recovery_state_digest(snapshot),
    )
    evidence = _runtime_evidence(failures, "dropped")
    events: list[str] = []
    endpoint = _StaticDigestAttestedSnapshotEndpoint(state, events)

    def readback(plan):
        return _predicate_context(
            failures,
            telemetry,
            attempt_overrides={
                "failure_seed": plan.failure_seed,
                "injected_category": plan.category,
                "transform_digest": plan.transform_digest,
            },
            grasp=False,
            pose_valid=False,
            xy_mismatch_m=0.3,
            ground_supported=False,
        )

    hooks = _InjectionHooks(events, readback)
    with pytest.raises(
        failures.RecoveryFailureVerificationError, match="readback"
    ) as exc_info:
        failures.inject_recovery_failure(
            _RawSceneTrapEnv(),
            snapshot,
            descriptor,
            evidence,
            snapshot_endpoint=endpoint,
            writer=hooks,
            settler=hooks,
            reader=hooks,
        )

    assert exc_info.value.failure_seed == 17
    assert events == ["restore", "write", "settle", "readback"]
    assert (hooks.write_count, hooks.settle_count, hooks.read_count) == (1, 1, 1)


def test_mdp_package_exports_recovery_failure_capabilities() -> None:
    tree = ast.parse(MDP_INIT_PATH.read_text(encoding="utf-8"))
    wildcard_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and any(alias.name == "*" for alias in node.names)
    }

    assert "recovery_failures" in wildcard_modules
