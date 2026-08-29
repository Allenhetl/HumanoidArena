from __future__ import annotations

import ast
import importlib.util
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
    grasp: bool = False,
    pose_valid: bool = True,
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
    left_pose = (0.15 if pose_valid else 1.0, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0)
    right_pose = (-0.15 if pose_valid else -1.0, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0)
    is_fall = terminal_reason == "fall"
    fall_streak = 5 if is_fall else (1 if fall_candidate else 0)
    control_step = 2000 if terminal_reason == "time_limit" else 20
    root_quat = (
        (0.0, 1.0, 0.0, 0.0) if (is_fall or fall_candidate) else (1.0, 0.0, 0.0, 0.0)
    )
    return telemetry.build_privileged_telemetry(
        env_index=0,
        box_center_w=box_center,
        box_linear_velocity_w=(0.0, 0.0, 0.0),
        box_angular_velocity_w=(0.0, 0.0, 0.0),
        support=support,
        left_ee_pose_w=left_pose,
        right_ee_pose_w=right_pose,
        left_contact=_hand_contact(telemetry, "left", grasp),
        right_contact=_hand_contact(telemetry, "right", grasp),
        root_quat_wxyz=root_quat,
        critical_body_contact=False,
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
    replay_records: tuple[object, ...] = (),
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
        replay_records=replay_records,
    )


def _replay_record(failures, plan, *, repeat_index: int, **overrides: object):
    values: dict[str, object] = {
        "schema_version": failures.RECOVERY_REPLAY_EVIDENCE_SCHEMA_VERSION,
        "repeat_index": repeat_index,
        "category": plan.category,
        "failure_seed": plan.failure_seed,
        "snapshot_digest": plan.snapshot_digest,
        "category_seed": plan.category_seed,
        "plan_transform_digest": plan.transform_digest,
        "runtime_evidence_digest": plan.runtime_evidence_digest,
        "readback_state_digest": "4" * 64,
        "continuation_digest": "5" * 64,
        "predicate_passed": True,
        "observed_category": plan.category,
        "category_passed": True,
    }
    values.update(overrides)
    return failures.FailureReplayRecord(**values)


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


def test_stage_fsm_separates_activation_completion_and_declares_fallbacks(
    modules,
) -> None:
    _state, _telemetry, failures = modules

    stages = failures.recovery_stage_fsm()

    assert tuple(stage.name for stage in stages) == ("approach", "acquire", "place")
    assert [
        (stage.activation_predicate, stage.completion_predicate) for stage in stages
    ] == [
        ("running_and_not_grasp", "bimanual_pose_evidence"),
        ("running_and_not_grasp_and_bimanual_pose", "grasp"),
        ("running_and_grasp", "placement"),
    ]
    assert [
        (transition.predicate, transition.target_stage)
        for transition in stages[1].fallbacks
    ] == [("lost_bimanual_pose", "approach")]
    assert [
        (transition.predicate, transition.target_stage)
        for transition in stages[2].fallbacks
    ] == [
        ("lost_grasp_with_bimanual_pose", "acquire"),
        ("lost_grasp_without_bimanual_pose", "approach"),
    ]


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
        entry.stage_fsm == failures.recovery_stage_fsm()
        and entry.reward_bindings == bindings
        for entry in failures.declared_failure_catalog().values()
    )


def test_reward_resolver_emits_component_scoped_numeric_task_truth_and_current_baselines(
    modules,
) -> None:
    _state, telemetry, failures = modules
    task_truth = _telemetry_state(
        telemetry,
        grasp=False,
        pose_valid=True,
        xy_mismatch_m=0.3,
        z_mismatch_m=0.4,
    )

    resolved = failures.resolve_recovery_reward_telemetry(task_truth)
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

    distance = resolved["components"][scopes["distance"]]
    grasp = resolved["components"][scopes["grasp"]]
    placement = resolved["components"][scopes["placement"]]
    assert distance["distance"] == pytest.approx(0.15)
    assert distance["d_init"] == distance["distance"]
    assert grasp["q_grasp"] == 0.0
    assert placement["distance"] == pytest.approx(0.5)
    assert placement["d_init"] == placement["distance"]
    assert len(resolved["components"]) == 3
    assert "articulation" not in repr(resolved).lower()
    assert "rgb" not in repr(resolved).lower()
    assert "vlm" not in repr(resolved).lower()
    assert "injected" not in repr(resolved).lower()

    closer = _telemetry_state(
        telemetry,
        grasp=True,
        pose_valid=True,
        xy_mismatch_m=0.12,
        z_mismatch_m=0.05,
    )
    closer_resolved = failures.resolve_recovery_reward_telemetry(closer)
    closer_placement = closer_resolved["components"][scopes["placement"]]
    closer_grasp = closer_resolved["components"][scopes["grasp"]]
    assert closer_placement["distance"] == pytest.approx(0.13)
    assert closer_placement["d_init"] == closer_placement["distance"]
    assert closer_grasp["q_grasp"] == 1.0


def test_reward_resolver_emits_activation_completion_and_fallback_gate_truth(
    modules,
) -> None:
    _state, telemetry, failures = modules
    pose_without_grasp = failures.resolve_recovery_reward_telemetry(
        _telemetry_state(telemetry, grasp=False, pose_valid=True)
    )
    grasped = failures.resolve_recovery_reward_telemetry(
        _telemetry_state(telemetry, grasp=True, pose_valid=True)
    )
    no_pose = failures.resolve_recovery_reward_telemetry(
        _telemetry_state(telemetry, grasp=False, pose_valid=False)
    )
    falling = failures.resolve_recovery_reward_telemetry(
        _telemetry_state(
            telemetry,
            grasp=False,
            pose_valid=True,
            fall_candidate=True,
        )
    )

    approach = pose_without_grasp["stage_gates"]["approach"]
    acquire = pose_without_grasp["stage_gates"]["acquire"]
    place = pose_without_grasp["stage_gates"]["place"]
    assert approach == {
        "activation_predicate": "running_and_not_grasp",
        "activation": True,
        "completion_predicate": "bimanual_pose_evidence",
        "completion": True,
        "fallbacks": {},
    }
    assert acquire["activation"] is True
    assert acquire["completion"] is False
    assert acquire["fallbacks"] == {"lost_bimanual_pose": False}
    assert place["activation"] is False
    assert place["fallbacks"] == {
        "lost_grasp_with_bimanual_pose": True,
        "lost_grasp_without_bimanual_pose": False,
    }

    assert grasped["stage_gates"]["acquire"]["completion"] is True
    assert grasped["stage_gates"]["place"]["activation"] is True
    assert no_pose["stage_gates"]["acquire"]["fallbacks"] == {
        "lost_bimanual_pose": True
    }
    assert no_pose["stage_gates"]["place"]["fallbacks"] == {
        "lost_grasp_with_bimanual_pose": False,
        "lost_grasp_without_bimanual_pose": True,
    }
    assert all(
        gate["activation"] is False
        and all(active is False for active in gate["fallbacks"].values())
        for gate in falling["stage_gates"].values()
    )


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


def test_effective_catalog_requires_two_identical_bound_replays_per_category(
    modules,
) -> None:
    _state, _telemetry, failures = modules
    pre_injection = {
        category: _runtime_evidence(failures, category)
        for category in sorted(DECLARED_CATEGORIES)
    }
    assert failures.effective_failure_catalog(pre_injection) == {}

    single_replay = {}
    verified = {}
    for category, evidence in pre_injection.items():
        descriptor = failures.RecoveryFailureDescriptor.from_mapping(
            _descriptor_payload(failures, category)
        )
        plan = failures.build_failure_injection_plan(descriptor, evidence)
        first = _replay_record(failures, plan, repeat_index=0)
        second = _replay_record(failures, plan, repeat_index=1)
        single_replay[category] = replace(evidence, replay_records=(first,))
        verified[category] = replace(
            evidence,
            replay_records=(first, second),
        )

    assert failures.effective_failure_catalog(single_replay) == {}
    assert set(failures.effective_failure_catalog(verified)) == DECLARED_CATEGORIES


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeat_index", 0),
        ("category", "dropped"),
        ("failure_seed", 18),
        ("snapshot_digest", "3" * 64),
        ("category_seed", 123),
        ("plan_transform_digest", "6" * 64),
        ("runtime_evidence_digest", "7" * 64),
        ("readback_state_digest", "8" * 64),
        ("continuation_digest", "9" * 64),
        ("predicate_passed", False),
        ("observed_category", "dropped"),
        ("category_passed", False),
    ],
)
def test_effective_catalog_fails_closed_on_replay_label_digest_or_distribution_drift(
    modules,
    field: str,
    value: object,
) -> None:
    _state, _telemetry, failures = modules
    category = "near-shelf-misplaced"
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, category)
    )
    evidence = _runtime_evidence(failures, category)
    plan = failures.build_failure_injection_plan(descriptor, evidence)
    first = _replay_record(failures, plan, repeat_index=0)
    drifted_repeat_index = value if field == "repeat_index" else 1
    drifted_overrides = {} if field == "repeat_index" else {field: value}
    drifted = _replay_record(
        failures,
        plan,
        repeat_index=drifted_repeat_index,
        **drifted_overrides,
    )

    assert (
        failures.effective_failure_catalog(
            {category: replace(evidence, replay_records=(first, drifted))}
        )
        == {}
    )


def test_replay_record_schema_rejects_unknown_category_invalid_digest_and_non_bool(
    modules,
) -> None:
    _state, _telemetry, failures = modules
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, "near-shelf-misplaced")
    )
    evidence = _runtime_evidence(failures, "near-shelf-misplaced")
    plan = failures.build_failure_injection_plan(descriptor, evidence)

    for overrides, message in (
        ({"category": "fall"}, "category"),
        ({"snapshot_digest": "caller-label"}, "digest"),
        ({"predicate_passed": 1}, "boolean"),
    ):
        with pytest.raises(failures.RecoveryFailureSchemaError, match=message):
            _replay_record(failures, plan, repeat_index=0, **overrides)


def test_runtime_evidence_digest_is_content_derived_and_stale_replays_fail_closed(
    modules,
) -> None:
    _state, _telemetry, failures = modules
    category = "misaligned"
    evidence = _runtime_evidence(failures, category)
    identical = _runtime_evidence(failures, category)
    descriptor = failures.RecoveryFailureDescriptor.from_mapping(
        _descriptor_payload(failures, category)
    )
    plan = failures.build_failure_injection_plan(descriptor, evidence)
    records = (
        _replay_record(failures, plan, repeat_index=0),
        _replay_record(failures, plan, repeat_index=1),
    )

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
        replace(evidence, target_shelf=shifted_shelf, replay_records=records),
        replace(evidence, verified_anchor=shifted_anchor, replay_records=records),
        replace(
            evidence,
            validated_capabilities=reduced_capabilities,
            replay_records=records,
        ),
    )

    for mutated in mutations:
        assert mutated.evidence_digest != evidence.evidence_digest
        assert failures.effective_failure_catalog({category: mutated}) == {}


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


def test_snapshot_digest_mismatch_fails_before_restore_rng_or_hooks(
    modules, monkeypatch
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
    monkeypatch.setattr(
        failures.recovery_state,
        "restore_recovery_state",
        lambda env, restored_snapshot, **kwargs: events.append("restore"),
    )
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


def test_injection_delegates_restore_before_explicit_write_settle_readback_without_raw_mutation(
    modules,
    monkeypatch,
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
    monkeypatch.setattr(
        failures.recovery_state,
        "restore_recovery_state",
        lambda env, snapshot, **kwargs: events.append("restore"),
    )

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
        writer=hooks,
        settler=hooks,
        reader=hooks,
    )

    assert result.category == "near-shelf-misplaced"
    assert result.readback_passed is True
    assert events == ["restore", "write", "settle", "readback"]
    assert (hooks.write_count, hooks.settle_count, hooks.read_count) == (1, 1, 1)

    continuation = {
        "control_step": 21,
        "observation": torch.tensor([0.25, -0.5], dtype=torch.float32),
    }
    replay = failures.build_failure_replay_record(
        result,
        repeat_index=0,
        continuation_state=continuation,
    )
    assert replay.category == result.category
    assert replay.failure_seed == result.plan.failure_seed
    assert replay.plan_transform_digest == result.plan.transform_digest
    assert replay.readback_state_digest == state.recovery_value_digest(result.readback)
    assert replay.continuation_digest == state.recovery_value_digest(continuation)
    assert replay.predicate_passed is True
    assert replay.observed_category == result.category
    assert replay.category_passed is True


def test_failed_readback_raises_once_without_retry_or_seed_change(
    modules, monkeypatch
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
    monkeypatch.setattr(
        failures.recovery_state,
        "restore_recovery_state",
        lambda env, snapshot, **kwargs: events.append("restore"),
    )

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
