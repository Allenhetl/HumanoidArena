from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
RECOVERY_TELEMETRY_PATH = MDP_DIR / "recovery_telemetry.py"
REWARDS_PATH = MDP_DIR / "rewards.py"
ENV_CFG_PATH = MDP_DIR.parent / "move_pickplace_box_g1_29dof_dex3_hw_env_cfg.py"

EXPECTED_HAND_CONTACT_BODIES = {
    side: tuple(
        f"{side}_hand_{link}_link"
        for link in (
            "palm",
            "index_0",
            "index_1",
            "middle_0",
            "middle_1",
            "thumb_0",
            "thumb_1",
            "thumb_2",
        )
    )
    for side in ("left", "right")
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def telemetry():
    package_name = "pp_box_recovery_mdp_test"
    package = ModuleType(package_name)
    package.__path__ = [str(MDP_DIR)]
    sys.modules[package_name] = package
    return _load_module(
        f"{package_name}.recovery_telemetry",
        RECOVERY_TELEMETRY_PATH,
    )


@pytest.fixture()
def rewards():
    return _load_module("pp_box_recovery_rewards", REWARDS_PATH)


def _support_case(rewards, center_xyz: tuple[float, float, float]):
    return rewards.evaluate_box_support(
        torch.tensor(center_xyz, dtype=torch.float64),
        [(-1.0, 1.0, -0.5, 0.5, 0.4)],
        target_support_top_z=0.4,
    )


def _single_link_hand_contact(telemetry, side: str, force: tuple[float, float, float]):
    link = telemetry.pairwise_contact_evidence(
        force,
        sensor_body=f"{side}_hand_palm_link",
        filtered_body="Box",
        threshold_n=1.0,
    )
    return telemetry.aggregate_hand_contact_evidence(side, (link,))


def _driver_terminal_context(
    telemetry,
    *,
    control_step_count: int = 10,
    max_control_steps: int = 2000,
    fall_streak: int = 0,
    fall_confirm_steps: int = 5,
):
    return telemetry.DriverTerminalContext(
        control_step_count=control_step_count,
        max_control_steps=max_control_steps,
        fall_streak=fall_streak,
        fall_confirm_steps=fall_confirm_steps,
        time_limit=control_step_count >= max_control_steps,
        fall_confirmed=fall_streak >= fall_confirm_steps,
    )


def _fall_lane(
    telemetry,
    *,
    env_index: int = 0,
    control_step_count: int = 10,
    root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    critical_body_contact: bool | None = False,
):
    alignment = telemetry.compute_root_up_alignment(root_quat_wxyz)
    return telemetry.LiveFallLaneEvidence(
        env_index=env_index,
        control_step_count=control_step_count,
        root_quat_wxyz=root_quat_wxyz,
        root_up_alignment=alignment,
        critical_body_contact=critical_body_contact,
        fall_candidate=telemetry.classify_fall(
            alignment,
            critical_body_contact=critical_body_contact,
        ),
    )


def _evaluator_fall_detector(env, **overrides):
    body_names = tuple(env.scene["robot"].data.body_names)
    critical_names = tuple(name for name in body_names if name == "pelvis")
    detector = {
        "soft_up_alignment": math.cos(math.radians(60.0)),
        "hard_up_alignment": math.cos(math.radians(75.0)),
        "contact_force_threshold": 50.0,
        "confirm_steps": 5,
        "critical_body_indices": tuple(
            body_names.index(name) for name in critical_names
        ),
        "critical_body_names": critical_names,
    }
    detector.update(overrides)
    return detector


@pytest.mark.parametrize(
    ("center_xyz", "inside_xy", "aligned_z", "placed"),
    [
        ((0.0, 0.0, 0.505), True, True, True),
        ((0.895, 0.395, 0.505), True, True, True),
        ((0.896, 0.0, 0.505), False, True, False),
        ((0.0, 0.396, 0.505), False, True, False),
        ((0.0, 0.0, 0.566), True, False, False),
    ],
)
def test_support_bounds_and_placement_truth_table(
    rewards,
    center_xyz: tuple[float, float, float],
    inside_xy: bool,
    aligned_z: bool,
    placed: bool,
) -> None:
    evidence = _support_case(rewards, center_xyz)

    assert evidence.inside_xy is inside_xy
    assert evidence.aligned_z is aligned_z
    assert evidence.placed is placed
    assert evidence.support_bounds_w == (-1.0, 1.0, -0.5, 0.5)
    assert evidence.target_support_top_z_m == pytest.approx(0.4)


def test_xy_and_z_mismatch_are_independent_and_metric(rewards) -> None:
    evidence = _support_case(rewards, (0.925, 0.415, 0.535))
    below_surface = _support_case(rewards, (0.0, 0.0, 0.475))

    assert evidence.dx_outside_m == pytest.approx(0.03)
    assert evidence.dy_outside_m == pytest.approx(0.02)
    assert evidence.xy_mismatch_m == pytest.approx(math.hypot(0.03, 0.02))
    assert evidence.z_mismatch_m == pytest.approx(0.03)
    assert below_surface.z_mismatch_m == pytest.approx(0.03)
    assert evidence.inside_xy is False
    assert evidence.aligned_z is True


@pytest.mark.parametrize(
    ("left_force", "right_force", "left_expected", "right_expected", "pair_expected"),
    [
        ((0.0, 0.0, 2.0), (0.0, 0.0, 3.0), True, True, True),
        ((0.0, 0.0, 2.0), (0.0, 0.0, 0.9), True, False, False),
        ((0.0, 0.0, 0.9), (0.0, 0.0, 3.0), False, True, False),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, False, False),
    ],
)
def test_pairwise_contact_requires_independent_left_and_right_force_evidence(
    telemetry,
    left_force: tuple[float, float, float],
    right_force: tuple[float, float, float],
    left_expected: bool,
    right_expected: bool,
    pair_expected: bool,
) -> None:
    left = telemetry.pairwise_contact_evidence(
        left_force,
        sensor_body="left_exact_body",
        filtered_body="box",
        threshold_n=1.0,
    )
    right = telemetry.pairwise_contact_evidence(
        right_force,
        sensor_body="right_exact_body",
        filtered_body="box",
        threshold_n=1.0,
    )

    assert left.in_contact is left_expected
    assert right.in_contact is right_expected
    assert telemetry.has_pairwise_bimanual_contact(left, right) is pair_expected


def test_hand_contact_aggregation_preserves_links_and_uses_any_verified_link(
    telemetry,
) -> None:
    quiet_palm = telemetry.pairwise_contact_evidence(
        (0.0, 0.0, 0.0),
        sensor_body="left_hand_palm_link",
        filtered_body="Box",
        threshold_n=1.0,
    )
    touching_finger = telemetry.pairwise_contact_evidence(
        (0.0, 0.0, 2.0),
        sensor_body="left_hand_index_0_link",
        filtered_body="Box",
        threshold_n=1.0,
    )
    left = telemetry.aggregate_hand_contact_evidence(
        "left",
        (quiet_palm, touching_finger),
    )
    right = telemetry.aggregate_hand_contact_evidence(
        "right",
        (
            telemetry.pairwise_contact_evidence(
                (0.0, 0.0, 0.0),
                sensor_body="right_hand_palm_link",
                filtered_body="Box",
                threshold_n=1.0,
            ),
        ),
    )

    assert tuple(link.sensor_body for link in left.links) == (
        "left_hand_palm_link",
        "left_hand_index_0_link",
    )
    assert left.contacting_bodies == ("left_hand_index_0_link",)
    assert left.in_contact is True
    assert left.total_magnitude_n == pytest.approx(2.0)
    assert right.in_contact is False
    assert telemetry.has_pairwise_bimanual_contact(left, right) is False


def test_hand_contact_aggregation_rejects_empty_duplicate_or_mixed_side_evidence(
    telemetry,
) -> None:
    left = telemetry.pairwise_contact_evidence(
        (0.0, 0.0, 2.0),
        sensor_body="left_hand_index_0_link",
        filtered_body="Box",
        threshold_n=1.0,
    )
    right = telemetry.pairwise_contact_evidence(
        (0.0, 0.0, 2.0),
        sensor_body="right_hand_index_0_link",
        filtered_body="Box",
        threshold_n=1.0,
    )

    with pytest.raises(ValueError, match="at least one"):
        telemetry.aggregate_hand_contact_evidence("left", ())
    with pytest.raises(ValueError, match="duplicate"):
        telemetry.aggregate_hand_contact_evidence("left", (left, left))
    with pytest.raises(ValueError, match="left hand"):
        telemetry.aggregate_hand_contact_evidence("left", (left, right))


def test_default_contact_bindings_are_exact_palm_and_finger_leaf_sensors(
    telemetry,
) -> None:
    bindings = telemetry.default_hand_contact_bindings()

    assert set(bindings) == {"left", "right"}
    all_scene_keys = []
    for side, expected_bodies in EXPECTED_HAND_CONTACT_BODIES.items():
        hand = bindings[side]
        assert hand.side == side
        assert hand.ee_body_name == f"{side}_hand_palm_link"
        assert (
            tuple(sensor.sensor_body_name for sensor in hand.sensors) == expected_bodies
        )
        assert (
            tuple(sensor.filtered_body_name for sensor in hand.sensors) == ("Box",) * 8
        )
        assert tuple(sensor.sensor_scene_key for sensor in hand.sensors) == tuple(
            f"{side}_box_contact_{body.removeprefix(f'{side}_hand_').removesuffix('_link')}"
            for body in expected_bodies
        )
        all_scene_keys.extend(sensor.sensor_scene_key for sensor in hand.sensors)

    assert len(all_scene_keys) == len(set(all_scene_keys)) == 16
    assert not any(
        "wrist" in body or "camera" in body
        for body in sum(EXPECTED_HAND_CONTACT_BODIES.values(), ())
    )


def test_env_cfg_declares_exact_one_body_box_filters_without_actor_terms() -> None:
    tree = ast.parse(ENV_CFG_PATH.read_text())
    scene_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PickPlaceBoxTaskSceneCfg"
    )
    configured = {}
    for statement in scene_cfg.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        call = statement.value
        if (
            isinstance(target, ast.Name)
            and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_box_contact_sensor_cfg"
        ):
            configured[target.id] = ast.literal_eval(call.args[0])

    expected = {
        f"{side}_box_contact_{body.removeprefix(f'{side}_hand_').removesuffix('_link')}": body
        for side, bodies in EXPECTED_HAND_CONTACT_BODIES.items()
        for body in bodies
    }
    assert configured == expected

    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_box_contact_sensor_cfg"
    )
    return_stmt = next(node for node in helper.body if isinstance(node, ast.Return))
    assert isinstance(return_stmt.value, ast.Call)
    keywords = {keyword.arg: keyword.value for keyword in return_stmt.value.keywords}
    assert ast.unparse(keywords["prim_path"]) == "f'{{ENV_REGEX_NS}}/Robot/{body_name}'"
    assert ast.literal_eval(keywords["filter_prim_paths_expr"]) == [
        "{ENV_REGEX_NS}/Box"
    ]
    assert ast.literal_eval(keywords["update_period"]) == 0.0
    assert ast.literal_eval(keywords["history_length"]) == 0
    assert ast.literal_eval(keywords["track_air_time"]) is False
    assert ast.literal_eval(keywords["debug_vis"]) is False

    observations_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ObservationsCfg"
    )
    policy_cfg = next(
        node
        for node in observations_cfg.body
        if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg"
    )
    actor_terms = {
        statement.targets[0].id
        for statement in policy_cfg.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "ObsTerm"
    }
    assert actor_terms == {"robot_joint_state", "robot_gipper_state", "camera_image"}


@pytest.mark.parametrize(
    ("left_contact", "right_contact", "left_pos", "right_pos", "expected"),
    [
        (True, True, (0.15, 0.0, 0.0), (-0.15, 0.0, 0.0), True),
        (True, False, (0.15, 0.0, 0.0), (-0.15, 0.0, 0.0), False),
        (False, True, (0.15, 0.0, 0.0), (-0.15, 0.0, 0.0), False),
        (False, False, (0.15, 0.0, 0.0), (-0.15, 0.0, 0.0), False),
        (True, True, (0.31, 0.0, 0.0), (-0.15, 0.0, 0.0), False),
        (True, True, (0.15, 0.0, 0.0), (-0.31, 0.0, 0.0), False),
    ],
)
def test_grasp_requires_both_pairwise_contacts_and_both_pose_checks(
    telemetry,
    left_contact: bool,
    right_contact: bool,
    left_pos: tuple[float, float, float],
    right_pos: tuple[float, float, float],
    expected: bool,
) -> None:
    evidence = telemetry.classify_bimanual_grasp(
        box_center_w=(0.0, 0.0, 0.0),
        left_ee_pose_w=(*left_pos, 1.0, 0.0, 0.0, 0.0),
        right_ee_pose_w=(*right_pos, 1.0, 0.0, 0.0, 0.0),
        left_contact=left_contact,
        right_contact=right_contact,
        max_ee_box_distance_m=0.3,
    )

    assert evidence.bimanual_grasp is expected
    assert evidence.pose_evidence is (
        math.dist(left_pos, (0.0, 0.0, 0.0)) <= 0.3
        and math.dist(right_pos, (0.0, 0.0, 0.0)) <= 0.3
    )
    assert evidence.pairwise_contact is (left_contact and right_contact)


@pytest.mark.parametrize(
    ("success", "fall", "time_limit", "expected"),
    [
        (False, False, False, "running"),
        (False, False, True, "time_limit"),
        (False, True, False, "fall"),
        (False, True, True, "fall"),
        (True, False, False, "success"),
        (True, True, True, "success"),
    ],
)
def test_terminal_priority_is_explicit(
    telemetry,
    success: bool,
    fall: bool,
    time_limit: bool,
    expected: str,
) -> None:
    assert (
        telemetry.classify_terminal(success=success, fall=fall, time_limit=time_limit)
        == expected
    )


def test_root_up_alignment_normalizes_quaternion_and_classifies_hard_fall(
    telemetry,
) -> None:
    identity = telemetry.compute_root_up_alignment((2.0, 0.0, 0.0, 0.0))
    side = telemetry.compute_root_up_alignment(
        (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    )
    inverted = telemetry.compute_root_up_alignment((0.0, 1.0, 0.0, 0.0))

    assert identity == pytest.approx(1.0)
    assert side == pytest.approx(0.0)
    assert inverted == pytest.approx(-1.0)
    assert telemetry.classify_fall(identity, critical_body_contact=None) is False
    assert telemetry.classify_fall(inverted, critical_body_contact=None) is True


def test_soft_tilt_requires_critical_contact_evidence_and_fails_closed(
    telemetry,
) -> None:
    soft_tilt_alignment = 0.4

    assert (
        telemetry.classify_fall(soft_tilt_alignment, critical_body_contact=False)
        is False
    )
    assert (
        telemetry.classify_fall(soft_tilt_alignment, critical_body_contact=True) is True
    )
    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.classify_fall(soft_tilt_alignment, critical_body_contact=None)
    assert exc_info.value.missing_capabilities == ("critical_body_contact",)


def test_versioned_privileged_record_contains_required_truth(
    telemetry, rewards
) -> None:
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    left_contact = _single_link_hand_contact(telemetry, "left", (0.0, 0.0, 2.0))
    right_contact = _single_link_hand_contact(telemetry, "right", (0.0, 0.0, 3.0))

    state = telemetry.build_privileged_telemetry(
        task_identity=telemetry.PP_BOX_TASK_IDENTITY,
        env_index=0,
        box_center_w=(0.0, 0.0, 0.505),
        box_linear_velocity_w=(0.1, 0.2, 0.3),
        box_angular_velocity_w=(0.4, 0.5, 0.6),
        support=support,
        left_ee_pose_w=(0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        right_ee_pose_w=(-0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        left_contact=left_contact,
        right_contact=right_contact,
        live_fall_evidence=_fall_lane(telemetry),
        terminal_context=_driver_terminal_context(telemetry),
        max_ee_box_distance_m=0.3,
    )

    assert state.schema_version == telemetry.RECOVERY_TELEMETRY_SCHEMA_VERSION == 2
    assert state.task_identity == telemetry.PP_BOX_TASK_IDENTITY
    assert state.box_center_w == (0.0, 0.0, 0.505)
    assert state.shelf_bounds_w == (-1.0, 1.0, -0.5, 0.5)
    assert state.support_surface_z_m == pytest.approx(0.4)
    assert state.left_box_contact.in_contact is True
    assert state.right_box_contact.in_contact is True
    assert state.grasp is True
    assert state.xy_mismatch_m == pytest.approx(0.0)
    assert state.z_mismatch_m == pytest.approx(0.0)
    assert state.placement is True
    assert state.success is True
    assert state.root_up_alignment == pytest.approx(1.0)
    assert state.control_step_count == 10
    assert state.max_control_steps == 2000
    assert state.fall_candidate is False
    assert state.fall_streak == 0
    assert state.fall_confirm_steps == 5
    assert state.fall is False
    assert state.terminal_reason == "success"


def test_privileged_record_rejects_unaggregated_or_wrong_side_contact(
    telemetry, rewards
) -> None:
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    pairwise = telemetry.pairwise_contact_evidence(
        (0.0, 0.0, 2.0),
        sensor_body="left_hand_palm_link",
        filtered_body="Box",
    )
    right_contact = _single_link_hand_contact(telemetry, "right", (0.0, 0.0, 2.0))
    common = {
        "task_identity": telemetry.PP_BOX_TASK_IDENTITY,
        "env_index": 0,
        "box_center_w": (0.0, 0.0, 0.505),
        "box_linear_velocity_w": (0.0, 0.0, 0.0),
        "box_angular_velocity_w": (0.0, 0.0, 0.0),
        "support": support,
        "left_ee_pose_w": (0.1, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        "right_ee_pose_w": (-0.1, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        "live_fall_evidence": _fall_lane(telemetry),
        "terminal_context": _driver_terminal_context(telemetry),
    }

    with pytest.raises(ValueError, match="aggregated left hand"):
        telemetry.build_privileged_telemetry(
            **common,
            left_contact=pairwise,
            right_contact=right_contact,
        )
    with pytest.raises(ValueError, match="aggregated left hand"):
        telemetry.build_privileged_telemetry(
            **common,
            left_contact=right_contact,
            right_contact=right_contact,
        )


def test_reward_output_remains_the_existing_binary_success_value(
    rewards, monkeypatch
) -> None:
    env = SimpleNamespace(num_envs=2, device="cpu")
    centers = torch.tensor(
        [[0.0, 0.0, 0.505], [0.0, 0.0, 0.7]],
        dtype=torch.float32,
    )
    surfaces = [
        [(-1.0, 1.0, -0.5, 0.5, 0.4)],
        [(-1.0, 1.0, -0.5, 0.5, 0.4)],
    ]
    monkeypatch.setattr(rewards, "_get_box_centers_world", lambda _env: centers)
    monkeypatch.setattr(
        rewards, "_get_shelf_support_surfaces_world", lambda _env: surfaces
    )
    monkeypatch.setattr(
        rewards,
        "_estimate_target_support_top_z",
        lambda _env, _idx, _center, _surfaces: (0.4, 0.4),
    )

    reward = rewards.compute_reward_pickplace_box(env)

    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0], dtype=torch.float32))


def test_actor_observation_leak_check_is_recursive_and_identity_aware(
    telemetry, rewards
) -> None:
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    left_contact = _single_link_hand_contact(telemetry, "left", (0.0, 0.0, 2.0))
    right_contact = _single_link_hand_contact(telemetry, "right", (0.0, 0.0, 2.0))
    privileged = telemetry.build_privileged_telemetry(
        task_identity=telemetry.PP_BOX_TASK_IDENTITY,
        env_index=0,
        box_center_w=(0.0, 0.0, 0.505),
        box_linear_velocity_w=(0.0, 0.0, 0.0),
        box_angular_velocity_w=(0.0, 0.0, 0.0),
        support=support,
        left_ee_pose_w=(0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        right_ee_pose_w=(-0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        left_contact=left_contact,
        right_contact=right_contact,
        live_fall_evidence=_fall_lane(telemetry),
        terminal_context=_driver_terminal_context(telemetry),
        max_ee_box_distance_m=0.3,
    )
    env = _complete_runtime_env(telemetry)
    safe_actor_observation = telemetry.issue_residual_actor_observation(env)

    telemetry.assert_actor_observation_isolated(
        safe_actor_observation,
        privileged,
        env=env,
    )

    env.observation_manager._obs_buffer["policy"]["camera_image"] = {
        "history": [{"box_center_w": tuple(privileged.box_center_w)}],
        "opaque_alias": privileged.left_box_contact,
    }
    leaking_actor_observation = telemetry.issue_residual_actor_observation(env)
    with pytest.raises(telemetry.PrivilegedObservationLeakError) as exc_info:
        telemetry.assert_actor_observation_isolated(
            leaking_actor_observation,
            privileged,
            env=env,
        )
    assert "$.policy.camera_image.history[0].box_center_w" in exc_info.value.leak_paths
    assert "$.policy.camera_image.opaque_alias" in exc_info.value.leak_paths


def test_runtime_extractor_rejects_unproven_pairwise_contact_bindings(
    telemetry,
) -> None:
    env = SimpleNamespace(
        num_envs=1,
        scene={},
        device="cpu",
        cfg=SimpleNamespace(
            env_name=telemetry.PP_BOX_TASK_IDENTITY,
            recovery_task_identity=telemetry.PP_BOX_TASK_IDENTITY,
        ),
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            actor_observation=_safe_actor_observation(),
            terminal_evidence=object(),
        )

    assert "left_box_pairwise_contact" in exc_info.value.missing_capabilities
    assert "right_box_pairwise_contact" in exc_info.value.missing_capabilities


def _complete_runtime_env(telemetry, *, include_contact_mapping_proofs: bool = True):
    box_root_state = torch.zeros(1, 13, dtype=torch.float64)
    box_root_state[0, 0:3] = torch.tensor([0.0, 0.0, 0.505], dtype=torch.float64)
    box_root_state[0, 3] = 1.0
    box_root_state[0, 7:10] = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    box_root_state[0, 10:13] = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64)
    box = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Box"),
        root_physx_view=SimpleNamespace(prim_paths=["/World/envs/env_0/Box"]),
        data=SimpleNamespace(root_state_w=box_root_state),
    )

    body_names = [
        *EXPECTED_HAND_CONTACT_BODIES["left"],
        *EXPECTED_HAND_CONTACT_BODIES["right"],
        "pelvis",
        "left_foot",
    ]
    body_state = torch.zeros(1, len(body_names), 13, dtype=torch.float64)
    body_state[0, body_names.index("left_hand_palm_link"), 0:3] = torch.tensor(
        [0.15, 0.0, 0.505], dtype=torch.float64
    )
    body_state[0, body_names.index("right_hand_palm_link"), 0:3] = torch.tensor(
        [-0.15, 0.0, 0.505], dtype=torch.float64
    )
    body_state[0, :, 3] = 1.0
    root_state = torch.zeros(1, 13, dtype=torch.float64)
    root_state[0, 3] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=body_names,
            body_state_w=body_state,
            root_state_w=root_state,
        )
    )

    bindings = telemetry.default_hand_contact_bindings()
    scene = {
        "box": box,
        "robot": robot,
        "contact_forces": SimpleNamespace(
            body_names=list(body_names),
            num_bodies=len(body_names),
            data=SimpleNamespace(
                net_forces_w=torch.zeros(1, len(body_names), 3, dtype=torch.float64)
            ),
        ),
    }
    contact_forces = {
        "left_hand_index_0_link": (0.0, 0.0, 2.0),
        "right_hand_thumb_2_link": (0.0, 0.0, 3.0),
    }
    for hand in bindings.values():
        for sensor_binding in hand.sensors:
            force = contact_forces.get(sensor_binding.sensor_body_name, (0.0, 0.0, 0.0))
            scene[sensor_binding.sensor_scene_key] = SimpleNamespace(
                body_names=[sensor_binding.sensor_body_name],
                num_bodies=1,
                body_physx_view=SimpleNamespace(
                    prim_paths=[
                        f"/World/envs/env_0/Robot/{sensor_binding.sensor_body_name}"
                    ]
                ),
                cfg=SimpleNamespace(
                    prim_path=(
                        f"{{ENV_REGEX_NS}}/Robot/{sensor_binding.sensor_body_name}"
                    ),
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
                ),
                contact_physx_view=SimpleNamespace(filter_count=1),
                data=SimpleNamespace(
                    force_matrix_w=torch.tensor([[[force]]], dtype=torch.float64)
                ),
            )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=scene,
        cfg=SimpleNamespace(
            env_name=telemetry.PP_BOX_TASK_IDENTITY,
            recovery_task_identity=telemetry.PP_BOX_TASK_IDENTITY,
            recovery_contact_bindings=bindings,
            recovery_telemetry_thresholds={
                "contact_force_n": 1.0,
                "max_ee_box_distance_m": 0.3,
            },
        ),
        episode_length_buf=torch.tensor([10]),
        max_episode_length=100,
        recovery_terminal_contexts=(_driver_terminal_context(telemetry),),
    )
    policy = {
        "robot_joint_state": torch.zeros(1, 29),
        "robot_gipper_state": torch.zeros(1, 14),
        "camera_image": torch.zeros(1, 2, 2, 3),
    }
    env.observation_manager = SimpleNamespace(
        _group_obs_term_names={"policy": list(policy)},
        _obs_buffer={"policy": policy},
    )
    env.obs_buf = env.observation_manager._obs_buffer
    if include_contact_mapping_proofs:
        return _attach_valid_contact_mapping_proofs(telemetry, env)
    return env


def _attach_valid_contact_mapping_proofs(telemetry, env):
    if not hasattr(env, "recovery_contact_mapping_receipt"):
        post_reset_forces = {
            sensor.sensor_scene_key: env.scene[
                sensor.sensor_scene_key
            ].data.force_matrix_w.clone()
            for hand in telemetry.default_hand_contact_bindings().values()
            for sensor in hand.sensors
        }
        _install_fake_contact_calibration_runtime(
            telemetry,
            env,
            emit_touch_force=True,
        )
        telemetry.execute_pp_box_contact_calibration(env)
        # Pure fixture analogue of the mandatory post-calibration episode reset.
        for sensor_key, forces in post_reset_forces.items():
            env.scene[sensor_key].data.force_matrix_w.copy_(forces)
    return env


def _safe_actor_observation(telemetry=None, env=None):
    if telemetry is not None and env is not None:
        return telemetry.issue_residual_actor_observation(env)
    return {"policy": {"joint_state": torch.zeros(1, 4)}}


def _terminal_evidence(
    telemetry,
    env,
    *,
    step_idx: int = 1,
    max_steps: int = 2000,
    fall_detector=None,
):
    return telemetry.produce_evaluator_terminal_evidence(
        env,
        step_idx=step_idx,
        max_steps=max_steps,
        fall_detector=(
            _evaluator_fall_detector(env) if fall_detector is None else fall_detector
        ),
    )


def _extract_runtime_telemetry(
    telemetry,
    env,
    *,
    support_resolver,
    terminal_evidence=None,
    step_idx: int = 1,
    max_steps: int = 2000,
    fall_detector=None,
    actor_observation=None,
):
    evidence = (
        _terminal_evidence(
            telemetry,
            env,
            step_idx=step_idx,
            max_steps=max_steps,
            fall_detector=fall_detector,
        )
        if terminal_evidence is None
        else terminal_evidence
    )
    return telemetry.extract_privileged_telemetry(
        env,
        support_resolver=support_resolver,
        terminal_evidence=evidence,
        actor_observation=(
            _safe_actor_observation(telemetry, env)
            if actor_observation is None
            else actor_observation
        ),
    )


@pytest.mark.parametrize("task_identity", [None, "wrong-task"])
def test_runtime_extractor_checks_task_identity_before_runtime_reads(
    telemetry,
    task_identity,
) -> None:
    class TrapCfg:
        recovery_task_identity = task_identity

        @property
        def recovery_contact_bindings(self):
            raise AssertionError("task identity must fail before contact binding reads")

    env = SimpleNamespace(cfg=TrapCfg())

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            actor_observation=_safe_actor_observation(),
            terminal_evidence=object(),
        )

    assert "task_identity" in exc_info.value.missing_capabilities


def test_live_fall_producer_reuses_instantaneous_root_and_contact_truth(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    detector = _evaluator_fall_detector(env)

    def produce(step: int):
        terminal = telemetry.produce_evaluator_terminal_evidence(
            env,
            step_idx=step,
            max_steps=20,
            fall_detector=detector,
        )
        evidence = terminal.live_fall_evidence
        assert evidence.task_identity == telemetry.PP_BOX_TASK_IDENTITY
        assert evidence.runtime_identity_digest == env.recovery_runtime_identity_digest
        assert len(evidence.evidence_digest) == 64
        assert evidence.lanes[0].control_step_count == step
        return evidence.lanes[0]

    assert produce(1).fall_candidate is False

    up_alignment = 0.4
    root_x = math.sqrt((1.0 - up_alignment) / 2.0)
    root_w = math.sqrt(1.0 - root_x * root_x)
    env.scene["robot"].data.root_state_w[0, 3:7] = torch.tensor(
        [root_w, root_x, 0.0, 0.0]
    )
    tilted = produce(2)
    assert tilted.root_up_alignment == pytest.approx(up_alignment)
    assert tilted.fall_candidate is False

    pelvis_index = env.scene["robot"].data.body_names.index("pelvis")
    env.scene["contact_forces"].data.net_forces_w[0, pelvis_index, 2] = 51.0
    contacted = produce(3)
    assert contacted.critical_body_contact is True
    assert contacted.fall_candidate is True

    env.scene["contact_forces"].data.net_forces_w = None
    env.scene["robot"].data.root_state_w[0, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert produce(4).fall_candidate is True


def test_evaluator_terminal_evidence_uses_step_idx_and_spec_max_steps_only(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.episode_length_buf[:] = env.max_episode_length
    detector = _evaluator_fall_detector(env)

    first = telemetry.produce_evaluator_terminal_evidence(
        env,
        step_idx=1,
        max_steps=3,
        fall_detector=detector,
    )
    terminal = telemetry.produce_evaluator_terminal_evidence(
        env,
        step_idx=2,
        max_steps=3,
        fall_detector=detector,
    )
    terminal = telemetry.produce_evaluator_terminal_evidence(
        env,
        step_idx=3,
        max_steps=3,
        fall_detector=detector,
    )

    assert first.contexts[0].time_limit is False
    assert terminal.contexts[0].time_limit is True
    assert terminal.contexts[0].control_step_count == 3
    assert terminal.contexts[0].max_control_steps == 3


def test_evaluator_terminal_evidence_defaults_to_five_consecutive_fall_steps(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.scene["robot"].data.root_state_w[0, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    detector = _evaluator_fall_detector(env)

    evidence = None
    for step_idx in range(1, 6):
        evidence = telemetry.produce_evaluator_terminal_evidence(
            env,
            step_idx=step_idx,
            max_steps=20,
            fall_detector=detector,
        )
        assert evidence.contexts[0].fall_streak == step_idx
        assert evidence.contexts[0].fall_confirm_steps == 5
        assert evidence.contexts[0].fall_confirmed is (step_idx == 5)

    assert evidence is not None
    assert evidence.contexts[0].fall_confirmed is True


def test_extractor_rejects_copied_or_stale_evaluator_terminal_evidence(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    detector = _evaluator_fall_detector(env)
    stale = telemetry.produce_evaluator_terminal_evidence(
        env,
        step_idx=1,
        max_steps=20,
        fall_detector=detector,
    )
    current = telemetry.produce_evaluator_terminal_evidence(
        env,
        step_idx=2,
        max_steps=20,
        fall_detector=detector,
    )
    forged = object.__new__(type(current))
    for name in current.__dataclass_fields__:
        object.__setattr__(forged, name, getattr(current, name))
    support = _support_case(rewards, (0.0, 0.0, 0.505))

    for invalid in (stale, forged):
        with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
            telemetry.extract_privileged_telemetry(
                env,
                support_resolver=lambda _env: [support],
                terminal_evidence=invalid,
                actor_observation=_safe_actor_observation(telemetry, env),
            )
        assert "evaluator_terminal_evidence" in exc_info.value.missing_capabilities


def test_runtime_contact_validator_rejects_complete_but_wrong_env_namespace(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.scene["box"].root_physx_view.prim_paths = ["/World/envs/env_7/Box"]
    for hand in telemetry.default_hand_contact_bindings().values():
        for binding in hand.sensors:
            env.scene[binding.sensor_scene_key].body_physx_view.prim_paths = [
                f"/World/envs/env_7/Robot/{binding.sensor_body_name}"
            ]

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert "runtime_contact_filter_asset" in exc_info.value.missing_capabilities


def test_runtime_contact_validator_rejects_candidate_only_filter_identity(
    telemetry,
) -> None:
    env = _complete_runtime_env(
        telemetry,
        include_contact_mapping_proofs=False,
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert (
        "runtime_contact_mapping_receipt:left_box_contact_palm"
        in exc_info.value.missing_capabilities
    )


def test_executor_receipt_digest_binds_actual_raw_measurement_bytes(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    execution = env.recovery_contact_mapping_receipt
    receipt = execution.sensor_receipts[0]
    baseline, touch, removed = receipt.phases

    assert (
        receipt.schema_version
        == telemetry.RUNTIME_CONTACT_MAPPING_RECEIPT_SCHEMA_VERSION
        == 3
    )
    assert tuple(phase.phase for phase in receipt.phases) == (
        "baseline",
        "target_touch",
        "target_removed",
    )
    assert (
        baseline.raw_force_sha256
        == hashlib.sha256(baseline.raw_force_bytes).hexdigest()
    )
    assert touch.raw_force_sha256 == hashlib.sha256(touch.raw_force_bytes).hexdigest()
    assert (
        removed.raw_force_sha256 == hashlib.sha256(removed.raw_force_bytes).hexdigest()
    )

    object.__setattr__(touch, "raw_force_bytes", b"\x00" * len(touch.raw_force_bytes))
    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)
    assert (
        f"runtime_contact_mapping_receipt:{receipt.sensor_scene_key}"
        in exc_info.value.missing_capabilities
    )


def test_runtime_contact_validator_rejects_miswired_executor_receipt(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    sensor_key = "right_box_contact_thumb_2"
    receipt = next(
        item
        for item in env.recovery_contact_mapping_receipt.sensor_receipts
        if item.sensor_scene_key == sensor_key
    )
    object.__setattr__(receipt, "target_asset_scene_key", "shelf")

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert (
        f"runtime_contact_mapping_receipt:{sensor_key}"
        in exc_info.value.missing_capabilities
    )


@pytest.mark.parametrize(
    ("phase_index", "forces"),
    [
        (0, (0.0, 0.0, 2.0)),
        (1, (0.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 2.0)),
    ],
)
def test_runtime_contact_validator_rejects_digest_valid_noncausal_receipt_bytes(
    telemetry,
    phase_index: int,
    forces: tuple[float, float, float],
) -> None:
    env = _complete_runtime_env(telemetry)
    sensor_key = "left_box_contact_index_0"
    execution = env.recovery_contact_mapping_receipt
    receipt = next(
        item
        for item in execution.sensor_receipts
        if item.sensor_scene_key == sensor_key
    )
    phase = receipt.phases[phase_index]
    raw = torch.tensor([[[forces]]], dtype=torch.float64).numpy().tobytes()
    object.__setattr__(phase, "raw_force_bytes", raw)
    object.__setattr__(phase, "raw_force_sha256", hashlib.sha256(raw).hexdigest())
    object.__setattr__(phase, "receipt_digest", telemetry._phase_receipt_digest(phase))
    object.__setattr__(
        receipt, "receipt_digest", telemetry._sensor_receipt_digest(receipt)
    )
    object.__setattr__(
        execution, "receipt_digest", telemetry._execution_receipt_digest(execution)
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert (
        f"runtime_contact_mapping_receipt:{sensor_key}"
        in exc_info.value.missing_capabilities
    )


def test_runtime_contact_validator_requires_exact_complete_receipt_set(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    missing_key = "right_box_contact_middle_1"
    execution = env.recovery_contact_mapping_receipt
    object.__setattr__(
        execution,
        "sensor_receipts",
        tuple(
            receipt
            for receipt in execution.sensor_receipts
            if receipt.sensor_scene_key != missing_key
        ),
    )
    object.__setattr__(
        execution, "receipt_digest", telemetry._execution_receipt_digest(execution)
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert "runtime_contact_mapping_receipts" in exc_info.value.missing_capabilities


def test_runtime_contact_validator_binds_receipts_to_derived_runtime_identity(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.scene[
        "left_box_contact_palm"
    ].cfg.prim_path = "{ENV_REGEX_NS}/Robot/left_hand_index_0_link"

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert (
        "runtime_contact_sensor:left_box_contact_palm"
        in exc_info.value.missing_capabilities
    )


def test_runtime_contact_validator_rejects_boolean_receipt_schema(telemetry) -> None:
    env = _complete_runtime_env(telemetry)
    sensor_key = "left_box_contact_palm"
    execution = env.recovery_contact_mapping_receipt
    receipt = next(
        item
        for item in execution.sensor_receipts
        if item.sensor_scene_key == sensor_key
    )
    object.__setattr__(receipt, "schema_version", True)
    object.__setattr__(
        receipt, "receipt_digest", telemetry._sensor_receipt_digest(receipt)
    )
    object.__setattr__(
        execution, "receipt_digest", telemetry._execution_receipt_digest(execution)
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert (
        f"runtime_contact_mapping_receipt:{sensor_key}"
        in exc_info.value.missing_capabilities
    )


def test_runtime_contact_sensor_report_records_all_materialized_identities(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)

    reports = telemetry.validate_runtime_hand_contact_sensors(env)

    assert len(reports) == 16
    assert tuple(report.sensor_scene_key for report in reports) == tuple(
        sensor.sensor_scene_key
        for hand in telemetry.default_hand_contact_bindings().values()
        for sensor in hand.sensors
    )
    palm = reports[0]
    assert (
        palm.schema_version
        == telemetry.RUNTIME_CONTACT_SENSOR_REPORT_SCHEMA_VERSION
        == 2
    )
    assert palm.side == "left"
    assert palm.sensor_scene_key == "left_box_contact_palm"
    assert (
        palm.sensor_prim_path_expression == "{ENV_REGEX_NS}/Robot/left_hand_palm_link"
    )
    assert palm.resolved_sensor_prim_paths == (
        "/World/envs/env_0/Robot/left_hand_palm_link",
    )
    assert palm.resolved_sensor_body_names == ("left_hand_palm_link",)
    assert palm.configured_filter_prim_path_expressions == ("{ENV_REGEX_NS}/Box",)
    assert palm.candidate_filter_asset_scene_key == "box"
    assert palm.candidate_filter_asset_prim_path_expression == "{ENV_REGEX_NS}/Box"
    assert palm.candidate_filter_asset_prim_paths == ("/World/envs/env_0/Box",)
    assert palm.proven_filter_prim_paths == ("/World/envs/env_0/Box",)
    assert palm.proven_filter_body_name == "Box"
    assert palm.filter_mapping_proof_schema_version == 3
    assert palm.filter_mapping_proof_source == "controlled_three_phase_executor"
    receipt = next(
        item
        for item in env.recovery_contact_mapping_receipt.sensor_receipts
        if item.sensor_scene_key == "left_box_contact_palm"
    )
    assert palm.filter_mapping_proof_digest == (receipt.receipt_digest)
    assert palm.num_bodies == 1
    assert palm.filter_count == 1
    assert palm.force_matrix_shape == (1, 1, 1, 3)
    assert palm.force_matrix_dtype == "torch.float64"
    assert palm.force_matrix_device == "cpu"
    assert palm.force_matrix_finite is True


def test_runtime_extractor_invokes_validator_and_actor_leak_assertion_once(
    telemetry,
    rewards,
    monkeypatch,
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    actor_observation = _safe_actor_observation(telemetry, env)
    original_validator = telemetry.validate_runtime_hand_contact_sensors
    original_assertion = telemetry.assert_actor_observation_isolated
    validator_reports: list[tuple[object, ...]] = []
    leak_calls: list[tuple[object, object]] = []

    def validate_once(runtime_env):
        reports = original_validator(runtime_env)
        validator_reports.append(reports)
        return reports

    def assert_once(actor, privileged, *, env):
        leak_calls.append((actor, privileged))
        return original_assertion(actor, privileged, env=env)

    monkeypatch.setattr(
        telemetry, "validate_runtime_hand_contact_sensors", validate_once
    )
    monkeypatch.setattr(telemetry, "assert_actor_observation_isolated", assert_once)

    states = _extract_runtime_telemetry(
        telemetry,
        env,
        support_resolver=lambda _env: [support],
        actor_observation=actor_observation,
    )

    assert len(validator_reports) == 1
    assert len(validator_reports[0]) == 16
    assert leak_calls == [(actor_observation, states[0])]


def test_runtime_extractor_rejects_actor_key_leak_and_stale_fall_evidence(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    evidence = _terminal_evidence(telemetry, env)

    with pytest.raises(telemetry.PrivilegedObservationLeakError):
        _extract_runtime_telemetry(
            telemetry,
            env,
            support_resolver=lambda _env: [support],
            actor_observation={"policy": {"box_center_w": torch.zeros(1, 3)}},
            terminal_evidence=evidence,
        )

    object.__setattr__(evidence, "evidence_digest", "f" * 64)
    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        _extract_runtime_telemetry(
            telemetry,
            env,
            support_resolver=lambda _env: [support],
            terminal_evidence=evidence,
        )
    assert "evaluator_terminal_evidence" in exc_info.value.missing_capabilities


def test_runtime_extractor_rejects_fall_evidence_from_another_control_step(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    stale = telemetry.produce_evaluator_terminal_evidence(
        env,
        step_idx=1,
        max_steps=20,
        fall_detector=_evaluator_fall_detector(env),
    )
    telemetry.produce_evaluator_terminal_evidence(
        env,
        step_idx=2,
        max_steps=20,
        fall_detector=_evaluator_fall_detector(env),
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        _extract_runtime_telemetry(
            telemetry,
            env,
            support_resolver=lambda _env: [support],
            terminal_evidence=stale,
        )
    assert "evaluator_terminal_evidence" in exc_info.value.missing_capabilities


@pytest.mark.parametrize(
    ("mutation", "expected_key"),
    [
        (
            lambda env: env.scene.pop("left_box_contact_middle_1"),
            "left_box_contact_middle_1",
        ),
        (
            lambda env: setattr(
                env.scene["left_box_contact_index_0"].body_physx_view,
                "prim_paths",
                [
                    "/World/envs/env_0/Robot/left_hand_index_0_link",
                    "/World/envs/env_0/Robot/left_hand_index_0_link",
                ],
            ),
            "left_box_contact_index_0",
        ),
        (
            lambda env: setattr(
                env.scene["right_box_contact_thumb_2"].body_physx_view,
                "prim_paths",
                ["/World/envs/env_0/Robot/right_hand_thumb_1_link"],
            ),
            "right_box_contact_thumb_2",
        ),
        (
            lambda env: setattr(
                env.scene["left_box_contact_palm"].data,
                "force_matrix_w",
                torch.full((1, 1, 1, 3), float("nan"), dtype=torch.float64),
            ),
            "left_box_contact_palm",
        ),
        (
            lambda env: setattr(
                env.scene["right_box_contact_index_1"].data,
                "force_matrix_w",
                torch.zeros((1, 1, 1, 3), dtype=torch.int64),
            ),
            "right_box_contact_index_1",
        ),
        (
            lambda env: setattr(env, "device", "cuda:0"),
            "left_box_contact_palm",
        ),
        (
            lambda env: setattr(
                env.scene["right_box_contact_middle_0"].data,
                "force_matrix_w",
                torch.zeros((1, 1, 1, 3), dtype=torch.float32),
            ),
            "right_box_contact_middle_0",
        ),
    ],
)
def test_runtime_contact_sensor_validator_fails_closed_on_identity_or_tensor_mismatch(
    telemetry,
    mutation,
    expected_key: str,
) -> None:
    env = _complete_runtime_env(telemetry)
    mutation(env)

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert (
        f"runtime_contact_sensor:{expected_key}" in exc_info.value.missing_capabilities
    )


def test_runtime_extractor_reads_exact_pairwise_force_matrix_and_dynamics(
    telemetry, rewards
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.505))

    states = _extract_runtime_telemetry(
        telemetry,
        env,
        support_resolver=lambda _env: [support],
    )

    assert len(states) == 1
    state = states[0]
    assert state.box_center_w == (0.0, 0.0, 0.505)
    assert state.box_linear_velocity_w == (0.1, 0.2, 0.3)
    assert state.box_angular_velocity_w == (0.4, 0.5, 0.6)
    assert state.left_ee_pose_w[:3] == (0.15, 0.0, 0.505)
    assert state.right_ee_pose_w[:3] == (-0.15, 0.0, 0.505)
    assert len(state.left_box_contact.links) == 8
    assert len(state.right_box_contact.links) == 8
    assert state.left_box_contact.contacting_bodies == ("left_hand_index_0_link",)
    assert state.right_box_contact.contacting_bodies == ("right_hand_thumb_2_link",)
    assert state.left_box_contact.resultant_force_w == (0.0, 0.0, 2.0)
    assert state.right_box_contact.resultant_force_w == (0.0, 0.0, 3.0)
    assert state.grasp is True
    assert state.success is True
    assert state.fall is False
    assert state.terminal_reason == "success"


@pytest.mark.parametrize(
    ("mutation", "missing"),
    [
        (
            lambda env: setattr(
                env.scene["left_box_contact_index_0"],
                "body_names",
                ["left_hand_index_0_link", "another_body"],
            ),
            "left_box_pairwise_contact",
        ),
        (
            lambda env: setattr(
                env.scene["right_box_contact_thumb_2"].data,
                "force_matrix_w",
                torch.zeros(1, 1, 2, 3),
            ),
            "right_box_pairwise_contact",
        ),
        (
            lambda env: env.scene["robot"].data.body_names.append(
                "left_hand_palm_link"
            ),
            "ee_body:left_hand_palm_link",
        ),
        (
            lambda env: setattr(
                env.scene["left_box_contact_index_0"].cfg,
                "filter_prim_paths_expr",
                ["{ENV_REGEX_NS}/Shelf"],
            ),
            "left_box_pairwise_contact",
        ),
        (
            lambda env: setattr(
                env.scene["right_box_contact_thumb_2"], "num_bodies", 2
            ),
            "right_box_pairwise_contact",
        ),
        (
            lambda env: setattr(
                env.scene["left_box_contact_index_0"].contact_physx_view,
                "filter_count",
                2,
            ),
            "left_box_pairwise_contact",
        ),
        (
            lambda env: env.cfg.recovery_contact_bindings.__setitem__(
                "right",
                replace(
                    env.cfg.recovery_contact_bindings["right"],
                    sensors=(
                        replace(
                            env.cfg.recovery_contact_bindings["right"].sensors[0],
                            filtered_body_name="Shelf",
                        ),
                        *env.cfg.recovery_contact_bindings["right"].sensors[1:],
                    ),
                ),
            ),
            "right_box_pairwise_contact",
        ),
    ],
)
def test_runtime_extractor_fails_closed_on_ambiguous_body_or_force_shape(
    telemetry,
    rewards,
    mutation,
    missing: str,
) -> None:
    env = _complete_runtime_env(telemetry)
    mutation(env)
    support = _support_case(rewards, (0.0, 0.0, 0.505))

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        _extract_runtime_telemetry(
            telemetry,
            env,
            support_resolver=lambda _env: [support],
        )

    assert missing in exc_info.value.missing_capabilities


@pytest.mark.parametrize(
    "thresholds",
    [
        {},
        {"contact_force_n": 1.0},
        {"contact_force_n": 0.0, "max_ee_box_distance_m": 0.3},
        {"contact_force_n": "invalid", "max_ee_box_distance_m": 0.3},
        {"contact_force_n": 1.0, "max_ee_box_distance_m": float("nan")},
        "not-a-mapping",
    ],
)
def test_runtime_extractor_fails_closed_on_missing_or_invalid_thresholds(
    telemetry,
    rewards,
    thresholds,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.cfg.recovery_telemetry_thresholds = thresholds
    support = _support_case(rewards, (0.0, 0.0, 0.505))

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        _extract_runtime_telemetry(
            telemetry,
            env,
            support_resolver=lambda _env: [support],
        )

    assert "recovery_telemetry_thresholds" in exc_info.value.missing_capabilities


def test_runtime_extractor_requires_factory_issued_evaluator_terminal_evidence(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.7))
    env.recovery_terminal_contexts = (_driver_terminal_context(telemetry),)

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            support_resolver=lambda _env: [support],
            actor_observation=_safe_actor_observation(telemetry, env),
            terminal_evidence=object(),
        )
    assert "evaluator_terminal_evidence" in exc_info.value.missing_capabilities

    state = _extract_runtime_telemetry(
        telemetry,
        env,
        support_resolver=lambda _env: [support],
        step_idx=1,
        max_steps=2,
    )[0]
    assert state.terminal_reason == "running"


def test_evaluator_terminal_producer_resets_streak_when_candidate_clears(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.scene["robot"].data.root_state_w[0, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    detector = _evaluator_fall_detector(env)
    first = _terminal_evidence(
        telemetry,
        env,
        step_idx=1,
        fall_detector=detector,
    )
    assert first.contexts[0].fall_streak == 1
    env.scene["robot"].data.root_state_w[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = _terminal_evidence(
        telemetry,
        env,
        step_idx=2,
        fall_detector=detector,
    )
    assert second.contexts[0].fall_streak == 0
    assert second.contexts[0].fall_confirmed is False


def test_evaluator_terminal_producer_rejects_skipped_control_step(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry)
    detector = _evaluator_fall_detector(env)
    _terminal_evidence(
        telemetry,
        env,
        step_idx=1,
        fall_detector=detector,
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        _terminal_evidence(
            telemetry,
            env,
            step_idx=3,
            fall_detector=detector,
        )
    assert exc_info.value.missing_capabilities == ("evaluator_terminal_sequence",)


def test_evaluator_terminal_evidence_context_is_digest_bound(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    evidence = _terminal_evidence(telemetry, env)
    object.__setattr__(
        evidence,
        "contexts",
        (
            _driver_terminal_context(
                telemetry,
                control_step_count=1,
                fall_streak=1,
            ),
        ),
    )
    support = _support_case(rewards, (0.0, 0.0, 0.7))

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        _extract_runtime_telemetry(
            telemetry,
            env,
            support_resolver=lambda _env: [support],
            terminal_evidence=evidence,
        )
    assert exc_info.value.missing_capabilities == ("evaluator_terminal_evidence",)


def test_runtime_extractor_uses_critical_body_contact_for_soft_tilt(
    telemetry, rewards
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.7))
    up_alignment = 0.4
    root_x = math.sqrt((1.0 - up_alignment) / 2.0)
    root_w = math.sqrt(1.0 - root_x * root_x)
    env.scene["robot"].data.root_state_w[0, 3:7] = torch.tensor(
        [root_w, root_x, 0.0, 0.0]
    )
    pelvis_index = env.scene["robot"].data.body_names.index("pelvis")
    env.scene["contact_forces"].data.net_forces_w[0, pelvis_index, 2] = 51.0
    detector = _evaluator_fall_detector(env)
    terminal_evidence = None
    for step_idx in range(1, 5):
        terminal_evidence = _terminal_evidence(
            telemetry,
            env,
            step_idx=step_idx,
            fall_detector=detector,
        )

    state = _extract_runtime_telemetry(
        telemetry,
        env,
        support_resolver=lambda _env: [support],
        terminal_evidence=terminal_evidence,
    )[0]

    assert state.root_up_alignment == pytest.approx(up_alignment)
    assert state.fall_candidate is True
    assert state.fall is False
    assert state.terminal_reason == "running"

    terminal_evidence = _terminal_evidence(
        telemetry,
        env,
        step_idx=5,
        fall_detector=detector,
    )
    state = _extract_runtime_telemetry(
        telemetry,
        env,
        support_resolver=lambda _env: [support],
        terminal_evidence=terminal_evidence,
    )[0]
    assert state.fall_candidate is True
    assert state.fall is True
    assert state.terminal_reason == "fall"


def test_runtime_extractor_classifies_time_limit_without_changing_task_reward(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.7))
    env.episode_length_buf[:] = env.max_episode_length

    state = _extract_runtime_telemetry(
        telemetry,
        env,
        support_resolver=lambda _env: [support],
        step_idx=1,
        max_steps=2,
    )[0]

    assert state.success is False
    assert state.fall is False
    assert state.time_limit is False
    assert state.terminal_reason == "running"

    terminal_evidence = _terminal_evidence(
        telemetry,
        env,
        step_idx=2,
        max_steps=2,
    )
    state = _extract_runtime_telemetry(
        telemetry,
        env,
        support_resolver=lambda _env: [support],
        terminal_evidence=terminal_evidence,
    )[0]
    assert state.time_limit is True
    assert state.terminal_reason == "time_limit"


def test_task_identity_requires_actual_runtime_env_name_before_scene_access(
    telemetry,
) -> None:
    class TrapCfg:
        recovery_task_identity = telemetry.PP_BOX_TASK_IDENTITY
        env_name = "Isaac-Another-Task"

        @property
        def recovery_contact_bindings(self):
            raise AssertionError("runtime task identity must fail before scene reads")

    env = SimpleNamespace(cfg=TrapCfg())

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            actor_observation=object(),
            terminal_evidence=object(),
        )

    assert exc_info.value.missing_capabilities == ("task_identity",)


def test_support_and_privileged_telemetry_expose_signed_gap_and_metric_distance(
    telemetry,
    rewards,
) -> None:
    above = _support_case(rewards, (0.925, 0.415, 0.535))
    below = _support_case(rewards, (0.0, 0.0, 0.475))

    assert above.z_gap_m == pytest.approx(0.03)
    assert below.z_gap_m == pytest.approx(-0.03)
    assert above.placement_distance_m == pytest.approx(
        math.hypot(math.hypot(0.03, 0.02), 0.03)
    )
    assert below.placement_distance_m == pytest.approx(0.03)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_live_fall_producer_fails_closed_on_nonfinite_critical_force(
    telemetry,
    bad_value: float,
) -> None:
    env = _complete_runtime_env(telemetry)
    pelvis_index = env.scene["robot"].data.body_names.index("pelvis")
    env.scene["contact_forces"].data.net_forces_w[0, pelvis_index, 2] = bad_value

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.produce_evaluator_terminal_evidence(
            env,
            step_idx=1,
            max_steps=20,
            fall_detector=_evaluator_fall_detector(env),
        )

    assert exc_info.value.missing_capabilities == ("critical_body_contact",)


def test_actor_isolation_requires_factory_provenance_and_rejects_copied_truth(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    privileged = telemetry.build_privileged_telemetry(
        task_identity=telemetry.PP_BOX_TASK_IDENTITY,
        env_index=0,
        box_center_w=(0.0, 0.0, 0.505),
        box_linear_velocity_w=(0.0, 0.0, 0.0),
        box_angular_velocity_w=(0.0, 0.0, 0.0),
        support=support,
        left_ee_pose_w=(0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        right_ee_pose_w=(-0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        left_contact=_single_link_hand_contact(telemetry, "left", (0.0, 0.0, 2.0)),
        right_contact=_single_link_hand_contact(telemetry, "right", (0.0, 0.0, 2.0)),
        live_fall_evidence=_fall_lane(telemetry),
        terminal_context=_driver_terminal_context(telemetry),
    )
    copied_privileged_value = tuple(privileged.box_center_w)

    with pytest.raises(telemetry.PrivilegedObservationLeakError):
        telemetry.assert_actor_observation_isolated(
            {"policy": {"robot_joint_state": copied_privileged_value}},
            privileged,
            env=env,
        )

    issued = telemetry.issue_residual_actor_observation(env)
    telemetry.assert_actor_observation_isolated(issued, privileged, env=env)


def test_public_contact_proof_builder_cannot_activate_runtime_mapping(
    telemetry,
) -> None:
    assert not hasattr(telemetry, "build_empirical_contact_mapping_proof")
    env = _complete_runtime_env(telemetry, include_contact_mapping_proofs=False)
    env.recovery_contact_mapping_proofs = {
        key: object()
        for hand in telemetry.default_hand_contact_bindings().values()
        for key in (sensor.sensor_scene_key for sensor in hand.sensors)
    }

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(env)

    assert any(
        capability.startswith("runtime_contact_mapping_receipt:")
        for capability in exc_info.value.missing_capabilities
    )


def _install_fake_contact_calibration_runtime(
    telemetry,
    env,
    *,
    emit_touch_force: bool,
    corrupt_roundtrip: bool = False,
):
    env.action_manager = SimpleNamespace(_action=torch.zeros(1, 1, dtype=torch.float64))
    env.common_step_counter = 0
    env.step_calls = 0
    box = env.scene["box"]

    def write_root_state_to_sim(root_state, *, env_ids):
        assert torch.equal(env_ids, torch.tensor([0]))
        box.data.root_state_w.copy_(root_state)

    box.write_root_state_to_sim = write_root_state_to_sim

    def step(action):
        torch.testing.assert_close(action, env.action_manager._action)
        env.common_step_counter += 1
        env.step_calls += 1
        box_position = box.data.root_state_w[:, :3]
        robot = env.scene["robot"]
        for hand in telemetry.default_hand_contact_bindings().values():
            for binding in hand.sensors:
                sensor = env.scene[binding.sensor_scene_key]
                sensor.data.force_matrix_w.zero_()
                body_index = robot.data.body_names.index(binding.sensor_body_name)
                body_position = robot.data.body_state_w[:, body_index, :3]
                if emit_touch_force and torch.equal(box_position, body_position):
                    sensor.data.force_matrix_w[:, 0, 0, 2] = 2.0
        return (
            {},
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    env.step = step

    def capture_snapshot(runtime_env):
        return {
            "box_root_state": runtime_env.scene["box"].data.root_state_w.clone(),
            "common_step_counter": runtime_env.common_step_counter,
        }

    def snapshot_digest(snapshot):
        payload = snapshot[
            "box_root_state"
        ].detach().cpu().contiguous().numpy().tobytes() + int(
            snapshot["common_step_counter"]
        ).to_bytes(8, "little", signed=True)
        return hashlib.sha256(payload).hexdigest()

    def restore_snapshot(runtime_env, snapshot, *, snapshot_digest: str):
        assert snapshot_digest == snapshot_digest_fn(snapshot)
        runtime_env.scene["box"].data.root_state_w.copy_(snapshot["box_root_state"])
        runtime_env.common_step_counter = snapshot["common_step_counter"]
        for hand in telemetry.default_hand_contact_bindings().values():
            for binding in hand.sensors:
                runtime_env.scene[binding.sensor_scene_key].data.force_matrix_w.zero_()

    snapshot_digest_fn = snapshot_digest

    class RecoveryStateCoordinator:
        def __init__(self):
            self.capture_count = 0
            self.events = []

        @property
        def binding_identity(self):
            return {
                "schema_version": 1,
                "coordinator_type": f"{type(self).__module__}.{type(self).__qualname__}",
                "task_identity": telemetry.PP_BOX_TASK_IDENTITY,
            }

        def capture(self, *, fidelity_tier, required_capabilities=None):
            assert fidelity_tier == "state_only"
            assert required_capabilities is None
            self.events.append("capture")
            self.capture_count += 1
            snapshot = capture_snapshot(env)
            if corrupt_roundtrip and self.capture_count == 2:
                snapshot["common_step_counter"] += 1
            return snapshot

        def digest(self, snapshot):
            self.events.append("digest")
            return snapshot_digest(snapshot)

        def preflight(self, snapshot, *, snapshot_digest, required_capabilities=None):
            assert required_capabilities is None
            self.events.append("preflight")
            assert snapshot_digest == snapshot_digest_fn(snapshot)

        def restore(self, snapshot, *, snapshot_digest, required_capabilities=None):
            assert required_capabilities is None
            self.events.append("restore")
            restore_snapshot(env, snapshot, snapshot_digest=snapshot_digest)

    coordinator = RecoveryStateCoordinator()
    env.recovery_state_coordinator = coordinator
    telemetry.RecoveryStateCoordinator = RecoveryStateCoordinator
    telemetry.install_pp_box_contact_calibration_executor(env)
    return env


def test_contact_calibration_installer_rejects_arbitrary_snapshot_callbacks(
    telemetry,
) -> None:
    env = _complete_runtime_env(telemetry, include_contact_mapping_proofs=False)

    with pytest.raises(TypeError):
        telemetry.install_pp_box_contact_calibration_executor(
            env,
            capture_snapshot=lambda _env: object(),
            restore_snapshot=lambda *_args, **_kwargs: None,
            snapshot_digest=lambda _snapshot: "0" * 64,
        )


def test_contact_calibration_roundtrip_fails_before_any_simulator_step(
    telemetry,
) -> None:
    env = _install_fake_contact_calibration_runtime(
        telemetry,
        _complete_runtime_env(telemetry, include_contact_mapping_proofs=False),
        emit_touch_force=True,
        corrupt_roundtrip=True,
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.execute_pp_box_contact_calibration(env)
    assert exc_info.value.missing_capabilities == (
        "runtime_contact_calibration_snapshot_roundtrip",
    )
    assert env.step_calls == 0


def test_contact_calibration_receipt_binds_coordinator_and_state_only_fidelity(
    telemetry,
) -> None:
    env = _install_fake_contact_calibration_runtime(
        telemetry,
        _complete_runtime_env(telemetry, include_contact_mapping_proofs=False),
        emit_touch_force=True,
    )

    receipt = telemetry.execute_pp_box_contact_calibration(env)

    assert receipt.snapshot_fidelity_tier == "state_only"
    assert receipt.coordinator_binding_identity == tuple(
        sorted(env.recovery_state_coordinator.binding_identity.items())
    )
    assert len(receipt.coordinator_binding_digest) == 64
    assert env.recovery_state_coordinator.events[:6] == [
        "capture",
        "digest",
        "preflight",
        "restore",
        "capture",
        "digest",
    ]


def test_controlled_contact_executor_binds_three_real_steps_per_sensor_and_allows_quiet_live_state(
    telemetry,
) -> None:
    env = _install_fake_contact_calibration_runtime(
        telemetry,
        _complete_runtime_env(telemetry, include_contact_mapping_proofs=False),
        emit_touch_force=True,
    )

    receipt = telemetry.execute_pp_box_contact_calibration(env)
    assert env.step_calls == 16 * 3
    assert len(receipt.sensor_receipts) == 16
    assert all(
        tuple(phase.phase for phase in sensor.phases)
        == ("baseline", "target_touch", "target_removed")
        for sensor in receipt.sensor_receipts
    )
    assert all(
        torch.count_nonzero(env.scene[binding.sensor_scene_key].data.force_matrix_w)
        == 0
        for hand in telemetry.default_hand_contact_bindings().values()
        for binding in hand.sensors
    )
    assert len(telemetry.validate_runtime_hand_contact_sensors(env)) == 16


def test_controlled_contact_executor_rejects_all_zero_claimed_touch_and_publishes_nothing(
    telemetry,
) -> None:
    env = _install_fake_contact_calibration_runtime(
        telemetry,
        _complete_runtime_env(telemetry, include_contact_mapping_proofs=False),
        emit_touch_force=False,
    )

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.execute_pp_box_contact_calibration(env)
    assert "runtime_contact_calibration_touch" in exc_info.value.missing_capabilities
    evidence = exc_info.value.runtime_evidence
    assert evidence["schema"] == "pp_box_contact_calibration_failure_evidence_v1"
    assert evidence["sensor_scene_key"] == "left_box_contact_palm"
    assert evidence["sensor_body_name"] == "left_hand_palm_link"
    assert tuple(phase.phase for phase in evidence["phase_receipts"]) == (
        "baseline",
        "target_touch",
        "target_removed",
    )
    assert evidence["filtered_force"]["shape"] == (1, 1, 1, 3)
    assert len(evidence["filtered_force"]["raw_sha256"]) == 64
    assert evidence["box_root_state_w"]["shape"] == (1, 13)
    assert evidence["sensor_body_pose_w"]["shape"] == (1, 7)

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError):
        telemetry.validate_runtime_hand_contact_sensors(env)


def test_contact_receipt_from_another_runtime_cannot_activate_mapping(
    telemetry,
) -> None:
    source = _install_fake_contact_calibration_runtime(
        telemetry,
        _complete_runtime_env(telemetry, include_contact_mapping_proofs=False),
        emit_touch_force=True,
    )
    target = _install_fake_contact_calibration_runtime(
        telemetry,
        _complete_runtime_env(telemetry, include_contact_mapping_proofs=False),
        emit_touch_force=True,
    )
    receipt = telemetry.execute_pp_box_contact_calibration(source)
    target.recovery_contact_mapping_receipt = receipt

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.validate_runtime_hand_contact_sensors(target)

    assert any(
        capability.startswith("runtime_contact_mapping_receipt:")
        for capability in exc_info.value.missing_capabilities
    )
