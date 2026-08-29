from __future__ import annotations

import ast
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    return _load_module("pp_box_recovery_telemetry", RECOVERY_TELEMETRY_PATH)


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


def test_hand_contact_aggregation_preserves_links_and_uses_any_verified_link(telemetry) -> None:
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


def test_hand_contact_aggregation_rejects_empty_duplicate_or_mixed_side_evidence(telemetry) -> None:
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


def test_default_contact_bindings_are_exact_palm_and_finger_leaf_sensors(telemetry) -> None:
    bindings = telemetry.default_hand_contact_bindings()

    assert set(bindings) == {"left", "right"}
    all_scene_keys = []
    for side, expected_bodies in EXPECTED_HAND_CONTACT_BODIES.items():
        hand = bindings[side]
        assert hand.side == side
        assert hand.ee_body_name == f"{side}_hand_palm_link"
        assert tuple(sensor.sensor_body_name for sensor in hand.sensors) == expected_bodies
        assert tuple(sensor.filtered_body_name for sensor in hand.sensors) == ("Box",) * 8
        assert tuple(sensor.sensor_scene_key for sensor in hand.sensors) == tuple(
            f"{side}_box_contact_{body.removeprefix(f'{side}_hand_').removesuffix('_link')}"
            for body in expected_bodies
        )
        all_scene_keys.extend(sensor.sensor_scene_key for sensor in hand.sensors)

    assert len(all_scene_keys) == len(set(all_scene_keys)) == 16
    assert not any("wrist" in body or "camera" in body for body in sum(EXPECTED_HAND_CONTACT_BODIES.values(), ()))


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
    assert ast.literal_eval(keywords["filter_prim_paths_expr"]) == ["{ENV_REGEX_NS}/Box"]
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
    assert telemetry.classify_terminal(success=success, fall=fall, time_limit=time_limit) == expected


def test_root_up_alignment_normalizes_quaternion_and_classifies_hard_fall(telemetry) -> None:
    identity = telemetry.compute_root_up_alignment((2.0, 0.0, 0.0, 0.0))
    side = telemetry.compute_root_up_alignment((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))
    inverted = telemetry.compute_root_up_alignment((0.0, 1.0, 0.0, 0.0))

    assert identity == pytest.approx(1.0)
    assert side == pytest.approx(0.0)
    assert inverted == pytest.approx(-1.0)
    assert telemetry.classify_fall(identity, critical_body_contact=None) is False
    assert telemetry.classify_fall(inverted, critical_body_contact=None) is True


def test_soft_tilt_requires_critical_contact_evidence_and_fails_closed(telemetry) -> None:
    soft_tilt_alignment = 0.4

    assert telemetry.classify_fall(soft_tilt_alignment, critical_body_contact=False) is False
    assert telemetry.classify_fall(soft_tilt_alignment, critical_body_contact=True) is True
    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.classify_fall(soft_tilt_alignment, critical_body_contact=None)
    assert exc_info.value.missing_capabilities == ("critical_body_contact",)


def test_versioned_privileged_record_contains_required_truth(telemetry, rewards) -> None:
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    left_contact = _single_link_hand_contact(telemetry, "left", (0.0, 0.0, 2.0))
    right_contact = _single_link_hand_contact(telemetry, "right", (0.0, 0.0, 3.0))

    state = telemetry.build_privileged_telemetry(
        env_index=0,
        box_center_w=(0.0, 0.0, 0.505),
        box_linear_velocity_w=(0.1, 0.2, 0.3),
        box_angular_velocity_w=(0.4, 0.5, 0.6),
        support=support,
        left_ee_pose_w=(0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        right_ee_pose_w=(-0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        left_contact=left_contact,
        right_contact=right_contact,
        root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        critical_body_contact=False,
        terminal_context=_driver_terminal_context(telemetry),
        max_ee_box_distance_m=0.3,
    )

    assert state.schema_version == telemetry.RECOVERY_TELEMETRY_SCHEMA_VERSION == 1
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


def test_privileged_record_rejects_unaggregated_or_wrong_side_contact(telemetry, rewards) -> None:
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    pairwise = telemetry.pairwise_contact_evidence(
        (0.0, 0.0, 2.0),
        sensor_body="left_hand_palm_link",
        filtered_body="Box",
    )
    right_contact = _single_link_hand_contact(telemetry, "right", (0.0, 0.0, 2.0))
    common = {
        "env_index": 0,
        "box_center_w": (0.0, 0.0, 0.505),
        "box_linear_velocity_w": (0.0, 0.0, 0.0),
        "box_angular_velocity_w": (0.0, 0.0, 0.0),
        "support": support,
        "left_ee_pose_w": (0.1, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        "right_ee_pose_w": (-0.1, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        "root_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
        "critical_body_contact": False,
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


def test_reward_output_remains_the_existing_binary_success_value(rewards, monkeypatch) -> None:
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
    monkeypatch.setattr(rewards, "_get_shelf_support_surfaces_world", lambda _env: surfaces)
    monkeypatch.setattr(
        rewards,
        "_estimate_target_support_top_z",
        lambda _env, _idx, _center, _surfaces: (0.4, 0.4),
    )

    reward = rewards.compute_reward_pickplace_box(env)

    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0], dtype=torch.float32))


def test_actor_observation_leak_check_is_recursive_and_identity_aware(telemetry, rewards) -> None:
    support = _support_case(rewards, (0.0, 0.0, 0.505))
    left_contact = _single_link_hand_contact(telemetry, "left", (0.0, 0.0, 2.0))
    right_contact = _single_link_hand_contact(telemetry, "right", (0.0, 0.0, 2.0))
    privileged = telemetry.build_privileged_telemetry(
        env_index=0,
        box_center_w=(0.0, 0.0, 0.505),
        box_linear_velocity_w=(0.0, 0.0, 0.0),
        box_angular_velocity_w=(0.0, 0.0, 0.0),
        support=support,
        left_ee_pose_w=(0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        right_ee_pose_w=(-0.15, 0.0, 0.505, 1.0, 0.0, 0.0, 0.0),
        left_contact=left_contact,
        right_contact=right_contact,
        root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        critical_body_contact=False,
        terminal_context=_driver_terminal_context(telemetry),
        max_ee_box_distance_m=0.3,
    )
    safe_actor_observation = {
        "robot_joint_state": torch.zeros(1, 29),
        "robot_gipper_state": torch.zeros(1, 14),
        "camera_image": {"rgb": torch.zeros(1, 2, 2, 3)},
        "frozen_vla": {"latent": torch.zeros(1, 8)},
    }

    telemetry.assert_actor_observation_isolated(safe_actor_observation, privileged)

    leaking_actor_observation = {
        **safe_actor_observation,
        "history": [{"box_center_w": privileged.box_center_w}],
        "opaque_alias": privileged.left_box_contact,
    }
    with pytest.raises(telemetry.PrivilegedObservationLeakError) as exc_info:
        telemetry.assert_actor_observation_isolated(leaking_actor_observation, privileged)
    assert "$.history[0].box_center_w" in exc_info.value.leak_paths
    assert "$.opaque_alias" in exc_info.value.leak_paths


def test_runtime_extractor_rejects_unproven_pairwise_contact_bindings(telemetry) -> None:
    env = SimpleNamespace(num_envs=1, scene={}, device="cpu")

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(env)

    assert "left_box_pairwise_contact" in exc_info.value.missing_capabilities
    assert "right_box_pairwise_contact" in exc_info.value.missing_capabilities


def _complete_runtime_env(telemetry):
    box_root_state = torch.zeros(1, 13, dtype=torch.float64)
    box_root_state[0, 0:3] = torch.tensor([0.0, 0.0, 0.505], dtype=torch.float64)
    box_root_state[0, 3] = 1.0
    box_root_state[0, 7:10] = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    box_root_state[0, 10:13] = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64)
    box = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Box"),
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
    return SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=scene,
        cfg=SimpleNamespace(
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


def test_runtime_extractor_reads_exact_pairwise_force_matrix_and_dynamics(telemetry, rewards) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.505))

    states = telemetry.extract_privileged_telemetry(
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
            lambda env: env.scene["robot"].data.body_names.append("left_hand_palm_link"),
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
            lambda env: setattr(env.scene["right_box_contact_thumb_2"], "num_bodies", 2),
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
        telemetry.extract_privileged_telemetry(
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
        telemetry.extract_privileged_telemetry(
            env,
            support_resolver=lambda _env: [support],
        )

    assert "recovery_telemetry_thresholds" in exc_info.value.missing_capabilities


def test_runtime_extractor_requires_self_consistent_driver_terminal_context(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    support = _support_case(rewards, (0.0, 0.0, 0.7))
    del env.recovery_terminal_contexts

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            support_resolver=lambda _env: [support],
        )
    assert "authoritative_terminal_context" in exc_info.value.missing_capabilities

    contradictory = telemetry.DriverTerminalContext(
        control_step_count=10,
        max_control_steps=2000,
        fall_streak=0,
        fall_confirm_steps=5,
        time_limit=True,
        fall_confirmed=True,
    )
    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            support_resolver=lambda _env: [support],
            terminal_contexts=(contradictory,),
        )
    assert "authoritative_terminal_context" in exc_info.value.missing_capabilities

    state = telemetry.extract_privileged_telemetry(
        env,
        support_resolver=lambda _env: [support],
        terminal_contexts=(_driver_terminal_context(telemetry),),
    )[0]
    assert state.terminal_reason == "running"


def test_runtime_extractor_rejects_nonzero_fall_streak_without_current_candidate(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.recovery_terminal_contexts = (
        _driver_terminal_context(telemetry, fall_streak=1),
    )
    support = _support_case(rewards, (0.0, 0.0, 0.7))

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            support_resolver=lambda _env: [support],
        )

    assert exc_info.value.missing_capabilities == ("authoritative_terminal_context",)


def test_runtime_extractor_rejects_current_fall_candidate_with_zero_streak(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.scene["robot"].data.root_state_w[0, 3:7] = torch.tensor(
        [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
    )
    env.recovery_terminal_contexts = (_driver_terminal_context(telemetry),)
    support = _support_case(rewards, (0.0, 0.0, 0.7))

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            support_resolver=lambda _env: [support],
        )

    assert exc_info.value.missing_capabilities == ("authoritative_terminal_context",)


def test_runtime_extractor_rejects_fall_streak_longer_than_control_history(
    telemetry,
    rewards,
) -> None:
    env = _complete_runtime_env(telemetry)
    env.scene["robot"].data.root_state_w[0, 3:7] = torch.tensor(
        [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
    )
    env.recovery_terminal_contexts = (
        _driver_terminal_context(
            telemetry,
            control_step_count=3,
            fall_streak=4,
        ),
    )
    support = _support_case(rewards, (0.0, 0.0, 0.7))

    with pytest.raises(telemetry.RecoveryTelemetryIncompleteError) as exc_info:
        telemetry.extract_privileged_telemetry(
            env,
            support_resolver=lambda _env: [support],
        )

    assert exc_info.value.missing_capabilities == ("authoritative_terminal_context",)


def test_runtime_extractor_uses_critical_body_contact_for_soft_tilt(telemetry, rewards) -> None:
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
    env.recovery_terminal_contexts = (
        _driver_terminal_context(telemetry, fall_streak=4),
    )

    state = telemetry.extract_privileged_telemetry(
        env,
        support_resolver=lambda _env: [support],
    )[0]

    assert state.root_up_alignment == pytest.approx(up_alignment)
    assert state.fall_candidate is True
    assert state.fall is False
    assert state.terminal_reason == "running"

    env.recovery_terminal_contexts = (
        _driver_terminal_context(telemetry, fall_streak=5),
    )
    state = telemetry.extract_privileged_telemetry(
        env,
        support_resolver=lambda _env: [support],
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

    state = telemetry.extract_privileged_telemetry(
        env,
        support_resolver=lambda _env: [support],
    )[0]

    assert state.success is False
    assert state.fall is False
    assert state.time_limit is False
    assert state.terminal_reason == "running"

    env.recovery_terminal_contexts = (
        _driver_terminal_context(
            telemetry,
            control_step_count=2000,
            max_control_steps=2000,
        ),
    )
    state = telemetry.extract_privileged_telemetry(
        env,
        support_resolver=lambda _env: [support],
    )[0]
    assert state.time_limit is True
    assert state.terminal_reason == "time_limit"
