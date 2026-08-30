from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TASK_ID = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"
MDP_DIR = (
    Path(__file__).resolve().parent
    / "tasks"
    / "g1_tasks"
    / "move_pickplace_box_g1_29dof_dex3_wholedoby"
    / "mdp"
)
PACKAGE_NAME = "pp_box_recovery_runtime_tests"


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
    _load_module(f"{PACKAGE_NAME}.recovery_state", MDP_DIR / "recovery_state.py")
    telemetry = _load_module(
        f"{PACKAGE_NAME}.recovery_telemetry", MDP_DIR / "recovery_telemetry.py"
    )
    _load_module(f"{PACKAGE_NAME}.rewards", MDP_DIR / "rewards.py")
    failures = _load_module(
        f"{PACKAGE_NAME}.recovery_failures", MDP_DIR / "recovery_failures.py"
    )
    runtime = _load_module(
        f"{PACKAGE_NAME}.recovery_runtime", MDP_DIR / "recovery_runtime.py"
    )
    return SimpleNamespace(
        recovery_failures=failures,
        recovery_runtime=runtime,
        recovery_telemetry=telemetry,
    )


def _contact(telemetry_api, side: str, active: bool):
    return telemetry_api.HandContactEvidence(
        side=side,
        links=(),
        contacting_bodies=("Box",) if active else (),
        resultant_force_w=(2.0, 0.0, 0.0) if active else (0.0, 0.0, 0.0),
        total_magnitude_n=2.0 if active else 0.0,
        in_contact=active,
    )


def _telemetry(
    telemetry_api,
    *,
    step: int,
    grasp: bool = False,
    placement_distance_m: float = 1.0,
):
    left = _contact(telemetry_api, "left", grasp)
    right = _contact(telemetry_api, "right", grasp)
    grasp_evidence = telemetry_api.BimanualGraspEvidence(
        left_ee_box_distance_m=0.1,
        right_ee_box_distance_m=0.1,
        max_ee_box_distance_m=0.1,
        left_pose_valid=True,
        right_pose_valid=True,
        pose_evidence=True,
        pairwise_contact=grasp,
        bimanual_grasp=grasp,
    )
    return telemetry_api.PrivilegedRecoveryTelemetry(
        schema_version=telemetry_api.RECOVERY_TELEMETRY_SCHEMA_VERSION,
        task_identity=TASK_ID,
        env_index=0,
        box_center_w=(-0.5, -4.0, 0.105),
        box_linear_velocity_w=(0.0, 0.0, 0.0),
        box_angular_velocity_w=(0.0, 0.0, 0.0),
        shelf_bounds_w=(-3.0, -2.0, -2.0, -1.0),
        support_surface_z_m=0.0,
        target_support_surface_z_m=0.65,
        left_ee_pose_w=(-0.55, -4.0, 0.2, 1.0, 0.0, 0.0, 0.0),
        right_ee_pose_w=(-0.45, -4.0, 0.2, 1.0, 0.0, 0.0, 0.0),
        left_box_contact=left,
        right_box_contact=right,
        grasp_evidence=grasp_evidence,
        grasp=grasp,
        xy_mismatch_m=1.0,
        z_gap_m=-0.545,
        z_mismatch_m=0.545,
        placement_distance_m=placement_distance_m,
        placement=False,
        success=False,
        root_up_alignment=1.0,
        control_step_count=step,
        max_control_steps=2000,
        fall_candidate=False,
        fall_streak=0,
        fall_confirm_steps=5,
        fall=False,
        time_limit=False,
        terminal_reason="running",
    )


def _live_env() -> tuple[SimpleNamespace, SimpleNamespace]:
    action = torch.zeros(40, dtype=torch.float32).numpy()
    action[38:40] = 1.0
    provider = SimpleNamespace(
        _latest_executed_canonical_action=action,
        _latest_executed_source_control_step=7,
    )
    box_state = torch.zeros((1, 13), dtype=torch.float32)
    box_state[0, :7] = torch.tensor([-0.5, -4.0, 0.105, 1.0, 0.0, 0.0, 0.0])
    env = SimpleNamespace(
        num_envs=1,
        cfg=SimpleNamespace(
            env_name=TASK_ID,
            seed=123,
            recovery_runtime_thresholds={
                "ground_surface_z_m": 0.0,
                "ground_support_tolerance_m": 0.02,
                "linear_stable_speed_mps": 0.02,
                "angular_stable_speed_radps": 0.05,
                "progress_epsilon_m": 0.005,
                "stall_confirm_steps": 1,
                "stable_confirm_steps": 1,
                "place_attempt_distance_m": 0.25,
                "axis_alignment_tolerance_deg": 10.0,
            },
        ),
        scene={"box": SimpleNamespace(data=SimpleNamespace(root_state_w=box_state))},
    )
    return env, provider


def test_live_runtime_builds_dropped_from_provider_and_task_state(modules) -> None:
    env, provider = _live_env()
    runtime = modules.recovery_runtime.PPBoxLiveRecoveryRuntime(env, provider)

    first = runtime.observe(_telemetry(modules.recovery_telemetry, step=1))
    second = runtime.observe(_telemetry(modules.recovery_telemetry, step=2))

    assert first.attempt.pickup_attempted is True
    assert first.attempt.stalled is False
    assert second.attempt.stalled is True
    assert second.attempt.stable is True
    assert second.ground_supported is True
    assert second.target_disjoint is True
    assert second.box_axis_aligned is True
    assert modules.recovery_failures.classify_recoverable_failure(second) == "dropped"
    identity = runtime.identity()
    assert identity["schema"] == "ha_pp_box_live_recovery_runtime_identity_v1"
    assert identity["task_identity"] == TASK_ID
    assert identity["thresholds_sha256"] == (
        "5bcdbad04f690c53375f63d7b3148df8d6b2204192f4a52f064f211228711529"
    )
    assert identity["exclusive_num_envs"] == 1


def test_runtime_snapshot_roundtrip_has_no_recovery_stage_or_fsm(modules) -> None:
    env, provider = _live_env()
    runtime = modules.recovery_runtime.PPBoxLiveRecoveryRuntime(env, provider)
    runtime.observe(_telemetry(modules.recovery_telemetry, step=1))
    expected = runtime.observe(_telemetry(modules.recovery_telemetry, step=2))
    snapshot = runtime.capture_state()
    forbidden = {"stage", "phase", "fsm", "current_stage", "phase_enter_step"}
    assert forbidden.isdisjoint(snapshot)

    runtime.clear()
    runtime.restore_state(snapshot)
    restored = runtime.observe(_telemetry(modules.recovery_telemetry, step=3))

    assert restored.attempt.attempt_count == expected.attempt.attempt_count
    assert restored.attempt.pickup_attempted is True
    assert restored.attempt.stable is True


def test_runtime_fails_closed_on_wrong_task_or_nonfinite_provider_action(
    modules,
) -> None:
    env, provider = _live_env()
    env.cfg.env_name = "Isaac-Another-Task"
    with pytest.raises(ValueError, match="env_name"):
        modules.recovery_runtime.PPBoxLiveRecoveryRuntime(env, provider)

    env, provider = _live_env()
    provider._latest_executed_canonical_action[20] = float("nan")
    runtime = modules.recovery_runtime.PPBoxLiveRecoveryRuntime(env, provider)
    with pytest.raises(ValueError, match="executed semantic40"):
        runtime.observe(_telemetry(modules.recovery_telemetry, step=1))
