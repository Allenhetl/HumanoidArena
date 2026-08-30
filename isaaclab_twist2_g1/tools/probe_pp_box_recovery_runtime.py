#!/usr/bin/env python3
"""Controlled headless PP-box recovery telemetry and snapshot probe."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

TASK_IDENTITY = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"
TASK_MODULE = "tasks.g1_tasks.move_pickplace_box_g1_29dof_dex3_wholedoby"
DEFAULT_ENV_CONFIG = "tasks/common_env_config/pickplace_box_sonic.yaml"
ACTION_CONTRACT = {
    "method": "ReCoVLA-GR00T-arms14",
    "base_horizon": 40,
    "base_commit": 40,
    "native_action_dim": 40,
    "residual_cadence": "primitive_control_step",
    "owned_indices": [
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
    ],
    "base_owned_hand_indices": [38, 39],
}


def build_evaluator_fall_args() -> SimpleNamespace:
    """Return the existing evaluator's default, enabled fall-detector settings."""

    return SimpleNamespace(
        disable_fall_detection=False,
        fall_tilt_deg=60.0,
        fall_hard_tilt_deg=75.0,
        fall_contact_force_threshold=50.0,
        fall_confirm_steps=5,
        verbose_startup=False,
    )


def initial_report(args: argparse.Namespace | SimpleNamespace) -> dict[str, object]:
    return {
        "schema": "ha_pp_box_recovery_runtime_probe_v1",
        "status": "running",
        "evidence_layer": "real_isaac_task_runtime_probe",
        "run_id": str(args.run_id),
        "source": {
            "git_sha": str(args.source_sha),
            "archive_sha256": str(args.source_archive_sha256),
        },
        "task": {
            "identity": str(args.task),
            "env_config_yaml": str(args.env_config_yaml),
            "num_envs": 1,
            "seed": int(args.seed),
            "device": str(args.device),
        },
        "action_contract": ACTION_CONTRACT,
        "claims": {
            "task_side_state_snapshot_roundtrip": False,
            "task_side_contact_mapping": False,
            "task_side_privileged_telemetry": False,
            "actor_observation_isolation": False,
            "gr00t_provider_exact_continuation": False,
            "semantic40_executed_action_comparison": False,
            "m4_training_or_efficacy": False,
        },
        "snapshot": {},
        "contact": {},
        "telemetry": {},
        "runtime": {},
    }


def write_report_exclusive(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    with path.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.write("\n")


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _nvidia_smi_identity() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return {"available": True, "rows": rows}


def _action_shape(env: object) -> tuple[int, ...]:
    shape = getattr(getattr(env, "action_space", None), "shape", None)
    if not shape:
        raise RuntimeError("PP-box runtime action_space has no fixed shape")
    normalized = tuple(int(value) for value in shape)
    if len(normalized) == 1:
        return (int(env.num_envs), normalized[0])
    return normalized


def _state_only_roundtrip(env: object, mdp: object) -> dict[str, object]:
    coordinator = getattr(env, "recovery_state_coordinator", None)
    if not isinstance(coordinator, mdp.RecoveryStateCoordinator):
        raise TypeError("PP-box recovery state coordinator was not installed")
    snapshot = coordinator.capture(fidelity_tier="state_only")
    digest = coordinator.digest(snapshot)
    coordinator.preflight(snapshot, snapshot_digest=digest)
    coordinator.restore(snapshot, snapshot_digest=digest)
    restored = coordinator.capture(fidelity_tier="state_only")
    restored_digest = coordinator.digest(restored)
    if restored_digest != digest:
        raise RuntimeError("state-only snapshot digest changed after restore")
    return {
        "status": "passed",
        "fidelity_tier": snapshot.fidelity_tier,
        "snapshot_digest": digest,
        "restored_snapshot_digest": restored_digest,
        "scope": _jsonable(snapshot.scope),
        "available_capabilities": dict(snapshot.capabilities.available),
    }


def _exact_continuation_capability(env: object, mdp: object) -> dict[str, object]:
    coordinator = env.recovery_state_coordinator
    try:
        snapshot = coordinator.capture(fidelity_tier="exact_continuation")
    except mdp.RecoveryStateIncompleteError as exc:
        return {
            "status": "unsupported_fidelity",
            "exception_type": type(exc).__name__,
            "operation": exc.operation,
            "missing_capabilities": list(exc.missing_capabilities),
            "available_capabilities": dict(exc.available),
        }
    digest = coordinator.digest(snapshot)
    coordinator.preflight(snapshot, snapshot_digest=digest)
    coordinator.restore(snapshot, snapshot_digest=digest)
    return {
        "status": "captured_and_restored",
        "snapshot_digest": digest,
        "scope": _jsonable(snapshot.scope),
        "warning": (
            "This task probe does not compare semantic40 transitions; integrated "
            "GR00T provider evidence is still required."
        ),
    }


def _run_contact_and_telemetry(
    env: object,
    mdp: object,
    *,
    max_steps: int,
) -> tuple[dict[str, object], dict[str, object]]:
    import torch

    mdp.install_pp_box_contact_calibration_executor(env)
    receipt = mdp.execute_pp_box_contact_calibration(env)
    reports = mdp.validate_runtime_hand_contact_sensors(env)
    if len(receipt.sensor_receipts) != 16 or len(reports) != 16:
        raise RuntimeError("PP-box contact calibration did not cover 16 sensors")
    if any(len(sensor.phases) != 3 for sensor in receipt.sensor_receipts):
        raise RuntimeError("PP-box contact calibration did not execute three phases")

    # The controlled calibration changes physical state for its proof. Start the
    # telemetry sample from a fresh task reset while retaining its bound receipt.
    env.reset()
    event_manager = getattr(getattr(env, "cfg", None), "event_manager", None)
    if not callable(getattr(event_manager, "trigger", None)):
        raise TypeError("PP-box task reset event manager is unavailable")
    event_manager.trigger("reset_all_self", env)
    action = torch.zeros(_action_shape(env), device=env.device, dtype=torch.float32)
    transition = env.step(action)
    if not isinstance(transition, tuple) or len(transition) != 5:
        raise RuntimeError("PP-box env.step did not return the Gymnasium 5-tuple")
    _observation, reward, terminated, truncated, _extras = transition

    from script.eval_scripts.sonic.sim_eval_vla import _build_fall_detector

    fall_detector = _build_fall_detector(env, build_evaluator_fall_args())
    if not isinstance(fall_detector, Mapping):
        raise TypeError("evaluator fall detector was not enabled")
    terminal = mdp.produce_evaluator_terminal_evidence(
        env,
        step_idx=1,
        max_steps=max_steps,
        fall_detector=fall_detector,
    )
    actor_observation = mdp.issue_residual_actor_observation(env)
    privileged = mdp.extract_privileged_telemetry(
        env,
        terminal_evidence=terminal,
        actor_observation=actor_observation,
    )
    if len(privileged) != 1:
        raise RuntimeError("PP-box telemetry must contain exactly one environment lane")

    contact = {
        "status": "passed",
        "executor": "controlled_three_phase_executor",
        "simulator_steps": 16 * 3,
        "execution_receipt": _jsonable(receipt),
        "sensor_reports": _jsonable(reports),
    }
    telemetry = {
        "status": "passed",
        "task_step": {
            "action_shape": list(action.shape),
            "reward": _jsonable(reward.detach().cpu().tolist()),
            "terminated": _jsonable(terminated.detach().cpu().tolist()),
            "truncated": _jsonable(truncated.detach().cpu().tolist()),
        },
        "fall_detector": _jsonable(fall_detector),
        "terminal_evidence": _jsonable(terminal),
        "actor_observation": {
            "schema_version": actor_observation.schema_version,
            "policy_term_names": list(actor_observation.policy_term_names),
            "payload_digest": actor_observation.payload_digest,
        },
        "privileged_state": _jsonable(privileged),
    }
    return contact, telemetry


def run_runtime_probe(args: argparse.Namespace, report: dict[str, object]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", str(project_root))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from isaaclab.app import AppLauncher

    profile_args = SimpleNamespace(
        task=args.task,
        task_runtime_profile="auto",
        input_source="pico_sonic",
        gmt_backend="sonic",
        replay_file="",
        replay_mode="",
        record_during_replay=False,
    )
    from task_runtime_profiles import apply_task_runtime_profile

    apply_task_runtime_profile(profile_args)
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    env = None
    try:
        import importlib

        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tasks.common_env_config import apply_env_config_yaml

        importlib.import_module(TASK_MODULE)
        task_module = importlib.import_module(f"{TASK_MODULE}.mdp")
        from script.eval_scripts.sonic.sim_eval_vla import (
            _configure_episode_seed_state,
        )

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.env_name = args.task
        env_cfg.recovery_task_identity = args.task
        resolved_yaml = apply_env_config_yaml(
            env_cfg,
            args.env_config_yaml,
            task_name=args.task,
            route_name="sonic",
        )
        _configure_episode_seed_state(env_cfg, args.seed)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env_cfg.initialize_task_scene(env, profile_args)
        env.sim.reset()
        env.reset()
        env_cfg.event_manager.trigger("reset_all_self", env)

        report["runtime"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(env.device),
            "nvidia_smi": _nvidia_smi_identity(),
            "env_cfg_type": type(env_cfg).__qualname__,
            "resolved_env_config_yaml": str(resolved_yaml),
            "task_action_space_shape": list(_action_shape(env)),
        }
        snapshot = report["snapshot"]
        assert isinstance(snapshot, dict)
        snapshot["state_only"] = _state_only_roundtrip(env, task_module)
        snapshot["exact_continuation"] = _exact_continuation_capability(
            env, task_module
        )
        claims = report["claims"]
        assert isinstance(claims, dict)
        claims["task_side_state_snapshot_roundtrip"] = True

        contact, telemetry = _run_contact_and_telemetry(
            env,
            task_module,
            max_steps=args.max_steps,
        )
        report["contact"] = contact
        report["telemetry"] = telemetry
        claims["task_side_contact_mapping"] = True
        claims["task_side_privileged_telemetry"] = True
        claims["actor_observation_isolation"] = True
        report["status"] = "passed"
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled HumanoidArena PP-box recovery runtime probe"
    )
    parser.add_argument("--task", default=TASK_IDENTITY)
    parser.add_argument("--env_config_yaml", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--source_sha", required=True)
    parser.add_argument("--source_archive_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = _base_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.task != TASK_IDENTITY:
        parser.error(f"--task must be {TASK_IDENTITY}")
    if args.seed < 0 or args.max_steps <= 0:
        parser.error("--seed must be non-negative and --max_steps must be positive")

    report = initial_report(args)
    try:
        run_runtime_probe(args, report)
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_report_exclusive(args.output, report)
        raise
    write_report_exclusive(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
