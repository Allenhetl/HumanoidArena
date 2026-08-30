from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROBE_PATH = Path(__file__).parent / "tools" / "probe_pp_box_recovery_runtime.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "pp_box_recovery_probe_tool", PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_contract_is_ha_h40_c40_arms14() -> None:
    probe = _load_probe_module()

    assert probe.TASK_IDENTITY == "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"
    assert probe.ACTION_CONTRACT == {
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


def test_build_fall_detector_args_reuses_evaluator_defaults() -> None:
    probe = _load_probe_module()
    args = probe.build_evaluator_fall_args()

    assert args.disable_fall_detection is False
    assert args.fall_tilt_deg == 60.0
    assert args.fall_hard_tilt_deg == 75.0
    assert args.fall_contact_force_threshold == 50.0
    assert args.fall_confirm_steps == 5
    assert math.isclose(math.cos(math.radians(args.fall_tilt_deg)), 0.5)


def test_probe_report_write_is_exclusive_and_ascii(tmp_path: Path) -> None:
    probe = _load_probe_module()
    output = tmp_path / "probe.json"
    report = probe.initial_report(
        SimpleNamespace(
            run_id="RECOVLA-HA-PPBOX-RUNTIME-TEST",
            source_sha="a" * 40,
            source_archive_sha256="b" * 64,
            seed=20260830,
            task=probe.TASK_IDENTITY,
            env_config_yaml="tasks/common_env_config/pickplace_box_sonic.yaml",
            device="cuda:0",
        )
    )

    probe.write_report_exclusive(output, report)
    loaded = json.loads(output.read_text(encoding="ascii"))
    assert loaded["evidence_layer"] == "real_isaac_task_runtime_probe"
    assert loaded["claims"]["gr00t_provider_exact_continuation"] is False
    assert loaded["action_contract"] == probe.ACTION_CONTRACT

    with pytest.raises(FileExistsError):
        probe.write_report_exclusive(output, report)


def test_probe_system_exit_before_completion_is_reported_and_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    output = tmp_path / "probe-system-exit.json"

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", default="cpu")

    fake_app = SimpleNamespace(AppLauncher=FakeAppLauncher)
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(app=fake_app))
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)
    monkeypatch.setattr(
        probe,
        "run_runtime_probe",
        lambda _args, _report, _progress: (_ for _ in ()).throw(SystemExit(0)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PROBE_PATH),
            "--run_id",
            "RECOVLA-HA-PPBOX-SYSTEM-EXIT-TEST",
            "--source_sha",
            "a" * 40,
            "--source_archive_sha256",
            "b" * 64,
            "--output",
            str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="SystemExit before completion"):
        probe.main()

    report = json.loads(output.read_text(encoding="ascii"))
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "SystemExit"
    assert report["failure"]["exit_code"] == 0


def test_probe_python_process_exit_is_intercepted_with_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    output = tmp_path / "probe-process-exit.json"
    original_exit_calls: list[int] = []

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", default="cpu")

    fake_app = SimpleNamespace(AppLauncher=FakeAppLauncher)
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(app=fake_app))
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    def original_exit(code: int) -> None:
        original_exit_calls.append(code)
        raise AssertionError("the unguarded os._exit implementation was called")

    monkeypatch.setattr(os, "_exit", original_exit)
    monkeypatch.setattr(
        probe,
        "run_runtime_probe",
        lambda _args, _report, _progress: os._exit(0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PROBE_PATH),
            "--run_id",
            "RECOVLA-HA-PPBOX-PROCESS-EXIT-TEST",
            "--source_sha",
            "a" * 40,
            "--source_archive_sha256",
            "b" * 64,
            "--output",
            str(output),
        ],
    )

    with pytest.raises(probe.ProcessExitAttempt, match=r"os\._exit\(0\)"):
        probe.main()

    assert original_exit_calls == []
    assert os._exit is original_exit
    report = json.loads(output.read_text(encoding="ascii"))
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "ProcessExitAttempt"
    assert "run_runtime_probe" in report["failure"]["traceback"]


def test_probe_progress_trace_is_append_only_and_sequenced(tmp_path: Path) -> None:
    probe = _load_probe_module()
    progress_path = tmp_path / "probe.progress.jsonl"
    recorder = probe.ProgressRecorder(
        path=progress_path,
        run_id="RECOVLA-HA-PPBOX-PROGRESS-TEST",
    )

    recorder.record("target_task_import", "entered")
    recorder.record("target_task_import", "completed")

    events = [
        json.loads(line)
        for line in progress_path.read_text(encoding="ascii").splitlines()
    ]
    assert [event["sequence"] for event in events] == [0, 1]
    assert [event["status"] for event in events] == ["entered", "completed"]
    assert all(event["run_id"] == recorder.run_id for event in events)

    occupied = probe.ProgressRecorder(path=progress_path, run_id="other-run")
    with pytest.raises(FileExistsError):
        occupied.record("probe_main", "entered")


def test_runtime_failure_report_exists_before_native_cleanup(tmp_path: Path) -> None:
    probe = _load_probe_module()
    output = tmp_path / "probe-runtime-failure.json"
    progress = probe.ProgressRecorder(
        path=tmp_path / "probe-runtime-failure.progress.jsonl",
        run_id="RECOVLA-HA-PPBOX-RUNTIME-FAILURE-TEST",
    )
    report = probe.initial_report(
        SimpleNamespace(
            run_id=progress.run_id,
            source_sha="a" * 40,
            source_archive_sha256="b" * 64,
            seed=20260830,
            task=probe.TASK_IDENTITY,
            env_config_yaml="tasks/common_env_config/pickplace_box_sonic.yaml",
            device="cuda:0",
        )
    )
    cleanup_observed_report: list[bool] = []

    with pytest.raises(ValueError, match="runtime import failed"):
        try:
            raise ValueError("runtime import failed")
        except BaseException as exc:
            probe.persist_runtime_failure_report(output, report, progress, exc)
            raise
        finally:
            cleanup_observed_report.append(output.exists())

    assert cleanup_observed_report == [True]
    loaded = json.loads(output.read_text(encoding="ascii"))
    assert loaded["status"] == "failed"
    assert loaded["failure"]["type"] == "ValueError"
    assert loaded["failure"]["message"] == "runtime import failed"


def test_main_does_not_overwrite_pre_persisted_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _load_probe_module()
    output = tmp_path / "probe-pre-persisted.json"

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", default="cpu")

    fake_app = SimpleNamespace(AppLauncher=FakeAppLauncher)
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(app=fake_app))
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    def fail_after_persist(args, report, progress) -> None:
        try:
            raise ValueError("persisted before cleanup")
        except BaseException as exc:
            probe.persist_runtime_failure_report(args.output, report, progress, exc)
            raise

    monkeypatch.setattr(probe, "run_runtime_probe", fail_after_persist)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PROBE_PATH),
            "--run_id",
            "RECOVLA-HA-PPBOX-PRE-PERSISTED-TEST",
            "--source_sha",
            "a" * 40,
            "--source_archive_sha256",
            "b" * 64,
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="persisted before cleanup"):
        probe.main()

    report = json.loads(output.read_text(encoding="ascii"))
    assert report["failure"]["type"] == "ValueError"
    assert report["failure"]["message"] == "persisted before cleanup"
