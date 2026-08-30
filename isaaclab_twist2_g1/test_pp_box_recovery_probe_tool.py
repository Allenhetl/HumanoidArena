from __future__ import annotations

import importlib.util
import json
import math
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
        lambda _args, _report: (_ for _ in ()).throw(SystemExit(0)),
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
