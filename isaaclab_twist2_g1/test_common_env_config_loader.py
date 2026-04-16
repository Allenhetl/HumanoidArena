from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


THIS_DIR = Path(__file__).resolve().parent


def _load_loader_module():
    loader_path = THIS_DIR / "tasks" / "common_env_config" / "loader.py"
    spec = importlib.util.spec_from_file_location("common_env_config_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_env_config_yaml_rejects_task_name_mismatch() -> None:
    loader = _load_loader_module()
    env_cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.001), decimation=10)

    with pytest.raises(ValueError, match="task_name mismatch"):
        loader.apply_env_config_yaml(
            env_cfg,
            "football_single_twist2.yaml",
            task_name="Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
            route_name="twist2",
        )
