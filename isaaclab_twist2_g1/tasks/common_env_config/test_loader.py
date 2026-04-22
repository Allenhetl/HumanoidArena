from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml


THIS_DIR = Path(__file__).resolve().parent
G1_TASKS_DIR = THIS_DIR.parent / "g1_tasks"


def _load_loader_module():
    loader_path = THIS_DIR / "loader.py"
    spec = importlib.util.spec_from_file_location("common_env_config_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _registered_g1_task_ids() -> set[str]:
    task_ids: set[str] = set()
    for init_path in G1_TASKS_DIR.glob("*/__init__.py"):
        text = init_path.read_text(encoding="utf-8")
        task_ids.update(re.findall(r'id="([^"]+)"', text))
    return task_ids


def test_get_env_config_task_name_reads_yaml_metadata():
    loader = _load_loader_module()
    task_name = loader.get_env_config_task_name("football_single_twist2.yaml")
    assert task_name == "Isaac-Move-Football-Single-G129-Dex3-Wholebody"


def test_every_g1_task_has_sonic_and_twist2_yaml():
    registered_task_ids = _registered_g1_task_ids()
    coverage: dict[str, dict[str, list[str]]] = {"sonic": {}, "twist2": {}}

    for yaml_path in THIS_DIR.glob("*.yaml"):
        raw_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        backend = raw_cfg.get("backend")
        task_name = raw_cfg.get("task_name")
        if backend not in coverage or not isinstance(task_name, str):
            continue
        coverage[backend].setdefault(task_name, []).append(yaml_path.name)

    for backend, backend_coverage in coverage.items():
        assert set(backend_coverage) == registered_task_ids, (
            f"{backend} yaml coverage mismatch: "
            f"missing={sorted(registered_task_ids - set(backend_coverage))}, "
            f"extra={sorted(set(backend_coverage) - registered_task_ids)}"
        )
        duplicates = {
            task_name: names for task_name, names in backend_coverage.items() if len(names) != 1
        }
        assert not duplicates, f"{backend} yaml files must map 1:1 to tasks: {duplicates}"
