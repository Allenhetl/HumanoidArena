#!/usr/bin/env python3
"""Validate a real-scene release against locked files and numeric gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--policy", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_value(data: dict[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_path)
        value = value[key]
    return value


def gate_passes(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "min":
        return actual >= expected
    if operator == "max":
        return actual <= expected
    if operator == "eq":
        return actual == expected
    raise ValueError(f"Unsupported gate operator: {operator}")


def check_files(entries: list[dict[str, Any]], root: Path, label: str) -> list[str]:
    errors: list[str] = []
    for entry in entries:
        path = root / entry["path"]
        required = bool(entry.get("required", True))
        if not path.is_file():
            message = f"{label} missing: {path}"
            if required:
                errors.append(message)
            else:
                print(f"WARN {message}")
            continue
        expected_bytes = entry.get("bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            errors.append(f"{label} size mismatch: {path}")
            continue
        actual_hash = sha256(path)
        if actual_hash != entry["sha256"]:
            errors.append(f"{label} SHA-256 mismatch: {path}")
            continue
        print(f"PASS {label}: {entry.get('role', path.name)}")
    return errors


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    repo_root = (
        args.repo_root.expanduser().resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[3]
    )
    policy_path = (
        args.policy.expanduser().resolve()
        if args.policy
        else repo_root / "isaaclab_twist2_g1/real_scenes/acceptance_policy.json"
    )

    lock = load_json(scene_dir / "manifest.lock.json")
    acceptance = load_json(scene_dir / "acceptance.json")
    policy = load_json(policy_path)
    scene_id = lock.get("scene_id")
    if not scene_id or acceptance.get("scene_id") != scene_id:
        raise ValueError("Scene IDs in manifest.lock.json and acceptance.json must match")
    if acceptance.get("policy_id") != policy.get("policy_id"):
        raise ValueError("acceptance.json references a different acceptance policy")

    required_failures: list[str] = []
    warning_failures: list[str] = []
    for gate in policy["gates"]:
        try:
            actual = nested_value(acceptance, gate["path"])
            passed = gate_passes(actual, gate["operator"], gate["value"])
            detail = f"{gate['id']}: actual={actual!r} {gate['operator']} expected={gate['value']!r}"
        except (KeyError, TypeError) as error:
            passed = False
            detail = f"{gate['id']}: unavailable ({error})"
        if passed:
            print(f"PASS gate: {detail}")
        elif gate.get("severity", "required") == "warning":
            warning_failures.append(detail)
            print(f"WARN gate: {detail}")
        else:
            required_failures.append(detail)
            print(f"FAIL gate: {detail}")

    required_failures.extend(check_files(lock["control_files"], repo_root, "control file"))
    if args.asset_dir:
        required_failures.extend(
            check_files(lock["artifacts"], args.asset_dir.expanduser().resolve(), "artifact")
        )
    else:
        print("SKIP artifact files: pass --asset-dir to verify deployed payloads")

    summary = {
        "scene_id": scene_id,
        "release_status": lock.get("release_status"),
        "required_failures": len(required_failures),
        "warnings": len(warning_failures),
        "artifacts_checked": bool(args.asset_dir),
    }
    print(json.dumps(summary, indent=2))
    if required_failures:
        for failure in required_failures:
            print(f"ERROR {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
