#!/usr/bin/env python3
"""Batch replay TWIST2 recordings and re-record them as multicam episodes."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_ROOT = REPO_ROOT / "isaaclab_twist2_g1"
SIM_MAIN = ISAACLAB_ROOT / "sim_main.py"
DEFAULT_ISAACLAB_PY = Path(
    "/home/dreams/miniconda3/envs/unitree_sim_env_isaaclab5_0/bin/python"
)
DEFAULT_BOX_ROBOT_USD = (
    ISAACLAB_ROOT
    / "assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"
).resolve()
DEFAULT_FOURPOINTS_ROBOT_USD = (
    ISAACLAB_ROOT
    / "assets/robots/g1-29dof_wholebody_dex3/temp/g1_29dof_with_dex3_rev_1_0_fourpoints.usd"
).resolve()
DEFAULT_INPUT_ROOT = ISAACLAB_ROOT / "recording_data/HOI_double_desk/twist2"
DEFAULT_ENV_CONFIG_YAML = (
    ISAACLAB_ROOT / "tasks/common_env_config/doubledesk_twist2.yaml"
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch replay TWIST2 recordings and re-record them as multicam episodes."
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        default=str(DEFAULT_INPUT_ROOT),
        type=str,
        help="Root directory containing source TWIST2 .npz files.",
    )
    parser.add_argument(
        "--env-config-yaml",
        type=str,
        default=str(DEFAULT_ENV_CONFIG_YAML),
        help="Environment config YAML used for every replay job.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="auto",
        help="Python interpreter used to launch sim_main.py.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--robot-type", type=str, default="g129")
    parser.add_argument(
        "--robot-collider-mode",
        type=str,
        default="box",
        choices=["box", "fourpoints", "default"],
    )
    parser.add_argument(
        "--replay-mode",
        type=str,
        default="direct",
        choices=["direct", "inference", "direct_replay", "inference_replay"],
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Explicit output root. Defaults to <input_root>_multicam_rerecord.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recording-save-workers", type=int, default=1)
    parser.add_argument("--recording-save-queue-size", type=int, default=4)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def resolve_python_bin(requested: str) -> str:
    if requested != "auto":
        return requested
    if DEFAULT_ISAACLAB_PY.is_file():
        return str(DEFAULT_ISAACLAB_PY)
    return "python"


def resolve_robot_usd_override(args: argparse.Namespace) -> str | None:
    if args.robot_collider_mode == "default":
        return None
    if args.robot_collider_mode == "fourpoints":
        return str(DEFAULT_FOURPOINTS_ROBOT_USD)
    return str(DEFAULT_BOX_ROBOT_USD)


def build_env(robot_usd_override: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(ISAACLAB_ROOT)
    env["PYTHONPATH"] = str(ISAACLAB_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if robot_usd_override:
        env["ROBOT_USD_OVERRIDE"] = robot_usd_override
    else:
        env.pop("ROBOT_USD_OVERRIDE", None)
    return env


def find_npz_files(input_root: Path) -> list[Path]:
    return sorted(path.resolve() for path in input_root.rglob("*.npz"))


def read_task_name(npz_path: Path) -> str:
    with np.load(npz_path, allow_pickle=True) as data:
        task = data.get("task")
        if task is None:
            raise KeyError(f"{npz_path} missing task field")
        if hasattr(task, "item"):
            task = task.item()
        return str(task)


def read_rerecord_final_reward(npz_path: Path) -> float | None:
    with np.load(npz_path, allow_pickle=True) as data:
        if "rerecord_final_reward" not in data:
            return None
        value = np.asarray(data["rerecord_final_reward"], dtype=np.float32).reshape(-1)
        if value.size == 0:
            return None
        return float(value[0])


def find_new_npz_files(output_dir: Path, before_files: set[Path]) -> list[Path]:
    after_files = {path.resolve() for path in output_dir.glob("*.npz")}
    return sorted(after_files - before_files, key=lambda path: path.stat().st_mtime)


def build_command(
    args: argparse.Namespace,
    *,
    python_bin: str,
    replay_file: Path,
    output_dir: Path,
    task_name: str,
) -> list[str]:
    command = [
        python_bin,
        "-u",
        str(SIM_MAIN),
        "--device",
        args.device,
        "--env_config_yaml",
        args.env_config_yaml,
        "--task",
        task_name,
        "--robot_type",
        args.robot_type,
        "--input_source",
        "replay",
        "--gmt_backend",
        "twist2",
        "--replay_file",
        str(replay_file),
        "--replay_mode",
        args.replay_mode,
        "--record_during_replay",
        "--exit_when_replay_complete",
        "--recording_save_dir",
        str(output_dir),
        "--recording_save_workers",
        str(args.recording_save_workers),
        "--recording_save_queue_size",
        str(args.recording_save_queue_size),
        "--seed",
        str(args.seed),
        "--enable_cameras",
        "--enable_wrist_cameras",
        "--enable_dex3_dds",
    ]
    if args.headless:
        command.append("--headless")
    return command


def wait_for_rerecord_completion(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_handle,
    output_dir: Path,
    timeout_seconds: float,
) -> tuple[int | None, bool, bool]:
    before_npz = {path.resolve() for path in output_dir.glob("*.npz")}
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    start_time = time.time()
    stable_count = 0
    detected_output = False
    timed_out = False
    last_output_file = ""
    last_size = -1

    while True:
        if timeout_seconds > 0 and time.time() - start_time > timeout_seconds:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            break

        new_npz_files = find_new_npz_files(output_dir, before_npz)
        if new_npz_files:
            latest_output = new_npz_files[-1]
            current_size = latest_output.stat().st_size
            if str(latest_output) == last_output_file and current_size == last_size:
                stable_count += 1
            else:
                stable_count = 0
                last_output_file = str(latest_output)
                last_size = current_size

            if stable_count >= 2:
                detected_output = True
                log_handle.write(f"[watchdog] detected stable rerecord output: {latest_output}\n")
                log_handle.flush()
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
                break

        if process.poll() is not None:
            return process.returncode, bool(new_npz_files), timed_out

        time.sleep(1.0)

    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10.0)

    new_npz_files = find_new_npz_files(output_dir, before_npz)
    return process.returncode, bool(new_npz_files), timed_out


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else input_root.with_name(input_root.name + "_multicam_rerecord")
    )
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise SystemExit(
            f"output_root must not be inside input_root:\n"
            f"  input_root={input_root}\n"
            f"  output_root={output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    python_bin = resolve_python_bin(args.python_bin)
    robot_usd_override = resolve_robot_usd_override(args)
    subprocess_env = build_env(robot_usd_override)
    npz_paths = find_npz_files(input_root)
    if args.limit > 0:
        npz_paths = npz_paths[: args.limit]
    if not npz_paths:
        print("No TWIST2 recordings found.")
        return 0

    print(f"python_bin={python_bin}")
    print(f"input_root={input_root}")
    print(f"output_root={output_root}")
    print(f"jobs={len(npz_paths)}")

    for index, npz_path in enumerate(npz_paths, start=1):
        rel_parent = npz_path.relative_to(input_root).parent
        output_dir = (output_root / rel_parent).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        task_name = read_task_name(npz_path)
        command = build_command(
            args,
            python_bin=python_bin,
            replay_file=npz_path,
            output_dir=output_dir,
            task_name=task_name,
        )
        log_dir = output_root / "rerecord_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{npz_path.stem}.log"

        print(f"[{index}/{len(npz_paths)}] rerecording {npz_path}")
        print(f"  output_dir={output_dir}")
        print(f"  log={log_path}")

        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("COMMAND: " + " ".join(command) + "\n")
            log_handle.flush()
            started_at = time.time()
            return_code, detected_output, timed_out = wait_for_rerecord_completion(
                command=command,
                cwd=REPO_ROOT,
                env=subprocess_env,
                log_handle=log_handle,
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
            elapsed = time.time() - started_at
        if timed_out:
            print(f"  FAILED timeout elapsed={elapsed:.1f}s")
            return 1
        if return_code not in {0, 130, -2, 143, -15} and not detected_output:
            print(f"  FAILED return_code={return_code} elapsed={elapsed:.1f}s")
            return int(return_code or 1)
        rerecorded_npz = sorted(output_dir.glob("*.npz"), key=lambda path: path.stat().st_mtime)[-1]
        final_reward = read_rerecord_final_reward(rerecorded_npz)
        if final_reward is None:
            print(f"  ok elapsed={elapsed:.1f}s")
        else:
            print(f"  ok elapsed={elapsed:.1f}s final_reward={final_reward:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
