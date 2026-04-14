#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = str(ISAACLAB_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from action_provider.lerobot_vla_http_client import LeRobotVLAHttpClient


TASK_FOOTBALL_SINGLE = "Isaac-Move-Football-Single-G129-Dex3-Wholebody"


def _sanitize_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _wait_for_server_ready(base_url: str, timeout_s: float, verify_ssl: bool) -> None:
    client = LeRobotVLAHttpClient(base_url=base_url, timeout_s=2.0, verify_ssl=verify_ssl)
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            client.reset()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"LeRobot server not ready at {base_url}: {last_error}")


def _start_server(args, model_path: str, log_path: Path):
    server_script = Path(args.server_script).expanduser().resolve()
    cmd = [
        args.server_python,
        str(server_script),
        "--policy-path",
        model_path,
        "--device",
        args.server_device,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
    ]
    if args.server_scheme == "https":
        if not args.tls_cert_file or not args.tls_key_file:
            raise ValueError("HTTPS server requires --tls_cert_file and --tls_key_file")
        cmd.extend(["--tls-cert-file", args.tls_cert_file, "--tls-key-file", args.tls_key_file])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=str(server_script.parent),
    )
    return process, log_fp


def _run_episode(args, server_url: str, model_path: str, model_label: str, seed: int, episode_index: int, run_dir: Path):
    episodes_dir = run_dir / "episodes"
    logs_dir = run_dir / "logs"
    recordings_dir = run_dir / "recordings" / "vla_outputs"
    success_video_dir = run_dir / "videos" / "success"
    failure_video_dir = run_dir / "videos" / "failure"
    for directory in (episodes_dir, logs_dir, recordings_dir, success_video_dir, failure_video_dir):
        directory.mkdir(parents=True, exist_ok=True)

    result_json = episodes_dir / f"{model_label}__seed_{seed}__episode_{episode_index}.json"
    sim_log = logs_dir / f"{model_label}__seed_{seed}__episode_{episode_index}.log"
    vla_trace_path = recordings_dir / f"{model_label}__seed_{seed}__episode_{episode_index}.jsonl"
    trace_enabled = os.environ.get("LEROBOT_VLA_RECORD_OUTPUTS", "1").strip() not in {"0", "false", "False"}
    if trace_enabled and vla_trace_path.exists():
        vla_trace_path.unlink()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "sim_eval_vla.py"),
        "--task",
        args.task,
        "--env_config_yaml",
        args.env_config_yaml,
        "--seed",
        str(seed),
        "--max_steps",
        str(args.max_steps),
        "--sonic_encoder_path",
        args.sonic_encoder_path,
        "--sonic_decoder_path",
        args.sonic_decoder_path,
        "--sonic_vla_root_rot6d_layout",
        args.sonic_vla_root_rot6d_layout,
        "--sonic_vla_root_max_delta_deg",
        str(args.sonic_vla_root_max_delta_deg),
        "--model_path",
        args.sonic_encoder_path,
        "--lerobot_server_url",
        server_url,
        "--lerobot_server_timeout",
        str(args.lerobot_server_timeout),
        "--robot_type",
        args.robot_type,
        "--result_json",
        str(result_json),
        "--success_video_dir",
        str(success_video_dir),
        "--failure_video_dir",
        str(failure_video_dir),
        "--video_fps",
        str(args.video_fps),
        "--post_termination_record_steps",
        str(args.post_termination_record_steps),
        "--episode_index",
        str(episode_index),
        "--model_label",
        model_label,
        "--eval_model_path",
        model_path,
        "--recording_save_dir",
        str(run_dir / "recordings"),
        "--device",
        args.isaac_device,
        "--enable_cameras",
    ]
    if args.lerobot_server_verify_ssl:
        cmd.append("--lerobot_server_verify_ssl")
    if args.headless:
        cmd.append("--headless")

    child_env = os.environ.copy()
    if trace_enabled:
        child_env["LEROBOT_VLA_TRACE_PATH"] = str(vla_trace_path)
    else:
        child_env.pop("LEROBOT_VLA_TRACE_PATH", None)

    with open(sim_log, "w", encoding="utf-8") as log_fp:
        completed = subprocess.run(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=str(ISAACLAB_ROOT),
            env=child_env,
            check=False,
        )

    if result_json.exists():
        result = json.loads(result_json.read_text())
    else:
        result = {
            "task": args.task,
            "model_path": model_path,
            "model_label": model_label,
            "seed": seed,
            "episode_index": episode_index,
            "success": False,
            "failure_reason": "process_error",
            "episode_steps": 0,
            "max_steps": args.max_steps,
            "final_reward": 0.0,
            "final_reward_scaled": 0.0,
            "max_reward": 0.0,
            "max_reward_scaled": 0.0,
            "video_path": "",
            "server_url": server_url,
            "returncode": completed.returncode,
        }
        result_json.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n")

    result["log_path"] = str(sim_log)
    result["vla_trace_path"] = str(vla_trace_path) if trace_enabled else ""
    result["returncode"] = completed.returncode
    return result


def _write_summary(run_dir: Path, results: list[dict]) -> None:
    jsonl_path = run_dir / "summary.jsonl"
    csv_path = run_dir / "summary.csv"
    summary_json_path = run_dir / "summary.json"

    with open(jsonl_path, "w", encoding="utf-8") as fp:
        for row in results:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    fieldnames = [
        "task",
        "model_path",
        "model_label",
        "seed",
        "episode_index",
        "success",
        "failure_reason",
        "episode_steps",
        "max_steps",
        "final_reward",
        "final_reward_scaled",
        "max_reward",
        "max_reward_scaled",
        "video_path",
        "log_path",
        "vla_trace_path",
        "server_url",
        "returncode",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    per_model = {}
    for row in results:
        model_label = row["model_label"]
        stats = per_model.setdefault(model_label, {"model_path": row["model_path"], "episodes": 0, "successes": 0})
        stats["episodes"] += 1
        stats["successes"] += int(bool(row.get("success")))
    for stats in per_model.values():
        stats["success_rate"] = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0

    summary = {
        "total_episodes": len(results),
        "total_successes": sum(int(bool(row.get("success"))) for row in results),
        "overall_success_rate": (
            sum(int(bool(row.get("success"))) for row in results) / len(results) if results else 0.0
        ),
        "per_model": per_model,
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-parallel football-single VLA evaluation suite")
    parser.add_argument("--task", type=str, default=TASK_FOOTBALL_SINGLE)
    parser.add_argument("--env_config_yaml", type=str, default="tasks/common_env_config/football_single_sonic.yaml")
    parser.add_argument("--model-path", dest="model_paths", action="append", required=True)
    parser.add_argument("--seed", dest="seeds", action="append", type=int, required=True)
    parser.add_argument("--repeats_per_seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--post_termination_record_steps", type=int, default=0)
    parser.add_argument("--robot_type", type=str, default="g129")
    parser.add_argument("--sonic_encoder_path", type=str, required=True)
    parser.add_argument("--sonic_decoder_path", type=str, required=True)
    parser.add_argument(
        "--sonic_vla_root_rot6d_layout",
        type=str,
        default="auto",
        choices=["auto", "row", "col"],
    )
    parser.add_argument(
        "--sonic_vla_root_max_delta_deg",
        type=float,
        default=26.0,
        help="Clamp max root orientation delta per step (degrees). <=0 to disable.",
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--isaac_device", type=str, default="cpu")
    parser.add_argument("--server_python", type=str, required=True)
    parser.add_argument(
        "--server_script",
        type=str,
        default=str(ISAACLAB_ROOT.parent / "lerobot" / "serve_lerobot_vla_http.py"),
    )
    parser.add_argument("--server_device", type=str, default="cuda:0")
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8443)
    parser.add_argument("--server_scheme", type=str, default="http", choices=["http", "https"])
    parser.add_argument("--tls_cert_file", type=str, default="")
    parser.add_argument("--tls_key_file", type=str, default="")
    parser.add_argument("--lerobot_server_timeout", type=float, default=5.0)
    parser.add_argument("--lerobot_server_verify_ssl", action="store_true", default=False)
    parser.add_argument("--server_ready_timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.task != TASK_FOOTBALL_SINGLE:
        raise ValueError(f"eval_vla_suite first version only supports {TASK_FOOTBALL_SINGLE}")

    run_dir = Path(args.results_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    server_url = f"{args.server_scheme}://{args.server_host}:{args.server_port}"

    try:
        for model_path in args.model_paths:
            resolved_model_path = str(Path(model_path).expanduser().resolve())
            model_label = _sanitize_label(Path(resolved_model_path).parent.parent.name + "__" + Path(resolved_model_path).parent.name)
            server_log_path = run_dir / "logs" / f"server__{model_label}.log"
            print(f"[eval_vla_suite] starting server for model={resolved_model_path}")
            server_proc, server_log_fp = _start_server(args, resolved_model_path, server_log_path)
            try:
                _wait_for_server_ready(
                    server_url,
                    timeout_s=args.server_ready_timeout,
                    verify_ssl=args.lerobot_server_verify_ssl,
                )
                episode_index = 0
                for seed in args.seeds:
                    for repeat_idx in range(args.repeats_per_seed):
                        print(
                            f"[eval_vla_suite] model={model_label} seed={seed} "
                            f"repeat={repeat_idx + 1}/{args.repeats_per_seed}"
                        )
                        result = _run_episode(
                            args=args,
                            server_url=server_url,
                            model_path=resolved_model_path,
                            model_label=model_label,
                            seed=seed,
                            episode_index=episode_index,
                            run_dir=run_dir,
                        )
                        results.append(result)
                        episode_index += 1
            finally:
                try:
                    server_proc.terminate()
                    server_proc.wait(timeout=10.0)
                except Exception:
                    server_proc.kill()
                server_log_fp.close()
    finally:
        _write_summary(run_dir, results)

    total_successes = sum(int(bool(row.get("success"))) for row in results)
    total_episodes = len(results)
    overall_rate = total_successes / total_episodes if total_episodes else 0.0
    print(
        f"[eval_vla_suite] completed episodes={total_episodes} successes={total_successes} "
        f"success_rate={overall_rate:.4f} results_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
