#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import math
from pathlib import Path

from eval_vla_suite import (
    TASK_FOOTBALL_SINGLE,
    _run_episode,
    _sanitize_label,
    _start_server,
    _wait_for_server_ready,
    _write_summary,
)


def _resolve_server_devices(args) -> list[str]:
    raw_values: list[str] = []
    raw_gpu_ids = str(getattr(args, "server_gpu_ids", "") or "").strip()
    if raw_gpu_ids:
        raw_values.extend(raw_gpu_ids.split(","))
    else:
        raw_server_device = str(getattr(args, "server_device", "") or "").strip()
        if raw_server_device:
            raw_values.extend(raw_server_device.split(","))

    devices: list[str] = []
    for raw in raw_values:
        token = raw.strip()
        if not token:
            continue
        if "-" in token and token.replace("-", "").isdigit():
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            for gpu_idx in range(start, end + step, step):
                devices.append(f"cuda:{gpu_idx}")
            continue
        if token.isdigit():
            devices.append(f"cuda:{token}")
            continue
        devices.append(token)

    deduped_devices: list[str] = []
    seen_devices: set[str] = set()
    for device in devices:
        if device in seen_devices:
            continue
        deduped_devices.append(device)
        seen_devices.add(device)

    if not deduped_devices:
        raise ValueError(
            "No valid server devices resolved. Use --server_gpu_ids 0,1,2,3 "
            "or --server_device cuda:0."
        )
    return deduped_devices


def _build_worker_slots(server_devices: list[str], workers_per_device: int) -> list[dict]:
    slots: list[dict] = []
    slot_id = 0
    # Interleave devices so small job counts still spread across GPUs.
    for device_worker_index in range(workers_per_device):
        for server_device in server_devices:
            slots.append(
                {
                    "task_id": slot_id,
                    "server_device": server_device,
                    "device_worker_index": device_worker_index,
                }
            )
            slot_id += 1
    return slots


def _build_jobs(args) -> tuple[list[dict], dict[str, str]]:
    jobs = []
    model_labels = {}
    episode_index = 0
    for model_path in args.model_paths:
        resolved_model_path = str(Path(model_path).expanduser().resolve())
        model_label = _sanitize_label(
            Path(resolved_model_path).parent.parent.name + "__" + Path(resolved_model_path).parent.name
        )
        model_labels[resolved_model_path] = model_label
        for seed in args.seeds:
            for repeat_idx in range(args.repeats_per_seed):
                jobs.append(
                    {
                        "model_path": resolved_model_path,
                        "model_label": model_label,
                        "seed": seed,
                        "repeat_idx": repeat_idx,
                        "episode_index": episode_index,
                    }
                )
                episode_index += 1
    return jobs, model_labels


def _partition_jobs(all_jobs: list[dict], total_worker_slots: int) -> list[dict]:
    jobs_by_model: dict[str, list[dict]] = {}
    for job in all_jobs:
        jobs_by_model.setdefault(job["model_path"], []).append(job)

    total_jobs = len(all_jobs)
    task_specs = []
    task_id = 0
    for model_path, jobs in jobs_by_model.items():
        proportional_workers = max(1, math.ceil(total_worker_slots * len(jobs) / total_jobs))
        chunk_count = min(len(jobs), proportional_workers)
        chunked_jobs = [jobs[idx::chunk_count] for idx in range(chunk_count)]
        for chunk in chunked_jobs:
            if not chunk:
                continue
            task_specs.append(
                {
                    "task_id": task_id,
                    "model_path": model_path,
                    "model_label": chunk[0]["model_label"],
                    "jobs": chunk,
                }
            )
            task_id += 1
    return task_specs


def _assign_worker_slots(task_specs: list[dict], worker_slots: list[dict]) -> list[dict]:
    if len(task_specs) > len(worker_slots):
        raise ValueError(
            f"Internal error: task_specs={len(task_specs)} exceeds worker_slots={len(worker_slots)}"
        )

    assigned_specs: list[dict] = []
    for task_spec, worker_slot in zip(task_specs, worker_slots):
        assigned = dict(task_spec)
        assigned["task_id"] = worker_slot["task_id"]
        assigned["server_device"] = worker_slot["server_device"]
        assigned["device_worker_index"] = worker_slot["device_worker_index"]
        assigned_specs.append(assigned)
    return assigned_specs


def _build_failure_result(args, server_url: str, run_dir: Path, task_spec: dict, job: dict, exc: Exception) -> dict:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    sim_log = logs_dir / f"{job['model_label']}__seed_{job['seed']}__episode_{job['episode_index']}.log"
    return {
        "task": args.task,
        "model_path": job["model_path"],
        "model_label": job["model_label"],
        "seed": job["seed"],
        "episode_index": job["episode_index"],
        "success": False,
        "failure_reason": "worker_error",
        "episode_steps": 0,
        "max_steps": args.max_steps,
        "final_reward": 0.0,
        "final_reward_scaled": 0.0,
        "max_reward": 0.0,
        "max_reward_scaled": 0.0,
        "video_path": "",
        "server_url": server_url,
        "log_path": str(sim_log),
        "returncode": -1,
        "error": str(exc),
        "worker_id": task_spec["task_id"],
        "worker_device": task_spec.get("server_device", ""),
    }


def _build_crashed_task_results(args, run_dir: Path, task_spec: dict, exc: Exception) -> list[dict]:
    server_port = args.server_port_base + task_spec["task_id"]
    server_url = f"{args.server_scheme}://{args.server_host}:{server_port}"
    return [
        _build_failure_result(args, server_url, run_dir, task_spec, job, exc)
        for job in task_spec["jobs"]
    ]


def _worker_run(task_spec: dict, args_dict: dict, run_dir_str: str) -> list[dict]:
    args = argparse.Namespace(**args_dict)
    run_dir = Path(run_dir_str)
    server_port = args.server_port_base + task_spec["task_id"]
    server_url = f"{args.server_scheme}://{args.server_host}:{server_port}"
    server_log_path = run_dir / "logs" / f"server__worker_{task_spec['task_id']}__{task_spec['model_label']}.log"

    args.server_port = server_port
    args.server_device = task_spec["server_device"]
    results = []
    server_proc = None
    server_log_fp = None
    try:
        print(
            f"[eval_vla_suite_parallel] worker={task_spec['task_id']} "
            f"starting server model={task_spec['model_path']} port={server_port} "
            f"device={args.server_device}"
        )
        server_proc, server_log_fp = _start_server(args, task_spec["model_path"], server_log_path)
        _wait_for_server_ready(
            server_url,
            timeout_s=args.server_ready_timeout,
            verify_ssl=args.lerobot_server_verify_ssl,
        )
        for job in task_spec["jobs"]:
            print(
                f"[eval_vla_suite_parallel] worker={task_spec['task_id']} "
                f"model={job['model_label']} seed={job['seed']} "
                f"repeat={job['repeat_idx'] + 1}/{args.repeats_per_seed} "
                f"episode_index={job['episode_index']}"
                )
            result = _run_episode(
                args=args,
                server_url=server_url,
                model_path=job["model_path"],
                model_label=job["model_label"],
                seed=job["seed"],
                episode_index=job["episode_index"],
                run_dir=run_dir,
            )
            result["worker_id"] = task_spec["task_id"]
            result["worker_device"] = task_spec["server_device"]
            results.append(result)
        return results
    except Exception as exc:
        print(f"[eval_vla_suite_parallel] worker={task_spec['task_id']} failed: {exc}")
        handled = {
            (row["model_label"], row["seed"], row["episode_index"])
            for row in results
        }
        for job in task_spec["jobs"]:
            job_key = (job["model_label"], job["seed"], job["episode_index"])
            if job_key in handled:
                continue
            results.append(_build_failure_result(args, server_url, run_dir, task_spec, job, exc))
        return results
    finally:
        try:
            if server_proc is not None:
                server_proc.terminate()
                server_proc.wait(timeout=10.0)
        except Exception:
            if server_proc is not None:
                server_proc.kill()
        try:
            if server_log_fp is not None:
                server_log_fp.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel football-single VLA evaluation suite")
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
    parser.add_argument("--server_script", type=str, required=True)
    parser.add_argument("--server_device", type=str, default="cuda:0")
    parser.add_argument(
        "--server_gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU ids/devices for model servers. Example: 0,1,2,3. "
        "Each device runs num_workers server processes.",
    )
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_scheme", type=str, default="http", choices=["http", "https"])
    parser.add_argument("--tls_cert_file", type=str, default="")
    parser.add_argument("--tls_key_file", type=str, default="")
    parser.add_argument("--lerobot_server_timeout", type=float, default=5.0)
    parser.add_argument("--lerobot_server_verify_ssl", action="store_true", default=False)
    parser.add_argument("--server_ready_timeout", type=float, default=60.0)
    parser.add_argument("--num_workers", type=int, default=2, help="Worker processes per server device.")
    parser.add_argument("--server_port_base", type=int, default=18443)
    args = parser.parse_args()

    if args.task != TASK_FOOTBALL_SINGLE:
        raise ValueError(f"eval_vla_suite_parallel first version only supports {TASK_FOOTBALL_SINGLE}")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be >= 1")

    run_dir = Path(args.results_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    all_jobs, _ = _build_jobs(args)
    if not all_jobs:
        raise ValueError("No evaluation jobs were generated")

    server_devices = _resolve_server_devices(args)
    worker_slots = _build_worker_slots(server_devices, args.num_workers)
    total_worker_slots = len(worker_slots)
    task_specs = _assign_worker_slots(
        _partition_jobs(all_jobs, total_worker_slots),
        worker_slots,
    )
    print(
        f"[eval_vla_suite_parallel] total_jobs={len(all_jobs)} "
        f"workers_per_device={args.num_workers} total_worker_slots={total_worker_slots} "
        f"task_chunks={len(task_specs)} server_devices={server_devices} "
        f"server_port_base={args.server_port_base}"
    )

    args_dict = vars(args).copy()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=total_worker_slots) as executor:
        future_to_task = {
            executor.submit(_worker_run, task_spec, args_dict, str(run_dir)): task_spec
            for task_spec in task_specs
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task_spec = future_to_task[future]
            try:
                task_results = future.result()
            except Exception as exc:
                print(
                    f"[eval_vla_suite_parallel] worker={task_spec['task_id']} "
                    f"crashed before returning results: {exc}"
                )
                task_results = _build_crashed_task_results(args, run_dir, task_spec, exc)
            print(
                f"[eval_vla_suite_parallel] worker={task_spec['task_id']} "
                f"completed episodes={len(task_results)}"
            )
            results.extend(task_results)

    results.sort(key=lambda row: int(row["episode_index"]))
    _write_summary(run_dir, results)

    total_successes = sum(int(bool(row.get("success"))) for row in results)
    total_episodes = len(results)
    overall_rate = total_successes / total_episodes if total_episodes else 0.0
    summary = {
        "total_jobs": len(all_jobs),
        "completed_results": len(results),
        "workers_per_device": args.num_workers,
        "total_worker_slots": total_worker_slots,
        "task_chunks": len(task_specs),
        "server_devices": server_devices,
        "server_port_base": args.server_port_base,
        "overall_success_rate": overall_rate,
    }
    (run_dir / "parallel_config.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(
        f"[eval_vla_suite_parallel] completed episodes={total_episodes} successes={total_successes} "
        f"success_rate={overall_rate:.4f} results_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
