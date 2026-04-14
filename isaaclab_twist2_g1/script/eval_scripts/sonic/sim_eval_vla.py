#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path

ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = str(ISAACLAB_ROOT)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


TASK_FOOTBALL_SINGLE = "Isaac-Move-Football-Single-G129-Dex3-Wholebody"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-episode VLA evaluation for football-single")
    parser.add_argument("--task", type=str, default=TASK_FOOTBALL_SINGLE)
    parser.add_argument(
        "--env_config_yaml",
        type=str,
        default="tasks/common_env_config/football_single_sonic.yaml",
        help="YAML file with env config overrides",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--model_path", type=str, default="", help="Compatibility argument (unused in SONIC VLA eval)")
    parser.add_argument("--sonic_encoder_path", type=str, required=True, help="SONIC encoder ONNX path (29DoF controller)")
    parser.add_argument("--sonic_decoder_path", type=str, required=True, help="SONIC decoder ONNX path (29DoF controller)")
    parser.add_argument(
        "--sonic_vla_root_rot6d_layout",
        type=str,
        default="row",
        choices=["auto", "row", "col"],
        help="Root rot6d decode layout for VLA action (auto=row/col continuity selection).",
    )
    parser.add_argument(
        "--sonic_vla_root_max_delta_deg",
        type=float,
        default=26.0,
        help="Clamp max root orientation delta per step (degrees). <=0 to disable.",
    )
    parser.add_argument("--sonic_effort_control", action="store_true", default=False)
    parser.add_argument("--sonic_debug", action="store_true", default=False)
    parser.add_argument("--sonic_log_every", type=int, default=50)
    parser.add_argument("--lerobot_server_url", type=str, required=True)
    parser.add_argument("--lerobot_server_timeout", type=float, default=5.0)
    parser.add_argument("--lerobot_server_verify_ssl", action="store_true", default=False)
    parser.add_argument("--lerobot_gripper_threshold", type=float, default=0.5)
    parser.add_argument("--robot_type", type=str, default="g129")
    parser.add_argument("--result_json", type=str, required=True)
    parser.add_argument("--success_video_dir", type=str, required=True)
    parser.add_argument("--failure_video_dir", type=str, required=True)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--post_termination_record_steps", type=int, default=0)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--model_label", type=str, default="")
    parser.add_argument("--eval_model_path", type=str, default="")
    parser.add_argument("--recording_save_dir", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)
    return parser

def _normalize_control_routing(args):
    args.input_source = "vla"
    args.gmt_backend = "sonic"
    args.action_source = "sonic_wholebody"
    args.enable_wholebody_dds = True
    args.enable_dex1_dds = False
    args.enable_dex3_dds = True
    args.enable_inspire_dds = False
    args.replay_file = ""
    args.replay_mode = "inference_replay"
    args.replay_loop = False
    args.replay_data = False
    args.language_instruction = ""
    args.lerobot_policy_path = ""
    args.lerobot_policy_device = ""
    args.input_source = "vla"
    args.gmt_backend = "sonic"
    args.action_source = "sonic_wholebody"
    args.sonic_pose_source = "redis"
    args.enable_world_camera = False
    args.enable_wholebody_dds = True
    if not args.recording_save_dir:
        args.recording_save_dir = str(Path(args.result_json).resolve().parent / "recordings")


def _initialize_task_scene(env, env_cfg, args, apply_optional_runtime_augments):
    try:
        initialize_task_scene = getattr(env_cfg, "initialize_task_scene", None)
        if callable(initialize_task_scene):
            initialize_task_scene(env, args)
            return
        apply_optional_runtime_augments(args)
        legacy_runtime_setup = getattr(env_cfg, "apply_runtime_setup", None)
        if callable(legacy_runtime_setup):
            legacy_runtime_setup(env, args)
    except Exception as exc:
        print(f"[eval_vla] init setup failed: {exc}")


def _trigger_task_reset_event(env_cfg, event_name, env):
    event_manager = getattr(env_cfg, "event_manager", None)
    if event_manager is None:
        return False
    event_manager.trigger(event_name, env)
    return True


def _notify_action_provider_env_reset(action_provider):
    if action_provider is None:
        return
    for method_name in ("on_env_reset", "_reset_internal_buffers"):
        method = getattr(action_provider, method_name, None)
        if callable(method):
            method()
            return


def _capture_front_camera_rgb(env):
    try:
        if "front_camera" not in env.scene.keys():
            return None
        camera = env.scene["front_camera"]
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        frame = rgb[0].detach().cpu().numpy()
        if frame.ndim != 3:
            return None
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != "uint8":
            frame = frame.clip(0, 255).astype("uint8")
        return frame
    except Exception:
        return None


def _extract_reward_info(env) -> dict:
    import torch

    reward_manager = getattr(env, "reward_manager", None)
    if reward_manager is not None:
        dt = getattr(env, "step_dt", None)
        if dt is None:
            dt = getattr(env, "physics_dt", None)
        if dt is None:
            raise RuntimeError("env.step_dt and env.physics_dt are both unavailable")
        scaled_reward = reward_manager.compute(dt=dt)
        if isinstance(scaled_reward, torch.Tensor):
            scaled_value = float(scaled_reward.detach().reshape(-1)[0].item())
        else:
            scaled_value = float(scaled_reward[0])

        raw_terms = []
        raw_total = 0.0
        get_terms = getattr(reward_manager, "get_active_iterable_terms", None)
        if callable(get_terms):
            for entry in get_terms(0):
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                term_name = str(entry[0])
                term_values = entry[1]
                if isinstance(term_values, (list, tuple)) and term_values:
                    term_value = float(term_values[0])
                else:
                    term_value = float(term_values)
                raw_terms.append((term_name, term_value))
                raw_total += term_value
        else:
            raw_total = scaled_value / dt

        return {
            "scaled_total": scaled_value,
            "raw_total": raw_total,
            "dt": float(dt),
            "raw_terms": raw_terms,
        }

    reward_buf = getattr(env, "reward_buf", None)
    if reward_buf is not None:
        if isinstance(reward_buf, torch.Tensor):
            value = float(reward_buf.detach().reshape(-1)[0].item())
        else:
            value = float(reward_buf[0])
        return {
            "scaled_total": value,
            "raw_total": value,
            "dt": 1.0,
            "raw_terms": [],
        }

    raise RuntimeError("env.reward_manager and env.reward_buf are both unavailable")


def _write_result(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def main() -> int:
    parser = _build_parser()
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import tasks
    from action_provider.create_action_provider import create_action_provider
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from layeredcontrol.robot_control_system import ControlConfig, RobotController
    from tasks.common_env_config import apply_env_config_yaml
    from tasks.common_runtime import apply_optional_runtime_augments
    from utils.video_recorder import SimpleVideoRecorder

    if args_cli.task != TASK_FOOTBALL_SINGLE:
        raise ValueError(f"sim_eval_vla.py first version only supports {TASK_FOOTBALL_SINGLE}")

    _normalize_control_routing(args_cli)

    result_path = Path(args_cli.result_json).expanduser().resolve()
    success_video_dir = Path(args_cli.success_video_dir).expanduser().resolve()
    failure_video_dir = Path(args_cli.failure_video_dir).expanduser().resolve()
    success_video_dir.mkdir(parents=True, exist_ok=True)
    failure_video_dir.mkdir(parents=True, exist_ok=True)

    model_label = args_cli.model_label or Path(args_cli.eval_model_path or args_cli.model_path).stem
    episode_name = f"{model_label}__seed_{args_cli.seed}__episode_{args_cli.episode_index}"
    temp_video_path = result_path.parent / f"{episode_name}__tmp.mp4"

    env = None
    controller = None
    action_provider = None
    recorder = SimpleVideoRecorder(str(temp_video_path), fps=args_cli.video_fps)
    started_at = time.time()
    step_idx = 0
    max_reward = float("-inf")
    max_reward_scaled = float("-inf")
    final_reward = 0.0
    final_reward_scaled = 0.0
    terminal_step_idx = 0
    success = False
    failure_reason = "unknown"
    video_path = ""
    post_termination_steps_remaining = 0

    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
        apply_env_config_yaml(
            env_cfg,
            args_cli.env_config_yaml,
            task_name=args_cli.task,
            route_name=args_cli.gmt_backend or args_cli.action_source,
        )
        env_cfg.seed = args_cli.seed
        print(f"[sim_eval_vla] seed={args_cli.seed}")

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        _initialize_task_scene(env, env_cfg, args_cli, apply_optional_runtime_augments)
        env.sim.reset()
        env.reset()
        if getattr(env_cfg, "startup_task_reset_enabled", True):
            _trigger_task_reset_event(env_cfg, "reset_all_self", env)

        physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
        control_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
        control_config = ControlConfig(
            step_hz=control_hz,
            replay_mode=False,
            use_rl_action_mode=True,
        )
        print(
            f"[sim_eval_vla] task={args_cli.task} control_hz={control_hz} "
            f"max_steps={args_cli.max_steps} lerobot_server={args_cli.lerobot_server_url}"
        )

        action_provider = create_action_provider(env, args_cli)
        if action_provider is None:
            raise RuntimeError("failed to create action provider")
        env.action_provider = action_provider
        _notify_action_provider_env_reset(action_provider)

        controller = RobotController(env, control_config)
        controller.set_action_provider(action_provider)
        controller.start()

        initial_frame = _capture_front_camera_rgb(env)
        if initial_frame is not None:
            recorder.add_frame(initial_frame)

        while simulation_app.is_running() and controller.is_running:
            controller.step()
            step_idx += 1

            frame = _capture_front_camera_rgb(env)
            if frame is not None:
                recorder.add_frame(frame)

            if post_termination_steps_remaining > 0:
                post_termination_steps_remaining -= 1
                if env.sim.is_stopped():
                    print(
                        f"[sim_eval_vla] post-termination recording interrupted at control_step={step_idx} "
                        "because simulation stopped"
                    )
                    break
                if post_termination_steps_remaining == 0:
                    print(f"[sim_eval_vla] post-termination recording finished at control_step={step_idx}")
                    break
                continue

            reward_info = _extract_reward_info(env)
            final_reward = reward_info["raw_total"]
            final_reward_scaled = reward_info["scaled_total"]
            if final_reward > max_reward:
                max_reward = final_reward
            if final_reward_scaled > max_reward_scaled:
                max_reward_scaled = final_reward_scaled

            terms_str = ", ".join(f"{name}={value:.4f}" for name, value in reward_info["raw_terms"])
            print(
                f"[sim_eval_vla] reward step={step_idx} "
                f"raw_total={final_reward:.4f} scaled_total={final_reward_scaled:.4f}"
                + (f" | {terms_str}" if terms_str else "")
            )

            if final_reward >= 1.0:
                success = True
                failure_reason = "success"
                terminal_step_idx = step_idx
                print(
                    f"[sim_eval_vla] success at control_step={step_idx} "
                    f"raw_reward={final_reward:.4f} scaled_reward={final_reward_scaled:.4f}"
                )
                post_termination_steps_remaining = args_cli.post_termination_record_steps
                if post_termination_steps_remaining <= 0:
                    break
                print(
                    f"[sim_eval_vla] recording {post_termination_steps_remaining} extra frames after success"
                )
                continue

            if step_idx >= args_cli.max_steps:
                failure_reason = "timeout"
                terminal_step_idx = step_idx
                print(
                    f"[sim_eval_vla] timeout at control_step={step_idx} "
                    f"raw_reward={final_reward:.4f} scaled_reward={final_reward_scaled:.4f}"
                )
                post_termination_steps_remaining = args_cli.post_termination_record_steps
                if post_termination_steps_remaining <= 0:
                    break
                print(
                    f"[sim_eval_vla] recording {post_termination_steps_remaining} extra frames after timeout"
                )
                continue

            if env.sim.is_stopped():
                failure_reason = "sim_stopped"
                terminal_step_idx = step_idx
                print(f"[sim_eval_vla] simulation stopped at control_step={step_idx}")
                break

        target_dir = success_video_dir if success else failure_video_dir
        recorder.save_path = target_dir / f"{episode_name}__{'success' if success else failure_reason}.mp4"
        if recorder.frames:
            recorder.save()
            video_path = str(recorder.save_path)
        else:
            print("[sim_eval_vla] no video frames captured")

        result = {
            "task": args_cli.task,
            "model_path": args_cli.eval_model_path or "",
            "model_label": model_label,
            "seed": args_cli.seed,
            "episode_index": args_cli.episode_index,
            "success": success,
            "failure_reason": failure_reason,
            "episode_steps": terminal_step_idx or step_idx,
            "max_steps": args_cli.max_steps,
            "final_reward": final_reward,
            "final_reward_scaled": final_reward_scaled,
            "max_reward": max_reward if max_reward != float("-inf") else 0.0,
            "max_reward_scaled": max_reward_scaled if max_reward_scaled != float("-inf") else 0.0,
            "video_path": video_path,
            "server_url": args_cli.lerobot_server_url,
            "started_at": started_at,
            "finished_at": time.time(),
            "duration_sec": time.time() - started_at,
        }
        _write_result(result_path, result)
        return 0
    except Exception as exc:
        print(f"[sim_eval_vla] episode failed: {exc}")
        result = {
            "task": args_cli.task,
            "model_path": args_cli.eval_model_path or "",
            "model_label": model_label,
            "seed": args_cli.seed,
            "episode_index": args_cli.episode_index,
            "success": False,
            "failure_reason": "sim_error",
            "error": str(exc),
            "episode_steps": step_idx,
            "max_steps": args_cli.max_steps,
            "final_reward": final_reward,
            "final_reward_scaled": final_reward_scaled,
            "max_reward": max_reward if max_reward != float("-inf") else 0.0,
            "max_reward_scaled": max_reward_scaled if max_reward_scaled != float("-inf") else 0.0,
            "video_path": "",
            "server_url": args_cli.lerobot_server_url,
            "started_at": started_at,
            "finished_at": time.time(),
            "duration_sec": time.time() - started_at,
        }
        _write_result(result_path, result)
        return 1
    finally:
        try:
            if controller is not None:
                controller.cleanup()
        except Exception as exc:
            print(f"[sim_eval_vla] controller cleanup failed: {exc}")
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            print(f"[sim_eval_vla] env close failed: {exc}")
        try:
            recorder.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception as exc:
            print(f"[sim_eval_vla] simulation_app close failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
