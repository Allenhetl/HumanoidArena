#!/usr/bin/env python3

import argparse
import base64
import json
import random
import signal
import ssl
import sys
import tempfile
import threading
import traceback
from contextlib import nullcontext
from copy import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch


_SERVER_STOP_REASON = "unknown"
_LEGACY_CHECKPOINT_ROOT = Path("/mnt/workspace/users/xujunzhe/yunhengwang/lerobot/lerobot/checkpoints")
_COMPAT_CHECKPOINT_ROOT = Path("/ai/Yichi/taowen/ckpts/checkpoints")


def _remap_legacy_checkpoint_ref(value):
    if not isinstance(value, str):
        return value, False

    try:
        path = Path(value)
    except Exception:
        return value, False

    if not path.is_absolute():
        return value, False
    if path.exists():
        return value, False

    try:
        path.relative_to(_LEGACY_CHECKPOINT_ROOT)
    except Exception:
        return value, False

    compat_candidate = _COMPAT_CHECKPOINT_ROOT / path.name
    if compat_candidate.exists():
        print(
            f"[lerobot_vla_server] remap legacy checkpoint ref {value} -> {compat_candidate}",
            flush=True,
        )
        return str(compat_candidate), True

    print(
        f"[lerobot_vla_server] unresolved legacy checkpoint ref {value}; compat candidate missing: {compat_candidate}",
        flush=True,
    )
    return value, False


def _remap_json_tree(obj):
    if isinstance(obj, dict):
        changed = False
        remapped = {}
        for key, value in obj.items():
            remapped_value, child_changed = _remap_json_tree(value)
            remapped[key] = remapped_value
            changed = changed or child_changed
        return remapped, changed

    if isinstance(obj, list):
        changed = False
        remapped = []
        for value in obj:
            remapped_value, child_changed = _remap_json_tree(value)
            remapped.append(remapped_value)
            changed = changed or child_changed
        return remapped, changed

    return _remap_legacy_checkpoint_ref(obj)


def _prepare_compat_policy_dir(policy_dir: Path):
    temp_dir = tempfile.TemporaryDirectory(prefix="lerobot_policy_compat_")
    compat_dir = Path(temp_dir.name)

    for child in policy_dir.iterdir():
        target = compat_dir / child.name
        target.symlink_to(child, target_is_directory=child.is_dir())

    changed_files = []
    for json_path in sorted(policy_dir.glob("*.json")):
        try:
            payload = json.loads(json_path.read_text())
        except Exception:
            continue

        remapped_payload, changed = _remap_json_tree(payload)
        if not changed:
            continue

        compat_json_path = compat_dir / json_path.name
        if compat_json_path.exists() or compat_json_path.is_symlink():
            compat_json_path.unlink()
        compat_json_path.write_text(json.dumps(remapped_payload, indent=2) + "\n")
        changed_files.append(json_path.name)

    if changed_files:
        print(
            "[lerobot_vla_server] prepared compat policy dir for "
            f"{policy_dir} with remapped files: " + ", ".join(changed_files),
            flush=True,
        )
    else:
        print(f"[lerobot_vla_server] compat policy dir not needed for {policy_dir}", flush=True)

    return temp_dir, compat_dir


def _interrupt_handler(signum, frame):
    global _SERVER_STOP_REASON
    _SERVER_STOP_REASON = signal.Signals(signum).name
    raise KeyboardInterrupt


def _install_interrupt_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _interrupt_handler)


def _feature_shape_dim(feature) -> tuple[int, ...] | None:
    if feature is None:
        return None
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _load_policy(policy_dir: Path, device_name: str):
    lerobot_src = Path(__file__).resolve().parents[1] / "src"
    if not lerobot_src.is_dir():
        raise FileNotFoundError(f"LeRobot src directory not found: {lerobot_src}")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import _reconnect_relative_absolute_steps, get_policy_class
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    from lerobot.utils.control_utils import predict_action

    compat_dir_ctx, effective_policy_dir = _prepare_compat_policy_dir(policy_dir)

    config = PreTrainedConfig.from_pretrained(effective_policy_dir)
    config.device = device_name
    policy_cls = get_policy_class(config.type)

    policy = policy_cls.from_pretrained(effective_policy_dir, config=config)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        effective_policy_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        effective_policy_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    _reconnect_relative_absolute_steps(preprocessor, postprocessor)
    return (
        config,
        policy,
        preprocessor,
        postprocessor,
        predict_action,
        prepare_observation_for_inference,
        compat_dir_ctx,
    )


class LeRobotServerState:
    def __init__(self, policy_dir: Path, device_name: str):
        (
            self.config,
            self.policy,
            self.preprocessor,
            self.postprocessor,
            self.predict_action,
            self.prepare_observation_for_inference,
            self._compat_policy_dir_ctx,
        ) = _load_policy(policy_dir, device_name)
        self.expected_state_shape = _feature_shape_dim(self.config.input_features.get("observation.state"))
        self.expected_action_shape = _feature_shape_dim(self.config.output_features.get("action"))
        self.device = torch.device(device_name)
        self.lock = threading.Lock()
        self.reset_count = 0
        self.infer_count = 0
        self.current_seed: int | None = None
        self.reset()

    def _seed_runtime(self, seed: int) -> None:
        normalized_seed = int(seed) & 0xFFFFFFFF
        random.seed(normalized_seed)
        np.random.seed(normalized_seed)
        torch.manual_seed(normalized_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(normalized_seed)

    def reset(self, seed: int | None = None):
        with self.lock:
            if seed is not None:
                self.current_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
                self._seed_runtime(self.current_seed)
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            self.reset_count += 1
            self.infer_count = 0
            return self.reset_count

    def _prepare_observation(self, observation: dict, robot_type: str):
        prepared = self.prepare_observation_for_inference(copy(observation), self.device, None, robot_type)
        return self.preprocessor(prepared)

    def infer(self, observation: dict, robot_type: str):
        with self.lock:
            return self.predict_action(
                observation=observation,
                policy=self.policy,
                device=self.device,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=self.device.type == "cuda",
                task=None,
                robot_type=robot_type,
            )

    def infer_chunk(self, observation: dict, robot_type: str) -> np.ndarray:
        with self.lock:
            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type) if self.device.type == "cuda" else nullcontext(),
            ):
                processed_observation = self._prepare_observation(observation, robot_type)
                action_chunk = self.policy.predict_action_chunk(processed_observation)
                if action_chunk.ndim != 3:
                    action_chunk = action_chunk.unsqueeze(0)

                processed_actions = []
                _, chunk_size, _ = action_chunk.shape
                for i in range(chunk_size):
                    processed_actions.append(self.postprocessor(action_chunk[:, i, :]))

                action_chunk = torch.stack(processed_actions, dim=1).squeeze(0)
                return action_chunk.detach().cpu().to(torch.float32).numpy()


def _decode_image(image_payload: dict) -> np.ndarray:
    shape = tuple(int(v) for v in image_payload["shape"])
    dtype = np.dtype(image_payload["dtype"])
    raw = base64.b64decode(image_payload["data_b64"].encode("ascii"))
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def make_handler(state: LeRobotServerState):
    class Handler(BaseHTTPRequestHandler):
        def _read_json(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, status_code: int, payload: dict):
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            try:
                if self.path == "/reset":
                    payload = self._read_json()
                    reset_seed = payload.get("seed")
                    reset_count = state.reset(seed=reset_seed)
                    print(
                        f"[lerobot_vla_server] reset count={reset_count} seed={state.current_seed} peer={self.client_address}",
                        flush=True,
                    )
                    self._send_json(200, {"ok": True})
                    return

                if self.path != "/infer":
                    self._send_json(404, {"error": f"unknown path: {self.path}"})
                    return

                payload = self._read_json()
                image = _decode_image(payload["observation"]["images"]["front"])
                observation_state = np.asarray(payload["observation"]["state"], dtype=np.float32).copy()
                if state.expected_state_shape is not None and observation_state.shape != state.expected_state_shape:
                    raise ValueError(
                        f"Expected observation.state shape {state.expected_state_shape}, got {observation_state.shape}"
                    )
                robot_type = payload.get("robot_type", "g129")
                observation = {
                    "observation.images.front": image,
                    "observation.state": observation_state,
                }
                infer_index = state.infer_count + 1
                if infer_index == 1:
                    print(
                        f"[lerobot_vla_server] first_infer peer={self.client_address} robot_type={robot_type} "
                        f"state_shape={tuple(observation_state.shape)} image_shape={tuple(image.shape)}",
                        flush=True,
                    )

                if bool(payload.get("return_chunk", False)):
                    action_chunk = np.asarray(state.infer_chunk(observation, robot_type), dtype=np.float32)
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk.reshape(1, -1)
                    if action_chunk.ndim != 2:
                        raise ValueError(f"Expected 2D action chunk, got {action_chunk.shape}")
                    first_action = action_chunk[0]
                    response = {
                        "action": first_action.tolist(),
                        "action_chunk": action_chunk.tolist(),
                        "chunk_size": int(action_chunk.shape[0]),
                    }
                    log_suffix = f"chunk_size={action_chunk.shape[0]} first_action={first_action.tolist()}"
                else:
                    action = state.infer(observation, robot_type)
                    if not isinstance(action, torch.Tensor):
                        action = torch.as_tensor(action)
                    action = action.detach().cpu().to(torch.float32).reshape(-1)
                    if state.expected_action_shape is not None and tuple(action.shape) != state.expected_action_shape:
                        raise ValueError(
                            f"Expected action shape {state.expected_action_shape}, got {tuple(action.shape)}"
                        )
                    response = {"action": action.tolist()}
                    log_suffix = f"action={action.tolist()}"

                state.infer_count = infer_index
                print(f"[lerobot_vla_server] infer count={infer_index} {log_suffix}", flush=True)
                self._send_json(200, response)
            except BrokenPipeError:
                print(
                    f"[lerobot_vla_server] client disconnected while responding "
                    f"path={self.path} peer={self.client_address}",
                    flush=True,
                )
            except Exception as exc:
                traceback.print_exc()
                try:
                    self._send_json(500, {"error": str(exc)})
                except BrokenPipeError:
                    print(
                        f"[lerobot_vla_server] client disconnected before error response "
                        f"path={self.path} peer={self.client_address} error={exc}",
                        flush=True,
                    )

        def log_message(self, format: str, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve LeRobot VLA policy over HTTP(S)")
    parser.add_argument("--policy-path", required=True, help="Path to LeRobot pretrained_model directory")
    parser.add_argument("--device", default="cuda:0", help="LeRobot inference device")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8443, help="Bind port")
    parser.add_argument("--tls-cert-file", default="", help="Optional TLS certificate file")
    parser.add_argument("--tls-key-file", default="", help="Optional TLS private key file")
    args = parser.parse_args()

    policy_dir = Path(args.policy_path).expanduser().resolve()
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

    _install_interrupt_handlers()

    server = None
    try:
        state = LeRobotServerState(policy_dir, args.device)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(state))

        if args.tls_cert_file and args.tls_key_file:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=args.tls_cert_file, keyfile=args.tls_key_file)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            scheme = "https"
        else:
            scheme = "http"

        print(f"[lerobot_vla_server] Serving on {scheme}://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[lerobot_vla_server] interrupted stop_reason={_SERVER_STOP_REASON}")
    finally:
        if server is not None:
            server.server_close()
            print("[lerobot_vla_server] server_closed")


if __name__ == "__main__":
    main()
