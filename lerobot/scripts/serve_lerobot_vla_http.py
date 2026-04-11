#!/usr/bin/env python3

import argparse
import base64
import json
import ssl
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch


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
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    from lerobot.utils.control_utils import predict_action

    config = PreTrainedConfig.from_pretrained(policy_dir)
    config.device = device_name
    policy_cls = get_policy_class(config.type)

    policy = policy_cls.from_pretrained(policy_dir, config=config)
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    _reconnect_relative_absolute_steps(preprocessor, postprocessor)
    return config, policy, preprocessor, postprocessor, predict_action


class LeRobotServerState:
    def __init__(self, policy_dir: Path, device_name: str):
        self.config, self.policy, self.preprocessor, self.postprocessor, self.predict_action = _load_policy(
            policy_dir, device_name
        )
        self.expected_state_shape = _feature_shape_dim(self.config.input_features.get("observation.state"))
        self.expected_action_shape = _feature_shape_dim(self.config.output_features.get("action"))
        self.device = torch.device(device_name)
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

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
                    state.reset()
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
                action = state.infer(observation, robot_type)
                if not isinstance(action, torch.Tensor):
                    action = torch.as_tensor(action)
                action = action.detach().cpu().to(torch.float32).reshape(-1)
                if state.expected_action_shape is not None and action.shape != state.expected_action_shape:
                    raise ValueError(
                        f"Expected action shape {state.expected_action_shape}, got {tuple(action.shape)}"
                    )
                print(f"[lerobot_vla_server] action={action.tolist()}")
                self._send_json(200, {"action": action.tolist()})
            except Exception as exc:
                traceback.print_exc()
                self._send_json(500, {"error": str(exc)})

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


if __name__ == "__main__":
    main()
