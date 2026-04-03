import base64
import json
import ssl
import urllib.parse
import urllib.request
import urllib.error
from typing import Any

import numpy as np


class LeRobotVLAHttpClient:
    """Thin HTTP(S) client for remote LeRobot VLA inference."""

    def __init__(self, base_url: str, timeout_s: float = 5.0, verify_ssl: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.verify_ssl = bool(verify_ssl)

    def _build_url(self, path: str) -> str:
        return urllib.parse.urljoin(self.base_url + "/", path.lstrip("/"))

    def _ssl_context(self):
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != "https":
            return None
        if self.verify_ssl:
            return ssl.create_default_context()
        return ssl._create_unverified_context()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._build_url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        handlers: list[Any] = [urllib.request.ProxyHandler({})]
        ssl_context = self._ssl_context()
        if ssl_context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=ssl_context))
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {self._build_url(path)}: {error_body}") from exc
        return json.loads(raw.decode("utf-8"))

    def infer(self, front_rgb: np.ndarray, observation_state: np.ndarray, robot_type: str) -> np.ndarray:
        rgb = np.asarray(front_rgb)
        state = np.asarray(observation_state, dtype=np.float32).reshape(-1)
        payload = {
            "observation": {
                "images": {
                    "front": {
                        "shape": list(rgb.shape),
                        "dtype": str(rgb.dtype),
                        "data_b64": base64.b64encode(rgb.tobytes()).decode("ascii"),
                    }
                },
                "state": state.tolist(),
            },
            "robot_type": robot_type,
        }
        response = self._post_json("/infer", payload)
        action = np.asarray(response["action"], dtype=np.float32)
        if action.ndim != 1:
            action = action.reshape(-1)
        return action

    def reset(self) -> None:
        self._post_json("/reset", {})
