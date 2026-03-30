"""Shared Redis reset helpers for IsaacLab action providers."""

from __future__ import annotations

import json
import time
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover - runtime dependency
    redis = None


RESET_TRIGGER_KEY = "isaac_reset_trigger"
RESET_COMPLETE_KEY = "isaac_reset_complete_unitree_g1_with_hands"


def create_redis_client(host: str = "localhost", port: int = 6379, *, decode_responses: bool = False):
    if redis is None:
        raise RuntimeError("redis package is not installed")
    return redis.Redis(host=host, port=port, db=0, decode_responses=decode_responses)


def publish_reset_command(
    reset_category: str = "3",
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> None:
    client = redis_client or create_redis_client(host=host, port=port)
    payload = {
        "reset_category": str(reset_category),
        "timestamp": int(time.time() * 1000),
    }
    client.set(RESET_TRIGGER_KEY, json.dumps(payload))
    client.expire(RESET_TRIGGER_KEY, 5)


def read_reset_trigger(
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> dict[str, Any] | None:
    client = redis_client or create_redis_client(host=host, port=port, decode_responses=True)
    raw = client.get(RESET_TRIGGER_KEY)
    if not raw:
        return None
    return json.loads(raw)


def clear_reset_trigger(
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> None:
    client = redis_client or create_redis_client(host=host, port=port)
    client.delete(RESET_TRIGGER_KEY)


def publish_reset_complete(
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> None:
    client = redis_client or create_redis_client(host=host, port=port)
    payload = {
        "status": "complete",
        "timestamp": int(time.time() * 1000),
    }
    client.set(RESET_COMPLETE_KEY, json.dumps(payload))
    client.expire(RESET_COMPLETE_KEY, 5)


def consume_reset_complete(
    *,
    host: str = "localhost",
    port: int = 6379,
    redis_client: Any | None = None,
) -> bool:
    client = redis_client or create_redis_client(host=host, port=port, decode_responses=True)
    raw = client.get(RESET_COMPLETE_KEY)
    if not raw:
        return False
    data = json.loads(raw)
    if data.get("status") != "complete":
        return False
    client.delete(RESET_COMPLETE_KEY)
    return True
