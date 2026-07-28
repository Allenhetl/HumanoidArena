"""Helpers for de-duplicating Redis recording control events."""

from __future__ import annotations

from typing import Any, Optional, Tuple


RecordingControlIdentity = Tuple[str, int]


def get_recording_control_identity(payload: dict[str, Any]) -> RecordingControlIdentity:
    """Return an event identity that remains stable across env resets."""
    session_id = payload.get("session_id") or payload.get("source") or "legacy"
    try:
        sequence = int(payload.get("sequence", -1))
    except (TypeError, ValueError):
        sequence = -1
    return str(session_id), sequence


def should_accept_recording_control(
    payload: dict[str, Any],
    *,
    last_identity: Optional[RecordingControlIdentity],
    waiting_for_reset_complete: bool,
) -> bool:
    """Return whether a recording command is a new event for this episode."""
    if waiting_for_reset_complete or str(payload.get("command", "none")) == "none":
        return False
    if get_recording_control_identity(payload) == last_identity:
        return False
    return True
