#!/usr/bin/env python3
"""Backward-compatible wrapper for the renamed SONIC multicam rerecord tool."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("rerecord_sonic_recordings_to_multicam.py")
    print(
        "[deprecated] migrate_sonic_recordings_v2_to_v3.py was renamed to "
        f"{target.name}; forwarding to the new entrypoint."
    )
    runpy.run_path(str(target), run_name="__main__")
