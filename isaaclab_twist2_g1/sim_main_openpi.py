#!/usr/bin/env python3
"""Compatibility wrapper for the unified sim_main.py VLA entrypoint."""

from __future__ import annotations

import os
import sys


def _ensure_flag(argv: list[str], flag: str, value: str) -> list[str]:
    if flag in argv:
        return argv
    return argv + [flag, value]


def main() -> None:
    argv = sys.argv[1:]
    argv = _ensure_flag(argv, "--input_source", "vla")
    argv = _ensure_flag(argv, "--gmt_backend", "twist2")
    sys.argv = [os.path.join(os.path.dirname(__file__), "sim_main.py"), *argv]

    from sim_main import main as sim_main_main

    sim_main_main()


if __name__ == "__main__":
    main()
