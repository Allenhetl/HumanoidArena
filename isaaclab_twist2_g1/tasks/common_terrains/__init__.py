# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Common terrain generation module for HumanoidArena.
Provides terrain generation utilities adapted from instinctlab.
"""

from .terrain_generator import HumanoidTerrainGenerator
from .terrain_generator_cfg import HumanoidTerrainGeneratorCfg

__all__ = [
    "HumanoidTerrainGenerator",
    "HumanoidTerrainGeneratorCfg",
]
