# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Height field terrain generation module.
"""

from .hf_terrains_cfg import (
    FlatTerrainCfg,
    SlopeTerrainCfg,
    StairsTerrainCfg,
    PyramidStairsTerrainCfg,
    WaveTerrainCfg,
    SteppingStonesTerrainCfg,
    GapTerrainCfg,
)

from .hf_terrains import (
    flat_terrain,
    slope_terrain,
    stairs_terrain,
    pyramid_stairs_terrain,
    wave_terrain,
    stepping_stones_terrain,
    gap_terrain,
)

__all__ = [
    # Configs
    "FlatTerrainCfg",
    "SlopeTerrainCfg",
    "StairsTerrainCfg",
    "PyramidStairsTerrainCfg",
    "WaveTerrainCfg",
    "SteppingStonesTerrainCfg",
    "GapTerrainCfg",
    # Functions
    "flat_terrain",
    "slope_terrain",
    "stairs_terrain",
    "pyramid_stairs_terrain",
    "wave_terrain",
    "stepping_stones_terrain",
    "gap_terrain",
]
