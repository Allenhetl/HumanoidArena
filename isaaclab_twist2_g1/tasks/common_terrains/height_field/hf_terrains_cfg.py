# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Height field terrain configuration classes.
Simplified versions of terrain configs for HumanoidArena.
"""

from dataclasses import MISSING
from typing import Tuple

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import hf_terrains


@configclass
class FlatTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a flat terrain with optional Perlin noise."""
    
    function = hf_terrains.flat_terrain
    
    noise_scale: float = 0.0
    noise_frequency: int = 10


@configclass
class SlopeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a sloped terrain."""
    
    function = hf_terrains.slope_terrain
    
    slope_range: Tuple[float, float] = (0.0, 0.3)
    platform_width: float = 1.5


@configclass
class StairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for stairs terrain."""
    
    function = hf_terrains.stairs_terrain
    
    step_height_range: Tuple[float, float] = (0.1, 0.2)
    step_width: float = 0.3
    platform_width: float = 1.5
    going_up: bool = True


@configclass
class PyramidStairsTerrainCfg(HfTerrainBaseCfg):
    """Configuration for pyramid stairs terrain."""
    
    function = hf_terrains.pyramid_stairs_terrain
    
    step_height_range: Tuple[float, float] = (0.05, 0.15)
    step_width: float = 0.3
    platform_width: float = 1.0
    inverted: bool = False


@configclass
class WaveTerrainCfg(HfTerrainBaseCfg):
    """Configuration for wave terrain."""
    
    function = hf_terrains.wave_terrain
    
    amplitude_range: Tuple[float, float] = (0.02, 0.1)
    num_waves: int = 3


@configclass
class SteppingStonesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for stepping stones terrain."""
    
    function = hf_terrains.stepping_stones_terrain
    
    stone_height_range: Tuple[float, float] = (0.05, 0.15)
    stone_size_range: Tuple[float, float] = (0.3, 0.6)
    stone_distance_range: Tuple[float, float] = (0.1, 0.3)
    platform_width: float = 1.5
    depth: float = -0.5


@configclass
class GapTerrainCfg(HfTerrainBaseCfg):
    """Configuration for gap terrain."""
    
    function = hf_terrains.gap_terrain
    
    gap_width_range: Tuple[float, float] = (0.2, 0.6)
    gap_depth: float = 0.5
    platform_width: float = 1.5
