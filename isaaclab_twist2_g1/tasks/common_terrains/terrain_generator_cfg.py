# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Terrain generator configuration for HumanoidArena.
"""

from isaaclab.terrains import TerrainGeneratorCfg as TerrainGeneratorCfgBase
from isaaclab.utils import configclass

from .terrain_generator import HumanoidTerrainGenerator


@configclass
class HumanoidTerrainGeneratorCfg(TerrainGeneratorCfgBase):
    """Configuration for humanoid terrain generator."""
    
    class_type: type = HumanoidTerrainGenerator
