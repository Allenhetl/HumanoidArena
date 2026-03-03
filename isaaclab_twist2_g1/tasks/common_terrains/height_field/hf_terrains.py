# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Height field terrain generation functions.
Simplified implementations for HumanoidArena.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def flat_terrain(difficulty: float, cfg: "hf_terrains_cfg.FlatTerrainCfg") -> np.ndarray:
    """Generate a flat terrain with optional Perlin noise.
    
    Args:
        difficulty: Terrain difficulty (0-1).
        cfg: Terrain configuration.
        
    Returns:
        Height field as 2D numpy array.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    hf_raw = np.zeros((width_pixels, length_pixels))
    
    if cfg.noise_scale > 0:
        from ..perlin import generate_fractal_noise_2d
        noise = generate_fractal_noise_2d(
            xSize=cfg.size[0],
            ySize=cfg.size[1],
            xSamples=width_pixels,
            ySamples=length_pixels,
            frequency=cfg.noise_frequency,
            zScale=cfg.noise_scale * difficulty,
        )
        hf_raw += noise / cfg.vertical_scale
    
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def slope_terrain(difficulty: float, cfg: "hf_terrains_cfg.SlopeTerrainCfg") -> np.ndarray:
    """Generate a sloped terrain.
    
    Args:
        difficulty: Terrain difficulty (0-1).
        cfg: Terrain configuration.
        
    Returns:
        Height field as 2D numpy array.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    
    hf_raw = np.zeros((width_pixels, length_pixels))
    
    # Create slope along y-axis
    for i in range(length_pixels):
        height = slope * i * cfg.horizontal_scale / cfg.vertical_scale
        hf_raw[:, i] = height
    
    # Add flat platform at center
    platform_pixels = int(cfg.platform_width / cfg.horizontal_scale / 2)
    center_x = width_pixels // 2
    center_y = length_pixels // 2
    
    x1 = max(0, center_x - platform_pixels)
    x2 = min(width_pixels, center_x + platform_pixels)
    y1 = max(0, center_y - platform_pixels)
    y2 = min(length_pixels, center_y + platform_pixels)
    
    platform_height = hf_raw[center_x, center_y]
    hf_raw[x1:x2, y1:y2] = platform_height
    
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def stairs_terrain(difficulty: float, cfg: "hf_terrains_cfg.StairsTerrainCfg") -> np.ndarray:
    """Generate stairs terrain.
    
    Args:
        difficulty: Terrain difficulty (0-1).
        cfg: Terrain configuration.
        
    Returns:
        Height field as 2D numpy array.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    step_width_pixels = int(cfg.step_width / cfg.horizontal_scale)
    step_height_pixels = int(step_height / cfg.vertical_scale)
    
    if not cfg.going_up:
        step_height_pixels = -step_height_pixels
    
    hf_raw = np.zeros((width_pixels, length_pixels))
    
    current_height = 0
    for y in range(0, length_pixels, step_width_pixels):
        y_end = min(y + step_width_pixels, length_pixels)
        hf_raw[:, y:y_end] = current_height
        current_height += step_height_pixels
    
    # Add flat platform at start
    platform_pixels = int(cfg.platform_width / cfg.horizontal_scale)
    hf_raw[:, :platform_pixels] = 0
    
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def pyramid_stairs_terrain(difficulty: float, cfg: "hf_terrains_cfg.PyramidStairsTerrainCfg") -> np.ndarray:
    """Generate pyramid stairs terrain.
    
    Args:
        difficulty: Terrain difficulty (0-1).
        cfg: Terrain configuration.
        
    Returns:
        Height field as 2D numpy array.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height = -step_height
    
    step_width_pixels = int(cfg.step_width / cfg.horizontal_scale)
    step_height_pixels = int(step_height / cfg.vertical_scale)
    platform_pixels = int(cfg.platform_width / cfg.horizontal_scale)
    
    hf_raw = np.zeros((width_pixels, length_pixels))
    
    current_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    
    while (stop_x - start_x) > platform_pixels and (stop_y - start_y) > platform_pixels:
        start_x += step_width_pixels
        stop_x -= step_width_pixels
        start_y += step_width_pixels
        stop_y -= step_width_pixels
        current_height += step_height_pixels
        
        if start_x < stop_x and start_y < stop_y:
            hf_raw[start_x:stop_x, start_y:stop_y] = current_height
    
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def wave_terrain(difficulty: float, cfg: "hf_terrains_cfg.WaveTerrainCfg") -> np.ndarray:
    """Generate wave terrain.
    
    Args:
        difficulty: Terrain difficulty (0-1).
        cfg: Terrain configuration.
        
    Returns:
        Height field as 2D numpy array.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    amplitude_pixels = int(amplitude / cfg.vertical_scale)
    
    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * np.pi / wave_length
    
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    hf_raw = amplitude_pixels * (np.sin(wave_number * xx) + np.cos(wave_number * yy))
    
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def stepping_stones_terrain(difficulty: float, cfg: "hf_terrains_cfg.SteppingStonesTerrainCfg") -> np.ndarray:
    """Generate stepping stones terrain.
    
    Args:
        difficulty: Terrain difficulty (0-1).
        cfg: Terrain configuration.
        
    Returns:
        Height field as 2D numpy array.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    stone_height = cfg.stone_height_range[0] + difficulty * (cfg.stone_height_range[1] - cfg.stone_height_range[0])
    stone_size = cfg.stone_size_range[1] - difficulty * (cfg.stone_size_range[1] - cfg.stone_size_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (cfg.stone_distance_range[1] - cfg.stone_distance_range[0])
    
    stone_height_pixels = int(stone_height / cfg.vertical_scale)
    stone_size_pixels = int(stone_size / cfg.horizontal_scale)
    stone_distance_pixels = int(stone_distance / cfg.horizontal_scale)
    depth_pixels = int(cfg.depth / cfg.vertical_scale)
    platform_pixels = int(cfg.platform_width / cfg.horizontal_scale / 2)
    
    # Start with pit
    hf_raw = np.ones((width_pixels, length_pixels)) * depth_pixels
    
    # Add stepping stones
    step_x = stone_size_pixels + stone_distance_pixels
    step_y = stone_size_pixels + stone_distance_pixels
    
    for x in range(0, width_pixels, step_x):
        for y in range(0, length_pixels, step_y):
            # Random offset
            offset_x = np.random.randint(-stone_distance_pixels // 2, stone_distance_pixels // 2 + 1)
            offset_y = np.random.randint(-stone_distance_pixels // 2, stone_distance_pixels // 2 + 1)
            
            stone_x = x + offset_x
            stone_y = y + offset_y
            
            x1 = max(0, stone_x)
            x2 = min(width_pixels, stone_x + stone_size_pixels)
            y1 = max(0, stone_y)
            y2 = min(length_pixels, stone_y + stone_size_pixels)
            
            if x1 < x2 and y1 < y2:
                hf_raw[x1:x2, y1:y2] = stone_height_pixels
    
    # Add flat platform at center
    center_x = width_pixels // 2
    center_y = length_pixels // 2
    x1 = max(0, center_x - platform_pixels)
    x2 = min(width_pixels, center_x + platform_pixels)
    y1 = max(0, center_y - platform_pixels)
    y2 = min(length_pixels, center_y + platform_pixels)
    hf_raw[x1:x2, y1:y2] = 0
    
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def gap_terrain(difficulty: float, cfg: "hf_terrains_cfg.GapTerrainCfg") -> np.ndarray:
    """Generate gap terrain.
    
    Args:
        difficulty: Terrain difficulty (0-1).
        cfg: Terrain configuration.
        
    Returns:
        Height field as 2D numpy array.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    gap_width_pixels = int(gap_width / cfg.horizontal_scale)
    gap_depth_pixels = int(cfg.gap_depth / cfg.vertical_scale)
    platform_pixels = int(cfg.platform_width / cfg.horizontal_scale / 2)
    
    hf_raw = np.zeros((width_pixels, length_pixels))
    
    # Add gap at center
    center_y = length_pixels // 2
    y1 = center_y - gap_width_pixels // 2
    y2 = center_y + gap_width_pixels // 2
    
    hf_raw[:, y1:y2] = -gap_depth_pixels
    
    # Ensure platform area is flat
    center_x = width_pixels // 2
    x1 = max(0, center_x - platform_pixels)
    x2 = min(width_pixels, center_x + platform_pixels)
    py1 = max(0, center_y - platform_pixels - gap_width_pixels)
    py2 = min(y1, center_y - gap_width_pixels // 2)
    
    if py1 < py2:
        hf_raw[x1:x2, py1:py2] = 0
    
    return np.rint(hf_raw).astype(np.int16)
