# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Configuration classes for visual zone markers.
"""

from dataclasses import dataclass, field
from typing import Tuple, Literal, List


@dataclass
class VisualZoneCfg:
    """Base configuration for visual zone markers."""
    
    # Zone identification
    name: str = "zone"
    
    # Zone position (relative to environment origin)
    position: Tuple[float, float, float] = (0.0, 2.0, 0.0)
    
    # Boundary line properties
    boundary_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # RGB
    boundary_width: float = 0.02  # Line width in meters
    boundary_height: float = 0.01  # Line height in meters
    
    # Corner marker properties
    enable_corner_markers: bool = True
    corner_marker_type: Literal["cone", "cylinder", "sphere"] = "cone"
    corner_marker_color: Tuple[float, float, float] = (1.0, 1.0, 0.0)
    corner_marker_size: float = 0.05
    corner_marker_height: float = 0.15


@dataclass
class RectangleZoneCfg(VisualZoneCfg):
    """Configuration for rectangular visual zone."""
    
    # Zone dimensions
    width: float = 1.0  # X-axis
    length: float = 1.0  # Y-axis
    
    # Rotation (around Z-axis, in radians)
    rotation: float = 0.0


@dataclass
class CircleZoneCfg(VisualZoneCfg):
    """Configuration for circular visual zone."""
    
    # Circle properties
    radius: float = 0.5
    num_segments: int = 16  # Number of line segments to approximate circle


@dataclass
class PathZoneCfg(VisualZoneCfg):
    """Configuration for path/corridor visual zone."""
    
    # Path properties
    path_width: float = 0.8  # Width of the path corridor
    waypoints: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 2.0),
    ])
