# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Visual zone marker generation utilities.
Creates boundary lines and corner markers for visual zones.
"""

from typing import List, Tuple, Dict, Any
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from .zone_markers_cfg import VisualZoneCfg, RectangleZoneCfg, CircleZoneCfg, PathZoneCfg


def create_rectangle_boundary_lines(
    cfg: RectangleZoneCfg,
    prim_path_prefix: str,
) -> List[AssetBaseCfg]:
    """Create boundary line assets for a rectangular zone.
    
    Args:
        cfg: Rectangle zone configuration.
        prim_path_prefix: Prefix for USD prim paths.
        
    Returns:
        List of AssetBaseCfg for boundary lines.
    """
    assets = []
    
    half_w = cfg.width / 2
    half_l = cfg.length / 2
    px, py, pz = cfg.position
    
    # Apply rotation
    cos_r = math.cos(cfg.rotation)
    sin_r = math.sin(cfg.rotation)
    
    # Define corners (before rotation)
    corners = [
        (-half_w, -half_l),
        (half_w, -half_l),
        (half_w, half_l),
        (-half_w, half_l),
    ]
    
    # Rotate corners
    rotated_corners = []
    for cx, cy in corners:
        rx = cx * cos_r - cy * sin_r + px
        ry = cx * sin_r + cy * cos_r + py
        rotated_corners.append((rx, ry))
    
    # Create lines between corners
    line_names = ["front", "right", "back", "left"]
    for i, name in enumerate(line_names):
        x1, y1 = rotated_corners[i]
        x2, y2 = rotated_corners[(i + 1) % 4]
        
        # Calculate line center and length
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate rotation angle
        angle = math.atan2(y2 - y1, x2 - x1)
        
        # Quaternion for rotation around Z
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        
        line_cfg = AssetBaseCfg(
            prim_path=f"{prim_path_prefix}/{cfg.name}_line_{name}",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[center_x, center_y, pz + cfg.boundary_height / 2],
                rot=[qw, 0.0, 0.0, qz],
            ),
            spawn=sim_utils.CuboidCfg(
                size=(line_length, cfg.boundary_width, cfg.boundary_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=cfg.boundary_color,
                    metallic=0.0,
                    roughness=0.8,
                ),
            ),
        )
        assets.append(line_cfg)
    
    return assets


def create_corner_markers(
    cfg: VisualZoneCfg,
    corners: List[Tuple[float, float]],
    prim_path_prefix: str,
) -> List[AssetBaseCfg]:
    """Create corner marker assets.
    
    Args:
        cfg: Zone configuration.
        corners: List of (x, y) corner positions.
        prim_path_prefix: Prefix for USD prim paths.
        
    Returns:
        List of AssetBaseCfg for corner markers.
    """
    if not cfg.enable_corner_markers:
        return []
    
    assets = []
    px, py, pz = cfg.position
    
    for i, (cx, cy) in enumerate(corners):
        if cfg.corner_marker_type == "cone":
            spawn_cfg = sim_utils.ConeCfg(
                radius=cfg.corner_marker_size,
                height=cfg.corner_marker_height,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=cfg.corner_marker_color,
                    metallic=0.0,
                ),
            )
        elif cfg.corner_marker_type == "cylinder":
            spawn_cfg = sim_utils.CylinderCfg(
                radius=cfg.corner_marker_size,
                height=cfg.corner_marker_height,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=cfg.corner_marker_color,
                    metallic=0.0,
                ),
            )
        else:  # sphere
            spawn_cfg = sim_utils.SphereCfg(
                radius=cfg.corner_marker_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=cfg.corner_marker_color,
                    metallic=0.0,
                ),
            )
        
        marker_cfg = AssetBaseCfg(
            prim_path=f"{prim_path_prefix}/{cfg.name}_corner_{i}",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[cx, cy, pz + cfg.corner_marker_height / 2],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=spawn_cfg,
        )
        assets.append(marker_cfg)
    
    return assets


def create_zone_boundary_assets(
    cfg: VisualZoneCfg,
    prim_path_prefix: str,
) -> List[AssetBaseCfg]:
    """Create all boundary assets for a visual zone.
    
    Args:
        cfg: Zone configuration.
        prim_path_prefix: Prefix for USD prim paths.
        
    Returns:
        List of AssetBaseCfg for all zone markers.
    """
    assets = []
    
    if isinstance(cfg, RectangleZoneCfg):
        # Create boundary lines
        assets.extend(create_rectangle_boundary_lines(cfg, prim_path_prefix))
        
        # Calculate corners for markers
        half_w = cfg.width / 2
        half_l = cfg.length / 2
        px, py, _ = cfg.position
        cos_r = math.cos(cfg.rotation)
        sin_r = math.sin(cfg.rotation)
        
        corners_local = [
            (-half_w, -half_l),
            (half_w, -half_l),
            (half_w, half_l),
            (-half_w, half_l),
        ]
        
        corners = []
        for cx, cy in corners_local:
            rx = cx * cos_r - cy * sin_r + px
            ry = cx * sin_r + cy * cos_r + py
            corners.append((rx, ry))
        
        assets.extend(create_corner_markers(cfg, corners, prim_path_prefix))
    
    elif isinstance(cfg, CircleZoneCfg):
        px, py, pz = cfg.position
        
        # Create line segments for circle
        for i in range(cfg.num_segments):
            angle1 = 2 * math.pi * i / cfg.num_segments
            angle2 = 2 * math.pi * (i + 1) / cfg.num_segments
            
            x1 = px + cfg.radius * math.cos(angle1)
            y1 = py + cfg.radius * math.sin(angle1)
            x2 = px + cfg.radius * math.cos(angle2)
            y2 = py + cfg.radius * math.sin(angle2)
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            line_angle = math.atan2(y2 - y1, x2 - x1)
            
            qw = math.cos(line_angle / 2)
            qz = math.sin(line_angle / 2)
            
            line_cfg = AssetBaseCfg(
                prim_path=f"{prim_path_prefix}/{cfg.name}_arc_{i}",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=[center_x, center_y, pz + cfg.boundary_height / 2],
                    rot=[qw, 0.0, 0.0, qz],
                ),
                spawn=sim_utils.CuboidCfg(
                    size=(line_length, cfg.boundary_width, cfg.boundary_height),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=cfg.boundary_color,
                        metallic=0.0,
                    ),
                ),
            )
            assets.append(line_cfg)
    
    return assets
