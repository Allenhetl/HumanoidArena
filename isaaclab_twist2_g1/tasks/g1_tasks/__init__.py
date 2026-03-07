
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module
contains various task implementations for the G1 robot, such as pick and place, motion control, etc.
"""

def _safe_import(name):
    """Import module, skip on ImportError (e.g. pinocchio/Assimp symbol conflict in GUI mode)."""
    try:
        __import__(name, fromlist=[""])
    except ImportError as e:
        import warnings
        warnings.warn(f"Skipping {name}: {e}", ImportWarning)

# Import move_football first (no pink dependency) so football test works when others fail
from . import move_football_g1_29dof_dex3_wholebody

# Other tasks may use pink/pinocchio - wrap in _safe_import to tolerate ImportError
_safe_import("tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex3")
_safe_import("tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex1")
_safe_import("tasks.g1_tasks.pick_place_cylinder_g1_29dof_inspire")
_safe_import("tasks.g1_tasks.pick_place_redblock_g1_29dof_dex1")
_safe_import("tasks.g1_tasks.pick_place_redblock_g1_29dof_dex3")
_safe_import("tasks.g1_tasks.stack_rgyblock_g1_29dof_dex1")
_safe_import("tasks.g1_tasks.stack_rgyblock_g1_29dof_dex3")
_safe_import("tasks.g1_tasks.stack_rgyblock_g1_29dof_inspire")
_safe_import("tasks.g1_tasks.pick_redblock_into_drawer_g1_29dof_dex1")
_safe_import("tasks.g1_tasks.pick_redblock_into_drawer_g1_29dof_dex3")
_safe_import("tasks.g1_tasks.pick_place_redblock_g1_29dof_inspire")
_safe_import("tasks.g1_tasks.move_cylinder_g1_29dof_dex1_wholebody")
_safe_import("tasks.g1_tasks.move_cylinder_g1_29dof_dex3_wholebody")
_safe_import("tasks.g1_tasks.move_cylinder_g1_29dof_inspire_wholebody")
_safe_import("tasks.g1_tasks.visual_zone_g1_29dof_dex3_wholebody")

# export all modules
__all__ = [
        "pick_place_cylinder_g1_29dof_dex3", "pick_place_cylinder_g1_29dof_dex1", 
        "pick_place_redblock_g1_29dof_dex1", "pick_place_redblock_g1_29dof_dex3", 
        "stack_rgyblock_g1_29dof_dex1", "stack_rgyblock_g1_29dof_dex3", 
        "stack_rgyblock_g1_29dof_inspire",
        "pick_redblock_into_drawer_g1_29dof_dex1","pick_redblock_into_drawer_g1_29dof_dex3",
        "pick_place_redblock_g1_29dof_inspire",
        "pick_place_cylinder_g1_29dof_inspire",
        "move_cylinder_g1_29dof_dex1_wholebody",
        "move_cylinder_g1_29dof_dex3_wholebody",
        "move_cylinder_g1_29dof_inspire_wholebody",
        "visual_zone_g1_29dof_dex3_wholebody",
        "move_football_g1_29dof_dex3_wholebody",
]