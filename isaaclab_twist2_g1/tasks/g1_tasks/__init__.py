
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
from . import move_football_single_g1_29dof_dex3_wholebody
from . import move_boxing_bag_g1_29dof_dex3_wholebody
from . import move_pickplace_doubledesk_g1_29dof_dex3_wholebody
from . import push_t_g1_29dof_dex3_wholebody
from . import move_three_step_platform_g1_29dof_dex3_wholebody
from . import move_artvip_livingroom_nosofa_g1_29dof_dex3_wholebody
from . import move_open_door_g1_29dof_dex3_wholebody
from . import move_pickplace_small_trolley_g1_29dof_dex3_wholebody
from . import move_sit_sofa_g1_29dof_dex3_wholebody

# Other tasks may use pink/pinocchio - wrap in _safe_import to tolerate ImportError

# export all modules
__all__ = [
        "move_football_g1_29dof_dex3_wholebody",
        "move_football_single_g1_29dof_dex3_wholebody",
        "move_boxing_bag_g1_29dof_dex3_wholebody",
        "move_pickplace_doubledesk_g1_29dof_dex3_wholebody",
        "push_t_g1_29dof_dex3_wholebody",
        "move_three_step_platform_g1_29dof_dex3_wholebody",
        "move_artvip_livingroom_nosofa_g1_29dof_dex3_wholebody",
        "move_open_door_g1_29dof_dex3_wholebody",
        "move_pickplace_small_trolley_g1_29dof_dex3_wholebody",
        "move_sit_sofa_g1_29dof_dex3_wholebody",
]
