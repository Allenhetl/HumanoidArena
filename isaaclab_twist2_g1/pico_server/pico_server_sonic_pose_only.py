"""Pico SMPL stream server for VR Whole-body Teleoperation (Pose Mode Only)

Simplified version that only supports POSE mode for VR whole-body teleoperation.
Removed planner and other modes.

Usage:
    python pico_server_sonic_pose_only.py --vis_vr3pt --vis_smpl
"""

from collections import defaultdict, deque
import os
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import zmq

from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)

try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
except ImportError:
    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")

try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: gear_sonic.isaac_utils.rotations not available.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    xrt = None

try:
    from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
        G1GripperInverseKinematicsSolver,
    )
except ImportError:
    print("Warning: G1GripperInverseKinematicsSolver not available.")
    G1GripperInverseKinematicsSolver = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer
except ImportError:
    print("Warning: VR3PtPoseVisualizer not available (pyvista may not be installed).")
    VR3PtPoseVisualizer = None

try:
    from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import get_g1_key_frame_poses
except ImportError:
    print("Warning: get_g1_key_frame_poses not available (pyvista may not be installed).")
    get_g1_key_frame_poses = None


# SMPL joint offsets for coordinate frame alignment
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Root
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),  # L-Wrist
    sRot.from_euler("xyz", [-90, 0, 180], degrees=True),  # R-Wrist
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),  # Neck
]


def _compute_rel_transform(pose, world_frame, scalar_first=True):