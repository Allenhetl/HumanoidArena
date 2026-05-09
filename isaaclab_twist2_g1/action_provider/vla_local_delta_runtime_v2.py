from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from action_provider.vla_smpl_runtime import (
    CANONICAL_G1_JOINT_NAMES_29,
    VLA_SMPL_ACTION_DIM,
    VLA_SMPL_HAND_DIM,
    VLA_SMPL_STATE_DIM,
    compute_yaw_rate_from_quats,
    quat_angle_between_wxyz,
    quat_normalize_wxyz,
    quat_slerp_shortest_wxyz,
    quat_to_roll_pitch_wxyz,
    reorder_canonical_to_twist2_29,
    rot6d_to_quat_wxyz_with_layout,
)


LOCAL_DELTA_V2_SCHEMA_VERSION = "local_delta_isaac_time_v2"
VLA_LOCAL_DELTA_V2_STATE_DIM = VLA_SMPL_STATE_DIM
VLA_LOCAL_DELTA_V2_ACTION_DIM = VLA_SMPL_ACTION_DIM
VLA_LOCAL_DELTA_V2_HAND_DIM = VLA_SMPL_HAND_DIM

CANONICAL_JOINT_NAMES_29 = CANONICAL_G1_JOINT_NAMES_29

STATE_NAMES = [
    *[f"state.root_rot6d.{idx}" for idx in range(6)],
    *[f"state.dof_pos.{name}" for name in CANONICAL_JOINT_NAMES_29],
    *[f"state.dof_vel.{name}" for name in CANONICAL_JOINT_NAMES_29],
]

ACTION_NAMES = [
    "action.root_local_xy_delta.x",
    "action.root_local_xy_delta.y",
    "action.root_z",
    *[f"action.root_rot6d.{idx}" for idx in range(6)],
    *[f"action.joint_pos.{name}" for name in CANONICAL_JOINT_NAMES_29],
    "action.hand_binary.left",
    "action.hand_binary.right",
]


def _rot_from_quat_wxyz(quat_wxyz: np.ndarray) -> R:
    quat_xyzw = quat_normalize_wxyz(np.asarray(quat_wxyz, dtype=np.float32).reshape(4))[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw)


def rotate_world_delta_to_root_local(
    root_quat_wxyz: np.ndarray,
    world_xy_delta: np.ndarray,
) -> np.ndarray:
    world_delta_3d = np.zeros((3,), dtype=np.float32)
    world_delta_3d[:2] = np.asarray(world_xy_delta, dtype=np.float32).reshape(2)
    return _rot_from_quat_wxyz(root_quat_wxyz).inv().apply(world_delta_3d)[:2].astype(np.float32)


def rotate_root_local_delta_to_world(
    root_quat_wxyz: np.ndarray,
    root_local_xy_delta: np.ndarray,
) -> np.ndarray:
    local_delta_3d = np.zeros((3,), dtype=np.float32)
    local_delta_3d[:2] = np.asarray(root_local_xy_delta, dtype=np.float32).reshape(2)
    return _rot_from_quat_wxyz(root_quat_wxyz).apply(local_delta_3d)[:2].astype(np.float32)


def build_vla_local_delta_action(
    *,
    root_local_xy_delta: np.ndarray,
    root_z: np.ndarray | float,
    root_rot6d: np.ndarray,
    joint_pos_canonical_29: np.ndarray,
    hand_binary: np.ndarray,
) -> np.ndarray:
    action = np.concatenate(
        [
            np.asarray(root_local_xy_delta, dtype=np.float32).reshape(2),
            np.asarray(root_z, dtype=np.float32).reshape(1),
            np.asarray(root_rot6d, dtype=np.float32).reshape(6),
            np.asarray(joint_pos_canonical_29, dtype=np.float32).reshape(29),
            np.asarray(hand_binary, dtype=np.float32).reshape(2),
        ],
        axis=0,
    ).astype(np.float32)
    if action.shape != (VLA_LOCAL_DELTA_V2_ACTION_DIM,):
        raise ValueError(f"Unexpected local-delta v2 action shape: {action.shape}")
    return action


@dataclass
class UnifiedLocalDeltaActionFrameV2:
    root_local_xy_delta: np.ndarray
    root_xy_delta_world: np.ndarray
    root_z: float
    body_pos_world: np.ndarray
    root_orient_rot6d: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos_canonical_29: np.ndarray
    hand_binary: np.ndarray
    prev_root_quat_wxyz: np.ndarray | None
    prev_joint_pos_canonical_29: np.ndarray | None


class UnifiedLocalDeltaActionRuntimeV2:
    def __init__(self, root_rot6d_layout: str = "row", max_root_delta_deg: float | None = None) -> None:
        self._root_rot6d_layout = str(root_rot6d_layout).strip().lower()
        if self._root_rot6d_layout not in {"row", "col", "auto"}:
            raise ValueError(
                f"Unsupported root_rot6d_layout={root_rot6d_layout}. "
                "Expected one of ['row', 'col', 'auto']."
            )
        self._max_root_delta_rad: float | None = None
        self.set_max_root_delta_deg(max_root_delta_deg)
        self.reset()

    def reset(self, body_xy_world: np.ndarray | None = None) -> None:
        if body_xy_world is None:
            self._body_xy_world = np.zeros((2,), dtype=np.float32)
        else:
            self._body_xy_world = np.asarray(body_xy_world, dtype=np.float32).reshape(2).copy()
        self._prev_root_quat_wxyz: np.ndarray | None = None
        self._prev_joint_pos_canonical_29: np.ndarray | None = None
        self._last_selected_root_rot6d_layout: str = "row"

    def prime_root_quat(self, root_quat_wxyz: np.ndarray) -> None:
        self._prev_root_quat_wxyz = quat_normalize_wxyz(
            np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4)
        )

    def set_max_root_delta_deg(self, max_root_delta_deg: float | None) -> None:
        if max_root_delta_deg is None:
            self._max_root_delta_rad = None
            return
        value = float(max_root_delta_deg)
        if not np.isfinite(value) or value <= 0.0:
            self._max_root_delta_rad = None
            return
        self._max_root_delta_rad = float(np.deg2rad(value))

    def _decode_root_quat(self, root_orient_rot6d: np.ndarray) -> np.ndarray:
        if self._root_rot6d_layout in {"row", "col"}:
            root_quat_wxyz = rot6d_to_quat_wxyz_with_layout(
                root_orient_rot6d,
                layout=self._root_rot6d_layout,
            ).reshape(4).astype(np.float32)
            self._last_selected_root_rot6d_layout = self._root_rot6d_layout
            return root_quat_wxyz

        quat_row = rot6d_to_quat_wxyz_with_layout(root_orient_rot6d, layout="row").reshape(4).astype(np.float32)
        quat_col = rot6d_to_quat_wxyz_with_layout(root_orient_rot6d, layout="col").reshape(4).astype(np.float32)
        if self._prev_root_quat_wxyz is None:
            self._last_selected_root_rot6d_layout = "row"
            return quat_row
        row_delta = quat_angle_between_wxyz(self._prev_root_quat_wxyz, quat_row)
        col_delta = quat_angle_between_wxyz(self._prev_root_quat_wxyz, quat_col)
        if col_delta < row_delta:
            self._last_selected_root_rot6d_layout = "col"
            return quat_col
        self._last_selected_root_rot6d_layout = "row"
        return quat_row

    def step(self, action: np.ndarray) -> UnifiedLocalDeltaActionFrameV2:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (VLA_LOCAL_DELTA_V2_ACTION_DIM,):
            raise ValueError(
                f"Expected local-delta v2 action dim {VLA_LOCAL_DELTA_V2_ACTION_DIM}, got {action.shape}"
            )

        root_local_xy_delta = action[0:2].astype(np.float32, copy=True)
        root_z = float(action[2])
        root_orient_rot6d = action[3:9].astype(np.float32, copy=True)
        joint_pos_canonical_29 = action[9:38].astype(np.float32, copy=True)
        hand_binary = action[38:40].astype(np.float32, copy=True)

        root_quat_wxyz = self._decode_root_quat(root_orient_rot6d)
        if self._max_root_delta_rad is not None and self._prev_root_quat_wxyz is not None:
            root_delta = quat_angle_between_wxyz(self._prev_root_quat_wxyz, root_quat_wxyz)
            if root_delta > self._max_root_delta_rad:
                blend = self._max_root_delta_rad / max(root_delta, 1e-8)
                root_quat_wxyz = quat_slerp_shortest_wxyz(
                    self._prev_root_quat_wxyz,
                    root_quat_wxyz,
                    blend,
                )

        root_xy_delta_world = rotate_root_local_delta_to_world(
            root_quat_wxyz=root_quat_wxyz,
            root_local_xy_delta=root_local_xy_delta,
        )
        prev_root_quat = None if self._prev_root_quat_wxyz is None else self._prev_root_quat_wxyz.copy()
        prev_joint_pos = (
            None if self._prev_joint_pos_canonical_29 is None else self._prev_joint_pos_canonical_29.copy()
        )

        self._body_xy_world = self._body_xy_world + root_xy_delta_world
        body_pos_world = np.array(
            [self._body_xy_world[0], self._body_xy_world[1], root_z],
            dtype=np.float32,
        )
        self._prev_root_quat_wxyz = root_quat_wxyz.copy()
        self._prev_joint_pos_canonical_29 = joint_pos_canonical_29.copy()

        return UnifiedLocalDeltaActionFrameV2(
            root_local_xy_delta=root_local_xy_delta,
            root_xy_delta_world=root_xy_delta_world,
            root_z=root_z,
            body_pos_world=body_pos_world,
            root_orient_rot6d=root_orient_rot6d,
            root_quat_wxyz=root_quat_wxyz,
            joint_pos_canonical_29=joint_pos_canonical_29,
            hand_binary=hand_binary,
            prev_root_quat_wxyz=prev_root_quat,
            prev_joint_pos_canonical_29=prev_joint_pos,
        )


def build_twist2_mimic_obs_v2(
    *,
    runtime_frame: UnifiedLocalDeltaActionFrameV2,
    control_dt: float,
) -> np.ndarray:
    dt = max(float(control_dt), 1e-6)
    root_quat_wxyz = quat_normalize_wxyz(runtime_frame.root_quat_wxyz.astype(np.float32, copy=False))
    xy_vel_local = (runtime_frame.root_local_xy_delta / dt).astype(np.float32)
    roll_pitch = quat_to_roll_pitch_wxyz(root_quat_wxyz)
    yaw_vel = compute_yaw_rate_from_quats(
        runtime_frame.prev_root_quat_wxyz,
        root_quat_wxyz,
        control_dt,
    )
    joint_pos_twist2 = reorder_canonical_to_twist2_29(runtime_frame.joint_pos_canonical_29)
    mimic_obs = np.concatenate(
        [
            xy_vel_local,
            np.array([runtime_frame.root_z], dtype=np.float32),
            roll_pitch,
            yaw_vel,
            joint_pos_twist2,
        ],
        axis=0,
    ).astype(np.float32)
    if mimic_obs.shape != (35,):
        raise ValueError(f"Unexpected TWIST2 mimic_obs shape: {mimic_obs.shape}")
    return mimic_obs


def build_sonic_joint29_payload_v2(
    *,
    runtime_frame: UnifiedLocalDeltaActionFrameV2,
    control_dt: float,
) -> dict[str, np.ndarray]:
    if runtime_frame.prev_joint_pos_canonical_29 is None:
        joint_vel = np.zeros((29,), dtype=np.float32)
    else:
        joint_vel = (
            (runtime_frame.joint_pos_canonical_29 - runtime_frame.prev_joint_pos_canonical_29)
            / max(float(control_dt), 1e-6)
        ).astype(np.float32)
    return {
        "body_pos": runtime_frame.body_pos_world.astype(np.float32, copy=True),
        "body_quat_w": runtime_frame.root_quat_wxyz.astype(np.float32, copy=True),
        "joint_pos": runtime_frame.joint_pos_canonical_29.astype(np.float32, copy=True),
        "joint_vel": joint_vel,
    }
