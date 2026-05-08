from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from tasks.common_scene.base_scene_open_door import DOOR_POS
except Exception:
    # Keep this module importable in lightweight unit tests where the runtime
    # package layout is not available.
    DOOR_POS = [-1.614, 2.314, 0.002]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_MIN_STANDING_ROOT_HEIGHT_M = 0.45
_MIN_STANDING_UP_AXIS_Z = 0.60
_DOOR_FRAME_HALF_WIDTH_M = 0.55
_DOOR_PASSING_FORWARD_CLEARANCE_M = 0.02
def _root_up_axis_z(root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = root_quat_wxyz / torch.linalg.vector_norm(root_quat_wxyz, dim=1, keepdim=True).clamp_min(1e-8)
    x = quat[:, 1]
    y = quat[:, 2]
    return 1.0 - 2.0 * (x * x + y * y)


def _compute_standing_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    root_state = env.scene["robot"].data.root_state_w
    root_height = root_state[:, 2]
    up_axis_z = _root_up_axis_z(root_state[:, 3:7])
    return (root_height >= _MIN_STANDING_ROOT_HEIGHT_M) & (up_axis_z >= _MIN_STANDING_UP_AXIS_Z)


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz / torch.linalg.vector_norm(quat_wxyz, dim=1, keepdim=True).clamp_min(1e-8)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _log_door_pose_source_once(source: str, detail: str) -> None:
    return


def _static_door_pose_batch(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    door_pos_w = torch.tensor(DOOR_POS, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    door_quat_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32).repeat(
        env.num_envs, 1
    )
    return door_pos_w, door_quat_wxyz


def _try_resolve_door_pose_from_asset_data(env: "ManagerBasedRLEnv", door_asset):
    data = getattr(door_asset, "data", None)
    if data is None:
        return None

    root_state_w = getattr(data, "root_state_w", None)
    if root_state_w is not None:
        _log_door_pose_source_once("asset_data", "using root_state_w")
        return root_state_w[:, 0:3], root_state_w[:, 3:7]

    root_pos_w = getattr(data, "root_pos_w", None)
    root_quat_w = getattr(data, "root_quat_w", None)
    if root_pos_w is not None and root_quat_w is not None:
        _log_door_pose_source_once("asset_data", "using root_pos_w/root_quat_w")
        return root_pos_w, root_quat_w

    root_pose_w = getattr(data, "root_pose_w", None)
    if root_pose_w is not None:
        _log_door_pose_SOURCE = "using root_pose_w"
        _log_door_pose_source_once("asset_data", _log_door_pose_SOURCE)
        return root_pose_w[:, 0:3], root_pose_w[:, 3:7]

    return None


def _try_resolve_door_pose_from_stage(env: "ManagerBasedRLEnv"):
    try:
        import omni.usd
        from pxr import UsdGeom
    except Exception as exc:
        _log_door_pose_source_once("stage_unavailable", f"imports_failed={exc}")
        return None

    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            _log_door_pose_source_once("stage_unavailable", "stage_is_none")
            return None

        cache = UsdGeom.XformCache()
        pos_rows: list[list[float]] = []
        quat_rows: list[list[float]] = []
        missing_paths: list[str] = []
        for env_idx in range(int(env.num_envs)):
            prim_path = f"/World/envs/env_{env_idx}/Door"
            prim = stage.GetPrimAtPath(prim_path)
            if prim is None or not prim.IsValid() or not prim.IsActive():
                missing_paths.append(prim_path)
                pos_rows.append([float(DOOR_POS[0]), float(DOOR_POS[1]), float(DOOR_POS[2])])
                quat_rows.append([1.0, 0.0, 0.0, 0.0])
                continue

            world_matrix = cache.GetLocalToWorldTransform(prim)
            translation = world_matrix.ExtractTranslation()
            quat_wxyz = [1.0, 0.0, 0.0, 0.0]
            try:
                quat = world_matrix.ExtractRotationQuat()
                imag = quat.GetImaginary()
                quat_wxyz = [float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]
            except Exception:
                try:
                    rotation = world_matrix.ExtractRotation().GetQuat()
                    imag = rotation.GetImaginary()
                    quat_wxyz = [float(rotation.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])]
                except Exception:
                    quat_wxyz = [1.0, 0.0, 0.0, 0.0]

            pos_rows.append([float(translation[0]), float(translation[1]), float(translation[2])])
            quat_rows.append(quat_wxyz)

        if missing_paths:
            _log_door_pose_source_once(
                "stage_world_transform_partial",
                f"using stage transform with static fallback for missing prims: {missing_paths}",
            )
        else:
            _log_door_pose_source_once("stage_world_transform", "using /World/envs/env_i/Door world transform")

        door_pos_w = torch.tensor(pos_rows, device=env.device, dtype=torch.float32)
        door_quat_wxyz = torch.tensor(quat_rows, device=env.device, dtype=torch.float32)
        return door_pos_w, door_quat_wxyz
    except Exception as exc:
        _log_door_pose_source_once("stage_world_transform_failed", str(exc))
        return None


def _resolve_door_root_pose_w(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    door_asset = None
    for scene_key in ("door", "Door"):
        try:
            door_asset = env.scene[scene_key]
            break
        except Exception:
            continue

    if door_asset is not None:
        pose_from_asset = _try_resolve_door_pose_from_asset_data(env, door_asset)
        if pose_from_asset is not None:
            return pose_from_asset

        pose_from_stage = _try_resolve_door_pose_from_stage(env)
        if pose_from_stage is not None:
            return pose_from_stage

        _log_door_pose_source_once(
            "static_fallback",
            "scene asset found but no runtime world-pose tensors or stage transform were available",
        )
        return _static_door_pose_batch(env)

    pose_from_stage = _try_resolve_door_pose_from_stage(env)
    if pose_from_stage is not None:
        return pose_from_stage

    _log_door_pose_source_once("static_fallback", "door scene asset missing; using DOOR_POS constant")
    return _static_door_pose_batch(env)


def compute_success_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    root_state = env.scene["robot"].data.root_state_w
    root_pos_w = root_state[:, 0:3]
    door_pos_w, door_quat_wxyz = _resolve_door_root_pose_w(env)

    delta_xy_w = root_pos_w[:, 0:2] - door_pos_w[:, 0:2]
    door_yaw_w = _yaw_from_quat_wxyz(door_quat_wxyz)
    cos_yaw = torch.cos(door_yaw_w)
    sin_yaw = torch.sin(door_yaw_w)
    root_x_local = cos_yaw * delta_xy_w[:, 0] + sin_yaw * delta_xy_w[:, 1]
    root_y_local = -sin_yaw * delta_xy_w[:, 0] + cos_yaw * delta_xy_w[:, 1]

    inside_frame_width = torch.abs(root_x_local) <= _DOOR_FRAME_HALF_WIDTH_M
    passed_door = root_y_local > _DOOR_PASSING_FORWARD_CLEARANCE_M
    standing = _compute_standing_mask(env)
    return inside_frame_width & passed_door & standing


def compute_reward_open_door(env: "ManagerBasedRLEnv") -> torch.Tensor:
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    success = compute_success_mask(env)
    reward[success] = 1.0
    return reward


__all__ = [
    "compute_reward_open_door",
    "compute_success_mask",
]
