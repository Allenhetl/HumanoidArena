from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from tasks.common_scene.base_scene_open_door import DOOR_POS
except Exception:
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


def _log_pose_source_once(env: "ManagerBasedRLEnv", source: str, detail: str) -> None:
    key = f"{source}:{detail}"
    logged = getattr(env, "_open_door_pose_source_logged", None)
    if logged is None:
        logged = set()
        setattr(env, "_open_door_pose_source_logged", logged)
    if key in logged:
        return
    logged.add(key)
    print(f"[open_door] door pose source={source}: {detail}")


def _static_fallback_pose(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    door_pos_w = torch.tensor(DOOR_POS, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    door_quat_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32).repeat(
        env.num_envs, 1
    )
    return door_pos_w, door_quat_wxyz


def _try_asset_pose_tensor(
    env: "ManagerBasedRLEnv",
    value,
    *,
    width: int,
) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=env.device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] != env.num_envs or tensor.shape[1] < width:
        return None
    return tensor[:, :width]


def _try_resolve_door_pose_from_asset_data(
    env: "ManagerBasedRLEnv",
    door_asset,
) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    data = getattr(door_asset, "data", None)
    if data is None:
        return None

    root_state_w = _try_asset_pose_tensor(env, getattr(data, "root_state_w", None), width=7)
    if root_state_w is not None:
        return root_state_w[:, 0:3], root_state_w[:, 3:7], "root_state_w"

    root_pos_w = _try_asset_pose_tensor(env, getattr(data, "root_pos_w", None), width=3)
    root_quat_w = _try_asset_pose_tensor(env, getattr(data, "root_quat_w", None), width=4)
    if root_pos_w is not None and root_quat_w is not None:
        return root_pos_w, root_quat_w, "root_pos_w+root_quat_w"

    root_pose_w = _try_asset_pose_tensor(env, getattr(data, "root_pose_w", None), width=7)
    if root_pose_w is not None:
        return root_pose_w[:, 0:3], root_pose_w[:, 3:7], "root_pose_w"

    return None


def _read_stage_door_pose_w(env_idx: int) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Door")
    if prim is None or not prim.IsValid() or not prim.IsActive():
        return None

    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        quat = matrix.ExtractRotationQuat()
        imag = quat.GetImaginary()
        return (
            (float(translation[0]), float(translation[1]), float(translation[2])),
            (float(quat.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])),
        )
    except Exception:
        return None


def _try_resolve_door_pose_from_stage_world_transform(
    env: "ManagerBasedRLEnv",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    positions = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    quats = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)

    for env_idx in range(env.num_envs):
        pose = _read_stage_door_pose_w(env_idx)
        if pose is None:
            return None
        pos, quat = pose
        positions[env_idx] = torch.tensor(pos, device=env.device, dtype=torch.float32)
        quats[env_idx] = torch.tensor(quat, device=env.device, dtype=torch.float32)

    return positions, quats


def _resolve_door_root_pose_w(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    door_asset = None
    for scene_key in ("door", "Door"):
        try:
            door_asset = env.scene[scene_key]
            break
        except Exception:
            continue

    if door_asset is not None:
        asset_pose = _try_resolve_door_pose_from_asset_data(env, door_asset)
        if asset_pose is not None:
            door_pos_w, door_quat_wxyz, detail = asset_pose
            _log_pose_source_once(env, "asset_data", detail)
            return door_pos_w, door_quat_wxyz

    stage_pose = _try_resolve_door_pose_from_stage_world_transform(env)
    if stage_pose is not None:
        _log_pose_source_once(env, "stage_world_transform", "/World/envs/env_{env_idx}/Door")
        return stage_pose

    _log_pose_source_once(env, "static_fallback", f"DOOR_POS={DOOR_POS}")
    return _static_fallback_pose(env)


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
    debug_hook = getattr(getattr(env, "cfg", None), "debug_joint_runtime_step", None)
    if callable(debug_hook):
        try:
            debug_hook(env)
        except Exception as exc:
            print(f"[open_door_joint_debug] phase=runtime_hook failed: {exc}")

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    success = compute_success_mask(env)
    reward[success] = 1.0
    return reward


__all__ = [
    "compute_reward_open_door",
    "compute_success_mask",
]
