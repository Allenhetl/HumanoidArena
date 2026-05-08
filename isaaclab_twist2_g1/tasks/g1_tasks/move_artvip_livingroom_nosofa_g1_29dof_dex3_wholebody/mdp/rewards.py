from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_BASKET_TOKEN_GROUPS = (
    ("basket",),
    ("waste", "basket"),
)
_BASKET_EXCLUDE_TOKENS = (
    "yellowbox",
    "targetbox",
    "skybox",
    "mailbox",
    "switchbox",
    "bbox",
)
_MIN_BASKET_SIZE_XY_M = 0.08
_MAX_BASKET_SIZE_XY_M = 1.50
_MIN_BASKET_SIZE_Z_M = 0.05
_MAX_BASKET_SIZE_Z_M = 1.50


def _resolve_object_asset(env: "ManagerBasedRLEnv"):
    for scene_key in ("object", "Object"):
        try:
            return env.scene[scene_key]
        except Exception:
            continue
    return None


def _get_root_positions_world(asset) -> torch.Tensor:
    data = asset.data

    root_com_pos_w = getattr(data, "root_com_pos_w", None)
    if root_com_pos_w is not None:
        return root_com_pos_w

    root_pos_w = getattr(data, "root_pos_w", None)
    if root_pos_w is not None:
        return root_pos_w

    root_state_w = getattr(data, "root_state_w", None)
    if root_state_w is not None:
        return root_state_w[:, 0:3]

    raise AttributeError("asset world-position tensor not found")


def _matches_basket_tokens(path_lower: str, name_lower: str) -> bool:
    haystack = f"{path_lower} {name_lower}"
    if any(token in haystack for token in _BASKET_EXCLUDE_TOKENS):
        return False
    return any(all(token in haystack for token in token_group) for token_group in _BASKET_TOKEN_GROUPS)


def _box_from_aligned_range(aligned_range) -> tuple[float, float, float, float, float, float] | None:
    mn = aligned_range.GetMin()
    mx = aligned_range.GetMax()
    size_x = float(mx[0] - mn[0])
    size_y = float(mx[1] - mn[1])
    size_z = float(mx[2] - mn[2])
    if size_x < _MIN_BASKET_SIZE_XY_M or size_y < _MIN_BASKET_SIZE_XY_M or size_z < _MIN_BASKET_SIZE_Z_M:
        return None
    if size_x > _MAX_BASKET_SIZE_XY_M or size_y > _MAX_BASKET_SIZE_XY_M or size_z > _MAX_BASKET_SIZE_Z_M:
        return None
    return (
        float(mn[0]),
        float(mx[0]),
        float(mn[1]),
        float(mx[1]),
        float(mn[2]),
        float(mx[2]),
    )


def _candidate_basket_score(path_lower: str, name_lower: str) -> int:
    haystack = f"{path_lower} {name_lower}"
    score = 0
    if name_lower == "basket":
        score += 200
    if "/basket" in path_lower:
        score += 160
    if "basket" in haystack:
        score += 120
    if "waste" in haystack:
        score += 40
    if "drink" in haystack or "bottle" in haystack:
        score -= 200
    return score


def _load_stage_basket_world(
    stage,
    bbox_cache,
    env_idx: int,
) -> tuple[float, float, float, float, float, float] | None:
    try:
        from pxr import Usd
    except Exception:
        return None

    room_root = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Room")
    if room_root is None or not room_root.IsValid() or not room_root.IsActive():
        return None

    candidates: list[tuple[int, float, tuple[float, float, float, float, float, float]]] = []
    for prim in Usd.PrimRange(room_root):
        if prim is None or not prim.IsActive():
            continue
        path_lower = str(prim.GetPath()).lower()
        name_lower = prim.GetName().lower()
        if not _matches_basket_tokens(path_lower, name_lower):
            continue
        try:
            basket_box = _box_from_aligned_range(bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        except Exception:
            continue
        if basket_box is None:
            continue
        area_xy = (basket_box[1] - basket_box[0]) * (basket_box[3] - basket_box[2])
        score = _candidate_basket_score(path_lower, name_lower)
        candidates.append((score, area_xy, basket_box))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _get_stage_baskets_world(
    env: "ManagerBasedRLEnv",
) -> list[tuple[float, float, float, float, float, float] | None]:
    cached = getattr(env, "_artvip_baskets_world", None)
    if isinstance(cached, list) and len(cached) == env.num_envs:
        return cached

    boxes: list[tuple[float, float, float, float, float, float] | None] = [None] * env.num_envs
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return boxes

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        for env_idx in range(env.num_envs):
            boxes[env_idx] = _load_stage_basket_world(stage, bbox_cache, env_idx)
    except Exception:
        boxes = [None] * env.num_envs

    if any(box is not None for box in boxes):
        env._artvip_baskets_world = boxes
    return boxes


def _object_in_box(
    object_center_w: torch.Tensor,
    box_w: tuple[float, float, float, float, float, float],
) -> bool:
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = box_w
    center_x = float(object_center_w[0].item())
    center_y = float(object_center_w[1].item())
    center_z = float(object_center_w[2].item())
    return bool(
        (x_lo <= center_x <= x_hi)
        and (y_lo <= center_y <= y_hi)
        and (z_lo <= center_z <= z_hi)
    )


def compute_success_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    object_asset = _resolve_object_asset(env)
    success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if object_asset is None:
        return success

    object_centers_w = _get_root_positions_world(object_asset)
    basket_boxes_world = _get_stage_baskets_world(env)

    for env_idx in range(env.num_envs):
        basket_box = basket_boxes_world[env_idx]
        if basket_box is None:
            continue
        success[env_idx] = _object_in_box(object_centers_w[env_idx], basket_box)

    return success


def compute_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    success = compute_success_mask(env)
    reward[success] = 1.0
    return reward


__all__ = ["compute_reward", "compute_success_mask"]
