#
# 实用函数：如何获取奖励值
##
from __future__ import annotations

import torch


def _default_reward_tensor(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def _extract_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return _extract_scalar(value[0])
    try:
        return float(value)
    except Exception:
        return 0.0


def get_step_reward_value(env) -> torch.Tensor:
    """
    快速获取当前环境的总奖励值。

    Returns:
        torch.Tensor: 当前总奖励，失败时返回零张量。
    """
    try:
        if hasattr(env, "reward_manager"):
            dt = env.physics_dt if hasattr(env, "physics_dt") else 0.02
            return env.reward_manager.compute(dt=dt)
        return _default_reward_tensor(env)
    except Exception as e:
        print(f"获取奖励值时出错: {e}")
        return _default_reward_tensor(env)


def get_current_rewards(env):
    """
    获取当前环境的奖励值。

    Returns:
        reward_manager.compute(dt) 的返回值；失败时返回零张量。
    """
    try:
        if hasattr(env, "reward_manager"):
            dt = env.physics_dt if hasattr(env, "physics_dt") else 0.02
            return env.reward_manager.compute(dt=dt)
    except Exception as e:
        print(f"获取奖励值时出错: {e}")
    return _default_reward_tensor(env)


def get_reward_debug_string(env) -> str:
    """
    生成易读的 reward 调试字符串。

    输出总 reward，并尽量列出每个 active reward term 的数值。
    """
    if not hasattr(env, "reward_manager"):
        return "[reward_debug] reward_manager unavailable"

    try:
        total_reward = get_step_reward_value(env)
        total_value = _extract_scalar(total_reward)

        term_parts = []
        get_terms = getattr(env.reward_manager, "get_active_iterable_terms", None)
        if callable(get_terms):
            for entry in get_terms(0):
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                name = str(entry[0])
                value = _extract_scalar(entry[1])
                term_parts.append(f"{name}={value:.4f}")

        if term_parts:
            return f"[reward_debug] total={total_value:.4f} | " + ", ".join(term_parts)
        return f"[reward_debug] total={total_value:.4f}"
    except Exception as e:
        return f"[reward_debug] failed: {e}"
