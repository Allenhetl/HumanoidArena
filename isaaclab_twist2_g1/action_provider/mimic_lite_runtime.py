"""Runtime adapter for exported MimicLite policies.

This module intentionally depends only on numpy/torch/onnxruntime and the IsaacLab
robot state exposed by HumanoidArena. It mirrors the observation order recorded in
MimicLite's exported deploy YAML instead of importing the full training stack.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable
import os
import re
import time

import numpy as np
import torch
import yaml

try:
    import onnxruntime as ort
except ImportError as exc:
    ort = None
    _ORT_IMPORT_ERROR = exc
else:
    _ORT_IMPORT_ERROR = None


MIMIC_LITE_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def _as_np(x, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"{name} expected {shape}, got flat size {arr.size}")
    return arr.reshape(shape).astype(np.float32, copy=False)


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return (q / np.clip(n, 1e-8, None)).astype(np.float32)


def quat_mul_raw_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = quat_mul_raw_wxyz(a, b)
    return quat_normalize_wxyz(out)


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize_wxyz(q)
    vq = np.concatenate([np.zeros((*v.shape[:-1], 1), dtype=np.float32), v.astype(np.float32)], axis=-1)
    return quat_mul_raw_wxyz(quat_mul_raw_wxyz(q, vq), quat_conjugate_wxyz(q))[..., 1:]


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize_wxyz(q)
    vq = np.concatenate([np.zeros((*v.shape[:-1], 1), dtype=np.float32), v.astype(np.float32)], axis=-1)
    return quat_mul_raw_wxyz(quat_mul_raw_wxyz(quat_conjugate_wxyz(q), vq), q)[..., 1:]


def matrix_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = quat_normalize_wxyz(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.stack(
        [
            ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz,
        ],
        axis=-1,
    ).reshape((*q.shape[:-1], 3, 3)).astype(np.float32)


def quat_yaw_wxyz(q: np.ndarray) -> np.ndarray:
    q = quat_normalize_wxyz(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


def yaw_quat_wxyz(q: np.ndarray) -> np.ndarray:
    yaw = quat_yaw_wxyz(q)
    half = 0.5 * yaw
    zeros = np.zeros_like(half, dtype=np.float32)
    out = np.stack([np.cos(half), zeros, zeros, np.sin(half)], axis=-1)
    return quat_normalize_wxyz(out.astype(np.float32))


def projected_yaw_quat_wxyz(q: np.ndarray, x_axis_xy_threshold: float = 0.1) -> np.ndarray:
    """Match mimic_lite.tasks.command.projected_yaw_quat for wxyz quaternions."""
    q = quat_normalize_wxyz(q)
    flat = q.reshape(-1, 4)
    basis_x = np.zeros((flat.shape[0], 3), dtype=np.float32)
    basis_z = np.zeros_like(basis_x)
    basis_x[:, 0] = 1.0
    basis_z[:, 2] = 1.0
    x_axis_w = quat_rotate_wxyz(flat, basis_x)
    z_axis_w = quat_rotate_wxyz(flat, basis_z)
    x_axis_xy = x_axis_w[:, :2]
    z_axis_xy = z_axis_w[:, :2]
    x_axis_xy_norm = np.linalg.norm(x_axis_xy, axis=-1, keepdims=True)
    z_axis_heading_xy = np.where(x_axis_w[:, 2:3] < 0.0, z_axis_xy, -z_axis_xy)
    heading_xy = np.where(x_axis_xy_norm > x_axis_xy_threshold, x_axis_xy, z_axis_heading_xy)
    yaw = np.arctan2(heading_xy[:, 1], heading_xy[:, 0])
    half = 0.5 * yaw
    out = np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=-1)
    return quat_normalize_wxyz(out.astype(np.float32)).reshape(q.shape)


def _resolve_regex_values(spec: dict | None, names: list[str], fallback: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(fallback, dtype=np.float32).copy()
    if not spec:
        return values
    matched = np.zeros((len(names),), dtype=bool)
    for pattern, value in spec.items():
        regex = re.compile(str(pattern))
        for i, joint_name in enumerate(names):
            if regex.fullmatch(joint_name):
                values[i] = float(value)
                matched[i] = True
    return values.astype(np.float32)


class MimicLitePolicyRuntime:
    """Build MimicLite ONNX inputs from HumanoidArena state and GMR refs."""

    def __init__(
        self,
        *,
        env,
        onnx_path: str | Path,
        yaml_path: str | Path,
        expected_joint_order: Iterable[str] = MIMIC_LITE_JOINT_ORDER,
        providers: list[str] | None = None,
    ):
        if ort is None:
            raise ImportError("onnxruntime is required for MimicLite inference") from _ORT_IMPORT_ERROR
        self.env = env
        self.device = env.device
        self.onnx_path = Path(onnx_path).expanduser().resolve()
        self.yaml_path = Path(yaml_path).expanduser().resolve()
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"MimicLite ONNX not found: {self.onnx_path}")
        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"MimicLite YAML not found: {self.yaml_path}")

        with self.yaml_path.open("r") as f:
            self.cfg = yaml.safe_load(f)

        self.joint_names = list(self.cfg.get("policy_joint_names") or [])
        expected = list(expected_joint_order)
        if self.joint_names != expected:
            raise ValueError(
                "MimicLite joint order mismatch with SONIC/GMR joint29 order:\n"
                f"mimic_lite={self.joint_names}\nexpected={expected}"
            )
        for key in ("joint_names_simulation",):
            names = list(self.cfg.get(key) or [])
            if names and names != expected:
                raise ValueError(f"MimicLite {key} mismatch with expected joint order")
        motion_names = list((self.cfg.get("motion") or {}).get("joint_names") or [])
        if motion_names and motion_names != expected:
            raise ValueError("MimicLite motion.joint_names mismatch with expected joint order")

        robot = self.env.scene["robot"].data
        all_joint_names = list(robot.joint_names)
        index_by_name = {name: i for i, name in enumerate(all_joint_names)}
        missing = [name for name in self.joint_names if name not in index_by_name]
        if missing:
            raise ValueError(f"MimicLite joints missing from Isaac robot: {missing}")
        self.joint_indices = torch.tensor([index_by_name[name] for name in self.joint_names], dtype=torch.long, device=self.device)
        robot_default_joint_pos = robot.default_joint_pos[0, self.joint_indices].detach().cpu().numpy().astype(np.float32)
        self.default_joint_pos = _resolve_regex_values(
            self.cfg.get("default_joint_pos"),
            self.joint_names,
            robot_default_joint_pos,
            name="default_joint_pos",
        )
        self.robot_default_joint_pos = robot_default_joint_pos
        self.action_scale = _as_np(self.cfg.get("action_scale"), (29,), name="action_scale")
        self.joint_kp = _resolve_regex_values(
            self.cfg.get("joint_kp"),
            self.joint_names,
            np.zeros((29,), dtype=np.float32),
            name="joint_kp",
        )
        self.joint_kd = _resolve_regex_values(
            self.cfg.get("joint_kd"),
            self.joint_names,
            np.zeros((29,), dtype=np.float32),
            name="joint_kd",
        )

        self.future_steps = list((self.cfg.get("motion") or {}).get("future_steps") or [-8, -4, -2, 0, 1, 2, 3, 4])
        if len(self.future_steps) != 8:
            raise ValueError(f"MimicLite expects 8 future steps, got {self.future_steps}")
        self.policy_history_steps = [0, 1, 2, 3, 4, 8, 16]
        self.prev_action_steps = 3
        self.prev_actions_newest_first = bool(int(os.environ.get("MIMIC_LITE_PREV_ACTIONS_NEWEST_FIRST", "1") or "1"))
        self.max_hist = max(self.policy_history_steps) + 1
        self.root_ang_vel_hist = deque(maxlen=self.max_hist)
        self.gravity_hist = deque(maxlen=self.max_hist)
        self.joint_pos_hist = deque(maxlen=self.max_hist)
        self.joint_vel_hist = deque(maxlen=self.max_hist)
        self.prev_action_hist = deque(maxlen=self.prev_action_steps)
        physics_dt = float(getattr(self.env, "physics_dt", 0.005) or 0.005)
        decimation = int(getattr(getattr(self.env, "cfg", None), "decimation", 4) or 4)
        self.motion_dt_s = float(os.environ.get("MIMIC_LITE_MOTION_DT_S", physics_dt * decimation) or (physics_dt * decimation))
        self.motion_tolerance_s = float(os.environ.get("MIMIC_LITE_MOTION_TOLERANCE_S", "0.04") or 0.04)
        self._motion_dt_ns = int(self.motion_dt_s * 1e9)
        self._future_steps_ns = np.asarray(self.future_steps, dtype=np.int64) * self._motion_dt_ns
        self._delay_ns = int(max(0, max(int(s) for s in self.future_steps)) * self._motion_dt_ns + self.motion_tolerance_s * 1e9)
        self._history_ns = self._delay_ns + int(abs(min(int(s) for s in self.future_steps)) * self._motion_dt_ns)
        self.ref_timestamps_ns: list[int] = []
        self.ref_pos_frames: list[np.ndarray] = []
        self.ref_quat_frames: list[np.ndarray] = []
        self.ref_joint_frames: list[np.ndarray] = []

        if providers is None:
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        input_names = [i.name for i in self.session.get_inputs()]
        if input_names != ["command", "policy"]:
            raise ValueError(f"Unexpected MimicLite ONNX inputs: {input_names}")
        self.last_action = np.zeros((29,), dtype=np.float32)
        self.last_target = self.default_joint_pos.copy()
        self.last_command = np.zeros((304,), dtype=np.float32)
        self.last_policy = np.zeros((535,), dtype=np.float32)
        self._heading_align_initialized = False
        self._root_quat_offset_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.last_debug = {}

    def reset(self):
        self.root_ang_vel_hist.clear()
        self.gravity_hist.clear()
        self.joint_pos_hist.clear()
        self.joint_vel_hist.clear()
        self.prev_action_hist.clear()
        self.ref_timestamps_ns.clear()
        self.ref_pos_frames.clear()
        self.ref_quat_frames.clear()
        self.ref_joint_frames.clear()
        self._heading_align_initialized = False
        self._root_quat_offset_wxyz[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.last_debug = {}
        self.last_action.fill(0.0)
        self.last_target = self.default_joint_pos.copy()

    def _append_current_robot_state(self):
        robot = self.env.scene["robot"].data
        root_ang_vel_src = getattr(robot, "root_com_ang_vel_b", robot.root_ang_vel_b)
        root_ang_vel = root_ang_vel_src[0].detach().cpu().numpy().astype(np.float32)
        gravity = robot.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32)
        # MimicLite upstream subtracts action_manager.offset, initialized to zero.
        joint_pos = robot.joint_pos[0, self.joint_indices].detach().cpu().numpy().astype(np.float32)
        joint_vel = robot.joint_vel[0, self.joint_indices].detach().cpu().numpy().astype(np.float32)
        for hist, value in (
            (self.root_ang_vel_hist, root_ang_vel),
            (self.gravity_hist, gravity),
            (self.joint_pos_hist, joint_pos),
            (self.joint_vel_hist, joint_vel),
        ):
            if not hist:
                for _ in range(self.max_hist):
                    hist.append(value.copy())
            else:
                hist.append(value.copy())
        if not self.prev_action_hist:
            for _ in range(self.prev_action_steps):
                self.prev_action_hist.append(np.zeros((29,), dtype=np.float32))

    @staticmethod
    def _select_history(hist: deque, steps: list[int]) -> np.ndarray:
        values = list(hist)
        out = []
        for step in steps:
            idx = max(0, len(values) - 1 - int(step))
            out.append(values[idx])
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def build_policy_input(self) -> np.ndarray:
        self._append_current_robot_state()
        policy = np.concatenate(
            [
                self._select_history(self.root_ang_vel_hist, self.policy_history_steps),
                self._select_history(self.gravity_hist, self.policy_history_steps),
                self._select_history(self.joint_pos_hist, self.policy_history_steps),
                self._select_history(self.joint_vel_hist, self.policy_history_steps),
                np.asarray(
                    list(reversed(self.prev_action_hist)) if self.prev_actions_newest_first else list(self.prev_action_hist),
                    dtype=np.float32,
                ).reshape(-1),
            ]
        ).astype(np.float32)
        if policy.shape != (535,):
            raise RuntimeError(f"MimicLite policy input shape mismatch: {policy.shape}")
        self.last_policy = policy.copy()
        return policy

    def _append_reference_state(
        self,
        ref_body_pos_w: np.ndarray,
        ref_body_quat_wxyz: np.ndarray,
        ref_joint_pos: np.ndarray,
        timestamp_ns: int | None,
    ) -> None:
        timestamp_ns = int(timestamp_ns or time.time_ns())
        if self.ref_timestamps_ns and timestamp_ns <= self.ref_timestamps_ns[-1]:
            if timestamp_ns == self.ref_timestamps_ns[-1]:
                self.ref_pos_frames[-1] = ref_body_pos_w.copy()
                self.ref_quat_frames[-1] = ref_body_quat_wxyz.copy()
                self.ref_joint_frames[-1] = ref_joint_pos.copy()
                return
            insert_idx = int(np.searchsorted(np.asarray(self.ref_timestamps_ns, dtype=np.int64), timestamp_ns, side="right"))
            self.ref_timestamps_ns.insert(insert_idx, timestamp_ns)
            self.ref_pos_frames.insert(insert_idx, ref_body_pos_w.copy())
            self.ref_quat_frames.insert(insert_idx, ref_body_quat_wxyz.copy())
            self.ref_joint_frames.insert(insert_idx, ref_joint_pos.copy())
        else:
            self.ref_timestamps_ns.append(timestamp_ns)
            self.ref_pos_frames.append(ref_body_pos_w.copy())
            self.ref_quat_frames.append(ref_body_quat_wxyz.copy())
            self.ref_joint_frames.append(ref_joint_pos.copy())

        cutoff_ns = time.time_ns() - self._history_ns
        while len(self.ref_timestamps_ns) > 1 and self.ref_timestamps_ns[1] < cutoff_ns:
            self.ref_timestamps_ns.pop(0)
            self.ref_pos_frames.pop(0)
            self.ref_quat_frames.pop(0)
            self.ref_joint_frames.pop(0)

    @staticmethod
    def _sample_linear_frames(timestamps_ns: np.ndarray, frames: list[np.ndarray], target_times_ns: np.ndarray) -> np.ndarray:
        if len(frames) == 1:
            return np.repeat(frames[0][None, ...], target_times_ns.shape[0], axis=0).astype(np.float32)
        right = np.searchsorted(timestamps_ns, np.clip(target_times_ns, timestamps_ns[0], timestamps_ns[-1]), side="right")
        right = np.clip(right, 1, timestamps_ns.shape[0] - 1)
        left = right - 1
        t0 = timestamps_ns[left]
        t1 = timestamps_ns[right]
        alpha = np.divide(
            np.clip(target_times_ns, timestamps_ns[0], timestamps_ns[-1]) - t0,
            t1 - t0,
            out=np.zeros_like(target_times_ns, dtype=np.float32),
            where=t1 > t0,
        ).astype(np.float32)
        left_values = np.stack([frames[int(i)] for i in left], axis=0)
        right_values = np.stack([frames[int(i)] for i in right], axis=0)
        out = left_values + alpha.reshape((-1,) + (1,) * (left_values.ndim - 1)) * (right_values - left_values)
        out[target_times_ns <= timestamps_ns[0]] = frames[0]
        out[target_times_ns >= timestamps_ns[-1]] = frames[-1]
        return out.astype(np.float32)

    @staticmethod
    def _quat_pow_wxyz(q: np.ndarray, power: float) -> np.ndarray:
        q = quat_normalize_wxyz(np.asarray(q, dtype=np.float32).reshape(4))
        w = float(np.clip(q[0], -1.0, 1.0))
        xyz = q[1:]
        sin_half = float(np.linalg.norm(xyz))
        if sin_half < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        half_angle = float(np.arctan2(sin_half, w))
        axis = xyz / sin_half
        new_half = half_angle * float(power)
        return quat_normalize_wxyz(np.concatenate([[np.cos(new_half)], axis * np.sin(new_half)]).astype(np.float32))

    @classmethod
    def _sample_quat_frames(cls, timestamps_ns: np.ndarray, frames: list[np.ndarray], target_times_ns: np.ndarray) -> np.ndarray:
        if len(frames) == 1:
            return quat_normalize_wxyz(np.repeat(frames[0][None, ...], target_times_ns.shape[0], axis=0))
        clamped = np.clip(target_times_ns, timestamps_ns[0], timestamps_ns[-1])
        right = np.searchsorted(timestamps_ns, clamped, side="right")
        right = np.clip(right, 1, timestamps_ns.shape[0] - 1)
        left = right - 1
        t0 = timestamps_ns[left]
        t1 = timestamps_ns[right]
        alpha = np.divide(
            clamped - t0,
            t1 - t0,
            out=np.zeros_like(target_times_ns, dtype=np.float32),
            where=t1 > t0,
        ).astype(np.float32)
        out = []
        for left_idx, right_idx, a in zip(left, right, alpha, strict=True):
            q0 = quat_normalize_wxyz(frames[int(left_idx)])
            q1 = quat_normalize_wxyz(frames[int(right_idx)])
            if float(np.dot(q0, q1)) < 0.0:
                q1 = -q1
            delta = quat_mul_wxyz(q1, quat_conjugate_wxyz(q0))
            out.append(quat_mul_wxyz(cls._quat_pow_wxyz(delta, float(a)), q0))
        out = quat_normalize_wxyz(np.asarray(out, dtype=np.float32))
        out[target_times_ns <= timestamps_ns[0]] = quat_normalize_wxyz(frames[0])
        out[target_times_ns >= timestamps_ns[-1]] = quat_normalize_wxyz(frames[-1])
        return quat_normalize_wxyz(out)

    def _sample_reference_future(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.future_steps)
        if not self.ref_timestamps_ns:
            return (
                np.zeros((n, 3), dtype=np.float32),
                np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), n, axis=0),
                np.repeat(self.default_joint_pos[None, :], n, axis=0).astype(np.float32),
            )
        timestamps_ns = np.asarray(self.ref_timestamps_ns, dtype=np.int64)
        target_base_ns = int(time.time_ns()) - self._delay_ns
        target_times_ns = target_base_ns + self._future_steps_ns
        return (
            self._sample_linear_frames(timestamps_ns, self.ref_pos_frames, target_times_ns),
            self._sample_quat_frames(timestamps_ns, self.ref_quat_frames, target_times_ns),
            self._sample_linear_frames(timestamps_ns, self.ref_joint_frames, target_times_ns),
        )

    def build_command_input(self, *, ref_joint_pos: np.ndarray, ref_body_pos_w: np.ndarray, ref_body_quat_wxyz: np.ndarray, timestamp_ns: int | None = None) -> np.ndarray:
        ref_joint_pos = _as_np(ref_joint_pos, (29,), name="ref_joint_pos")
        ref_body_pos_w = _as_np(ref_body_pos_w, (3,), name="ref_body_pos_w")
        ref_body_quat_wxyz = quat_normalize_wxyz(_as_np(ref_body_quat_wxyz, (4,), name="ref_body_quat_wxyz"))
        self._append_reference_state(ref_body_pos_w, ref_body_quat_wxyz, ref_joint_pos, timestamp_ns)

        n = len(self.future_steps)
        ref_pos_future_w, ref_quat_future_w, ref_joint_future = self._sample_reference_future()

        robot = self.env.scene["robot"].data
        # Match SONIC's root orientation source in this codebase.
        robot_root_quat_wxyz = quat_normalize_wxyz(robot.root_state_w[0, 3:7].detach().cpu().numpy().astype(np.float32))
        current_idx = self.future_steps.index(0)

        # ref_body_pos_future_local is invariant to global yaw alignment only if
        # positions and orientations are transformed together. Compute it from the
        # unaligned live reference frame, matching sim2real's segment-transform math.
        ref_anchor_pos_w_z0 = ref_pos_future_w[current_idx].copy()
        ref_anchor_pos_w_z0[2] = 0.0
        ref_anchor_yaw_wxyz = projected_yaw_quat_wxyz(ref_quat_future_w[current_idx])
        ref_root_pos_future_local = quat_rotate_inverse_wxyz(
            np.repeat(ref_anchor_yaw_wxyz[None, :], n, axis=0),
            ref_pos_future_w - ref_anchor_pos_w_z0[None, :],
        )

        if not self._heading_align_initialized:
            ref_heading = projected_yaw_quat_wxyz(ref_quat_future_w[current_idx])
            robot_heading = projected_yaw_quat_wxyz(robot_root_quat_wxyz)
            # Match sim2real.ref_root_ori_future_b exactly: align the robot root
            # into the initial reference yaw frame before computing root-relative ref orientation.
            self._root_quat_offset_wxyz = quat_mul_wxyz(ref_heading, quat_conjugate_wxyz(robot_heading))
            self._heading_align_initialized = True

        robot_root_quat_aligned_wxyz = quat_mul_wxyz(self._root_quat_offset_wxyz, robot_root_quat_wxyz)
        ref_root_quat_future_b = quat_mul_wxyz(
            np.repeat(quat_conjugate_wxyz(robot_root_quat_aligned_wxyz)[None, :], n, axis=0),
            ref_quat_future_w,
        )
        ref_root_ori_6d = matrix_from_quat_wxyz(ref_root_quat_future_b)[:, :2, :].reshape(-1)
        self.last_debug = {
            "robot_root_quat_wxyz": robot_root_quat_wxyz.copy(),
            "robot_root_quat_aligned_wxyz": robot_root_quat_aligned_wxyz.copy(),
            "ref_root_quat_current_wxyz": ref_quat_future_w[current_idx].copy(),
            "root_quat_offset_wxyz": self._root_quat_offset_wxyz.copy(),
            "ref_root_quat_current_b_wxyz": ref_root_quat_future_b[current_idx].copy(),
            "ref_root_pos_future_local": ref_root_pos_future_local.copy(),
            "robot_yaw": float(quat_yaw_wxyz(robot_root_quat_wxyz)),
            "robot_aligned_yaw": float(quat_yaw_wxyz(robot_root_quat_aligned_wxyz)),
            "ref_current_yaw": float(quat_yaw_wxyz(ref_quat_future_w[current_idx])),
            "ref_current_b_yaw": float(quat_yaw_wxyz(ref_root_quat_future_b[current_idx])),
        }
        command = np.concatenate([ref_root_pos_future_local.reshape(-1), ref_root_ori_6d, ref_joint_future.reshape(-1)]).astype(np.float32)
        if command.shape != (304,):
            raise RuntimeError(f"MimicLite command input shape mismatch: {command.shape}")
        self.last_command = command.copy()
        return command

    def step(self, *, ref_joint_pos: np.ndarray, ref_body_pos_w: np.ndarray, ref_body_quat_wxyz: np.ndarray, timestamp_ns: int | None = None) -> dict[str, np.ndarray]:
        command = self.build_command_input(ref_joint_pos=ref_joint_pos, ref_body_pos_w=ref_body_pos_w, ref_body_quat_wxyz=ref_body_quat_wxyz, timestamp_ns=timestamp_ns)
        policy = self.build_policy_input()
        action = np.asarray(self.session.run(["action"], {"command": command, "policy": policy})[0], dtype=np.float32).reshape(29)
        target = self.default_joint_pos + action * self.action_scale
        self.last_action = action.copy()
        self.last_target = target.astype(np.float32, copy=True)
        self.prev_action_hist.append(action.copy())
        return {
            "target_joint_pos": self.last_target.copy(),
            "policy_action": self.last_action.copy(),
            "command": self.last_command.copy(),
            "policy": self.last_policy.copy(),
        }
