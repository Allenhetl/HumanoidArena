#!/usr/bin/env python3
"""AnyDexRetarget Adaptive-based inspire hand retargeting module (Phase-A core).

Replaces the fixed three-point-angle ``hand_keypoints_to_a_hw`` with the
AnyDexRetarget AdaptiveOptimizerAnalytical (pinch-aware, per-finger segment
scaling) and maps its 12-joint output back to the 6-motor a_hw_6 space used by
the HumanoidArena inspire actuator pipeline (and by the DFX hardware).

Pipeline (per hand, 26x7 OpenXR -> a_hw_6):
  OpenXR 26x7  ->  MediaPipe 21x3  (drop quat, reorder)
  MediaPipe 21 ->  AdaptiveOptimizerAnalytical (12 joints, rad)
  12 joints    ->  a_hw_6 = [index_flex, middle_flex, ring_flex, pinky_flex,
                             thumb_flex, thumb_rotation]

Reference implementation: reference/AnyDexRetarget
  config: example/config/adaptive/pico4/pico4_inspire_hand.yaml
  (segment_scaling + pinch_thresholds + mediapipe_rotation tuned for inspire)

Run in the `retarget` conda env (has pinocchio + nlopt + torch):
  LD_LIBRARY_PATH=.../retarget/lib python inspire_retarget_anydex.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
# reference/ sits at the repo root (inspire-ws), i.e. parents[2] of tools/
_ANYDEX_ROOT = os.path.join(str(Path(__file__).resolve().parents[2]), "reference", "AnyDexRetarget")
_RETARGET_LIB = os.environ.get(
    "RETARGET_LIB", "/home/dreams/miniconda3/envs/retarget/lib"
)
if _RETARGET_LIB and os.path.isdir(_RETARGET_LIB):
    os.environ["LD_LIBRARY_PATH"] = (
        _RETARGET_LIB + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )
    sys.path.insert(0, _RETARGET_LIB)
if os.path.isdir(_ANYDEX_ROOT):
    sys.path.insert(0, _ANYDEX_ROOT)

# a_hw_6 order (matches HumanoidArena actuator + DFX 6-motor API)
A_HW_6_NAMES = ["index_flex", "middle_flex", "ring_flex", "pinky_flex",
                "thumb_flex", "thumb_rotation"]

# AnyDex 12-joint output order (dof_joint_names, from inspire_hand_right.urdf
# minus the fixed base joint).  Verified at runtime.
_JOINT12_NAMES = [
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint", "thumb_distal_joint",
    "index_proximal_joint", "index_intermediate_joint",
    "middle_proximal_joint", "middle_intermediate_joint",
    "ring_proximal_joint", "ring_intermediate_joint",
    "pinky_proximal_joint", "pinky_intermediate_joint",
]

# 6-motor DFX API joints (proximal only; intermediate/distal are hardware
# coupled, mirrors xr_teleoperate's inspire_hand.yml target_joint_names).
_API_JOINTS = [
    "index_proximal_joint", "middle_proximal_joint", "ring_proximal_joint",
    "pinky_proximal_joint", "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
]


class InspireRetargetAnyDex:
    """Thread-safe wrapper around AnyDexRetarget Adaptive optimizer."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        side: str = "right",
    ):
        from anydexretarget import Retargeter

        if config_path is None:
            config_path = os.path.join(
                _ANYDEX_ROOT, "example", "config", "adaptive", "pico4",
                "pico4_inspire_hand.yaml",
            )
        self.retargeter = Retargeter.from_yaml(config_path, hand_side=side)
        opt = self.retargeter.optimizer

        # Verify 12-joint order and build a_hw_6 index map
        dof_names = list(opt.robot.dof_joint_names)
        self._joint12_idx = {n: i for i, n in enumerate(dof_names)}
        # AnyDex URDF: right hand uses bare joint names, left hand uses L_
        prefix = "" if side == "right" else "L"
        api_prefixed = [f"{prefix}{n}" for n in _API_JOINTS]
        self._api_idx = [self._joint12_idx[n] for n in api_prefixed]
        # a_hw_6 order: [index, middle, ring, pinky, thumb_pitch, thumb_yaw]
        self._a_hw6_idx = [
            self._joint12_idx[f"{prefix}index_proximal_joint"],
            self._joint12_idx[f"{prefix}middle_proximal_joint"],
            self._joint12_idx[f"{prefix}ring_proximal_joint"],
            self._joint12_idx[f"{prefix}pinky_proximal_joint"],
            self._joint12_idx[f"{prefix}thumb_proximal_pitch_joint"],
            self._joint12_idx[f"{prefix}thumb_proximal_yaw_joint"],
        ]
        self.side = side

    @staticmethod
    def openxr26_to_mp21(kp26x3: np.ndarray) -> np.ndarray:
        """OpenXR 26x3 -> MediaPipe-style 21x3 (positions only).

        26 order: 0=Palm,1=Wrist,2-5=Thumb,6-10=Index,11-15=Middle,
                  16-20=Ring,21-25=Pinky
        21 order: 0=Wrist,1-4=Thumb,5-8=Index,9-12=Middle,13-16=Ring,
                  17-20=Pinky
        """
        mp = np.zeros((21, 3), dtype=np.float64)
        mp[0] = kp26x3[1]
        mp[1:5] = kp26x3[2:6]
        mp[5:9] = kp26x3[6:10]
        mp[9:13] = kp26x3[11:15]
        mp[13:17] = kp26x3[16:20]
        mp[17:21] = kp26x3[21:25]
        return mp

    def retarget_a_hw6(self, kp26x7: np.ndarray, apply_filter: bool = True) -> np.ndarray:
        """26x7 OpenXR hand pose -> a_hw_6 (rad, DFX 6-motor order)."""
        kp = np.asarray(kp26x7, dtype=np.float64).reshape(26, 7)[:, :3]
        mp21 = self.openxr26_to_mp21(kp)
        q12 = self.retargeter.retarget(mp21, apply_filter=apply_filter)
        q12 = np.asarray(q12, dtype=np.float64).reshape(-1)
        if q12.size < 12:
            raise ValueError(f"expected 12 joints, got {q12.size}")
        return q12[self._a_hw6_idx]

    def a_hw6_to_dfx(self, a_hw6: np.ndarray) -> np.ndarray:
        """Map a_hw_6 (rad) to the DFX hardware command range [0,1].

        Mirrors xr_teleoperate's robot_hand_inspire.py normalize():
          idx 0..3 (index/middle/ring/pinky flex): rad range [0, 1.7]
          idx 4    (thumb flex / bend)           : rad range [0, 0.5]
          idx 5    (thumb rotation)              : rad range [-0.1, 1.3]
        Output: [0,1] where 0 = fully closed, 1 = fully open (DFX convention).
        """
        a = np.asarray(a_hw6, dtype=np.float64).reshape(6)
        ranges = [
            (0.0, 1.7), (0.0, 1.7), (0.0, 1.7), (0.0, 1.7),
            (0.0, 0.5), (-0.1, 1.3),
        ]
        out = np.zeros(6, dtype=np.float64)
        for i, (lo, hi) in enumerate(ranges):
            out[i] = float(np.clip((hi - a[i]) / (hi - lo), 0.0, 1.0))
        return out


def _self_test() -> int:
    """Run a quick offline sanity check with synthetic + optional real data."""
    hr = InspireRetargetAnyDex(side="right")
    print("[retarget] a_hw6_idx:", hr._a_hw6_idx)

    # synthetic open -> fist
    def make(flex):
        kp = np.zeros((26, 3))
        kp[0] = [0, 0.02, 0]
        kp[1] = [0, 0, 0]
        kp[2:6] = [[0.02, 0, 0], [0.032, 0, 0], [0.041, 0, 0],
                   [0.041 + 0.009 * (1 - flex), 0, -0.03 * flex]]
        base = [[0.01, 0.015, 0], [0.012, 0.02, 0], [-0.003, 0.02, 0], [-0.012, 0.015, 0]]
        for fi, off in zip([6, 11, 16, 21], base):
            kp[fi] = kp[0] + off
            kp[fi + 1] = kp[fi] + [0.01, 0, 0]
            kp[fi + 2] = kp[fi + 1] + [0.012, 0, 0]
            kp[fi + 3] = kp[fi + 2] + [0.012, 0, 0]
            kp[fi + 4] = kp[fi + 3] + [0.012 * (1 - flex), 0, -0.04 * flex]
        out = np.zeros((26, 7)); out[:, :3] = kp; out[:, 3] = 1.0
        return out

    for name, flex in [("open", 0.0), ("fist", 1.0), ("half", 0.5)]:
        a6 = hr.retarget_a_hw6(make(flex), apply_filter=False)
        print(f"[retarget] {name}: a_hw_6={np.round(a6, 3)}")

    # Optional real data check
    rec = os.path.join(_PROJECT_ROOT, "recording_data", "HOI_pickplace_inspire",
                       "mimic_lite", "zz")
    if os.path.isdir(rec):
        import glob
        npz = sorted(glob.glob(os.path.join(rec, "*.npz")))
        for f in npz:
            d = np.load(f, allow_pickle=True)
            raw = np.asarray(d.get("inspire_raw_hand_right", np.zeros(0)))
            if raw.size == 0:
                continue
            raw = raw.reshape(-1, 26, 7)
            valid = [i for i in range(len(raw))
                     if np.linalg.norm(raw[i, 10, :3] - raw[i, 0, :3]) > 0.05]
            if not valid:
                continue
            print(f"[retarget] real data: {os.path.basename(f)} valid={len(valid)}")
            gt_all = np.asarray(d.get("inspire_a_hw_right", np.zeros(0)))
            for i in valid[::max(1, len(valid) // 5)][:5]:
                a6 = hr.retarget_a_hw6(raw[i], apply_filter=False)
                dfx = hr.a_hw6_to_dfx(a6)
                gt = gt_all[i] if gt_all.size else np.zeros(6)
                print(f"[retarget]   f{i}: a_hw_6={np.round(a6, 2)} dfx[0,1]={np.round(dfx, 2)} gt3pt={np.round(gt, 2)}")
            break
    print("[retarget] SELF-TEST DONE")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
