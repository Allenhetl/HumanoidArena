"""Unit tests for inspire_mapping (a_hw_6 <-> q_sim_12), no Isaac runtime needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from action_provider.inspire_mapping import (  # noqa: E402
    A_HW_6_NAMES,
    COUPLING,
    RIGHT,
    LEFT,
    actuator_joint_names,
    expand_a_hw_to_q_sim,
    fold_q_sim_to_a_hw,
    normalize_hw_command,
    q_sim_joint_names,
)


def test_names_order() -> None:
    assert A_HW_6_NAMES == (
        "pinky_flex",
        "ring_flex",
        "middle_flex",
        "index_flex",
        "thumb_flex",
        "thumb_rotation",
    )
    names = q_sim_joint_names(RIGHT)
    assert len(names) == 12
    assert names[0] == "R_pinky_proximal_joint"
    assert names[5] == "R_thumb_proximal_yaw_joint"
    assert actuator_joint_names(RIGHT) == [
        "R_pinky_proximal_joint",
        "R_ring_proximal_joint",
        "R_middle_proximal_joint",
        "R_index_proximal_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_proximal_yaw_joint",
    ]


def test_expand_and_fold_roundtrip() -> None:
    a = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.1], dtype=np.float64)
    q = expand_a_hw_to_q_sim(a, side=RIGHT, unit="normalized")
    assert len(q) == 12
    # Mimic: intermediate == proximal for fingers
    assert np.isclose(q["R_index_intermediate_joint"], q["R_index_proximal_joint"])
    assert np.isclose(
        q["R_thumb_intermediate_joint"],
        q["R_thumb_proximal_pitch_joint"] * COUPLING["thumb_intermediate_multiplier"],
    )
    assert np.isclose(
        q["R_thumb_distal_joint"],
        q["R_thumb_proximal_pitch_joint"] * COUPLING["thumb_distal_multiplier"],
    )

    a2 = fold_q_sim_to_a_hw(q, side=RIGHT, out_unit="normalized")
    assert np.allclose(a, a2, atol=1e-6)


def test_angle_set_normalization() -> None:
    cmd = [0, 250, 500, 750, 1000, 100]
    norm = normalize_hw_command(cmd, source="angle_set")
    assert np.allclose(norm, [0.0, 0.25, 0.5, 0.75, 1.0, 0.1])


def test_left_side_names() -> None:
    q = expand_a_hw_to_q_sim([0.5] * 6, side=LEFT, unit="normalized")
    assert "L_index_proximal_joint" in q
    assert "R_index_proximal_joint" not in q


if __name__ == "__main__":
    test_names_order()
    test_expand_and_fold_roundtrip()
    test_angle_set_normalization()
    test_left_side_names()
    print("OK")
