#!/usr/bin/env python3
"""Deprecated guard for a diagnostic-only SONIC script.

This filename used to run a synthetic fixed-reference decoder path. That path
does not use the validated SonicActionProvider input chain and can make G1
twist or fall like an invalid no-input inference test.
"""
import sys


MESSAGE = """
ERROR: test_real_scene_lab_sonic_v3.py is deprecated and intentionally blocked.

Do not use this script for real-scene-lab SONIC standing or repaired-collision
validation. It used a diagnostic synthetic fixed-reference decoder path and was
the source of a misleading failed experiment.

Use instead:
  isaaclab_twist2_g1/tools/test_real_scene_lab_sonic_provider_joint29_static_ref.py

For repaired-collision A/B tests, load the known-good reference:
  --load_reference_npz /home/lab/zikang/HumanoidArena/isaaclab_twist2_g1/analysis_outputs/real_scene_provider_static_ref_ccm1.npz

The original diagnostic file was archived as:
  isaaclab_twist2_g1/tools/deprecated/test_real_scene_lab_sonic_v3_synthetic_negative_do_not_use.py
"""


if __name__ == "__main__":
    print(MESSAGE.strip(), file=sys.stderr)
    raise SystemExit(2)
