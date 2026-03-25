from pathlib import Path


env_cfg_text = Path(
    "tasks/g1_tasks/move_football_g1_29dof_dex3_wholebody/move_football_g1_29dof_dex3_hw_env_cfg.py"
).read_text()
scene_test_text = Path("script/test_move_football_scene_env.py").read_text()

assert "def log_foot_collision_status" in env_cfg_text
assert "env_cfg.setup_foot_collisions()" not in scene_test_text
assert "env_cfg.log_foot_collision_status()" in scene_test_text
assert '"left_ankle_pitch_link"' in env_cfg_text
assert '"right_ankle_pitch_link"' in env_cfg_text
assert '"left_ankle_roll_link"' in env_cfg_text
assert '"right_ankle_roll_link"' in env_cfg_text
assert "CONVEX_HULL" in env_cfg_text
assert "FOOT_COLLISION_CONVEX_HULL_APPROXIMATION" in env_cfg_text
assert 'f"{foot_prim_path}/collisions"' in env_cfg_text
assert "FOOT_COLLISION_EXPECTED_APPROXIMATIONS" in env_cfg_text
assert '"left_ankle_pitch_link": FOOT_COLLISION_CONVEX_HULL_APPROXIMATION' in env_cfg_text
assert '"right_ankle_pitch_link": FOOT_COLLISION_CONVEX_HULL_APPROXIMATION' in env_cfg_text
assert '"left_ankle_roll_link": FOOT_COLLISION_TARGET_APPROXIMATION' in env_cfg_text
assert '"right_ankle_roll_link": FOOT_COLLISION_TARGET_APPROXIMATION' in env_cfg_text
assert "actual_prim=" in env_cfg_text

print("Foot collision validation avoids pre-reset live collision edits.")
