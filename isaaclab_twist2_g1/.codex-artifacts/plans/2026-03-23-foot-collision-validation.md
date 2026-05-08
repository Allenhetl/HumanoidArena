# Foot Collision Validation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the football environment explicitly apply foot collision approximation settings at startup, print the validation result, and save the same output to a log file.

**Architecture:** Keep the collision-setting and readback logic in the football environment config so the behavior stays close to the task definition. Trigger that logic from the dedicated football scene test script right after environment creation and again after reset so startup logs show both the initial application and the persisted state.

**Tech Stack:** Python, Isaac Lab, USD Physics API, standard-library `logging`

---

## Chunk 1: Minimal verification and implementation

### Task 1: Add startup collision validation logging

**Files:**
- Modify: `tasks/g1_tasks/move_football_g1_29dof_dex3_wholebody/move_football_g1_29dof_dex3_hw_env_cfg.py`
- Modify: `script/test_move_football_scene_env.py`
- Verification (local-only; formal test files never go into git): `.codex-artifacts/tests/test_foot_collision_validation_hook.py`

- [ ] **Step 1: Write the failing verification artifact**

```python
from pathlib import Path

env_cfg = Path("tasks/g1_tasks/move_football_g1_29dof_dex3_wholebody/move_football_g1_29dof_dex3_hw_env_cfg.py").read_text()
scene_test = Path("script/test_move_football_scene_env.py").read_text()

assert "def log_foot_collision_status" in env_cfg
assert "env_cfg.setup_foot_collisions()" in scene_test
assert "env_cfg.log_foot_collision_status()" in scene_test
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python .codex-artifacts/tests/test_foot_collision_validation_hook.py`
Expected: `AssertionError` because the helper and startup calls do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def setup_foot_collisions(self):
    ...

def log_foot_collision_status(self):
    ...
```

```python
env = ManagerBasedRLEnv(cfg=env_cfg)
env_cfg.setup_foot_collisions()
env.reset()
env_cfg.log_foot_collision_status()
```

- [ ] **Step 4: Run it again to verify it passes**

Run: `python .codex-artifacts/tests/test_foot_collision_validation_hook.py`
Expected: exit code 0 with a success message.

- [ ] **Step 5: Run runtime verification**

Run: `python script/test_move_football_scene_env.py --headless --device cuda --num_steps 10`
Expected: stdout includes foot collision validation lines and `logs/foot_collision_validation.log` is appended with the same results.

## Chunk 2: Expand ankle coverage

### Task 2: Cover both ankle pitch and ankle roll links

**Files:**
- Modify: `tasks/g1_tasks/move_football_g1_29dof_dex3_wholebody/move_football_g1_29dof_dex3_hw_env_cfg.py`
- Verification (local-only; formal test files never go into git): `.codex-artifacts/tests/test_foot_collision_validation_hook.py`

- [ ] **Step 1: Write the failing verification artifact**

```python
assert '"left_ankle_pitch_link"' in env_cfg_text
assert '"right_ankle_pitch_link"' in env_cfg_text
assert '"left_ankle_roll_link"' in env_cfg_text
assert '"right_ankle_roll_link"' in env_cfg_text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python .codex-artifacts/tests/test_foot_collision_validation_hook.py`
Expected: `AssertionError` because only ankle roll links are covered today.

- [ ] **Step 3: Write minimal implementation**

```python
foot_links = [
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
```

- [ ] **Step 4: Run it again to verify it passes**

Run: `python .codex-artifacts/tests/test_foot_collision_validation_hook.py`
Expected: exit code 0.

- [ ] **Step 5: Run runtime verification**

Run: `python script/test_move_football_scene_env.py --headless --device cuda --enable_cameras --num_steps 10`
Expected: exit code 0 and `logs/foot_collision_validation.log` contains four `verify` lines, one for each ankle pitch/roll link on both sides.
