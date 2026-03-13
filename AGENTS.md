# Repository Guidelines

## Project Structure & Module Organization

This repository combines three related Python-first robotics stacks. `TWIST2/` contains teleoperation, training, and retargeting code; focus on `deploy_real/`, `legged_gym/`, `rsl_rl/`, `pose/`, and `tools/`. `isaaclab_twist2_g1/` contains the Isaac Lab simulation bridge, with runtime code in `action_provider/`, `tasks/`, `image_server/`, `pico_server/`, and helper scripts in `script/`. `GR00T-WholeBodyControl/` is a separate whole-body control stack with packages under `decoupled_wbc/`, `gear_sonic/`, and `gear_sonic_deploy/`. Root-level docs live in `README.md` and `code_structure.md`.

## Build, Test, and Development Commands

- `bash TWIST2/teleop.sh` starts the Redis-backed teleoperation publisher.
- `bash isaaclab_twist2_g1/run.sh` clears runtime Redis keys and launches Isaac Lab.
- `cd TWIST2 && bash train.sh <run_name> <device>` trains a TWIST2 policy.
- `cd GR00T-WholeBodyControl && make run-checks` runs `isort`, `black`, and `ruff`.
- `cd GR00T-WholeBodyControl && make format` applies the configured Python formatting.
- `cd GR00T-WholeBodyControl && pytest decoupled_wbc/tests -q` runs the main automated test suite.
- For targeted Isaac Lab smoke checks, run files such as `python isaaclab_twist2_g1/pico_server/test_import.py`.

## Coding Style & Naming Conventions

Use 4-space indentation in Python, `snake_case` for modules, files, and functions, and `CapWords` for classes. Prefer small, focused modules that match the existing package layout. `GR00T-WholeBodyControl/pyproject.toml` defines the clearest style baseline: Black-compatible formatting, isort-managed imports, and Ruff linting. Other areas do not have a shared root formatter, so match the surrounding style before refactoring. Keep machine-specific settings such as Redis hosts, headset IPs, and ffmpeg paths in shell variables or CLI flags.

## Testing Guidelines

Use `pytest` for Python changes and add tests near the edited subsystem, for example under `decoupled_wbc/tests/`, `isaaclab_twist2_g1/action_provider/`, or `isaaclab_twist2_g1/pico_server/`. No repository-wide coverage threshold is enforced, but every behavior change should include at least one regression test or a documented smoke test. For hardware-dependent teleop or simulation flows, record the exact manual command sequence used to validate the change.

## Commit & Pull Request Guidelines

The current history is sparse, so keep commit messages short and scope-first, for example `isaaclab: add replay seed` or `twist2: tune teleop smoothing`. Keep each commit focused on one subsystem. Pull requests should describe the behavior change, list validation commands, link related issues, and include screenshots or video for UI, simulator, camera, or teleoperation updates.

## Security & Configuration Tips

Do not commit generated assets, recordings, logs, model checkpoints, or machine-local paths. Double-check changes to `run.sh`, `teleop.sh`, and similar scripts for hard-coded IP addresses, Redis endpoints, or absolute paths before opening a PR.
