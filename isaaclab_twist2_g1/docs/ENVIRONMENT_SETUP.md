# HumanoidArena Environment Setup

This document describes the recommended runtime layout for deploying HumanoidArena on a new machine. The project uses two conda environments because Isaac Sim/Isaac Lab and LeRobot have different Python and dependency constraints.

## 1. System Requirements

Recommended platform:

- Ubuntu 22.04 or newer for the pip-based Isaac Sim 5.0 installation.
- NVIDIA GPU with a driver compatible with Isaac Sim 5.0 and CUDA 12.x.
- Miniconda or Anaconda.
- Git LFS for pulling repositories that contain or reference LFS files.
- The HumanoidArena asset package restored under `isaaclab_twist2_g1/assets`.
- Network access to GitHub and HuggingFace for external runtime repositories and SONIC policy artifacts.

The minimal asset package used by the current seven tasks is expected to include:

```text
isaaclab_twist2_g1/assets/objects/small_warehouse
isaaclab_twist2_g1/assets/objects/semantic
isaaclab_twist2_g1/assets/robots
```

Isaac/Omniverse materials such as `OmniPBR.mdl` are resolved by the Isaac Sim runtime. Run at least one smoke test on a clean machine to verify material and texture loading.

## 2. Environment Matrix

| Environment | Python | Main use | Source of dependencies |
| --- | --- | --- | --- |
| `unitree_sim_env` | 3.11 | Isaac Sim, Isaac Lab, live teleop, replay, rerecord, eval workers | Isaac Sim/Isaac Lab install flow plus `isaaclab_twist2_g1/requirements.txt` |
| `lerobot` | 3.12+ | LeRobot training and VLA HTTP policy server | Local `lerobot/pyproject.toml` |

The recommended current simulation stack is Isaac Sim 5.0.0 with Isaac Lab's `release/2.2.0` branch. The Isaac Sim 4.5 notes are kept for historical compatibility, but new deployments should start with the 5.0 path unless there is a specific reason to reproduce older experiments.

Reference install notes:

- `isaaclab_twist2_g1/doc/isaacsim5.0_install.md`
- `isaaclab_twist2_g1/doc/isaacsim4.5_install.md`

## 3. Optional Dry-Run Installer

The repository provides a conservative setup helper:

```bash
bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --dry-run
```

The script prints the commands required to create both conda environments and install project dependencies. It does not install anything unless `--execute` is passed explicitly:

```bash
CONDA_BASE=/path/to/miniconda3 \
bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --execute
```

Use this script as a deployment checklist first. Review the generated commands, then run with `--execute` only on a machine where creating conda environments, cloning IsaacLab, and installing Python packages is intended.

The helper installs the pip distribution of Isaac Sim 5.0.0 inside `unitree_sim_env`. It does not auto-detect or reuse a pre-existing binary/system Isaac Sim installation. If you want to validate against an externally installed Isaac Sim 5.0/5.1 build, follow NVIDIA's Isaac Sim and Isaac Lab matching-version instructions separately and use this script only as a dependency checklist.

For non-interactive smoke tests after installation, set NVIDIA's EULA acceptance environment variable explicitly:

```bash
OMNI_KIT_ACCEPT_EULA=YES python -c "import isaacsim; print('isaacsim import ok')"
```

The script also writes an Isaac Sim 5.0 constraint file under `external/humanoidarena_isaacsim5_constraints.txt` and uses it while installing IsaacLab/project packages. This prevents transitive dependencies from upgrading `torch`, `click`, `psutil`, and related packages beyond the versions required by Isaac Sim 5.0.0.

The helper also restores two repository-adjacent runtime dependencies that are intentionally not tracked by the main HumanoidArena git repository:

```text
GMR
GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
```

`GMR` is cloned from `https://github.com/YanjieZe/GMR.git`. The SONIC release files are downloaded from HuggingFace repo `nvidia/GEAR-SONIC`:

```text
model_encoder.onnx
model_decoder.onnx
observation_config.yaml
```

## 4. Install `unitree_sim_env`

Create the conda environment:

```bash
conda create -n unitree_sim_env python=3.11
conda activate unitree_sim_env
python -m pip install --upgrade pip
```

Install PyTorch for your CUDA/driver setup. IsaacLab `release/2.2.0` installs PyTorch 2.7.0 with CUDA 12.8 wheels, so use the same index unless you have a specific compatibility reason to override it:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install Isaac Sim 5.0.0:

```bash
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
isaacsim
```

Accept the EULA on the first launch.

Install Isaac Lab:

```bash
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout release/2.2.0
sudo apt install cmake build-essential cyclonedds-dev
./isaaclab.sh --install
```

If you do not have sudo access on the target machine, install CycloneDDS and basic build tools into the conda environment instead:

```bash
conda install -n unitree_sim_env -c conda-forge cyclonedds=0.10.5 cmake make pkg-config -y
```

Verify Isaac Lab:

```bash
python scripts/tutorials/00_sim/create_empty.py
# or
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

Install Unitree SDK Python bindings:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python
cd unitree_sdk2_python
CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}" pip install -e .
```

Install HumanoidArena simulation-side Python dependencies:

```bash
cd /path/to/HumanoidArena
pip install -r isaaclab_twist2_g1/requirements.txt
```

Install GMR for live teleoperation and any runtime path that reconstructs or consumes GMR retargeting data:

```bash
cd /path/to/HumanoidArena
git clone https://github.com/YanjieZe/GMR.git GMR
```

Download the SONIC policy release artifacts:

```bash
cd /path/to/HumanoidArena
mkdir -p GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
hf download nvidia/GEAR-SONIC \
  model_encoder.onnx model_decoder.onnx observation_config.yaml \
  --local-dir GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
```

Maintained SONIC scripts resolve these files through `SONIC_POLICY_ROOT`, defaulting to:

```text
/path/to/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
```

If the target machine cannot access HuggingFace directly, place the same three files in that directory by another artifact transfer method. The setup helper skips the download when all three files already exist.

Quick import check:

```bash
python - <<'PY'
import onnxruntime
import redis
import zmq
print("unitree_sim_env imports ok")
PY
```

## 5. Install `lerobot`

Create the LeRobot environment:

```bash
conda create -n lerobot python=3.12
conda activate lerobot
python -m pip install --upgrade pip
```

Install the local LeRobot package from this repository. For the current VLA/PI-style policy server, install the `pi` extra:

```bash
cd /path/to/HumanoidArena/lerobot
pip install -e ".[pi]"
```

If you only train or evaluate ACT/Diffusion policies, the base editable install may be enough:

```bash
pip install -e .
```

Quick import check:

```bash
PYTHONPATH=src python - <<'PY'
import torch
import torchvision
import lerobot
print("lerobot imports ok")
PY
```

## 6. Runtime Path Detection

Most maintained runtime scripts source:

```text
isaaclab_twist2_g1/script/common/runtime_paths.sh
```

That helper derives repository-relative paths and tries to locate conda automatically. Defaults:

```text
ISAACLAB_CONDA_ENV_NAME=unitree_sim_env
LEROBOT_CONDA_ENV_NAME=lerobot
```

It checks common conda roots such as:

```text
$CONDA_EXE
$CONDA_PREFIX
/ai/Yichi/0_Systems/miniconda3
~/miniconda3
/opt/conda
/root/miniconda3
```

If auto-detection fails, pass paths explicitly:

```bash
CONDA_BASE=/path/to/miniconda3 \
ISAACLAB_CONDA_ENV_NAME=unitree_sim_env \
LEROBOT_CONDA_ENV_NAME=lerobot \
bash isaaclab_twist2_g1/batch_1_test_v31_sonic.sh
```

You can also bypass conda discovery by providing Python executables directly:

```bash
ISAACLAB_PYTHON=/path/to/miniconda3/envs/unitree_sim_env/bin/python \
SERVER_PYTHON=/path/to/miniconda3/envs/lerobot/bin/python \
bash isaaclab_twist2_g1/batch_1_test_v31_sonic.sh
```

`ISAACLAB_PYTHON` is used for simulation/eval workers. `SERVER_PYTHON` is used for the LeRobot HTTP policy server.

## 7. Smoke Tests on a New Machine

After installing environments and restoring assets, run these checks from the repository root.

Check key asset directories:

```bash
test -d isaaclab_twist2_g1/assets/objects/small_warehouse
test -d isaaclab_twist2_g1/assets/objects/semantic
test -d isaaclab_twist2_g1/assets/robots
```

Check runtime path detection:

```bash
bash -lc 'source isaaclab_twist2_g1/script/common/runtime_paths.sh
echo "CONDA_BASE=${CONDA_BASE}"
echo "ISAACLAB_PYTHON=${ISAACLAB_PYTHON}"
echo "SERVER_PYTHON=${SERVER_PYTHON}"
echo "SONIC_POLICY_ROOT=${SONIC_POLICY_ROOT}"'
```

Use `bash` to source `runtime_paths.sh`. The helper relies on Bash `BASH_SOURCE`; sourcing it directly from `zsh` or `/bin/sh` can resolve the repository root incorrectly.

Check external runtime artifacts:

```bash
test -d GMR/general_motion_retargeting
test -f GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
test -f GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
test -f GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml
```

Run a short replay or base-test smoke before launching long evaluations. Example:

```bash
EVAL_BACKEND=sonic \
TEST_MODE=base_test \
MODEL_GLOB="*" \
MODEL_LIMIT=1 \
REPEATS_PER_SEED=1 \
SEEDS_OVERRIDE="0" \
NUM_WORKERS=1 \
MAX_STEPS=300 \
RESULTS_TAG_PREFIX=env_smoke \
RESUME_LATEST=0 \
bash isaaclab_twist2_g1/batch_1_test_v31_sonic.sh
```

Replay/rerecording smoke examples:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
HEADLESS=1 \
EXIT_WHEN_REPLAY_COMPLETE=1 \
REPLAY_FILE=/path/to/episode.npz \
bash isaaclab_twist2_g1/run_replay_sonic.sh
```

```bash
OMNI_KIT_ACCEPT_EULA=YES \
REPLAY_FILE=/path/to/episode.npz \
RECORDING_SAVE_DIR=/tmp/humanoidarena_sonic_rerecord \
bash isaaclab_twist2_g1/run_rerecord_sonic.sh
```

For long recordings, avoid short external `timeout` wrappers unless you are only checking startup. Rerecording with cameras can run much slower than real time on CPU-bound machines, and killing only the parent shell can leave `sim_main.py` running.

Expected evidence:

- The script prints resolved `ISAACLAB_ROOT`, `CONDA_BASE`, `ISAACLAB_PYTHON`, and `SERVER_PYTHON`.
- Isaac Sim starts without missing critical assets.
- The LeRobot server starts with the selected checkpoint when running inference eval.
- Result logs are written under `isaaclab_twist2_g1/script/eval_scripts/*/eval_results/`.

## 8. Notes for Maintainers

- Keep Isaac Sim/Isaac Lab dependencies isolated from LeRobot dependencies. Trying to merge them into one conda environment is fragile because the Python versions differ.
- Prefer the maintained batch entrypoints in `isaaclab_twist2_g1/` over older experiment scripts under `docs/eval_script_examples/`.
- If adding new eval scripts, source `isaaclab_twist2_g1/script/common/runtime_paths.sh` and avoid hard-coded machine paths.
