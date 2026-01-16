# OpenPI0.5 Integration with IsaacLab TWIST2 G1

This integration enables verification of OpenPI0.5 models in IsaacLab simulation using TWIST2 motion tracking.

## Overview

The system integrates:
1. **OpenPI0.5**: First-person vision + language → SMPL actions
2. **SMPL-X**: Forward kinematics for human motion
3. **GMR**: IK-based motion retargeting from human to robot
4. **TWIST2**: Low-level motion tracking policy
5. **IsaacLab**: Physics simulation environment

## Data Flow

```
Language Instruction + First-Person Camera
    ↓
OpenPI0.5 Inference → Action Diffs (16, 75)
    ↓
Cumulative Sum → SMPL Trajectory (16, 75)
    ↓
SMPL Forward Kinematics → Human Joint Positions/Orientations
    ↓
GMR IK Retargeting → Robot qpos (29 DOFs)
    ↓
Extract Mimic Observations (35 dims)
    ↓
TWIST2 Motion Tracker → Low-Level Actions
    ↓
Isaac Lab Execution + Video Recording
```

## File Structure

### New Files Created

```
isaaclab_twist2_g1/
├── utils/
│   ├── smpl_utils.py          # SMPL utility functions
│   ├── smpl_visualizer.py     # SMPL skeleton visualization
│   └── video_recorder.py      # Multi-view video recording
├── action_provider/
│   └── action_provider_openpi.py  # OpenPI action provider
└── sim_main_openpi.py         # Main simulation script
```

### Modified Files

```
isaaclab_twist2_g1/
└── action_provider/
    └── create_action_provider.py  # Added OpenPI entry
```

## Installation

### 1. Dependencies

Install required packages:

```bash
# OpenPI dependencies (in OpenPI environment)
cd /home/hcl4070-1/Desktop/taowen/projects/openpi
pip install -e .

# GMR dependencies
cd /home/hcl4070-1/Desktop/taowen/projects/GMR
pip install -r requirements.txt
pip install mink  # IK solver

# SMPL-X
pip install smplx

# Video processing
pip install opencv-python
pip install matplotlib

# Rotation math
pip install scipy

# ONNX runtime (for TWIST2)
pip install onnxruntime-gpu  # or onnxruntime for CPU
```

### 2. SMPL-X Models

Download SMPL-X models from https://smpl-x.is.tue.mpg.de/ and place them in a directory, e.g.:
```
/home/hcl4070-1/Desktop/taowen/projects/smplx_models/
```

### 3. Model Checkpoints

Ensure you have:
- **OpenPI checkpoint**: e.g., `/path/to/openpi/checkpoints/pi05_nymeria/33000`
- **TWIST2 model**: e.g., `/path/to/twist2/assets/ckpts/twist2_1017_20k.onnx`

## Usage

### Basic Command

```bash
cd /home/hcl4070-1/Desktop/taowen/projects/isaaclab_twist2_g1

python sim_main_openpi.py \
    --task Isaac-Wholebody-G129-Dex3 \
    --action_source openpi \
    --openpi_checkpoint /hpc2hdd/home/hchen858/taowen/projects/openpi/checkpoints/pi05_nymeria/pi05_nymeria_stages_2_1e-5/33000 \
    --language_instruction "walk forward slowly" \
    --smplx_model_path /path/to/smplx/models \
    --human_height 1.75 \
    --twist2_model_path /home/hcl4070-1/Desktop/taowen/projects/TWIST2/assets/ckpts/twist2_1017_20k.onnx \
    --video_save_dir ./videos/openpi_walk \
    --video_fps 30 \
    --enable_smpl_vis \
    --device cuda:0 \
    --enable_cameras
```

### Command Line Arguments

#### Required Arguments

- `--openpi_checkpoint`: Path to OpenPI checkpoint directory
- `--language_instruction`: Language instruction (e.g., "walk forward", "sit down")

#### OpenPI Arguments

- `--smplx_model_path`: Path to SMPL-X models (default: `/home/hcl4070-1/Desktop/taowen/projects/smplx_models`)
- `--human_height`: Human height in meters for GMR scaling (default: 1.75)

#### TWIST2 Arguments

- `--twist2_model_path`: Path to TWIST2 policy (.onnx or .pt)

#### Video Recording Arguments

- `--video_save_dir`: Directory to save videos (default: `./videos/openpi`)
- `--video_fps`: Video frame rate (default: 30)
- `--enable_smpl_vis`: Enable SMPL skeleton visualization

#### Simulation Arguments

- `--task`: Isaac Lab task name
- `--device`: Device (e.g., `cuda:0`)
- `--step_hz`: Control frequency (default: 500)
- `--enable_cameras`: Enable camera rendering
- `--enable_world_camera`: Enable third-person camera

### Example Commands

**Walk Forward:**
```bash
python sim_main_openpi.py \
    --openpi_checkpoint /path/to/checkpoint \
    --language_instruction "walk forward" \
    --video_save_dir ./videos/walk_forward
```

**Pick up Object:**
```bash
python sim_main_openpi.py \
    --openpi_checkpoint /path/to/checkpoint \
    --language_instruction "pick up the object" \
    --enable_dex3_dds \
    --video_save_dir ./videos/pick_object
```

**Custom Height:**
```bash
python sim_main_openpi.py \
    --openpi_checkpoint /path/to/checkpoint \
    --language_instruction "wave hand" \
    --human_height 1.80 \
    --video_save_dir ./videos/wave_hand
```

## Output

### Video Files

Videos are saved to `{video_save_dir}/{language_instruction}.mp4` with three views:
1. **Left**: First-person view (robot camera)
2. **Middle**: Third-person view (world camera, if enabled)
3. **Right**: SMPL skeleton visualization (if enabled)

### Console Output

During execution, you'll see:
```
[OpenPIActionProvider] Initializing OpenPI Action Provider...
[OpenPIActionProvider] OpenPI policy loaded from ...
[OpenPIActionProvider] SMPL-X model loaded
[OpenPIActionProvider] GMR retargeter initialized
[OpenPIActionProvider] TWIST2 policy loaded
[OpenPIActionProvider] Refilling action buffer...
[OpenPIActionProvider] OpenPI inference complete: (16, 75)
[OpenPIActionProvider] Action buffer refilled: 16 frames in 2.31s
```

## Troubleshooting

### Import Errors

If you get import errors for OpenPI or GMR:
```python
# Check if paths are correctly added
import sys
print(sys.path)
```

Solution: Ensure the paths are in your PYTHONPATH or add them explicitly:
```bash
export PYTHONPATH=/home/hcl4070-1/Desktop/taowen/projects/openpi:$PYTHONPATH
export PYTHONPATH=/home/hcl4070-1/Desktop/taowen/projects/GMR:$PYTHONPATH
```

### SMPL-X Model Not Found

Error: `FileNotFoundError: SMPL-X model path not found`

Solution: Download SMPL-X models and specify correct path with `--smplx_model_path`

### GPU Out of Memory

Error: `CUDA out of memory`

Solutions:
1. Use a smaller batch size (OpenPI is already batch_size=1)
2. Use CPU for some components: `--device cpu`
3. Close other GPU-intensive applications

### Video Not Saving

If video doesn't save:
1. Check `--video_save_dir` exists and is writable
2. Check console for error messages during cleanup
3. Ensure opencv-python is installed

### Slow Performance

If simulation is slow:
1. Reduce `--step_hz` (e.g., from 500 to 100)
2. Disable video recording for testing
3. Use `--enable_profiling` to identify bottlenecks

## Technical Details

### SMPL Format

- **Input**: 75 dims = 72 (joint angles) + 3 (root translation)
  - 72 dims: 24 joints × 3 (rotation vectors)
  - 3 dims: root position (x, y, z)
- **Output**: Action differences per timestep
- **Accumulation**: Uses `angle_wrap` for rotations, linear sum for translation

### GMR Retargeting

- **Input**: Human joint positions and orientations (dict format)
- **Output**: Robot qpos (29 DOFs)
- **Method**: Two-stage IK optimization with mink solver
- **Config**: `/home/hcl4070-1/Desktop/taowen/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json`

### TWIST2 Observation

- **Size**: 1402 dims
- **Structure**: current(127) + history(127×10) + future(35)
- **Current**: mimic_obs(35) + proprio(92)
- **Mimic Obs**: [xy_vel(2), z_pos(1), roll/pitch(2), yaw_vel(1), joints(29)]

## Performance Metrics

Typical performance on RTX 3090:
- **OpenPI inference**: ~2-3s for 16 frames
- **SMPL FK per frame**: ~10ms
- **GMR IK per frame**: ~50ms
- **TWIST2 inference**: ~5ms
- **Overall FPS**: Limited by OpenPI inference (refill every 16 steps)

## Citation

If you use this code, please cite:
- OpenPI
- TWIST2
- GMR
- IsaacLab

## License

Follow the licenses of the respective projects:
- OpenPI: [License]
- TWIST2: [License]
- GMR: [License]
- IsaacLab: BSD 3-Clause

## Contact

For questions or issues, please contact [your contact info].
