"""
OpenPI Action Provider for IsaacLab TWIST2 G1 simulation.

Integrates OpenPI0.5 model with TWIST2 motion tracking:
1. OpenPI infers SMPL actions from vision + language
2. SMPL FK converts to human joint positions
3. GMR IK retargets to robot qpos
4. Extract mimic observations
5. TWIST2 motion tracker generates low-level actions
"""

import os
import sys
import numpy as np
import torch
from typing import Optional
from pathlib import Path
import time

# Add project roots to path
OPENPI_ROOT = "/home/hcl4070-1/Desktop/taowen/projects/openpi"
GMR_ROOT = "/home/hcl4070-1/Desktop/taowen/projects/GMR"
if OPENPI_ROOT not in sys.path:
    sys.path.insert(0, OPENPI_ROOT)
if GMR_ROOT not in sys.path:
    sys.path.insert(0, GMR_ROOT)

from action_provider.action_base import ActionProvider

# Import OpenPI
try:
    from openpi.policies import policy_config
    from openpi.training import config as train_config
    OPENPI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenPI not available: {e}")
    OPENPI_AVAILABLE = False

# Import SMPLX
try:
    import smplx
    from scipy.spatial.transform import Rotation as R
    SMPLX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SMPLX not available: {e}")
    SMPLX_AVAILABLE = False

# Import GMR
try:
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
    GMR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GMR not available: {e}")
    GMR_AVAILABLE = False

# Import TWIST2 policy loader
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Import utils
from utils.smpl_utils import (
    angle_wrap, cumsum_smpl_actions, get_default_smpl_state,
    extract_mimic_obs_whole_body, parse_smpl_state
)
from utils.video_recorder import VideoRecorder
from utils.smpl_visualizer import create_smpl_visualizer


class OpenPIActionProvider(ActionProvider):
    """
    Action provider using OpenPI0.5 + GMR + TWIST2 motion tracker.
    """

    def __init__(self, env, args_cli):
        super().__init__("OpenPIActionProvider")

        if not OPENPI_AVAILABLE:
            raise ImportError("OpenPI is required but not available")
        if not SMPLX_AVAILABLE:
            raise ImportError("SMPLX is required but not available")
        if not GMR_AVAILABLE:
            raise ImportError("GMR is required but not available")

        self.env = env
        self.args = args_cli
        self.device = args_cli.device if hasattr(args_cli, 'device') else 'cuda:0'

        # Language instruction
        self.language_instruction = args_cli.language_instruction

        print(f"[{self.name}] Initializing OpenPI Action Provider...")
        print(f"  - Language: {self.language_instruction}")
        print(f"  - Device: {self.device}")

        # Initialize components
        self._init_openpi_policy(args_cli)
        self._init_smplx_model(args_cli)
        self._init_gmr_retargeter(args_cli)
        self._init_twist2_policy(args_cli)
        self._init_video_recorder(args_cli)

        # Action buffer (stores 16 frames of mimic observations)
        self.action_buffer = []
        self.action_buffer_index = 0
        self.action_horizon = 16  # OpenPI outputs 16 frames

        # SMPL state tracking
        self.last_smpl_state = get_default_smpl_state()
        self.last_qpos_full = None  # (36,) for velocity calculation

        # TWIST2 observation buffers (following action_provider_wh_twist2.py)
        self.n_mimic_obs = 35
        self.n_obs_single = 127  # 35 + 92
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs  # 1402

        self._twist2_history = torch.zeros(
            self.history_len, self.n_obs_single,
            device=env.device, dtype=torch.float32
        )
        self._twist2_last_action = torch.zeros(1, 29, device=env.device, dtype=torch.float32)
        self._twist2_obs_buf = torch.zeros(1, self.total_obs_size, device=env.device, dtype=torch.float32)

        # Joint indices for TWIST2 (29 DOFs in MuJoCo actuator order)
        self._setup_joint_indices()

        # Decimation (sync simulation steps per control step)
        cfg = getattr(self.env, "cfg", None)
        self._twist2_decimation = int(getattr(cfg, "decimation", 4))

        print(f"[{self.name}] Initialization complete!")

    def _init_openpi_policy(self, args_cli):
        """Initialize OpenPI policy."""
        print(f"[{self.name}] Loading OpenPI policy...")

        checkpoint_dir = args_cli.openpi_checkpoint
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"OpenPI checkpoint not found: {checkpoint_dir}")

        # Load config and create policy
        config = train_config.get_config("pi05_nymeria")
        self.openpi_policy = policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            pytorch_device=self.device,
            sample_kwargs={"num_steps": 10}  # Flow matching steps
        )

        print(f"[{self.name}] OpenPI policy loaded from {checkpoint_dir}")

    def _init_smplx_model(self, args_cli):
        """Initialize SMPL-X body model."""
        print(f"[{self.name}] Loading SMPL-X model...")

        smplx_model_path = args_cli.smplx_model_path
        if not os.path.exists(smplx_model_path):
            raise FileNotFoundError(f"SMPL-X model path not found: {smplx_model_path}")

        self.smplx_model = smplx.create(
            model_path=smplx_model_path,
            model_type='smplx',
            gender='neutral',
            use_pca=False,
            batch_size=1
        )
        self.smplx_model = self.smplx_model.to(self.device)

        print(f"[{self.name}] SMPL-X model loaded")

    def _init_gmr_retargeter(self, args_cli):
        """Initialize GMR retargeting system."""
        print(f"[{self.name}] Initializing GMR retargeter...")

        human_height = args_cli.human_height if hasattr(args_cli, 'human_height') else 1.75

        # Find G1 URDF/XML file
        gmr_xml_path = os.path.join(GMR_ROOT, "general_motion_retargeting/data/unitree_g1/mjcf/scene_humanoid_smplx_T.xml")
        if not os.path.exists(gmr_xml_path):
            raise FileNotFoundError(f"GMR XML file not found: {gmr_xml_path}")

        self.gmr = GMR(
            actual_human_height=human_height,
            src_human="smplx",
            tgt_robot="unitree_g1",
            xml_file=gmr_xml_path,
            visualizable=False  # No visualization during inference
        )

        print(f"[{self.name}] GMR retargeter initialized (human height: {human_height}m)")

    def _init_twist2_policy(self, args_cli):
        """Initialize TWIST2 motion tracker policy."""
        print(f"[{self.name}] Loading TWIST2 policy...")

        twist2_model_path = args_cli.twist2_model_path
        if not os.path.exists(twist2_model_path):
            raise FileNotFoundError(f"TWIST2 model not found: {twist2_model_path}")

        ext = os.path.splitext(twist2_model_path)[1].lower()
        if ext == ".onnx":
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX runtime not available")
            self.twist2_policy = ort.InferenceSession(twist2_model_path)
            self.policy_type = "onnx"
        elif ext == ".pt":
            self.twist2_policy = torch.jit.load(twist2_model_path)
            self.twist2_policy.eval()
            self.policy_type = "jit"
        else:
            raise ValueError(f"Unsupported policy format: {ext}")

        print(f"[{self.name}] TWIST2 policy loaded ({self.policy_type})")

    def _init_video_recorder(self, args_cli):
        """Initialize video recorder."""
        if hasattr(args_cli, 'video_save_dir') and args_cli.video_save_dir:
            enable_smpl_vis = getattr(args_cli, 'enable_smpl_vis', True)
            video_fps = getattr(args_cli, 'video_fps', 30)

            # Create SMPL visualizer if needed
            smpl_visualizer = None
            if enable_smpl_vis:
                smplx_path = args_cli.smplx_model_path if hasattr(args_cli, 'smplx_model_path') else None
                smpl_visualizer = create_smpl_visualizer(
                    smplx_model_path=smplx_path,
                    resolution=(640, 480),
                    use_simple=False  # Try to use full SMPL-X rendering
                )

            self.video_recorder = VideoRecorder(
                save_dir=args_cli.video_save_dir,
                fps=video_fps,
                enable_smpl_vis=enable_smpl_vis,
                smpl_visualizer=smpl_visualizer
            )
            print(f"[{self.name}] Video recorder enabled: {args_cli.video_save_dir}")
        else:
            self.video_recorder = None

    def _setup_joint_indices(self):
        """Setup joint indices for TWIST2 (29 DOFs in MuJoCo actuator order)."""
        # TWIST2 expects 29 DOFs in specific MuJoCo order
        # Reference: action_provider_wh_twist2.py
        twist2_joint_names = [
            # Left leg (6)
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            # Right leg (6)
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            # Waist (3)
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            # Left arm (7)
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            # Right arm (7)
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]

        # Build mapping from joint names to Isaac Lab indices
        all_joint_names = self.env.scene["robot"].data.joint_names
        self.twist2_action_indices = []
        for name in twist2_joint_names:
            if name in all_joint_names:
                self.twist2_action_indices.append(all_joint_names.index(name))
            else:
                raise ValueError(f"Joint {name} not found in robot")

        # Default joint positions (from TWIST2)
        self.twist2_default_pos = torch.tensor([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg
            0.0, 0.0, 0.0,                   # waist
            0.0, 0.4, 0.0, 0.05, 0.0, 0.0, 0.0,  # left arm
            0.0, -0.4, 0.0, 0.05, 0.0, 0.0, 0.0, # right arm
        ], device=self.env.device, dtype=torch.float32).unsqueeze(0)  # [1, 29]

        # Ankle indices (for zeroing velocities in TWIST2 observation)
        self._twist2_ankle_idx = [4, 5, 10, 11]

        # Full action buffer
        self._full_action_buf = torch.zeros(
            len(all_joint_names),
            device=self.env.device,
            dtype=torch.float32
        )

        # Convert indices to tensors for efficient indexing
        self._twist2_action_idx_t = torch.tensor(
            self.twist2_action_indices,
            dtype=torch.long,
            device=self.env.device
        )

    def get_action(self, env) -> Optional[torch.Tensor]:
        """
        Get action for current step.

        Returns:
            action: (N,) torch tensor of joint positions, or None
        """
        try:
            # Check if we need to refill action buffer
            if self.action_buffer_index >= len(self.action_buffer):
                print(f"[{self.name}] Refilling action buffer (step {self.action_buffer_index})...")
                self._refill_action_buffer(env)
                self.action_buffer_index = 0

            # Get current mimic observation from buffer
            current_mimic_obs = self.action_buffer[self.action_buffer_index]
            self.action_buffer_index += 1

            # Build TWIST2 observation
            twist2_obs = self._build_twist2_observation(current_mimic_obs, env)

            # Run TWIST2 policy
            with torch.no_grad():
                if self.policy_type == "onnx":
                    obs_numpy = twist2_obs.cpu().numpy()
                    action_numpy = self.twist2_policy.run(None, {"obs": obs_numpy})[0]
                    action = torch.from_numpy(action_numpy).to(env.device, dtype=torch.float32)
                else:  # jit
                    action = self.twist2_policy(twist2_obs)

            # Ensure shape [1, 29]
            if action.dim() == 1:
                action = action.unsqueeze(0)

            # Update last action for next observation
            if action.shape[-1] == 29:
                self._twist2_last_action.copy_(action)

            # Build full action vector
            full_action = self._full_action_buf
            full_action.zero_()
            full_action.copy_(env.scene["robot"].data.default_joint_pos.squeeze(0))

            # Apply TWIST2 action (scale and add to default)
            raw_action = torch.clip(action, -10.0, 10.0)
            target_29 = raw_action * 0.5 + self.twist2_default_pos

            # Copy to full action
            full_action.index_copy_(0, self._twist2_action_idx_t, target_29.squeeze(0))

            # Execute with decimation (sync multiple sim steps)
            for _ in range(self._twist2_decimation):
                env.scene["robot"].set_joint_position_target(full_action)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

            env.sim.render()

            # Record video frame if enabled
            if self.video_recorder is not None:
                self._record_video_frame(env)

            return full_action

        except Exception as e:
            print(f"[{self.name}] Error in get_action: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _refill_action_buffer(self, env):
        """
        Refill action buffer by running OpenPI inference and processing.

        Steps:
        1. Get observation (image + state + prompt)
        2. OpenPI inference -> (16, 75) action diffs
        3. Cumsum -> SMPL trajectory (16, 75)
        4. For each frame:
           - SMPL FK -> human joint dict
           - GMR IK -> robot qpos (29,)
           - Extract mimic obs (35,)
        5. Store 16 mimic observations in buffer
        """
        print(f"[{self.name}] Running OpenPI inference...")
        start_time = time.time()

        # Step 1: Get OpenPI observation
        openpi_obs = self._get_openpi_observation(env)

        # Step 2: OpenPI inference
        result = self.openpi_policy.infer(openpi_obs)
        action_diffs = result["actions"]  # (16, 75) numpy array

        print(f"[{self.name}] OpenPI inference complete: {action_diffs.shape}")

        # Step 3: Accumulate to SMPL trajectory
        smpl_traj = cumsum_smpl_actions(action_diffs, self.last_smpl_state)  # (16, 75)

        # Update last SMPL state
        self.last_smpl_state = smpl_traj[-1].copy()

        # Step 4: Process each frame
        self.action_buffer = []
        for t in range(len(smpl_traj)):
            smpl_state = smpl_traj[t]  # (75,)

            # SMPL FK
            human_data = self._smpl_forward_kinematics(smpl_state)

            # GMR IK
            qpos_robot = self.gmr.retarget(human_data, offset_to_ground=True)  # (29,) numpy

            # Build full qpos for velocity calculation
            # qpos format: [root_pos(3), root_quat(4), dof_pos(29)]
            # For simplicity, use SMPL trans as root_pos and root_orient as root_quat
            smpl_parsed = parse_smpl_state(smpl_state)
            root_pos = smpl_parsed['trans']  # (3,)
            root_orient_rotvec = smpl_parsed['root_orient']  # (3,)
            root_quat = R.from_rotvec(root_orient_rotvec).as_quat(scalar_first=True)  # (4,) w,x,y,z

            qpos_full = np.concatenate([root_pos, root_quat, qpos_robot])  # (36,)

            # Extract mimic observation
            if self.last_qpos_full is None:
                self.last_qpos_full = qpos_full.copy()

            mimic_obs = extract_mimic_obs_whole_body(
                qpos_full, self.last_qpos_full, dt=1/30
            )  # (35,)

            self.last_qpos_full = qpos_full.copy()

            # Store in buffer
            self.action_buffer.append(mimic_obs)

            # Record SMPL state for video
            if self.video_recorder is not None and t == 0:  # Record first frame
                self.video_recorder.add_frame(smpl_state=smpl_state)

        elapsed = time.time() - start_time
        print(f"[{self.name}] Action buffer refilled: {len(self.action_buffer)} frames in {elapsed:.2f}s")

    def _get_openpi_observation(self, env):
        """
        Build observation for OpenPI inference.

        Returns:
            dict with keys:
                'image': (224, 224, 3) RGB image uint8
                'state': (72,) joint angles
                'state_transl': (3,) root translation
                'prompt': str language instruction
        """
        # Get first-person camera image
        # Assuming camera is named "front_camera" or similar
        # TODO: Make camera name configurable
        camera_names = list(env.scene.sensors.keys())
        if len(camera_names) == 0:
            raise ValueError("No cameras found in scene")

        camera = env.scene.sensors[camera_names[0]]
        image_data = camera.data.output["rgb"]  # (1, H, W, 3) tensor

        # Convert to numpy and resize to 224x224
        image = image_data[0].cpu().numpy()  # (H, W, 3)
        if image.dtype == np.float32 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Resize to 224x224 (OpenPI input size)
        import cv2
        image = cv2.resize(image, (224, 224))

        # Get current SMPL state (use last predicted state)
        state = self.last_smpl_state[:72]       # (72,) joint angles
        state_transl = self.last_smpl_state[72:75]  # (3,) root translation

        return {
            'image': image,
            'state': state,
            'state_transl': state_transl,
            'prompt': self.language_instruction
        }

    def _smpl_forward_kinematics(self, smpl_state):
        """
        Perform SMPL forward kinematics.

        Args:
            smpl_state: (75,) SMPL state array

        Returns:
            human_data: dict {joint_name: (pos(3), quat(4))}
        """
        smpl_parsed = parse_smpl_state(smpl_state)

        body_pose = smpl_parsed['body_pose']      # (63,)
        root_orient = smpl_parsed['root_orient']  # (3,)
        trans = smpl_parsed['trans']              # (3,)

        # Run SMPL-X forward kinematics
        with torch.no_grad():
            body_pose_tensor = torch.from_numpy(body_pose).float().reshape(1, 63).to(self.device)
            root_orient_tensor = torch.from_numpy(root_orient).float().reshape(1, 3).to(self.device)
            trans_tensor = torch.from_numpy(trans).float().reshape(1, 3).to(self.device)

            smplx_output = self.smplx_model(
                global_orient=root_orient_tensor,
                body_pose=body_pose_tensor,
                transl=trans_tensor,
                left_hand_pose=torch.zeros(1, 45).float().to(self.device),
                right_hand_pose=torch.zeros(1, 45).float().to(self.device),
                jaw_pose=torch.zeros(1, 3).float().to(self.device),
                leye_pose=torch.zeros(1, 3).float().to(self.device),
                reye_pose=torch.zeros(1, 3).float().to(self.device),
                return_full_pose=True
            )

            joints = smplx_output.joints[0].detach().cpu().numpy()  # (N, 3)
            full_pose = smplx_output.full_pose[0].reshape(-1, 3).detach().cpu().numpy()  # (N, 3)

        # Convert to GMR format
        # Reference: GMR/utils/smpl.py:get_smplx_data_offline_fast
        joint_names = self.smplx_model.joint_names
        parents = self.smplx_model.parents.cpu().numpy()

        human_data = {}
        joint_orientations = []

        for i in range(len(joints)):
            if i == 0:
                # Root joint
                rot = R.from_rotvec(full_pose[i])
            else:
                # Child joint: parent_rotation * local_rotation
                parent_idx = parents[i]
                rot = joint_orientations[parent_idx] * R.from_rotvec(full_pose[i])

            joint_orientations.append(rot)

            # Store as (position, quaternion_wxyz)
            joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
            human_data[joint_name] = (joints[i], rot.as_quat(scalar_first=True))

        return human_data

    def _build_twist2_observation(self, mimic_obs, env):
        """
        Build TWIST2 observation (1402 dims).

        Structure: [current_obs(127) + history(127*10) + future_mimic(35)]
        where current_obs = [mimic_obs(35) + proprio(92)]

        Args:
            mimic_obs: (35,) numpy array
            env: Isaac Lab environment

        Returns:
            obs: (1, 1402) torch tensor
        """
        # Convert mimic_obs to tensor
        mimic_obs_tensor = torch.from_numpy(mimic_obs).float().unsqueeze(0).to(env.device)  # [1, 35]

        # Build proprioception (92 dims)
        # Following action_provider_wh_twist2.py: compute_current_observations
        root_state = env.scene["robot"].data.root_state_w  # [1, 13]
        ang_vel = root_state[:, 10:13]  # [1, 3]
        quat = root_state[:, 3:7]       # [1, 4] xyzw format

        # Convert to roll/pitch
        roll, pitch = self._roll_pitch_from_quaternion(quat)

        # Get 29-DOF joint states
        joint_pos = env.scene["robot"].data.joint_pos  # [1, N]
        joint_vel = env.scene["robot"].data.joint_vel  # [1, N]

        dof_pos = joint_pos[:, self.twist2_action_indices]  # [1, 29]
        dof_vel = joint_vel[:, self.twist2_action_indices]  # [1, 29]

        # Zero ankle velocities (TWIST2 convention)
        dof_vel = dof_vel.clone()
        dof_vel[:, self._twist2_ankle_idx] = 0.0

        # Compute position delta from default
        dof_pos_delta = dof_pos - self.twist2_default_pos

        # Build proprio: [ang_vel*0.25(3), roll/pitch(2), dof_pos_delta(29), dof_vel*0.05(29), last_action(29)] = 92
        proprio = torch.cat([
            ang_vel * 0.25,
            roll, pitch,
            dof_pos_delta,
            dof_vel * 0.05,
            self._twist2_last_action
        ], dim=-1)  # [1, 92]

        # Current observation
        current_obs = torch.cat([mimic_obs_tensor, proprio], dim=-1)  # [1, 127]

        # Update history
        self._twist2_history = torch.roll(self._twist2_history, -1, dims=0)
        self._twist2_history[-1] = current_obs.squeeze(0)

        # Flatten history
        obs_hist = self._twist2_history.reshape(1, -1)  # [1, 1270]

        # Future mimic (use current as target)
        future_mimic = mimic_obs_tensor  # [1, 35]

        # Concatenate all
        obs = torch.cat([current_obs, obs_hist, future_mimic], dim=-1)  # [1, 1402]

        return obs

    def _roll_pitch_from_quaternion(self, quat):
        """
        Extract roll and pitch from quaternion.

        Args:
            quat: (N, 4) quaternion in xyzw format

        Returns:
            roll: (N, 1) roll angle
            pitch: (N, 1) pitch angle
        """
        # Convert xyzw to wxyz for computation
        qw = quat[:, 3:4]
        qx = quat[:, 0:1]
        qy = quat[:, 1:2]
        qz = quat[:, 2:3]

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        return roll, pitch

    def _record_video_frame(self, env):
        """Record current frame for video."""
        if self.video_recorder is None:
            return

        try:
            # Get camera images
            camera_names = list(env.scene.sensors.keys())
            if len(camera_names) >= 1:
                first_person_img = env.scene.sensors[camera_names[0]].data.output["rgb"][0].cpu().numpy()
            else:
                first_person_img = None

            if len(camera_names) >= 2:
                third_person_img = env.scene.sensors[camera_names[1]].data.output["rgb"][0].cpu().numpy()
            else:
                third_person_img = first_person_img

            # SMPL state is recorded during refill_action_buffer
            # So we don't add it here to avoid duplication

        except Exception as e:
            print(f"[{self.name}] Warning: Failed to record video frame: {e}")

    def cleanup(self):
        """Clean up resources."""
        print(f"[{self.name}] Cleaning up...")

        # Save video if recorder exists
        if self.video_recorder is not None:
            try:
                output_name = self.language_instruction.replace(" ", "_")
                self.video_recorder.save(name=output_name)
                self.video_recorder.close()
            except Exception as e:
                print(f"[{self.name}] Warning: Failed to save video: {e}")

        # Close SMPL visualizer
        if hasattr(self, 'smplx_model'):
            del self.smplx_model

        print(f"[{self.name}] Cleanup complete")

    def __del__(self):
        """Destructor."""
        self.cleanup()
