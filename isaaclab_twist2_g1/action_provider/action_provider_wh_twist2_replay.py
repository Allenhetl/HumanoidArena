# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""
Replay action provider for TWIST2 recorded data.
Based on action_provider_wh_twist2.py, replacing Redis data source with npz file.
"""

from action_provider.action_base import ActionProvider
from action_provider.replay_debug_logger import ReplayDebugLogger
from typing import Optional
import torch
import os
import numpy as np
import onnxruntime as ort


class ReplayActionProvider(ActionProvider):
    """Action provider for replaying recorded TWIST2 data"""

    def __init__(self, env, args_cli):
        super().__init__("ReplayActionProvider")

        # Set random seed for reproducibility
        if hasattr(args_cli, 'seed') and args_cli.seed is not None:
            import random
            import numpy as np
            seed = args_cli.seed
            print(f"[{self.name}] Setting random seed: {seed}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            # Enable deterministic mode for PyTorch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Store seed for ONNX Runtime configuration
            self.onnx_seed = seed
        else:
            self.onnx_seed = None
            print(f"[{self.name}] No seed specified, using non-deterministic mode")

        self.env = env
        self.replay_file = args_cli.replay_file
        self.replay_mode = args_cli.replay_mode  # "inference" or "direct"
        self.replay_loop = args_cli.replay_loop

        # Debug logging (enabled by default, can be disabled via args)
        self.enable_debug_log = getattr(args_cli, 'enable_replay_debug_log', True)
        if self.enable_debug_log:
            log_dir = getattr(args_cli, 'replay_debug_log_dir', './replay_debug_logs')
            log_name = os.path.splitext(os.path.basename(self.replay_file))[0]
            self.debug_logger = ReplayDebugLogger(log_dir=log_dir, log_name=log_name)
            print(f"[{self.name}] Debug logging enabled: {log_dir}/{log_name}")
        else:
            self.debug_logger = None
            print(f"[{self.name}] Debug logging disabled")

        # Validate replay file
        if not os.path.exists(self.replay_file):
            raise FileNotFoundError(f"[{self.name}] Replay file not found: {self.replay_file}")

        # Load replay data
        print(f"[{self.name}] Loading replay data from: {self.replay_file}")
        self._load_replay_data()

        # Initialize replay state
        # IMPORTANT: Frame 0 is used for initialization only, replay starts from frame 1
        self.current_frame = 1  # Start from frame 1 (frame 0 is for initialization)
        self.total_frames = len(self.replay_data_qpos)
        print(f"[{self.name}] Loaded {self.total_frames} frames")
        print(f"[{self.name}] Frame 0 will be used for initialization, replay starts from frame 1")
        print(f"[{self.name}] Replay mode: {self.replay_mode}")
        print(f"[{self.name}] Loop: {self.replay_loop}")

        # Setup joint mapping (same as original)
        self._setup_joint_mapping()

        # Initialize history buffers for inference mode (CRITICAL for correct observation!)
        # These are needed to maintain observation history like the recording script
        self.n_obs_single = 127  # Size of single observation (35 mimic + 92 proprio)
        self.history_len = 10
        self._twist2_history = torch.zeros(
            self.history_len, self.n_obs_single,
            device=self.env.device, dtype=torch.float32
        )
        self._twist2_last_action = torch.zeros(
            1, 29, device=self.env.device, dtype=torch.float32
        )
        print(f"[{self.name}] History buffers initialized: history={self._twist2_history.shape}, last_action={self._twist2_last_action.shape}")

        # Initialize ONNX model if in inference mode
        if self.replay_mode == "inference":
            self.policy_path = args_cli.model_path
            if not os.path.exists(self.policy_path):
                raise FileNotFoundError(f"[{self.name}] Policy file not found: {self.policy_path}")
            print(f"[{self.name}] Loading ONNX model: {self.policy_path}")
            self._load_onnx_model()
        else:
            self.policy = None

        # Decimation and rendering settings (same as original)
        self._twist2_decimation = getattr(env.cfg, 'decimation', 10)
        self._render_interval = 1  # Render every N policy steps
        self._render_counter = 0
        self._obs_interval = 1  # Compute observations every N policy steps
        self._obs_counter = 0

        # Action buffer
        self._full_action_buf = torch.zeros(
            self.env.scene["robot"].data.default_joint_pos.shape[1],
            device=self.env.device,
            dtype=torch.float32
        )

        # Flag to track if initial state has been set
        self._initial_state_set = False

        print(f"[{self.name}] Replay action provider initialized successfully")

    def _load_replay_data(self):
        """Load replay data from npz file"""
        data = np.load(self.replay_file, allow_pickle=True)

        # Load observation data (for inference mode)
        if 'robot_obs_buf' in data:
            self.replay_data_obs = data['robot_obs_buf']  # [N, 1432]
            print(f"[{self.name}] Loaded observation data: {self.replay_data_obs.shape}")
        else:
            self.replay_data_obs = None
            print(f"[{self.name}] Warning: No observation data found in replay file")

        # Load qpos data (for direct mode)
        # Use twist2_inference_qpos which is the ONNX output (target positions for PD controller)
        # This is: target_29 = raw_action * 0.5 + default_pos (after clip)
        if 'robot_twist2_inference_qpos' in data:
            self.replay_data_qpos = data['robot_twist2_inference_qpos']  # [N, 29]
            print(f"[{self.name}] Loaded qpos data (ONNX inference output): {self.replay_data_qpos.shape}")
        else:
            raise ValueError(f"[{self.name}] No qpos data found in replay file (expected 'robot_twist2_inference_qpos')")

        # Also load actual positions for comparison/debugging
        if 'robot_qpos_before_decimation' in data:
            self.replay_data_qpos_actual = data['robot_qpos_before_decimation']  # [N, 29]
            print(f"[{self.name}] Loaded actual qpos for debugging: {self.replay_data_qpos_actual.shape}")
        else:
            self.replay_data_qpos_actual = None

        # Load root state data (CRITICAL for correct replay!)
        if 'robot_root_position' in data and 'robot_root_orientation' in data:
            self.replay_data_root_pos = data['robot_root_position']  # [N, 3]
            self.replay_data_root_quat = data['robot_root_orientation']  # [N, 4] (w,x,y,z)
            print(f"[{self.name}] Loaded root position: {self.replay_data_root_pos.shape}")
            print(f"[{self.name}] Loaded root orientation: {self.replay_data_root_quat.shape}")
            print(f"[{self.name}] Initial root position: {self.replay_data_root_pos[0]}")
            print(f"[{self.name}] Initial root orientation: {self.replay_data_root_quat[0]}")
        else:
            print(f"[{self.name}] ⚠️ WARNING: No root state data found in replay file!")
            print(f"[{self.name}] Robot will use default spawn position/orientation")
            self.replay_data_root_pos = None
            self.replay_data_root_quat = None

        # Load velocity data (CRITICAL for physics simulation!)
        # Prefer world frame velocities if available (from new recording format)
        if 'robot_root_lin_vel_world' in data:
            self.replay_data_root_lin_vel = data['robot_root_lin_vel_world']  # [N, 3] world frame
            print(f"[{self.name}] Loaded root linear velocity (world frame): {self.replay_data_root_lin_vel.shape}")
        elif 'robot_root_lin_vel_local' in data:
            self.replay_data_root_lin_vel = data['robot_root_lin_vel_local']  # [N, 3] local frame (legacy)
            print(f"[{self.name}] Loaded root linear velocity (local frame - legacy): {self.replay_data_root_lin_vel.shape}")
            print(f"[{self.name}] ⚠️  WARNING: Using local frame velocity, may need coordinate conversion")
        else:
            print(f"[{self.name}] ⚠️ WARNING: No root linear velocity data found!")
            self.replay_data_root_lin_vel = None

        if 'robot_root_ang_vel_world' in data:
            self.replay_data_root_ang_vel = data['robot_root_ang_vel_world']  # [N, 3] world frame
            print(f"[{self.name}] Loaded root angular velocity (world frame): {self.replay_data_root_ang_vel.shape}")
        elif 'robot_root_ang_vel_local' in data:
            self.replay_data_root_ang_vel = data['robot_root_ang_vel_local']  # [N, 3] local frame (legacy)
            print(f"[{self.name}] Loaded root angular velocity (local frame - legacy): {self.replay_data_root_ang_vel.shape}")
            print(f"[{self.name}] ⚠️  WARNING: Using local frame velocity, may need coordinate conversion")
        else:
            print(f"[{self.name}] ⚠️ WARNING: No root angular velocity data found!")
            self.replay_data_root_ang_vel = None

        if 'robot_qvel_before_decimation' in data:
            self.replay_data_qvel = data['robot_qvel_before_decimation']  # [N, 29]
            print(f"[{self.name}] Loaded joint velocities: {self.replay_data_qvel.shape}")
        else:
            print(f"[{self.name}] ⚠️ WARNING: No joint velocity data found!")
            self.replay_data_qvel = None

        print(f"[{self.name}] Replay data loaded successfully")

    def _load_onnx_model(self):
        """Load ONNX model for inference mode"""
        try:
            # Create ONNX runtime session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Configure for deterministic inference if seed is set
            if self.onnx_seed is not None:
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                print(f"[{self.name}] ONNX Runtime configured for deterministic inference (seed={self.onnx_seed})")

            # Check available providers and use CUDA if available
            available_providers = ort.get_available_providers()
            print(f"[{self.name}] Available ONNX providers: {available_providers}")

            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"[{self.name}] Using CUDAExecutionProvider")
            else:
                providers = ['CPUExecutionProvider']
                print(f"[{self.name}] CUDA not available, using CPUExecutionProvider")

            self.policy = ort.InferenceSession(
                self.policy_path,
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output names
            self.input_name = self.policy.get_inputs()[0].name
            self.output_name = self.policy.get_outputs()[0].name

            print(f"[{self.name}] ONNX model loaded successfully")
            print(f"[{self.name}] Input: {self.input_name}, Output: {self.output_name}")
            print(f"[{self.name}] Active provider: {self.policy.get_providers()}")

        except Exception as e:
            raise RuntimeError(f"[{self.name}] Failed to load ONNX model: {e}")

    def _setup_joint_mapping(self):
        """Setup joint mapping for TWIST2 (29 DOF) - same as original"""
        # TWIST2 29-dof action order (MuJoCo actuator order) - copied from recording script
        self.twist2_action_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # Get all joint names and create mapping (same as recording script)
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}

        # Dynamically map joint names to Isaac Lab indices (CRITICAL FIX!)
        missing = [n for n in self.twist2_action_joint_names if n not in self.joint_to_index]
        if missing:
            raise ValueError(f"TWIST2 joints missing in Isaac asset: {missing}")
        self.twist2_action_indices = [self.joint_to_index[n] for n in self.twist2_action_joint_names]

        # Get default joint positions (keep [1, 29] shape like recording script)
        self.twist2_default_pos = self.env.scene["robot"].data.default_joint_pos[:, self.twist2_action_indices]

        print(f"[{self.name}] Joint mapping setup complete (29 DOF)")
        print(f"[{self.name}] Joint order: {self.twist2_action_joint_names[:5]}... (showing first 5)")
        print(f"[{self.name}] Isaac indices: {self.twist2_action_indices[:5]}... (showing first 5)")
        print(f"[{self.name}] Default pos shape: {self.twist2_default_pos.shape}")

    def set_initial_state_from_recording(self):
        """Set initial state from recording data (called once after env.reset())

        Uses Frame 0 data for initialization. Replay will start from Frame 1.
        This ensures state-action alignment: Frame 0 state + Frame 1 action.
        """
        if self._initial_state_set:
            print(f"[{self.name}] Initial state already set, skipping")
            return

        print(f"[{self.name}] 🔧 Setting initial state from Frame 0 (replay starts from Frame 1)...")

        # 1. Set robot root state (position, orientation, velocities)
        if self.replay_data_root_pos is not None and self.replay_data_root_quat is not None:
            robot = self.env.scene["robot"]
            root_state = robot.data.default_root_state.clone()

            # Set position and orientation
            root_state[0, 0:3] = torch.from_numpy(self.replay_data_root_pos[0]).to(self.env.device, dtype=torch.float32)
            root_state[0, 3:7] = torch.from_numpy(self.replay_data_root_quat[0]).to(self.env.device, dtype=torch.float32)

            # Set velocities (prefer world frame)
            if self.replay_data_root_lin_vel is not None:
                root_state[0, 7:10] = torch.from_numpy(self.replay_data_root_lin_vel[0]).to(self.env.device, dtype=torch.float32)
            else:
                root_state[0, 7:10] = torch.zeros(3, device=self.env.device, dtype=torch.float32)

            if self.replay_data_root_ang_vel is not None:
                root_state[0, 10:13] = torch.from_numpy(self.replay_data_root_ang_vel[0]).to(self.env.device, dtype=torch.float32)
            else:
                root_state[0, 10:13] = torch.zeros(3, device=self.env.device, dtype=torch.float32)

            robot.write_root_state_to_sim(root_state)
            print(f"[{self.name}]   ✓ Set root pos: {self.replay_data_root_pos[0]}")
            print(f"[{self.name}]   ✓ Set root quat: {self.replay_data_root_quat[0]}")
            print(f"[{self.name}]   ✓ Set root lin_vel: {self.replay_data_root_lin_vel[0] if self.replay_data_root_lin_vel is not None else 'zeros'}")
            print(f"[{self.name}]   ✓ Set root ang_vel: {self.replay_data_root_ang_vel[0] if self.replay_data_root_ang_vel is not None else 'zeros'}")
        else:
            print(f"[{self.name}]   ⚠️  WARNING: No root state data, using spawn default")

        # 2. Set joint positions and velocities
        # CRITICAL: For initialization, use ACTUAL positions (qpos_actual), not target positions (qpos)
        # qpos = PD controller target (ONNX output)
        # qpos_actual = actual joint positions after physics simulation
        # For initial state, we need the actual state, not the target!
        if self.replay_data_qpos_actual is not None:
            initial_qpos = self.replay_data_qpos_actual[0]  # Use ACTUAL positions for initialization
            initial_qpos_tensor = torch.from_numpy(initial_qpos).to(self.env.device, dtype=torch.float32).unsqueeze(0)

            # Create full joint position array
            full_initial_pos = self.env.scene["robot"].data.default_joint_pos.clone()
            full_initial_pos[0, self.twist2_action_indices] = initial_qpos_tensor

            # Get initial joint velocities
            if self.replay_data_qvel is not None:
                initial_qvel = self.replay_data_qvel[0]
                initial_qvel_tensor = torch.from_numpy(initial_qvel).to(self.env.device, dtype=torch.float32).unsqueeze(0)
                full_initial_vel = torch.zeros_like(full_initial_pos)
                full_initial_vel[0, self.twist2_action_indices] = initial_qvel_tensor
            else:
                full_initial_vel = torch.zeros_like(full_initial_pos)

            # Set joint state
            self.env.scene["robot"].write_joint_state_to_sim(
                position=full_initial_pos,
                velocity=full_initial_vel
            )
            print(f"[{self.name}]   ✓ Set joint pos (actual, 前5个): {initial_qpos[:5]}")
            print(f"[{self.name}]   ✓ Set joint vel (前5个): {initial_qvel[:5] if self.replay_data_qvel is not None else 'zeros'}")
        elif self.replay_data_qpos is not None:
            # Fallback to target positions if actual positions not available
            print(f"[{self.name}]   ⚠️  WARNING: Using target positions (qpos) for initialization - actual positions (qpos_actual) not available")
            initial_qpos = self.replay_data_qpos[0]
            initial_qpos_tensor = torch.from_numpy(initial_qpos).to(self.env.device, dtype=torch.float32).unsqueeze(0)
            full_initial_pos = self.env.scene["robot"].data.default_joint_pos.clone()
            full_initial_pos[0, self.twist2_action_indices] = initial_qpos_tensor
            full_initial_vel = torch.zeros_like(full_initial_pos)
            if self.replay_data_qvel is not None:
                initial_qvel = self.replay_data_qvel[0]
                initial_qvel_tensor = torch.from_numpy(initial_qvel).to(self.env.device, dtype=torch.float32).unsqueeze(0)
                full_initial_vel[0, self.twist2_action_indices] = initial_qvel_tensor
            self.env.scene["robot"].write_joint_state_to_sim(
                position=full_initial_pos,
                velocity=full_initial_vel
            )
            print(f"[{self.name}]   ✓ Set joint pos (target, 前5个): {initial_qpos[:5]}")
        else:
            print(f"[{self.name}]   ⚠️  WARNING: No joint position data available")

        # 3. Set football state if available
        try:
            if "object" in self.env.scene.keys():
                # Check if football data exists in recording
                data = np.load(self.replay_file, allow_pickle=True)
                if 'env_obj_football_position' in data:
                    football = self.env.scene["object"]
                    football_state = football.data.default_root_state.clone()

                    # Set position
                    football_pos = data['env_obj_football_position'][0]
                    football_state[0, 0:3] = torch.from_numpy(football_pos).to(self.env.device, dtype=torch.float32)

                    # Set velocities if available
                    if 'env_obj_football_linear_velocity' in data:
                        football_lin_vel = data['env_obj_football_linear_velocity'][0]
                        football_state[0, 7:10] = torch.from_numpy(football_lin_vel).to(self.env.device, dtype=torch.float32)
                    else:
                        football_state[0, 7:10] = torch.zeros(3, device=self.env.device, dtype=torch.float32)

                    if 'env_obj_football_angular_velocity' in data:
                        football_ang_vel = data['env_obj_football_angular_velocity'][0]
                        football_state[0, 10:13] = torch.from_numpy(football_ang_vel).to(self.env.device, dtype=torch.float32)
                    else:
                        football_state[0, 10:13] = torch.zeros(3, device=self.env.device, dtype=torch.float32)

                    football.write_root_state_to_sim(football_state)
                    print(f"[{self.name}]   ✓ Set football pos: {football_pos}")
                else:
                    print(f"[{self.name}]   ⚠️  No football data in recording")
        except Exception as e:
            print(f"[{self.name}]   ⚠️  Failed to set football state: {e}")

        # Apply all changes
        self.env.scene.write_data_to_sim()

        # Mark as set
        self._initial_state_set = True
        print(f"[{self.name}]   ✅ Initial state set from Frame 0, replay will start from Frame 1")

    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from replay data - same structure as original get_action"""
        try:
            # Set initial state on first call (after env.reset())
            if not self._initial_state_set:
                self.set_initial_state_from_recording()

            # Check if replay finished
            if self.current_frame >= self.total_frames:
                # Save debug data immediately when replay completes
                if self.debug_logger is not None:
                    print(f"[{self.name}] Replay completed, saving debug logs...")
                    self.debug_logger.close()
                    self.debug_logger = None
                    print(f"[{self.name}] ✅ Debug logs saved successfully")

                if self.replay_loop:
                    print(f"[{self.name}] Replay finished, looping back to frame 1")
                    self.current_frame = 1  # Loop back to frame 1 (frame 0 is initialization)
                    # Reset initial state flag for loop
                    self._initial_state_set = False
                else:
                    print(f"[{self.name}] Replay finished")
                    return None

            # Monitor root movement and detect potential falls
            if self.current_frame > 0 and self.current_frame % 10 == 0:
                current_root_state = self.env.scene["robot"].data.root_state_w[0]
                current_root_pos = current_root_state[:3].cpu().numpy()
                current_root_quat = current_root_state[3:7].cpu().numpy()
                current_root_lin_vel = current_root_state[7:10].cpu().numpy()
                current_root_ang_vel = current_root_state[10:13].cpu().numpy()

                print(f"[{self.name}] Frame {self.current_frame}: Root pos = {current_root_pos}")

                # Check for potential fall (z position too low or extreme tilt)
                if current_root_pos[2] < 0.3:  # Robot base below 30cm
                    print(f"[{self.name}]   ⚠️  WARNING: Low z position {current_root_pos[2]:.4f}m - possible fall!")

                # Check orientation (w component should be close to 1 for upright)
                if abs(current_root_quat[0]) < 0.7:  # w < 0.7 means >45° tilt
                    print(f"[{self.name}]   ⚠️  WARNING: Large tilt detected! quat = {current_root_quat}")

                # Check velocities
                lin_speed = np.linalg.norm(current_root_lin_vel)
                ang_speed = np.linalg.norm(current_root_ang_vel)
                if lin_speed > 2.0:  # >2 m/s
                    print(f"[{self.name}]   ⚠️  WARNING: High linear velocity {lin_speed:.4f} m/s")
                if ang_speed > 5.0:  # >5 rad/s
                    print(f"[{self.name}]   ⚠️  WARNING: High angular velocity {ang_speed:.4f} rad/s")

            full_action = self._full_action_buf
            full_action.zero_()

            # 1. Get action based on replay mode
            if self.replay_mode == "inference":
                # Inference mode: use recorded observation directly for ONNX inference
                if self.replay_data_obs is None:
                    raise ValueError(f"[{self.name}] No observation data available for inference mode")

                # Get recorded observation buffer [1432] - use it directly!
                recorded_obs_buf = self.replay_data_obs[self.current_frame]  # [1432]
                obs_tensor = torch.from_numpy(recorded_obs_buf).unsqueeze(0).float()  # [1, 1432]

                # ONNX inference with recorded observation
                ort_inputs = {self.input_name: obs_tensor.cpu().numpy()}
                ort_outputs = self.policy.run(None, ort_inputs)
                action_data = torch.from_numpy(ort_outputs[0])  # [1, 29] - raw ONNX output

                # 2. Action preparation (same as original)
                raw_action = torch.clip(action_data.to(self.env.device, dtype=torch.float32), -10.0, 10.0)
                target_29 = raw_action * 0.5 + self.twist2_default_pos  # [1,29]

                # Debug: Print action statistics
                if self.current_frame % 100 == 0:
                    print(f"[{self.name}] Frame {self.current_frame}: Inference mode")
                    print(f"  raw_action range: [{raw_action.min():.4f}, {raw_action.max():.4f}]")
                    print(f"  target_29 range: [{target_29.min():.4f}, {target_29.max():.4f}]")

            else:
                # Direct mode: use recorded ACTUAL positions (qpos_before_decimation)
                # These are the actual joint positions after physics simulation, not target positions
                # This ensures consistency with the recorded observations
                target_29 = torch.from_numpy(self.replay_data_qpos[self.current_frame]).unsqueeze(0).to(
                    self.env.device, dtype=torch.float32)  # [1, 29]

            # Progress reporting with detailed debug info
            if self.current_frame % 100 == 0:
                print(f"[{self.name}] Replay progress: {self.current_frame}/{self.total_frames} (Frame 0 used for init)")
                # Print target_29 statistics
                target_29_np = target_29.cpu().numpy().squeeze(0)
                print(f"[{self.name}]   target_29 range: [{target_29_np.min():.4f}, {target_29_np.max():.4f}]")
                print(f"[{self.name}]   target_29 mean: {target_29_np.mean():.4f}, std: {target_29_np.std():.4f}")
                # Print first 5 joint targets
                print(f"[{self.name}]   First 5 joints: {target_29_np[:5]}")
                # If in direct mode, compare with recorded qpos
                if self.replay_mode == "direct":
                    recorded_qpos = self.replay_data_qpos[self.current_frame]
                    qpos_diff = np.abs(target_29_np - recorded_qpos).max()
                    print(f"[{self.name}]   Direct mode: qpos diff = {qpos_diff:.6f} (should be ~0)")


            # 3. Fill full action buffer
            # Fill defaults first
            full_action.copy_(self.env.scene["robot"].data.default_joint_pos.squeeze(0))
            # Overwrite the 29 controlled joints
            full_action.index_copy_(0, torch.tensor(self.twist2_action_indices, device=self.env.device,
                                                    dtype=torch.long), target_29.squeeze(0))

            # Note: We don't set root state for frames > 0
            # The initial state (frame 0) is set correctly, and physics evolves naturally
            # This avoids the timing mismatch issue and provides better stability


            # 4. Physics simulation loop (same as original)
            for i in range(self._twist2_decimation):
                # Set joint targets and write to sim
                self.env.scene["robot"].set_joint_position_target(full_action)
                self.env.scene.write_data_to_sim()

                # Physics step with optional rendering
                is_last_step = (i == self._twist2_decimation - 1)
                should_render = is_last_step and (self._render_counter % self._render_interval == 0)

                if should_render:
                    self.env.sim.step(render=True)
                else:
                    self.env.sim.step(render=False)

                # Scene update
                self.env.scene.update(dt=self.env.physics_dt)

            self._render_counter += 1

            # 5. Debug logging - compare recorded vs simulated state
            if self.debug_logger is not None:
                self._log_state_comparison(self.current_frame)

            # 6. Observation computation (same as original)
            self._obs_counter += 1
            if self._obs_counter % self._obs_interval == 0:
                self.env.observation_manager.compute()

            # Advance to next frame
            self.current_frame += 1

            return full_action

        except Exception as e:
            print(f"[{self.name}] Get replay action failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_state_comparison(self, frame_idx: int):
        """记录录制数据与仿真状态的对比

        Args:
            frame_idx: 当前帧索引
        """
        if frame_idx >= self.total_frames:
            return

        robot = self.env.scene["robot"]

        # 获取仿真状态
        sim_root_state = robot.data.root_state_w[0]  # [13]: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        sim_joint_pos = robot.data.joint_pos[0, self.twist2_action_indices]  # [29]
        sim_joint_vel = robot.data.joint_vel[0, self.twist2_action_indices]  # [29]

        # 准备仿真数据字典
        simulated = {
            'root_pos': sim_root_state[:3],
            'root_quat': sim_root_state[3:7],  # (w,x,y,z)
            'root_lin_vel': sim_root_state[7:10],
            'root_ang_vel': sim_root_state[10:13],
            'joint_pos': sim_joint_pos,
            'joint_vel': sim_joint_vel,
        }

        # 准备录制数据字典
        # CRITICAL: Use actual positions (qpos_actual) for comparison, not target positions (qpos)
        recorded = {
            'root_pos': torch.from_numpy(self.replay_data_root_pos[frame_idx]).to(self.env.device),
            'root_quat': torch.from_numpy(self.replay_data_root_quat[frame_idx]).to(self.env.device),
        }

        # Use actual joint positions for comparison (not target positions)
        if self.replay_data_qpos_actual is not None:
            recorded['joint_pos'] = torch.from_numpy(self.replay_data_qpos_actual[frame_idx]).to(self.env.device)
        elif self.replay_data_qpos is not None:
            # Fallback to target positions if actual not available (will show larger errors)
            recorded['joint_pos'] = torch.from_numpy(self.replay_data_qpos[frame_idx]).to(self.env.device)
            if frame_idx == 1:  # Warn only once
                print(f"[{self.name}] ⚠️  WARNING: Using target positions (qpos) for comparison - actual positions not available")

        # 添加速度数据（如果有）
        if self.replay_data_root_lin_vel is not None:
            recorded['root_lin_vel'] = torch.from_numpy(self.replay_data_root_lin_vel[frame_idx]).to(self.env.device)
        else:
            recorded['root_lin_vel'] = torch.zeros(3, device=self.env.device)

        if self.replay_data_root_ang_vel is not None:
            recorded['root_ang_vel'] = torch.from_numpy(self.replay_data_root_ang_vel[frame_idx]).to(self.env.device)
        else:
            recorded['root_ang_vel'] = torch.zeros(3, device=self.env.device)

        if self.replay_data_qvel is not None:
            recorded['joint_vel'] = torch.from_numpy(self.replay_data_qvel[frame_idx]).to(self.env.device)
        else:
            recorded['joint_vel'] = torch.zeros(29, device=self.env.device)

        # 记录到日志
        self.debug_logger.log_frame(frame_idx, recorded, simulated)

    def cleanup(self):
        """Cleanup resources"""
        print(f"[{self.name}] Cleaning up replay action provider")

        # Close debug logger (if not already closed)
        if self.debug_logger is not None:
            print(f"[{self.name}] Saving debug logs in cleanup...")
            self.debug_logger.close()
            self.debug_logger = None

        super().cleanup()
