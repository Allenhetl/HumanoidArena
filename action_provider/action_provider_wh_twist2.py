# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from action_provider.action_base import ActionProvider
from typing import Optional
import torch
import os
import json
import redis
import math
import onnxruntime as ort
from dds.dds_master import dds_manager
from dds.sharedmemorymanager import SharedMemoryManager
import time
import threading
from isaaclab.utils.buffers import CircularBuffer,DelayBuffer
import ast
project_root = os.environ.get("PROJECT_ROOT")
class DDSRLActionProvider(ActionProvider):
    """Action provider based on DDS"""

    def __init__(self,env, args_cli):
        super().__init__("DDSActionProvider")
        self.enable_robot = args_cli.robot_type
        self.enable_gripper = args_cli.enable_dex1_dds
        self.enable_dex3 = args_cli.enable_dex3_dds
        self.enable_inspire = args_cli.enable_inspire_dds
        self.wh = args_cli.enable_wholebody_dds
        self.policy_path = self._resolve_policy_path(args_cli.model_path)
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"[{self.name}] Policy file not found: {self.policy_path}")
        self.env = env
        # Initialize DDS communication
        self.robot_dds = None
        self.gripper_dds = None
        self.dex3_dds = None
        self.inspire_dds = None
        self.run_command = None
        self._setup_dds()
        self._setup_joint_mapping()

        # --- TWIST2 (Redis teleop / motion tracker) quick integration ---
        self.redis_client = None
        self.redis_pipeline = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_pipeline = self.redis_client.pipeline()
        except Exception as e:
            print(f"[{self.name}] Redis init failed: {e}")

        # TWIST2 observation sizes (from server_low_level_g1_sim.py)
        self.n_mimic_obs = 35
        self.n_obs_single = 127  # 35 + 92
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs  # 1402

        # Buffers
        self._twist2_history = torch.zeros(self.history_len, self.n_obs_single, device=self.env.device, dtype=torch.float32)
        self._twist2_last_action = torch.zeros(1, 29, device=self.env.device, dtype=torch.float32)
        self._twist2_obs_buf = torch.zeros(1, self.total_obs_size, device=self.env.device, dtype=torch.float32)
        self._twist2_hand_dim = 7
        self._twist2_neck_dim = 2
        self._twist2_action_hand_left = torch.zeros(1, self._twist2_hand_dim, device=self.env.device, dtype=torch.float32)
        self._twist2_action_hand_right = torch.zeros(1, self._twist2_hand_dim, device=self.env.device, dtype=torch.float32)
        self._twist2_action_neck = torch.zeros(1, self._twist2_neck_dim, device=self.env.device, dtype=torch.float32)
        self._twist2_hand_valid = False
        self._twist2_action_timeout_ms = 300

        # TWIST2 default mimic obs (unitree_g1_with_hands), used when teleop is inactive
        default_mimic = [
            0.0, 0.0, 0.8, 0.0, 0.0, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        ]
        self._twist2_default_mimic_obs = torch.tensor(
            default_mimic, device=self.env.device, dtype=torch.float32
        ).unsqueeze(0)

        # Indices used in TWIST2
        self._twist2_ankle_idx = [4, 5, 10, 11]

        self.policy = self.load_policy(self.policy_path)

        # 预计算索引张量与复用缓冲
        device = self.env.device
        if hasattr(self, "arm_joint_mapping") and self.arm_joint_mapping:
            self._arm_target_indices = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
            self._arm_source_indices = [idx + 15 for idx in self.arm_joint_mapping.values()]
            self._arm_target_idx_t = torch.tensor(self._arm_target_indices, dtype=torch.long, device=device)
            self._arm_source_idx_t = torch.tensor(self._arm_source_indices, dtype=torch.long, device=device)
        if self.enable_gripper:
            self._gripper_target_indices = [self.joint_to_index[name] for name in self.gripper_joint_mapping.keys()]
            self._gripper_source_indices = [idx for idx in self.gripper_joint_mapping.values()]
            self._gripper_target_idx_t = torch.tensor(self._gripper_target_indices, dtype=torch.long, device=device)
            self._gripper_source_idx_t = torch.tensor(self._gripper_source_indices, dtype=torch.long, device=device)
        if self.enable_dex3:
            self._left_hand_target_indices = [self.joint_to_index[name] for name in self.left_hand_joint_mapping.keys()]
            self._left_hand_source_indices = [idx for idx in self.left_hand_joint_mapping.values()]
            self._right_hand_target_indices = [self.joint_to_index[name] for name in self.right_hand_joint_mapping.keys()]
            self._right_hand_source_indices = [idx for idx in self.right_hand_joint_mapping.values()]
            self._left_hand_target_idx_t = torch.tensor(self._left_hand_target_indices, dtype=torch.long, device=device)
            self._left_hand_source_idx_t = torch.tensor(self._left_hand_source_indices, dtype=torch.long, device=device)
            self._right_hand_target_idx_t = torch.tensor(self._right_hand_target_indices, dtype=torch.long, device=device)
            self._right_hand_source_idx_t = torch.tensor(self._right_hand_source_indices, dtype=torch.long, device=device)
        if self.enable_inspire:
            self._inspire_target_indices = [self.joint_to_index[name] for name in self.inspire_hand_joint_mapping.keys()]
            self._inspire_source_indices = [idx for idx in self.inspire_hand_joint_mapping.values()]
            self._inspire_special_target_indices = [self.joint_to_index[name] for name in self.special_joint_mapping.keys()]
            self._inspire_special_source_indices = [spec[0] for spec in self.special_joint_mapping.values()]
            self._inspire_special_scales = torch.tensor([spec[1] for spec in self.special_joint_mapping.values()], dtype=torch.float32)
            self._inspire_target_idx_t = torch.tensor(self._inspire_target_indices, dtype=torch.long, device=device)
            self._inspire_source_idx_t = torch.tensor(self._inspire_source_indices, dtype=torch.long, device=device)
            self._inspire_special_target_idx_t = torch.tensor(self._inspire_special_target_indices, dtype=torch.long, device=device)
            self._inspire_special_source_idx_t = torch.tensor(self._inspire_special_source_indices, dtype=torch.long, device=device)
            self._inspire_special_scales_t = self._inspire_special_scales.to(device)
        if hasattr(self, "twist2_action_indices"):
            self._twist2_action_idx_t = torch.tensor(self.twist2_action_indices, dtype=torch.long, device=device)

        self._full_action_buf = torch.zeros(len(self.all_joint_names), device=device, dtype=torch.float32)
        self._positions_buf = torch.empty(29, device=device, dtype=torch.float32)
        if self.enable_gripper:
            self._gripper_buf = torch.empty(2, device=device, dtype=torch.float32)
        if self.enable_dex3:
            self._left_hand_buf = torch.empty(len(self._left_hand_source_indices), device=device, dtype=torch.float32)
            self._right_hand_buf = torch.empty(len(self._right_hand_source_indices), device=device, dtype=torch.float32)
        if self.enable_inspire:
            self._inspire_buf = torch.empty(12, device=device, dtype=torch.float32)

    def _setup_dds(self):
        """Setup DDS communication"""
        print(f"enable_robot: {self.enable_robot}")
        print(f"enable_gripper: {self.enable_gripper}")
        print(f"enable_dex3: {self.enable_dex3}")
        try:
            if self.enable_robot == "g129":
                self.robot_dds = dds_manager.get_object("g129")
            if self.enable_gripper:
                self.gripper_dds = dds_manager.get_object("dex1")
            elif self.enable_dex3:
                self.dex3_dds = dds_manager.get_object("dex3")
            elif self.enable_inspire:
                self.inspire_dds = dds_manager.get_object("inspire")
            if self.wh:
                self.run_command_dds = dds_manager.get_object("run_command")
            print(f"[{self.name}] DDS communication initialized")
        except Exception as e:
            print(f"[{self.name}] DDS initialization failed: {e}")

    def _setup_joint_mapping(self):
        """Setup joint mapping"""
        if self.wh:
            self.action_joint_names = [
            'left_hip_pitch_joint',
            'right_hip_pitch_joint',
            'left_hip_roll_joint',
            'right_hip_roll_joint',
            'left_hip_yaw_joint',
            'right_hip_yaw_joint',
            'left_knee_joint',
            'right_knee_joint',
            'left_ankle_pitch_joint',
            'right_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_ankle_roll_joint'
            ]
            self.waist_joint_mapping = [
                'waist_yaw_joint',
                'waist_roll_joint',
                'waist_pitch_joint',
            ]
            self.arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # right arm (7)
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            ]

            # TWIST2 29-dof action order (MuJoCo actuator order)
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

            self.old_action_joints_names = [
            'left_hip_pitch_joint',
            'right_hip_pitch_joint',
            'waist_yaw_joint',
            'left_hip_roll_joint',
            'right_hip_roll_joint',
            'waist_roll_joint',
            'left_hip_yaw_joint',
            'right_hip_yaw_joint',
            'waist_pitch_joint',
            'left_knee_joint',
            'right_knee_joint',
            'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint',
            'left_ankle_pitch_joint',
            'right_ankle_pitch_joint',
            'left_shoulder_roll_joint',
            'right_shoulder_roll_joint',
            'left_ankle_roll_joint',
            'right_ankle_roll_joint',
            'left_shoulder_yaw_joint',
            'right_shoulder_yaw_joint',
            'left_elbow_joint',
            'right_elbow_joint',
            'left_wrist_roll_joint',
            'right_wrist_roll_joint',
            'left_wrist_pitch_joint',
            'right_wrist_pitch_joint',
            'left_wrist_yaw_joint',
            'right_wrist_yaw_joint',]
        if self.enable_robot == "g129":
            self.arm_joint_mapping = {
                "left_shoulder_pitch_joint": 0,
                "left_shoulder_roll_joint": 1,
                "left_shoulder_yaw_joint": 2,
                "left_elbow_joint": 3,
                "left_wrist_roll_joint": 4,
                "left_wrist_pitch_joint": 5,
                "left_wrist_yaw_joint": 6,
                "right_shoulder_pitch_joint": 7,
                "right_shoulder_roll_joint": 8,
                "right_shoulder_yaw_joint": 9,
                "right_elbow_joint": 10,
                "right_wrist_roll_joint": 11,
                "right_wrist_pitch_joint": 12,
                "right_wrist_yaw_joint": 13
            }
        if self.enable_gripper:
            self.gripper_joint_mapping = {
                "left_hand_Joint1_1": 1,
                "left_hand_Joint2_1": 1,
                "right_hand_Joint1_1": 0,
                "right_hand_Joint2_1": 0,
            }
        if self.enable_dex3:
            self.left_hand_joint_mapping = {
                "left_hand_thumb_0_joint":0,
                "left_hand_thumb_1_joint":1,
                "left_hand_thumb_2_joint":2,
                "left_hand_middle_0_joint":3,
                "left_hand_middle_1_joint":4,
                "left_hand_index_0_joint":5,
                "left_hand_index_1_joint":6}
            self.right_hand_joint_mapping = {
                "right_hand_thumb_0_joint":0,
                "right_hand_thumb_1_joint":1,
                "right_hand_thumb_2_joint":2,
                "right_hand_middle_0_joint":3,
                "right_hand_middle_1_joint":4,
                "right_hand_index_0_joint":5,
                "right_hand_index_1_joint":6}
        if self.enable_inspire:
            self.inspire_hand_joint_mapping = {
                "R_pinky_proximal_joint":0,
                "R_ring_proximal_joint":1,
                "R_middle_proximal_joint":2,
                "R_index_proximal_joint":3,
                "R_thumb_proximal_pitch_joint":4,
                "R_thumb_proximal_yaw_joint":5,
                "L_pinky_proximal_joint":6,
                "L_ring_proximal_joint":7,
                "L_middle_proximal_joint":8,
                "L_index_proximal_joint":9,
                "L_thumb_proximal_pitch_joint":10,
                "L_thumb_proximal_yaw_joint":11,
            }
            self.special_joint_mapping = {
                "L_index_intermediate_joint":[9,1],
                "L_middle_intermediate_joint":[8,1],
                "L_pinky_intermediate_joint":[6,1],
                "L_ring_intermediate_joint":[7,1],
                "L_thumb_intermediate_joint":[10,1.5],
                "L_thumb_distal_joint":[10,2.4],

                "R_index_intermediate_joint":[3,1],
                "R_middle_intermediate_joint":[2,1],
                "R_pinky_intermediate_joint":[0,1],
                "R_ring_intermediate_joint":[1,1],
                "R_thumb_intermediate_joint":[4,1.5],
                "R_thumb_distal_joint":[4,2.4],
            }
        self.all_joint_names = self.env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}
        # Precompute Isaac indices for TWIST2 29-dof order
        if hasattr(self, "twist2_action_joint_names"):
            missing = [n for n in self.twist2_action_joint_names if n not in self.joint_to_index]
            if missing:
                raise ValueError(f"TWIST2 joints missing in Isaac asset: {missing}")
            self.twist2_action_indices = [self.joint_to_index[n] for n in self.twist2_action_joint_names]
            self.twist2_default_pos = self.env.scene["robot"].data.default_joint_pos[:, self.twist2_action_indices]
        self.arm_action_pose = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
        self.arm_action_pose_indices = [self.arm_joint_mapping[name] for name in self.arm_joint_mapping.keys()]
        self.action_to_indices=[]
        for action_joint in self.action_joint_names:
            if action_joint in self.all_joint_names:
                self.action_to_indices.append(self.all_joint_names.index(action_joint))
            else:
                raise ValueError(f"action joint '{action_joint}' not in all joint list")
        self.waist_to_all_indices = []
        for waist_joint in self.waist_joint_mapping:
            if waist_joint in self.all_joint_names:
                self.waist_to_all_indices.append(self.all_joint_names.index(waist_joint))
            else:
                raise ValueError(f"waist joint '{waist_joint}' not in all joint list")

        self.arm_to_all_indices=[]
        for arm_joint in self.arm_joint_names:
            if arm_joint in self.all_joint_names:
                self.arm_to_all_indices.append(self.all_joint_names.index(arm_joint))
            else:
                raise ValueError(f"arm joint '{arm_joint}' not in all joint list")
        self.default_waist_positions = self.env.scene["robot"].data.default_joint_pos[:, self.waist_to_all_indices]
        self.default_action_positions = self.env.scene["robot"].data.default_joint_pos
        self.default_action_velocities = self.env.scene["robot"].data.default_joint_vel
        self.all_obs_indices = self.action_to_indices + self.arm_to_all_indices
        self.old_action_indices = []
        for old_action_joint in self.old_action_joints_names:
            if old_action_joint in self.all_joint_names:
                self.old_action_indices.append(self.all_joint_names.index(old_action_joint))
            else:
                raise ValueError(f"action joint '{old_action_joint}' not in all joint list")
        self.arm_action = []
        self.obs_scales = {"ang_vel":1.0, "projected_gravity":1.0, "commands":1.0,
                           "joint_pos":1.0, "joint_vel":1.0, "actions":1.0}
        self.ang_vel = self.env.scene["robot"].data.root_ang_vel_b
        self.projected_gravity = self.env.scene["robot"].data.projected_gravity_b
        self.joint_pos = self.env.scene["robot"].data.joint_pos
        self.joint_vel = self.env.scene["robot"].data.joint_vel
        self.actor_obs_buffer = CircularBuffer(
            max_len=10, batch_size=1, device=self.env.device
        )
        self.num_envs =1
        self.clip_obs = 100
        self.num_actions_all = self.env.scene["robot"].data.default_joint_pos[:,self.old_action_indices].shape[1]
        self.action_buffer = DelayBuffer(
            5, self.num_envs, device=self.env.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions_all, dtype=torch.float, device=self.env.device, requires_grad=False)
        )
        self.clip_actions = 100
        self.action_scale = 0.25
        self.sim_step_counter = 0
        cfg = getattr(self.env, "cfg", None)
        self._twist2_decimation = int(getattr(cfg, "decimation", 4))

    def _resolve_policy_path(self, model_path: str) -> str:
        if os.path.isabs(model_path):
            return model_path
        if project_root:
            return os.path.join(project_root, model_path)
        return model_path

    def load_policy(self,path):
        ext = os.path.splitext(path)[1].lower()
        if ext==".onnx":
            return self.load_onnx_policy(path)
        elif ext==".pt":
            return self.load_jit_pt_policy(path)

    def load_jit_pt_policy(self,path):
        return torch.jit.load(path)

    def load_onnx_policy(self, path):
        available = []
        try:
            available = ort.get_available_providers()
        except Exception:
            available = []

        providers = []
        if str(self.env.device).startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        model = ort.InferenceSession(path, providers=providers)
        input_name = model.get_inputs()[0].name

        def run_inference(input_tensor: torch.Tensor):
            # Keep it simple: ORT expects numpy
            ort_inputs = {input_name: input_tensor.detach().cpu().numpy()}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device=self.env.device, dtype=torch.float32)

        print(f"[{self.name}] ONNX policy loaded with providers: {model.get_providers()}")
        return run_inference

    def _twist2_parse_list(self, value, expected_len: int) -> list:
        if value is None:
            return [0.0] * expected_len
        try:
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")
            data = json.loads(value)
        except Exception:
            return [0.0] * expected_len
        if not isinstance(data, list):
            return [0.0] * expected_len
        if len(data) < expected_len:
            data = data + [0.0] * (expected_len - len(data))
        elif len(data) > expected_len:
            data = data[:expected_len]
        return data

    def _twist2_fetch_actions(self) -> torch.Tensor:
        """Fetch TWIST2 actions (body + hand + neck) from Redis."""
        if self.redis_pipeline is None:
            self._twist2_hand_valid = False
            return self._twist2_default_mimic_obs.clone()
        try:
            keys = [
                "action_body_unitree_g1_with_hands",
                "action_hand_left_unitree_g1_with_hands",
                "action_hand_right_unitree_g1_with_hands",
                "action_neck_unitree_g1_with_hands",
                "t_action",
            ]
            for key in keys:
                self.redis_pipeline.get(key)
            res = self.redis_pipeline.execute()
            action_body_raw = res[0] if len(res) > 0 else None
            action_left_raw = res[1] if len(res) > 1 else None
            action_right_raw = res[2] if len(res) > 2 else None
            action_neck_raw = res[3] if len(res) > 3 else None
            t_action_raw = res[4] if len(res) > 4 else None

            action_body = self._twist2_parse_list(action_body_raw, self.n_mimic_obs)
            action_left = self._twist2_parse_list(action_left_raw, self._twist2_hand_dim)
            action_right = self._twist2_parse_list(action_right_raw, self._twist2_hand_dim)
            action_neck = self._twist2_parse_list(action_neck_raw, self._twist2_neck_dim)

            t_action = None
            if t_action_raw is not None:
                try:
                    if isinstance(t_action_raw, (bytes, bytearray)):
                        t_action_raw = t_action_raw.decode("utf-8")
                    t_action = int(t_action_raw)
                except Exception:
                    t_action = None
            now_ms = int(time.time() * 1000)
            is_fresh = t_action is not None and (now_ms - t_action) <= self._twist2_action_timeout_ms

            if not is_fresh:
                action_body = self._twist2_default_mimic_obs.squeeze(0).tolist()
            elif not any(abs(v) > 1e-6 for v in action_body):
                action_body = self._twist2_default_mimic_obs.squeeze(0).tolist()

            self._twist2_hand_valid = is_fresh and action_left_raw is not None and action_right_raw is not None
            self._twist2_action_hand_left.copy_(torch.tensor(action_left, device=self.env.device, dtype=torch.float32).unsqueeze(0))
            self._twist2_action_hand_right.copy_(torch.tensor(action_right, device=self.env.device, dtype=torch.float32).unsqueeze(0))
            self._twist2_action_neck.copy_(torch.tensor(action_neck, device=self.env.device, dtype=torch.float32).unsqueeze(0))
            return torch.tensor(action_body, device=self.env.device, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"[{self.name}] Redis action fetch failed: {e}")
            self._twist2_hand_valid = False
            return self._twist2_default_mimic_obs.clone()

    def _twist2_publish_state(self, state_body, state_hand_left, state_hand_right, state_neck) -> None:
        if self.redis_pipeline is None:
            return
        try:
            self.redis_pipeline.set("state_body_unitree_g1_with_hands", json.dumps(state_body))
            self.redis_pipeline.set("state_hand_left_unitree_g1_with_hands", json.dumps(state_hand_left))
            self.redis_pipeline.set("state_hand_right_unitree_g1_with_hands", json.dumps(state_hand_right))
            self.redis_pipeline.set("state_neck_unitree_g1_with_hands", json.dumps(state_neck))
            self.redis_pipeline.set("t_state", int(time.time() * 1000))
            self.redis_pipeline.execute()
        except Exception as e:
            print(f"[{self.name}] Redis state publish failed: {e}")

    def _twist2_roll_pitch_from_projected_gravity(self, g_b: torch.Tensor) -> torch.Tensor:
        """Approximate (roll, pitch) from projected gravity in body frame. g_b: [N,3]."""
        gx, gy, gz = g_b[:, 0], g_b[:, 1], g_b[:, 2]
        roll = torch.atan2(gy, gz)
        pitch = torch.atan2(-gx, torch.sqrt(gy * gy + gz * gz + 1e-8))
        return torch.stack([roll, pitch], dim=-1)

    def _twist2_roll_pitch_from_quaternion(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute (roll, pitch) from quaternion (w, x, y, z)."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)
        return torch.stack([roll, pitch], dim=-1)

    def compute_current_observations(self):
        # Proprio from Isaac
        root_state = self.env.scene["robot"].data.root_state_w
        self.ang_vel = root_state[:, 10:13]  # [1,3]
        quat = root_state[:, 3:7]
        self.joint_pos = self.env.scene["robot"].data.joint_pos
        self.joint_vel = self.env.scene["robot"].data.joint_vel

        # roll/pitch from quaternion (TWIST2 convention)
        rp = self._twist2_roll_pitch_from_quaternion(quat)

        # TWIST2 29-dof vectors in the correct order
        idx = self.twist2_action_indices
        dof_pos = self.joint_pos[:, idx]
        dof_vel = self.joint_vel[:, idx]
        dof_pos_delta = dof_pos - self.twist2_default_pos

        # Publish state to Redis for teleop/data record
        state_body = torch.cat([self.ang_vel, rp, dof_pos], dim=-1)
        if self.enable_dex3 and hasattr(self, "_left_hand_target_idx_t"):
            left_state = self.joint_pos.index_select(1, self._left_hand_target_idx_t)
            right_state = self.joint_pos.index_select(1, self._right_hand_target_idx_t)
            state_hand_left = left_state.squeeze(0).tolist()
            state_hand_right = right_state.squeeze(0).tolist()
        else:
            state_hand_left = [0.0] * self._twist2_hand_dim
            state_hand_right = [0.0] * self._twist2_hand_dim
        state_neck = [0.0] * self._twist2_neck_dim
        self._twist2_publish_state(state_body.squeeze(0).tolist(), state_hand_left, state_hand_right, state_neck)

        # TWIST2 mimic from Redis (and hand actions)
        action_mimic = self._twist2_fetch_actions()  # [1,35]

        # zero ankle velocities (TWIST2 convention)
        if len(self._twist2_ankle_idx) > 0:
            dof_vel = dof_vel.clone()
            dof_vel[:, self._twist2_ankle_idx] = 0.0

        obs_proprio = torch.cat(
            [
                self.ang_vel * 0.25,
                rp,
                dof_pos_delta,
                dof_vel * 0.05,
                self._twist2_last_action,
            ],
            dim=-1,
        )  # [1,92]

        obs_full = torch.cat([action_mimic, obs_proprio], dim=-1)  # [1,127]

        # History: flatten previous frames (127*10)
        obs_hist = self._twist2_history.reshape(1, -1)

        # Future: current mimic (35)
        future_obs = action_mimic

        obs_buf = torch.cat([obs_full, obs_hist, future_obs], dim=-1)  # [1,1402]
        # Update history AFTER forming obs_buf (match server semantics)
        self._twist2_history = torch.roll(self._twist2_history, shifts=-1, dims=0)
        self._twist2_history[-1].copy_(obs_full.squeeze(0))

        return obs_buf

    def compute_observations(self):
        obs = self.compute_current_observations()
        obs = torch.clip(obs, -self.clip_obs, self.clip_obs)
        return obs

    def run_policy(self):
        obs = self.compute_observations()
        with torch.no_grad():
            action = self.policy(obs)
        # Ensure shape [1,29]
        if isinstance(action, torch.Tensor) and action.dim() == 1:
            action = action.unsqueeze(0)
        # Update last_action (TWIST2)
        if isinstance(action, torch.Tensor) and action.shape[-1] == 29:
            self._twist2_last_action.copy_(action.to(self.env.device, dtype=torch.float32))
        return action

    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from DDS"""
        try:
            full_action = self._full_action_buf
            full_action.zero_()
            action_data = self.run_policy()

            # --- TWIST2 full-body action (29-dof, MuJoCo actuator order) ---
            raw_action = torch.clip(action_data.to(self.env.device, dtype=torch.float32), -10.0, 10.0)
            # Server uses per-joint action_scale=0.5; keep it simple for quick dev
            target_29 = raw_action * 0.5 + self.twist2_default_pos  # [1,29]

            # Fill defaults first
            full_action.copy_(self.env.scene["robot"].data.default_joint_pos.squeeze(0))
            # Overwrite the 29 controlled joints
            if hasattr(self, "_twist2_action_idx_t"):
                full_action.index_copy_(0, self._twist2_action_idx_t, target_29.squeeze(0))
            else:
                full_action.index_copy_(0, torch.tensor(self.twist2_action_indices, device=self.env.device, dtype=torch.long), target_29.squeeze(0))

            # 夹爪/手指（若有）
            hand_from_redis = False
            if self.enable_dex3 and self._twist2_hand_valid and hasattr(self, "_left_hand_source_idx_t"):
                self._left_hand_buf.copy_(self._twist2_action_hand_left.squeeze(0))
                self._right_hand_buf.copy_(self._twist2_action_hand_right.squeeze(0))
                l_vals = self._left_hand_buf.index_select(0, self._left_hand_source_idx_t)
                r_vals = self._right_hand_buf.index_select(0, self._right_hand_source_idx_t)
                full_action.index_copy_(0, self._left_hand_target_idx_t, l_vals)
                full_action.index_copy_(0, self._right_hand_target_idx_t, r_vals)
                hand_from_redis = True

            if not hand_from_redis:
                if self.gripper_dds and hasattr(self, "_gripper_source_idx_t"):
                    gripper_cmd = self.gripper_dds.get_gripper_command()
                    if gripper_cmd:
                        left_gripper_cmd = gripper_cmd.get('left_gripper_cmd', {})
                        right_gripper_cmd = gripper_cmd.get('right_gripper_cmd', {})
                        left_gripper_positions = left_gripper_cmd.get('positions', [])
                        right_gripper_positions = right_gripper_cmd.get('positions', [])
                        gripper_positions = right_gripper_positions + left_gripper_positions
                        if len(gripper_positions) >= 2:
                            self._gripper_buf.copy_(torch.tensor(gripper_positions[:2], dtype=torch.float32, device=self.env.device))
                            gp_vals = self._gripper_buf.index_select(0, self._gripper_source_idx_t)
                            full_action.index_copy_(0, self._gripper_target_idx_t, gp_vals)
                elif self.dex3_dds and hasattr(self, "_left_hand_source_idx_t"):
                    hand_cmds = self.dex3_dds.get_hand_commands()
                    if hand_cmds:
                        left_hand_cmd = hand_cmds.get('left_hand_cmd', {})
                        right_hand_cmd = hand_cmds.get('right_hand_cmd', {})
                        if left_hand_cmd and right_hand_cmd:
                            left_positions = left_hand_cmd.get('positions', [])
                            right_positions = right_hand_cmd.get('positions', [])
                            if len(left_positions) >= len(self._left_hand_buf) and len(right_positions) >= len(self._right_hand_buf):
                                self._left_hand_buf.copy_(torch.tensor(left_positions[:len(self._left_hand_buf)], dtype=torch.float32, device=self.env.device))
                                self._right_hand_buf.copy_(torch.tensor(right_positions[:len(self._right_hand_buf)], dtype=torch.float32, device=self.env.device))
                                l_vals = self._left_hand_buf.index_select(0, self._left_hand_source_idx_t)
                                r_vals = self._right_hand_buf.index_select(0, self._right_hand_source_idx_t)
                                full_action.index_copy_(0, self._left_hand_target_idx_t, l_vals)
                                full_action.index_copy_(0, self._right_hand_target_idx_t, r_vals)
                elif self.inspire_dds and hasattr(self, "_inspire_source_idx_t"):
                    inspire_cmds = self.inspire_dds.get_inspire_hand_command()
                    if inspire_cmds and 'positions' in inspire_cmds:
                        inspire_cmds_positions = inspire_cmds['positions']
                        if len(inspire_cmds_positions) >= 12:
                            self._inspire_buf.copy_(torch.tensor(inspire_cmds_positions[:12], dtype=torch.float32, device=self.env.device))
                            base_vals = self._inspire_buf.index_select(0, self._inspire_source_idx_t)
                            full_action.index_copy_(0, self._inspire_target_idx_t, base_vals)
                            special_vals = self._inspire_buf.index_select(0, self._inspire_special_source_idx_t) * self._inspire_special_scales_t
                            full_action.index_copy_(0, self._inspire_special_target_idx_t, special_vals)

            # 同步仿真多步
            for _ in range(self._twist2_decimation):
                self.env.scene["robot"].set_joint_position_target(full_action)
                self.env.scene.write_data_to_sim()
                self.env.sim.step(render=False)
                self.env.scene.update(dt=self.env.physics_dt)

            self.env.sim.render()
            self.env.observation_manager.compute()
            return full_action
        except Exception as e:
            print(f"[{self.name}] Get DDS action failed: {e}")
            return None
    
    def _convert_to_joint_range(self, value):
        """Convert gripper control value to joint angle"""
        input_min, input_max = 0.0, 5.6
        output_min, output_max = 0.03, -0.02
        value = max(input_min, min(input_max, value))
        return output_min + (output_max - output_min) * (value - input_min) / (input_max - input_min)
    
    def cleanup(self):
        """Clean up DDS resources"""
        try:
            if self.robot_dds:
                self.robot_dds.stop_communication()
            if self.gripper_dds:
                self.gripper_dds.stop_communication()
            if self.dex3_dds:
                self.dex3_dds.stop_communication()
            if self.inspire_dds:
                self.inspire_dds.stop_communication()
        except Exception as e:
            print(f"[{self.name}] Clean up DDS resources failed: {e}")
