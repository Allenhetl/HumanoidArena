# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from action_provider.action_base import ActionProvider
from typing import Optional
import torch
from dds.dds_master import dds_manager
class DDSActionProvider(ActionProvider):
    """Action provider based on DDS"""
    
    def __init__(self,env, args_cli):
        super().__init__("DDSActionProvider")
        self.enable_robot = args_cli.robot_type
        self.enable_gripper = args_cli.enable_dex1_dds
        self.enable_dex3 = args_cli.enable_dex3_dds
        self.enable_inspire = args_cli.enable_inspire_dds
        self.env = env
        # Initialize DDS communication
        self.robot_dds = None
        self.gripper_dds = None
        self.dex3_dds = None
        self.inspire_dds = None
        self._setup_dds()
        self._setup_joint_mapping()
    
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
            print(f"[{self.name}] DDS communication initialized")
        except Exception as e:
            print(f"[{self.name}] DDS initialization failed: {e}")
    
    def _setup_joint_mapping(self):
        """Setup joint mapping"""
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
        self.arm_action_pose = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
        self.arm_action_pose_indices = [self.arm_joint_mapping[name] for name in self.arm_joint_mapping.keys()]
    def get_action(self, env) -> Optional[torch.Tensor]:
        """Get action from DDS"""
        try:

            full_action = torch.zeros(len(self.all_joint_names), device=self.env.device)
            # Get robot command
            if self.enable_robot == "g129" and self.robot_dds:
                cmd_data = self.robot_dds.get_robot_command()
                if cmd_data and 'motor_cmd' in cmd_data:
                    positions = cmd_data['motor_cmd']['positions']
                    if len(positions) >= 29:
                        # arm_action = torch.tensor(positions[15:29], device=env.device, dtype=torch.float32)
                        for joint_name, arm_idx in self.arm_joint_mapping.items():
                            if joint_name in self.joint_to_index:
                                full_action[self.joint_to_index[joint_name]] = positions[arm_idx+15]
            
            # Get gripper command
            if self.gripper_dds:
                gripper_cmd = self.gripper_dds.get_gripper_command()
                if gripper_cmd:
                    left_gripper_cmd = gripper_cmd.get('left_gripper_cmd', {})
                    right_gripper_cmd = gripper_cmd.get('right_gripper_cmd', {})
                    left_gripper_positions = left_gripper_cmd.get('positions', [])
                    right_gripper_positions = right_gripper_cmd.get('positions', [])
                    gripper_positions = right_gripper_positions + left_gripper_positions
                    # print(f"gripper_positions: {gripper_positions}")
                    if len(gripper_positions) >= 2:
                        for joint_name, gripper_idx in self.gripper_joint_mapping.items():
                            if joint_name in self.joint_to_index:
                                # Convert gripper range
                                gripper_value = gripper_positions[gripper_idx]
                                # print(f"gripper_value: {gripper_value}")
                                full_action[self.joint_to_index[joint_name]] = gripper_value
            
            # Get hand command
            elif self.dex3_dds:
                hand_cmds = self.dex3_dds.get_hand_commands()
                if hand_cmds:
                    left_hand_cmd = hand_cmds.get('left_hand_cmd', {})
                    right_hand_cmd = hand_cmds.get('right_hand_cmd', {})
                    if left_hand_cmd and right_hand_cmd:
                        left_positions = left_hand_cmd.get('positions', [])
                        right_positions = right_hand_cmd.get('positions', [])
                        for joint_name, left_idx in self.left_hand_joint_mapping.items():
                            if joint_name in self.joint_to_index:
                                full_action[self.joint_to_index[joint_name]] = left_positions[left_idx]
                        for joint_name, right_idx in self.right_hand_joint_mapping.items():
                            if joint_name in self.joint_to_index:
                                full_action[self.joint_to_index[joint_name]] = right_positions[right_idx]
            elif self.inspire_dds:
                inspire_cmds = self.inspire_dds.get_inspire_hand_command()
                if inspire_cmds and 'positions' in inspire_cmds:
                        inspire_cmds_positions = inspire_cmds['positions']
                        # print(f"gripper_positions: {gripper_positions}")
                        if len(inspire_cmds_positions) >= 12:
                            for joint_name, inspire_hand_idx in self.inspire_hand_joint_mapping.items():
                                if joint_name in self.joint_to_index:
                                    # Convert gripper range
                                    inspire_value = inspire_cmds_positions[inspire_hand_idx]
                                    # print(f"gripper_value: {gripper_value}")
                                    full_action[self.joint_to_index[joint_name]] = inspire_value
                            for joint_name, special_idx in self.special_joint_mapping.items():
                                if joint_name in self.joint_to_index:
                                    inspire_value = inspire_cmds_positions[special_idx[0]]
                                    full_action[self.joint_to_index[joint_name]] = inspire_value * special_idx[1]
            return full_action.unsqueeze(0)
            
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