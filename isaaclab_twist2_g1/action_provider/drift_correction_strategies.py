"""
Improved replay strategy to minimize drift.

Key improvements:
1. Use root velocity instead of root position (less drift)
2. Implement periodic state correction
3. Add hybrid mode: use recorded root state but let physics evolve naturally
"""

# Add this method to ReplayActionProvider class

def get_action_with_drift_correction(self, env) -> Optional[torch.Tensor]:
    """
    Improved get_action with drift correction strategies.

    Strategies:
    1. POSITION_ONLY: Only set joint positions, let root evolve naturally (minimal intervention)
    2. ROOT_VELOCITY: Set root velocity instead of position (smoother)
    3. PERIODIC_CORRECTION: Correct root state every N frames
    4. FULL_STATE: Set both root and joints every frame (current implementation)
    """

    # Strategy selection (can be set via args_cli)
    strategy = getattr(self, 'drift_correction_strategy', 'PERIODIC_CORRECTION')
    correction_interval = getattr(self, 'correction_interval', 10)  # Correct every 10 frames

    # ... existing code to get target_29 ...

    # Apply drift correction based on strategy
    if strategy == 'POSITION_ONLY':
        # Don't set root state at all - let physics evolve naturally
        # This gives maximum stability but may drift from recorded trajectory
        pass

    elif strategy == 'ROOT_VELOCITY':
        # Set root velocity instead of position (smoother, less jarring)
        if self.replay_data_root_pos is not None and self.current_frame > 0:
            # Compute velocity from position difference
            pos_curr = self.replay_data_root_pos[self.current_frame]
            pos_prev = self.replay_data_root_pos[self.current_frame - 1]
            root_vel = (pos_curr - pos_prev) / (self._twist2_decimation * self.env.physics_dt)

            # Set root velocity
            root_vel_tensor = torch.from_numpy(root_vel).to(self.env.device, dtype=torch.float32).unsqueeze(0)
            self.env.scene["robot"].write_root_velocity_to_sim(root_vel_tensor)

    elif strategy == 'PERIODIC_CORRECTION':
        # Only correct root state every N frames to reduce jarring
        if self.current_frame % correction_interval == 0:
            # Full root state correction
            if self.replay_data_root_pos is not None and self.replay_data_root_quat is not None:
                root_pos = self.replay_data_root_pos[self.current_frame]
                root_quat = self.replay_data_root_quat[self.current_frame]

                root_pos_tensor = torch.from_numpy(root_pos).to(self.env.device, dtype=torch.float32).unsqueeze(0)
                root_quat_tensor = torch.from_numpy(root_quat).to(self.env.device, dtype=torch.float32).unsqueeze(0)

                self.env.scene["robot"].write_root_pose_to_sim(
                    root_pose=torch.cat([root_pos_tensor, root_quat_tensor], dim=-1)
                )

                if self.current_frame % 100 == 0:
                    print(f"[{self.name}] 🔧 Periodic correction at frame {self.current_frame}")

    elif strategy == 'FULL_STATE':
        # Current implementation - set root state every frame
        # This is most accurate but may cause jarring/instability
        if self.replay_data_root_pos is not None and self.replay_data_root_quat is not None:
            root_pos = self.replay_data_root_pos[self.current_frame]
            root_quat = self.replay_data_root_quat[self.current_frame]

            root_pos_tensor = torch.from_numpy(root_pos).to(self.env.device, dtype=torch.float32).unsqueeze(0)
            root_quat_tensor = torch.from_numpy(root_quat).to(self.env.device, dtype=torch.float32).unsqueeze(0)

            self.env.scene["robot"].write_root_pose_to_sim(
                root_pose=torch.cat([root_pos_tensor, root_quat_tensor], dim=-1)
            )

    # ... rest of physics simulation ...
