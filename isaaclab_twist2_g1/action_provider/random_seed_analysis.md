# Random Seed 对齐分析

## 问题发现

在 `move_football_g1_29dof_dex3_hw_env_cfg.py` 中发现环境重置时存在随机性：

```python
# Line 151-161
self.event_manager.register(
    "reset_object_self",
    SimpleEvent(
        func=lambda env: base_mdp.reset_root_state_uniform(
            env,
            torch.arange(env.num_envs, device=env.device),
            pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},  # ⚠️ 随机范围
            velocity_range={},
            asset_cfg=SceneEntityCfg("object"),
        )
    ),
)
```

## 影响

1. **足球初始位置随机化**：每次环境重置时，足球会在 x 方向 ±5cm、y 方向 0-5cm 范围内随机偏移
2. **录制和回放不一致**：如果没有设置相同的 random seed，回放时足球的初始位置会与录制时不同
3. **导致行为差异**：即使机器人动作完全相同，由于足球位置不同，物理交互结果也会不同

## 当前状态

### 录制脚本 (action_provider_wh_twist2.py)
- ❌ 没有显式设置 random seed
- 使用系统默认的随机状态

### 回放脚本 (action_provider_wh_twist2_replay.py)
- ❌ 没有显式设置 random seed
- 使用系统默认的随机状态

### 环境配置
- ❌ `MoveFootballG129Dex3WholebodyEnvCfg` 中没有 seed 参数
- ❌ Isaac Lab 环境初始化时没有传入 seed

## Isaac Lab 2.3.0 随机种子机制

根据 Isaac Lab 文档，框架提供了确定性仿真支持：

### 核心方法
```python
isaaclab.utils.seed.configure_seed(seed: int | None, torch_deterministic: bool = False) → int
```

### 环境配置
- `ManagerBasedEnvCfg.seed` - 基于管理器的环境
- `DirectRLEnvCfg.seed` - 直接环境

### 确定性保证
- ✅ 相同硬件 + 相同 Isaac Sim/PhysX 版本 → 完全相同的仿真结果
- ✅ 刚体和关节场景完全确定
- ⚠️ 不同硬件配置可能因浮点精度差异产生不同结果
- ❌ 非刚体（布料、软体）场景不保证确定性

## 解决方案

### 方案 1：设置固定 seed（推荐）

#### 步骤 1：在环境配置中添加 seed

```python
# move_football_g1_29dof_dex3_hw_env_cfg.py
@configclass
class MoveFootballG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """
    Environment configuration for G1 29DOF Dex3 wholebody robot with football task.
    """

    # 添加固定 seed
    seed = 42  # 或任何固定值

    scene: FootballTableSceneCfg = FootballTableSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    # ... 其他配置
```

#### 步骤 2：在录制数据中保存 seed

```python
# action_provider_wh_twist2.py
# 在初始化时保存环境的 seed
self.env_seed = self.env.cfg.seed if hasattr(self.env.cfg, 'seed') else None

# 在保存录制数据时添加 seed
organized['env_seed'] = self.env_seed
```

#### 步骤 3：在回放时验证 seed

```python
# action_provider_wh_twist2_replay.py
# 加载录制数据时检查 seed
if 'env_seed' in data:
    recorded_seed = data['env_seed']
    current_seed = self.env.cfg.seed if hasattr(self.env.cfg, 'seed') else None
    if recorded_seed != current_seed:
        print(f"⚠️ Warning: Seed mismatch! Recorded: {recorded_seed}, Current: {current_seed}")
```

### 方案 2：录制完整场景状态

在录制数据中添加所有物体的初始状态：

```python
# action_provider_wh_twist2.py
# 在 collect_recording_data 中添加：
robot_data["football_position"] = self.env.scene["object"].data.root_pos_w[0].cpu().numpy()
robot_data["football_orientation"] = self.env.scene["object"].data.root_quat_w[0].cpu().numpy()
robot_data["football_lin_vel"] = self.env.scene["object"].data.root_lin_vel_w[0].cpu().numpy()
robot_data["football_ang_vel"] = self.env.scene["object"].data.root_ang_vel_w[0].cpu().numpy()
```

在回放时恢复场景状态：

```python
# action_provider_wh_twist2_replay.py
# 在 Frame 0 初始化时：
if self.current_frame == 0:
    # 恢复足球状态
    football_pos = torch.from_numpy(self.replay_data_football_pos[0]).to(self.env.device)
    football_quat = torch.from_numpy(self.replay_data_football_quat[0]).to(self.env.device)
    football_lin_vel = torch.from_numpy(self.replay_data_football_lin_vel[0]).to(self.env.device)
    football_ang_vel = torch.from_numpy(self.replay_data_football_ang_vel[0]).to(self.env.device)

    # 设置足球位置和姿态
    root_pose = torch.cat([football_pos, football_quat]).unsqueeze(0)
    self.env.scene["object"].write_root_pose_to_sim(root_pose)

    # 设置足球速度
    root_velocity = torch.cat([football_lin_vel, football_ang_vel]).unsqueeze(0)
    self.env.scene["object"].write_root_velocity_to_sim(root_velocity)

    # 同步到仿真
    self.env.scene.write_data_to_sim()
```

### 方案 3：禁用随机化（不推荐）

修改环境配置，移除足球位置的随机化：

```python
# move_football_g1_29dof_dex3_hw_env_cfg.py
pose_range={"x": [0.0, 0.0], "y": [0.0, 0.0]},  # 禁用随机化
```

**缺点**：改变了训练环境的行为，可能影响策略的泛化能力。

## 推荐行动

### 短期方案（立即可用）
1. ✅ 在环境配置中添加固定 `seed = 42`
2. ✅ 重新录制数据（使用固定 seed）
3. ✅ 回放时使用相同的环境配置（相同 seed）

### 长期方案（更健壮）
1. ✅ 在录制数据中保存环境 seed
2. ✅ 在录制数据中保存所有物体的初始状态（足球、机器人等）
3. ✅ 回放时验证 seed 并恢复完整场景状态
4. ✅ 在回放脚本中添加 seed 不匹配的警告

## 验证步骤

1. 修改环境配置添加 seed
2. 录制新数据
3. 多次回放同一录制文件
4. 验证每次回放的足球轨迹完全相同
5. 验证机器人行为完全相同

## 相关文件

- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/g1_tasks/move_football_g1_29dof_dex3_wholebody/move_football_g1_29dof_dex3_hw_env_cfg.py:106-126`
- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/g1_tasks/move_football_g1_29dof_dex3_wholebody/move_football_g1_29dof_dex3_hw_env_cfg.py:151-161`
- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py:1863-1872`
- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_wh_twist2_replay.py:127-158`

## Isaac Lab 文档参考

- Isaac Lab 2.3.0 Reproducibility: `isaaclab.utils.seed.configure_seed()`
- Environment Config: `ManagerBasedEnvCfg.seed`
- PhysX Determinism: 刚体场景完全确定，相同硬件+版本保证相同结果

