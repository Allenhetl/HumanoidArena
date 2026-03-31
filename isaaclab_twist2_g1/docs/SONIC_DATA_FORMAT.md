# SONIC 数据格式

本文档描述 `isaaclab_twist2_g1` 中 `SONIC` 录制生成的 `.npz` 文件格式。

实现位置：
- 录制组织：[action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)
- 单帧采集：[action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)

## 文件命名

默认文件名形如：

```text
<task_name>_sonic_<timestamp_us>.npz
```

## Schema

- `schema_version = "sonic_episode_v1"`

与 `TWIST2` 不同，`SONIC` 是显式带版本号的 episode 格式。

## 顶层字段

### 元信息

- `schema_version`
- `task`
- `episode_id`
- `save_timestamp_us`
- `num_frames`
- `meta_control_dt`
- `meta_physics_dt`
- `meta_decimation`
- `meta_pose_source`
- `meta_encoder_path`
- `meta_decoder_path`

### 帧标记

- `frame_index`
  形状 `(T,)`，来自输入流的原始 frame index。
- `episode_step`
  形状 `(T,)`，当前 episode 内的录制步号。
- `timestamp_wall`
- `timestamp_monotonic`
- `timestamp_realtime`
- `recording_command`
  逐帧字符串数组，如 `none`、`start`、`save_and_reset`。
- `reset_requested`
- `reset_completed`
- `save_triggered`

### 人体处理后数据

- `human_left_hand`
  形状 `(T, 7)`。
- `human_right_hand`
  形状 `(T, 7)`。
- `human_smpl_joints`
  形状 `(T, 24, 3)`，当前帧 SMPL joints。
- `human_smpl_pose`
  形状 `(T, 21, 3)`。
- `human_body_quat_w`
  形状 `(T, 4)`，当前帧 body quaternion。
- `human_vr_position`
  形状 `(T, 9)`。
- `human_vr_orientation`
  形状 `(T, 12)`。
- `human_heading_increment`
  形状 `(T, 1)`。

### Anchor / heading 对齐状态

- `anchor_heading_initialized`
- `anchor_use_heading_align`
- `anchor_init_base_quat_wxyz`
- `anchor_init_ref_quat_wxyz`
- `anchor_heading_align_quat_wxyz`

这些字段对 replay 很关键，尤其是 `SONIC inference_replay`。

### SONIC 模型输入输出缓存

#### Encoder 侧

- `encoder_input`
  形状 `(T, 1762)`，录制时真正送入 encoder 的输入。
- `encoder_smpl_joint_window`
  形状 `(T, 10, 24, 3)`。
- `encoder_anchor_window`
  形状 `(T, 10, 6)`。
- `encoder_wrist_window`
  形状 `(T, 10, W)`，`W` 对应 wrist 相关维度。
- `encoder_motion_joint_pos_hist`
  形状 `(T, H, 29)`，参考 motion joint pos 历史。
- `encoder_motion_joint_vel_hist`
  形状 `(T, H, 29)`。
- `encoder_motion_root_z_hist`
  形状 `(T, H)`。
- `encoder_motion_anchor_rot6d_hist`
  形状 `(T, H, 6)`。
- `encoder_robot_joint_pos_hist`
  形状 `(T, 10, 29)`。
- `encoder_robot_joint_vel_hist`
  形状 `(T, 10, 29)`。
- `encoder_latent`
  形状 `(T, 64)`。

其中 `H` 是 step5 history 长度，当前实现由 provider 内部常量决定。

#### Decoder 侧

- `decoder_obs`
  形状 `(T, 994)`，录制时真正送入 decoder 的 observation。
- `decoder_ang_vel_hist`
  形状 `(T, 10, 3)`。
- `decoder_gravity_dir_hist`
  形状 `(T, 10, 3)`。
- `decoder_last_action_hist`
  形状 `(T, 10, 29)`。
- `decoder_raw_action`
  形状 `(T, 29)`，decoder 原始输出。
- `decoder_target_action`
  形状 `(T, 29)`，乘上 action scale 并加 default 后的目标关节角。

### 机器人状态

- `robot_qpos_before_decimation`
  形状 `(T, 29)`。
- `robot_qvel_before_decimation`
  形状 `(T, 29)`。
- `robot_root_position`
- `robot_root_orientation`
- `robot_root_lin_vel_local`
- `robot_root_ang_vel_local`
- `robot_root_lin_vel_world`
- `robot_root_ang_vel_world`

### 最终动作

- `final_body_action_29dof`
  形状 `(T, 29)`，最终 body joint target。
- `final_full_action`
  形状 `(T, N)`，完整机器人 joint target。
- `body_effort_target`
  形状 `(T, 29)`，若使用 torque/effort 路径时的 body effort。
- `hand_action_left`
  形状 `(T, 7)`。
- `hand_action_right`
  形状 `(T, 7)`。

### 原始输入 JSON

这些字段是 JSON 字符串，按帧打包：

- `human_raw_smplx_json`
- `human_controller_json`
- `human_recording_control_json`

### 环境对象

按需写入，可不存在：

- `env_obj_football_position`
- `env_obj_football_linear_velocity`
- `env_obj_football_angular_velocity`
- `env_obj_table_drink_position`
- `env_obj_table_drink_linear_velocity`
- `env_obj_table_drink_angular_velocity`

### 图像

- `vision_rgb`
- `vision_depth`
- `vision_frame_indices`

## Replay 依赖字段

### direct replay

核心依赖：

- `final_body_action_29dof`
  或旧兼容名 `decoder_target_action`

辅助依赖：

- `hand_action_left`
- `hand_action_right`
- anchor 状态字段

### inference replay

核心依赖：

- `encoder_input`
- `decoder_obs`
- encoder / decoder 历史缓冲字段
- 当前帧人体与 anchor 状态字段

当前 `SONIC inference_replay` 的正确做法是恢复录制时保存的历史和模型输入，而不是冷启动重建窗口。

## 读取示例

```python
import numpy as np

with np.load("foo.npz", allow_pickle=True) as data:
    print(data["schema_version"].item())
    print(data["task"].item())
    print(data["encoder_input"].shape)
    print(data["decoder_target_action"].shape)
```

## 备注

- 第二段及之后的 episode 是否可 replay，和录制边界是否在 reset 后立刻开始密切相关。
- 如果修改 encoder/decoder 输入定义，必须同步更新录制字段和 `inference_replay` 的恢复逻辑。
