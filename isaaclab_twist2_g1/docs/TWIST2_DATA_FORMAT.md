# TWIST2 数据格式

本文档描述 `isaaclab_twist2_g1` 中 `TWIST2` 录制生成的 `.npz` 文件格式。

实现位置：
- 录制组织：[action_provider_wh_twist2.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py)
- 单帧采集：[action_provider_wh_twist2.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py)

## 文件命名

默认文件名形如：

```text
<task_name>_<timestamp_us>.npz
```

例如：

```text
Isaac-Move-Football-Single-G129-Dex3-Wholebody_1774962955107099.npz
```

## 顶层字段

### 标量元信息

- `task`
  任务名字符串。
- `num_frames`
  帧数。
- `observation_semantics`
  JSON 字符串，描述 `robot_obs_buf` 的语义切分。

### 人体输入

- `human_hand_left`
  形状 `(T, 7)`，左手 7 维输入。
- `human_hand_right`
  形状 `(T, 7)`，右手 7 维输入。
- `human_neck`
  形状 `(T, 2)`，颈部 2 维输入。
- `human_smplx_data`
  JSON 字符串，逐帧保存 `smplx_data_before_gmr`。
- `human_info_data`
  JSON 字符串，逐帧保存 `human_info`。

### 机器人状态

- `robot_qpos_before_decimation`
  形状 `(T, 29)`，每个 control frame 开始时的 29 个 body joint 角度。
- `robot_qvel_before_decimation`
  形状 `(T, 29)`，对应关节速度。
- `robot_root_position`
  形状 `(T, 3)`，世界系根位置。
- `robot_root_orientation`
  形状 `(T, 4)`，根四元数，`wxyz`。
- `robot_root_lin_vel_local`
  形状 `(T, 3)`，机体系线速度。
- `robot_root_ang_vel_local`
  形状 `(T, 3)`，机体系角速度。
- `robot_root_lin_vel_world`
  形状 `(T, 3)`，世界系线速度。
- `robot_root_ang_vel_world`
  形状 `(T, 3)`，世界系角速度。
- `robot_twist2_inference_qpos`
  形状 `(T, 29)`，`TWIST2` 推理输出的 29 维目标动作。
- `robot_obs_buf`
  形状 `(T, 1432)`，录制时真正送入 policy 的观测。

### 可选机器人动力学量

这些字段只有在运行时可读到时才会写入：

- `robot_applied_torque_before_decimation`
  形状 `(T, 29)`，关节已施加力矩。
- `robot_body_net_contact_forces`
  形状 `(T, B, 3)`，每个 body 的净接触力。

### 环境对象

- `env_obj_football_position`
  形状 `(T, 3)`。
- `env_obj_football_linear_velocity`
  形状 `(T, 3)`。
- `env_obj_football_angular_velocity`
  形状 `(T, 3)`。
- `env_obj_table_drink_position`
  形状 `(T, 3)`。
- `env_obj_table_drink_linear_velocity`
  形状 `(T, 3)`。
- `env_obj_table_drink_angular_velocity`
  形状 `(T, 3)`。

如果对象不存在，会写零数组。

### 图像

- `vision_rgb`
  形状 `(N, H, W, C)`。
- `vision_depth`
  形状 `(N, H, W)`。
- `vision_frame_indices`
  形状 `(N,)`，对应这些图像来自哪几个控制帧。

当前实现一般会保存全部帧图像；如果后续改策略，要以 `vision_frame_indices` 为准。

### 系统信息

- `system_control_frequency`
  形状 `(T,)`，控制频率。
- `system_decimation`
  形状 `(T,)`，decimation。
- `system_physics_dt`
  形状 `(T,)`，physics dt。
- `system_timestamp`
  形状 `(T,)`，wall clock 时间戳。

## Replay 依赖字段

### direct replay

最关键字段：

- `robot_twist2_inference_qpos`
- `robot_qpos_before_decimation`

辅助字段：

- `human_hand_left`
- `human_hand_right`
- `human_neck`

### inference replay

最关键字段：

- `robot_obs_buf`
- `robot_twist2_inference_qpos`

`TWIST2` 的 inference replay 直接拿录制的 `robot_obs_buf` 再跑一次模型，不重建 observation。

## 读取示例

```python
import numpy as np

with np.load("foo.npz", allow_pickle=True) as data:
    print(data["task"].item())
    print(data["robot_twist2_inference_qpos"].shape)
    print(data["robot_obs_buf"].shape)
```

## 备注

- 该格式当前没有单独的 `schema_version` 字段，兼容性主要依赖字段名本身。
- 如果修改 `robot_obs_buf` 维度或观测语义，应同步更新 `observation_semantics` 和 replay 逻辑。
