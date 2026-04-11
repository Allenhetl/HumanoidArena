# VLA 统一 SMPL 数据格式方案

## 1. 目标

本方案的目标是为当前遥操作数据建立一套统一的 VLA 训练格式，使同一套 VLA 模型输出既可以适配到 `SONIC`，也可以适配到 `TWIST2`。

核心原则：

- 两套 GMT 数据来源本质一致，统一应优先发生在人体语义层，而不是机器人执行层。
- `SMPL` 比 `qpos` 更接近共同来源空间。
- `qpos` 更适合作为派生执行结果，而不是统一训练主标签。

## 2. 结论

建议分成两层：

- 统一存档源格式：保留 `SMPL pose`
- VLA 实际训练输出格式：将 `pose` 转成 `rot6d`

不建议直接将以下任一格式作为唯一统一训练标签：

- `TWIST2 29 qpos`
- `SONIC 86D semantic token`

原因：

- `29 qpos` 过于依赖具体 retargeter 和机器人执行语义。
- `86D` 过于偏向 `SONIC` 的 encoder 输入定义。
- 两者都不是最稳定、最通用的共同人体语义空间。

## 3. 统一存档源格式

建议每帧保存以下字段：

- `root_trans_ref`: `float32[3]`
  含义：root/pelvis 平移，建议使用 episode-local 坐标。
- `root_orient_aa`: `float32[3]`
  含义：root/global orientation，axis-angle 表示。
- `body_pose_local_aa`: `float32[21, 3]`
  含义：SMPL 局部关节旋转，父关节相对旋转。
- `hand_binary`: `float32[2]`
  含义：左右手开合语义。

总维度约为 `71D`。

这套格式适合作为长期保存格式，原因是：

- `pose -> joints` 可通过 SMPL 前向过程计算。
- `joints -> pose` 不唯一，尤其会丢失 wrist twist 信息。
- `SONIC` 和 `TWIST2` 都可从这套人体语义派生。

## 4. 为什么主格式应保留 pose

`pose` 在这里是关节旋转参数，通常是 axis-angle。

`joints` 是关节三维位置。

`pose -> joints` 需要通过 SMPL 前向过程或 FK。

仅保留 `joints` 不足以唯一恢复：

- 肘部和手腕的 twist
- 稳定的 wrist reference
- 完整局部关节旋转语义

因此：

- 统一主格式应以 `pose` 为核心。
- `joints` 应作为派生特征，而不是唯一训练真值。

## 5. VLA 训练 action 格式

训练时不建议直接使用 axis-angle，建议改用 `rot6d`。

推荐的 VLA action 为：

- `root_trans_ref`: `float32[3]`
- `root_orient_rot6d`: `float32[6]`
- `body_pose_local_rot6d`: `float32[21, 6]`
- `hand_binary`: `float32[2]`

总维度约为 `137D`。

理由：

- `rot6d` 比 axis-angle 更平滑，训练更稳定。
- `body_pose_local_rot6d` 保留了完整局部姿态信息。
- 不需要直接学习 `anchor_rot6d` 或 `wrist_ref`，它们应由适配层在线派生。

## 6. 是否需要做时间差

不建议将全身 `pose` 训练成逐帧差分形式。

推荐做法：

- 身体关节姿态：使用当前帧绝对局部姿态
- root 运动：在适配层按时间差分得到速度类量

原因：

- 全身 `delta_pose` 需要积分恢复，容易累计漂移。
- 静止和动态阶段的数值分布不稳定。
- 下游控制器更关心当前人体姿态，而不是上一帧基础上改多少。

## 7. VLA proprioception 定义

建议统一使用一套后端无关的机器人状态输入，而不是沿用 `SONIC` 或 `TWIST2` 各自网络内部定义。

推荐：

- `root_ang_vel_local`: `float32[3]`
- `gravity_dir`: `float32[3]`
- `joint_pos_delta_29`: `float32[29]`
- `joint_vel_29`: `float32[29]`
- `last_body_target_delta_29`: `float32[29]`
- `hand_binary`: `float32[2]`

总维度为 `95D`。

说明：

- `joint_pos_delta_29 = current_joint_pos - default_joint_pos`
- `last_body_target_delta_29` 表示上一帧实际 body target 相对默认姿态的偏移

不建议直接使用：

- `SONIC decoder_raw_action`
- `TWIST2 raw policy action`

因为这两者属于 backend-specific 网络内部语义，不适合作为统一 VLA 输入。

## 8. 到 SONIC 的转换

从统一 VLA action 到 `SONIC` 的流程：

1. `rot6d -> rotation matrix -> axis-angle / quat`
2. 用 `root_orient + body_pose_local` 运行 SMPL 前向，得到 `smpl_joints_local`
3. 用 `root_orient` 得到 `body_quat_w`
4. 结合当前 robot base quaternion，在线计算 `anchor_rot6d`
5. 用 elbow/wrist 的 pose 旋转按当前规则计算 `wrist_ref`
6. 得到 `SONIC` 当前帧语义：
   - `smpl_joints_local`
   - `anchor_rot6d`
   - `wrist_ref`
   - `hand_binary`
7. 写入 `SONIC` 的时序缓冲，继续走 encoder/decoder

这里的关键点：

- `anchor_rot6d` 不应作为 VLA 直接学习目标
- `wrist_ref` 不应作为统一主标签直接保存
- 它们都应该由 `pose` 在线派生

## 9. 到 TWIST2 的转换

从统一 VLA action 到 `TWIST2` 的流程：

1. `rot6d -> axis-angle`
2. 组装一帧 canonical `SMPL` 人体姿态：
   - `root_trans_ref`
   - `root_orient`
   - `body_pose_local`
   - `hand_binary`
3. 送入统一 GMR / retargeter，得到 `G1 qpos_ref29`
4. 根据连续两帧 root pose / qpos 在线差分得到：
   - `xy_vel`
   - `z_pos`
   - `roll_pitch`
   - `yaw_vel`
   - `qpos29`
5. 组成 `TWIST2 mimic_obs35`
6. 手部 binary 映射到 7+7 hand joints

对 `TWIST2` 而言，VLA 不应直接输出 `29 qpos` 作为统一训练目标，而应输出人体姿态，再经 GMR 派生到 `qpos`。

## 10. qpos 顺序统一建议

虽然统一主标签不应使用 `qpos`，但 proprioception 和适配层仍会使用 29 维机器人关节状态。

建议统一采用：

- `SONIC / IsaacLab order` 作为 canonical 29 joint order

原因：

- `SONIC` 原生使用该顺序。
- `TWIST2` 当前使用 MuJoCo action order，与 `SONIC` 不同。
- 因此由 `TWIST2` 侧做固定 reorder 更合理。

## 11. 为什么不直接用 SONIC 86D 作为统一标签

`SONIC 86D` 为：

- `smpl_joints 72D`
- `anchor_rot6d 6D`
- `wrist_ref 6D`
- `hand_binary 2D`

它适合作为 `SONIC` 适配层输入，但不适合作为唯一统一训练主标签。

原因：

- `smpl_joints` 丢失部分旋转信息。
- `wrist_ref` 是从 `pose` 派生出来的特殊补充量。
- `anchor_rot6d` 带有 `SONIC` 在线对齐语义。
- 对 `TWIST2` 而言，`pose` 经过 GMR 的路径更自然。

因此：

- `86D` 应作为 `SONIC` 的中间适配格式
- `pose_rot6d` 应作为 VLA 的统一训练输出格式

## 12. 最终推荐

### 12.1 长期保存格式

- `root_trans_ref 3`
- `root_orient_aa 3`
- `body_pose_local_aa 63`
- `hand_binary 2`
- `meta`
  - `human_height`
  - `tracker_type`
  - `fps`
  - 可选 `betas/calibration`

### 12.2 VLA 训练 action

- `root_trans_ref 3`
- `root_orient_rot6d 6`
- `body_pose_local_rot6d 126`
- `hand_binary 2`

### 12.3 VLA proprioception

- `root_ang_vel_local 3`
- `gravity_dir 3`
- `joint_pos_delta_29 29`
- `joint_vel_29 29`
- `last_body_target_delta_29 29`
- `hand_binary 2`

### 12.4 到 SONIC

- `pose_rot6d -> SMPL joints + body_quat`
- `root_orient -> anchor_rot6d`
- `elbow/wrist pose -> wrist_ref`
- 组成 `SONIC semantic token`
- 再走 `SONIC encoder/decoder`

### 12.5 到 TWIST2

- `pose_rot6d -> canonical SMPL frame`
- `SMPL frame -> GMR -> qpos_ref29`
- 连续帧差分得到 `mimic_obs35`
- 再走 `TWIST2`

## 13. 一句话总结

统一方案应为：

- 统一主语义：`SMPL pose`
- 训练表示：`rot6d`
- `SONIC` 和 `TWIST2` 都通过适配层从同一套 `pose` 派生
- VLA 本体感知使用统一 `95D` 机器人状态，不使用 backend-specific raw action
