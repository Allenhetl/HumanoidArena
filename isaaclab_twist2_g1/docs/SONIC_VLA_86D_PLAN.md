# SONIC VLA 86D 方案设计

## 1. 目标

本方案的目标不是让 VLA 直接输出可执行的 G1 低层关节动作，而是让 VLA 预测一组适合作为 `SONIC encoder` 条件输入的高层语义动作，并继续保留现有 `SONIC encoder -> SONIC decoder -> IsaacLab -> G1` 的稳定控制链路。

最终推理链路定义为：

```text
VLA -> 86D semantic action -> SONIC encoder active blocks -> SONIC encoder
    -> latent -> SONIC decoder -> 29D body target -> IsaacLab

VLA -> 2D hand position -> hand pose interpolation -> 7+7 hand joints -> IsaacLab
```

这意味着：

- VLA 不直接替代 `SONIC decoder`
- VLA 不预测完整 `encoder_input(1762)` 或 `decoder_obs(994)`
- VLA 负责提供 `SONIC` 稳定控制所需的上游人体语义条件

## 2. 为什么不是 31D 低层动作

如果 VLA 直接预测 `29D body + 2D hand` 并交给 IsaacLab 执行，会绕过 `SONIC encoder/decoder` 这条已经具备稳定性的控制路径。对于 G1 这种全身控制问题，这样做会显著增加跌倒风险。

当前 `SONIC` 的关键稳定性来自两点：

1. `encoder` 负责把人体语义输入编码成适合机器人执行的 latent
2. `decoder` 结合机器人历史状态、历史动作、重力方向等低层信息输出稳定的 body action

因此，VLA 的正确位置应在 `encoder` 之前，而不是 `decoder` 之后。

## 3. 当前 SONIC 中真正需要 VLA 预测的部分

### 3.1 encoder 的有效动态输入

当前 `SMPL mode` 下，`SONIC encoder_input` 虽然总维度是 `1762`，但实际启用的动态块只有三组，其余块在当前实现中都被置零：

- `smpl_joints_10frame_step1`: `10 x 24 x 3`
- `smpl_anchor_orientation_10frame_step1`: `10 x 6`
- `motion_joint_positions_wrists_10frame_step1`: `10 x 6`

对应代码在：

- `isaaclab_twist2_g1/action_provider/action_provider_sonic.py`

因此，不需要让 VLA 预测完整 `1762D encoder_input`，只需要预测这三组有效块在“当前帧”的语义值，并由在线历史缓冲区恢复成 `10` 帧窗口。

### 3.2 86D 动作定义

定义单步 VLA 动作为：

```text
body semantic token:
  smpl_joints_t           72D = 24 x 3
  anchor_rot6d_t           6D
  wrist_ref_t              6D

hand semantic token:
  left_hand_position       1D
  right_hand_position      1D

total = 72 + 6 + 6 + 2 = 86D
```

其中：

- `smpl_joints_t` 对应 `human_smpl_joints[t]`
- `anchor_rot6d_t` 由 `human_body_quat_w[t]` 或 anchor 对齐后的姿态转换而来
- `wrist_ref_t` 对应 encoder 中实际使用的 6 维 wrist reference
- `left/right_hand_position` 是手部开合连续量，不是 7 维手关节

## 4. 为什么手要用 2D position，而不是 14D joint

当前 `SONIC` / `pico_server` 的手部逻辑不是 14 自由度独立控制，而是：

1. 控制器输入先更新一个持续累积的 `hand_position in [0,1]`
2. 再用 open pose 和 close pose 线性插值出 7 维手关节

因此，手的本质语义是：

- 左手开合程度 `left_hand_position`
- 右手开合程度 `right_hand_position`

手部 7 维关节可以在推理时通过现有插值逻辑恢复，不应作为 VLA 的主控制语义。

这有三个直接好处：

- 动作空间更稳，更容易学
- 和现有 teleop 语义一致
- 后续更容易扩展不同手型映射

## 5. 什么不应该作为 VLA 训练目标

### 5.1 不学完整 `encoder_input(1762)`

原因：

- 当前大量维度恒为零
- 会让模型浪费容量去拟合无效块
- 一旦 encoder 配置调整，数据定义就会随之大幅波动

### 5.2 不学 `decoder_obs(994)`

原因：

- `decoder_obs` 包含机器人低层状态历史
- 包含 `last_action_hist`
- 包含重力方向、关节速度、机器人当前姿态等在线状态

这些量应该由仿真运行时和 `SONIC action provider` 在线维护，而不是让 VLA 预测。

### 5.3 不学 `29D decoder_raw_action`

原因：

- 这会把 VLA 放到 `decoder` 之后，绕过 `SONIC` 稳定控制链路
- 对稳定性和泛化都不利

## 6. 推理时的系统逻辑

### 6.1 body 路径

单步推理流程：

1. VLA 根据图像和状态输出 `84D body semantic token`
2. 将 `smpl_joints_t / anchor_rot6d_t / wrist_ref_t` 写入本地 ring buffer
3. 用最近 `10` 帧构造 encoder active blocks：
   - `smpl_joints_10frame_step1`
   - `smpl_anchor_orientation_10frame_step1`
   - `motion_joint_positions_wrists_10frame_step1`
4. 其余 encoder blocks 保持当前实现：
   - 常量块保持常量
   - 零块继续为零
5. 继续调用现有 `SONIC encoder`
6. 继续调用现有 `SONIC decoder`
7. 用 decoder 输出得到 `29D body target`
8. 送入 IsaacLab body joints

### 6.2 hand 路径

单步推理流程：

1. VLA 输出 `left_hand_position/right_hand_position`
2. 调用当前 hand interpolation 逻辑恢复 `7+7` 手关节
3. 送入 IsaacLab hand joints

### 6.3 reset 语义

episode reset 时必须同步清空：

- body semantic history ring buffer
- `SONIC` 内部 `last_action_hist`
- hand position 状态
- 任何 VLA preprocessor/postprocessor 的 temporal state

否则第二段 episode 的输入相位会错位。

## 7. 训练数据组织

### 7.1 LeRobot 数据结构

建议对齐 `twist2 -> lerobot` 的组织方式，仍然使用：

- `observation.images.front`
- `observation.state`
- `action`

但 `action` 改为 `86D`。

### 7.2 推荐 action 命名

建议命名为：

```text
action.body.smpl_joint.<joint_name>.x
action.body.smpl_joint.<joint_name>.y
action.body.smpl_joint.<joint_name>.z
action.body.anchor_rot6d.0..5
action.body.wrist_ref.0..5
action.hand.left_position
action.hand.right_position
```

### 7.3 推荐 observation.state

为了让 VLA 拥有最小但足够的机器人上下文，建议 `observation.state` 使用低层稳定相关量，而不是人体原始缓存：

```text
ang_vel_local                3
gravity_dir                  3
joint_pos_delta_29          29
joint_vel_29                29
last_body_raw_action_29     29
hand_position_2              2

total = 95D
```

这个状态定义的原则是：

- 保留 `decoder` 稳定性相关上下文
- 不让 VLA 背负重建整套 `SONIC` 内部缓存的负担
- 与推理时可在线获得的量一致

## 8. 从现有 sonic 录制中抽取 86D label 的方法

### 8.1 body 部分

从每个录制帧提取：

- `human_smpl_joints[t]` -> `72D`
- `human_body_quat_w[t]` 或等价 anchor 对齐结果 -> `6D rot`
- `encoder_wrist_window[t, -1]` 或等价当前帧 wrist ref -> `6D`

这里建议优先直接使用录制中已经保存的模型输入侧缓存，避免离线重建和在线实现不一致：

- `encoder_smpl_joint_window[t, -1]`
- `encoder_anchor_window[t, -1]`
- `encoder_wrist_window[t, -1]`

这样能保证 label 和线上 `SONIC encoder` 看到的当前帧语义完全一致。

### 8.2 hand 部分

录制中需要显式保存：

- `hand_left_position`
- `hand_right_position`

如果当前录制尚未保存这两个标量，则有两种策略：

1. 在录制侧补写这两个字段，之后重新录制
2. 用 `hand_action_left/right` 反推 position，作为兼容方案

推荐采用方案 1，因为它更稳，也和线上控制语义一致。

## 9. sonic2lerobot 的实现规划

新增脚本建议路径：

- `isaaclab_twist2_g1/tools/data_tools/sonic2lerobot.py`

### 9.1 输入

- 递归读取 `SONIC .npz`
- 每个 `.npz` 为一个 episode

### 9.2 输出 features

- `observation.images.front`
- `observation.state` shape=`(95,)`
- `action` shape=`(86,)`

### 9.3 单帧转换逻辑

对每个控制帧：

1. 取当前 RGB
2. 取当前机器人状态构造 `95D observation.state`
3. 从录制缓存中取当前帧 `86D action`
4. 写入 LeRobot dataset

### 9.4 数据过滤

建议加入以下过滤选项：

- 跳过手部信号全零的 episode
- 跳过 `vision_rgb` 缺失 episode
- 跳过 `encoder_*` 缓存缺失 episode
- 支持 `--limit` 和 `--overwrite`

## 10. 训练规划

第一阶段建议直接复用 `twist2` 现有 LeRobot 训练入口，先用 diffusion policy 跑通。

推荐起始参数：

```text
policy.type=diffusion
policy.n_obs_steps=2
policy.horizon=16
policy.n_action_steps=8
batch_size=32
optimizer_lr=1e-4
```

理由：

- 86D 动作比 35D/37D 更重，不宜一开始用过长 horizon
- `SONIC` 后面还有 encoder/decoder 稳定器，VLA 本身不需要直接承担长时低层控制

## 11. 推理接入规划

### 11.1 总体原则

不要新增一套脱离 `SonicActionProvider` 的控制器。  
应在现有 `SonicActionProvider` 内增加 `input_source=vla + gmt_backend=sonic` 分支。

### 11.2 provider 需要新增的能力

1. `LeRobot VLA client`
2. `86D action` 解包
3. `body semantic history ring buffer`
4. `hand position -> 7D hand joints` 插值
5. reset 时同步清理所有状态

### 11.3 provider 的执行流程

每步：

1. 读取前视图和 `95D state`
2. 调用 VLA 服务得到 `86D`
3. 更新 body semantic history
4. 构造 encoder active blocks
5. 调用 `SONIC encoder`
6. 调用 `SONIC decoder`
7. body 输出送入原有 body control
8. hand position 恢复成手关节并送入原有 hand control

## 12. 评测规划

建议完全仿照 `twist2` 的评测体系，新增一套 `sonic` 版本：

- `script/eval_scripts/sonic/run_vla_eval.sh`
- `script/eval_scripts/sonic/eval_vla_suite.py`
- `script/eval_scripts/sonic/sim_eval_vla.py`
- `tasks/common_env_config/football_single_sonic_vla.yaml`

### 12.1 批测指标

- 成功率
- 平均 episode 步数
- 跌倒率
- object interaction 成功率
- body tracking 误差
- hand usage 触发率

### 12.2 必做对照实验

至少做四组：

1. 原生 `SONIC teleop -> SONIC`
2. `VLA 86D -> SONIC`
3. `VLA 84D body only -> SONIC`
4. `VLA 29D low-level baseline -> IsaacLab`

目标是明确证明：

- `86D -> SONIC` 比直接 low-level 更稳
- hand 监督对 HOI 确实有增益

## 13. 实施顺序

### Phase 0: 数据审计

- 确认现有 episode 中哪些包含有效 hand 信号
- 确认 `encoder_*` 缓存字段齐全
- 确认 `vision_frame_indices` 与控制帧对齐

### Phase 1: 录制增强

- 在 `SONIC` 录制中新增 `hand_left_position/right_position`
- 必要时补充 `current_frame` wrist ref 显式字段

### Phase 2: 数据转换

- 实现 `sonic2lerobot.py`
- 生成人可检查的 metadata
- 做少量 episode smoke test

### Phase 3: 训练

- 跑小规模训练确认 loss 收敛
- 导出 checkpoint
- 跑单 episode 在线 smoke test

### Phase 4: 推理接入

- 在 `SonicActionProvider` 中接 VLA 分支
- 确认 reset、history、hand interpolation 都正确

### Phase 5: 批测

- 跑多 seed、多 repeat
- 输出 success rate、视频、summary

## 14. 主要风险

### 14.1 hand 录制无效

如果录制中手部信号长期全零，即使把手维度放进监督，训练也学不到有效手策略。  
这个问题必须在 Phase 0 先确认。

### 14.2 线上离线语义不一致

如果训练 label 不是直接取自录制时的 `encoder_*` 当前帧缓存，而是离线重建，容易与线上 provider 的实现发生偏差。

### 14.3 reset 相位错位

`SONIC` 对历史窗口敏感。  
如果 reset 后 ring buffer、`last_action_hist`、hand state 没有同步清空，第二段 episode 结果会失真。

## 15. 最终结论

本项目中，`SONIC VLA` 的正确设计不是直接学低层 `31D` 动作，而是学习：

- `84D body semantic token`
- `2D hand semantic token`

合计 **86D**。

这套设计的核心价值是：

- 保留 `SONIC encoder/decoder` 的稳定性兜底
- 避免预测完整 `SONIC` 内部缓存
- 保留手部监督并使用与 teleop 一致的语义
- 便于后续在 HOI 任务上扩展和做批量评测

后续实现应围绕这一定义展开，不建议再回到“VLA 直接输出低层 body action”的路线。
