# Object Randomization And Replay

本文总结 `isaaclab_twist2_g1` 当前关于环境物体随机化与 replay 可复现性的实现方案。

## 目标

- 录制数据时，环境物体初始位置要有足够随机性，降低人工重复采集负担。
- 随机化不能污染全局 RNG 状态，避免影响别的模块或 PhysX 初始化路径。
- replay 时必须能够精确恢复录制时的环境初始状态，保证 direct replay 的可复现性。
- 方案要能扩展到未来更多任务，以及一个任务中一个或多个物体。

## 当前方案

### 1. 录制模式使用 episode 级时间种子

每次对象 reset 时，都会生成一个 `episode_object_seed`。

- 当前默认来源：`time.time_ns()`
- 配置字段：`object_reset_seed_source: time`
- 实现位置：[common_env_objects.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/common_env_objects.py)

这个 seed 只用于当前 episode 的对象初始化，不会写回全局 `np.random` / `torch` / `random` 状态。

### 2. 用局部 Generator 做对象采样

对象位置不是从全局 RNG 流中取值，而是这样生成：

1. 先得到 `episode_object_seed`
2. 再结合 `object_name` 和 `env_index`
3. 派生出该对象本次 reset 专属的局部 `np.random.default_rng(...)`

因此：

- 不会污染全局 RNG 状态
- 不会受其他模块额外消费随机数影响
- 多个对象之间互不影响

### 3. replay 不重新采样

replay 时完全不依赖当前时间，也不重新走对象随机化采样。

流程是：

1. 录制时保存每个 episode 的初始环境对象状态
2. replay/reset 时直接读取这些 `episode_init_env_obj_*` 字段
3. 把对应对象 root state 写回仿真

实现位置：

- 录制组织与字段写入：
  - [action_provider_wh_twist2.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py)
  - [action_provider_sonic.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_sonic.py)
- replay 初始状态恢复：
  - [sim_main.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/sim_main.py)

如果旧录制文件没有 `episode_init_env_obj_*` 字段，当前会退回到 `env_obj_*` 的第 0 帧。

## 通用扩展结构

### 任务配置入口

每个任务、每个 GMT 使用自己的 YAML，例如：

- [football_single_twist2.yaml](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/common_env_config/football_single_twist2.yaml)
- [football_single_sonic.yaml](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/common_env_config/football_single_sonic.yaml)
- [football_single_vla.yaml](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/common_env_config/football_single_vla.yaml)

默认 YAML 只保留通用默认值：

- [twist2_default.yaml](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/common_env_config/twist2_default.yaml)
- [sonic_default.yaml](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/common_env_config/sonic_default.yaml)

### 通用对象随机化配置

任务 YAML 中通过 `deterministic_object_resets` 配置可录制、可恢复的环境对象：

```yaml
overrides:
  sim:
    dt: 0.001
  decimation: 10
  object_reset_seed_source: time
  deterministic_object_resets:
    - record_name: football
      scene_keys: [object, football]
      pose_range:
        x: [-0.05, 0.05]
        y: [0.0, 0.05]
      zero_velocity_on_reset: true
```

字段说明：

- `record_name`
  录制文件里的对象名，同时用于 replay 恢复匹配。
- `scene_keys`
  在 Isaac scene 中可尝试匹配的 asset key 列表。
- `pose_range`
  本次 reset 的偏移范围，支持 `x/y/z`。
- `zero_velocity_on_reset`
  reset 后是否把线速度和角速度清零。

### 多物体任务

如果未来一个任务有多个物体，只需要在 YAML 里写多个条目，例如：

```yaml
deterministic_object_resets:
  - record_name: football
    scene_keys: [object, football]
    pose_range:
      x: [-0.2, 0.2]
      y: [-0.2, 0.2]
  - record_name: cup
    scene_keys: [cup, object_cup]
    pose_range:
      x: [-0.1, 0.1]
      y: [-0.1, 0.1]
```

录制端会自动生成：

- `env_obj_football_*`
- `env_obj_cup_*`
- `episode_init_env_obj_football_*`
- `episode_init_env_obj_cup_*`

replay 也会自动按这些字段恢复。

## 当前关键文件

- 通用对象随机化与录制 helper：
  - [common_env_objects.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/common_env_objects.py)
- replay 初始环境对象恢复：
  - [sim_main.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/sim_main.py)
- 当前 football single task env cfg：
  - [move_football_single_g1_29dof_dex3_hw_env_cfg.py](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/g1_tasks/move_football_single_g1_29dof_dex3_wholebody/move_football_single_g1_29dof_dex3_hw_env_cfg.py)

## 当前结论

- 录制模式下，物体随机化现在默认来自时间种子，因此每个 episode 会自然变化。
- 这个随机化不会污染全局 RNG 状态。
- replay 不依赖当前时间，而是严格恢复录制保存的物体初始状态，因此不会破坏 replay 可复现性。

## VLA 评测建议

- `run_vla.sh` 不适合使用 `object_reset_seed_source: time`。
- 更适合使用独立配置文件 [football_single_vla.yaml](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/tasks/common_env_config/football_single_vla.yaml)。
- 该配置使用 `object_reset_seed_source: env_seed`，并由 `run_vla.sh` 显式传 `--seed 42`。

这样同一个模型重复测试、以及不同模型之间横向对比时，只要 `SEED` 相同，足球出生点序列就一致；同一进程内多次 reset 的变化则由 `seed + reset_counter` 决定，仍然是可复现的。
