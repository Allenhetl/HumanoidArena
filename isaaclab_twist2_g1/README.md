# IsaacLab TWIST2 G1

`isaaclab_twist2_g1/` 是当前仓库里连接 Isaac Lab 仿真、Pico 遥操作、录制与 replay 的主目录。

## 目录说明

- `action_provider/`
  `TWIST2` / `SONIC` 控制、录制、replay 的核心逻辑。
- `tasks/`
  Isaac Lab 任务定义与场景配置。
- `pico_server/`
  Pico 侧桥接与 Redis/ZMQ 数据发布。
- `image_server/`
  图像输出与串流。
- `recording_data/`
  默认录制输出目录。
- `logs/`
  replay 或运行日志。

## 常用脚本

所有启动参数都直接写在脚本顶部，按需手改。

### Live

- [run_twist2.sh](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/run_twist2.sh)
  启动 `TWIST2` live 遥操作录制。
- [run_sonic.sh](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/run_sonic.sh)
  启动 `SONIC` live 遥操作录制。

### Replay

- [run_replay_twist2.sh](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/run_replay_twist2.sh)
  启动 `TWIST2` replay。
- [run_replay_sonic.sh](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/run_replay_sonic.sh)
  启动 `SONIC` replay。

## Replay 模式

两条链都统一到了 `sim_main.py` 的 replay 入口：

- `direct_replay`
  直接执行录制文件里保存的目标动作。
- `inference_replay`
  使用录制文件中的模型输入或观测，再跑一次推理。

`TWIST2` 的脚本里显示为 `direct / inference`，内部会归一化成统一 replay 模式。

## 文档索引

- [REPLAY_DEBUG_SUMMARY.md](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/REPLAY_DEBUG_SUMMARY.md)
  当前 replay 问题、经验和修复总结。
- [SCENE_RANDOMIZATION_SEED_RULES.md](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/docs/SCENE_RANDOMIZATION_SEED_RULES.md)
  场景随机化、录制和 replay 必须共用单一 `episode_object_seed` 的约束说明。
- [TWIST2_DATA_FORMAT.md](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/docs/TWIST2_DATA_FORMAT.md)
  `TWIST2` 录制 `.npz` 数据格式说明。
- [SONIC_DATA_FORMAT.md](/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/docs/SONIC_DATA_FORMAT.md)
  `SONIC` 录制 `.npz` 数据格式说明。

## 备注

- 当前推荐直接修改各 `run_*.sh` 文件顶部参数，不再额外走共享 YAML/配置脚本。
- `TWIST2` 与 `SONIC` 都已经加入输入 ready barrier，避免启动或 reset 前的 Redis 数据污染录制首段。
- 带随机场景初始化的任务只允许一颗场景主 seed。不要再为 obstacle layout 或局部 scene 初始化额外维护第二颗 seed。
