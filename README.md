<div align="center">
  <h1>HumanoidArena 遥操作与 Replay 指南</h1>
  <p>TWIST2 / SONIC / Isaac Lab G1 全身遥操作、录制与 replay</p>
  <p><i>Taowen Wang</i></p>
</div>

---

## 目录

- [概述](#概述)
- [安装](#安装)
- [使用流程](#使用流程)
  - [Step 1：打开 XRobotToolkit](#step-1打开-xrobottoolkit)
  - [Step 2：运行 teleop 或 Pico server](#step-2运行-teleop-或-pico-server)
  - [Step 3：运行 Isaac Lab](#step-3运行-isaac-lab)
  - [Step 4：运行 replay](#step-4运行-replay)

---

## 概述

本文档介绍当前仓库里两条主要链路：

- `TWIST2 + IsaacLab`
- `SONIC + IsaacLab`

| 步骤 | 组件                    | 作用                          |
| ---- | ----------------------- | ----------------------------- |
| 1    | XRobotToolkit（Linux）  | 接收头显姿态数据并串流        |
| 2    | TWIST2 / SONIC 上游输入 | 姿态解算，通过 Redis 发布动作 |
| 3    | IsaacLab `run_*.sh`     | 仿真接收动作并驱动机器人      |
| 4    | IsaacLab `run_replay_*.sh` | 从录制 `.npz` 做 replay   |

---

## 安装

新机器部署建议先阅读项目级环境说明：

- [环境部署说明](./isaaclab_twist2_g1/docs/ENVIRONMENT_SETUP.md)

运行前请先完成以下组件的安装：

| 组件      | 安装文档 |
| --------- | -------- |
| Conda / Isaac Sim / Isaac Lab / LeRobot 环境 | [环境部署说明](./isaaclab_twist2_g1/docs/ENVIRONMENT_SETUP.md) |
| IsaacLab 仿真桥接 | [isaaclab_twist2_g1/README.md](./isaaclab_twist2_g1/README.md) |
| TWIST2    | [TWIST2/README.md](./TWIST2/README.md) |

---

## 使用流程

常用命令已经单独整理到：

- [isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md](./isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md)

### Step 1：打开 XRobotToolkit

在 **Linux 机器**上启动 XRobotToolkit，用于接收 PICO 头显的姿态数据。

![XRobotToolkit](xrobotoolkit.png)

> 确认头显已连接并在界面中显示正常后，进入下一步。

---

### Step 2：运行 teleop 或 Pico server

如果走 `TWIST2`，进入 `TWIST2` 目录，启动遥操作脚本：

```bash
cd TWIST2
bash teleop.sh
```

脚本会自动激活 `gmr` conda 环境，并运行 `xrobot_teleop_to_robot_w_hand.py`。

如果走 `SONIC`，通常需要先启动 `isaaclab_twist2_g1/pico_server/` 下对应的 Pico server。

#### 调整人体高度

在运行前，根据实际情况修改 `teleop.sh` 中的身高参数：

```bash
actual_human_height=1.79   # 单位：米，根据实际身高调整
```

> **注意**：由于 PICO 对高度估计存在误差，建议将该值设置为**略小于**实际身高。

启动后终端会以 1Hz 打印帧率，确认数据正常流动后进入下一步。

---

### Step 3：运行 Isaac Lab

常用 live / 录制命令见：

- [isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md#1-遥操作录制](./isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md#1-遥操作录制)

在 `isaaclab_twist2_g1` 根目录下启动仿真：

```bash
bash run_twist2.sh
# 或
bash run_sonic.sh
```

脚本顶部参数区可直接修改：

- 任务名
- 机器人 USD / 碰撞模式
- 录制目录
- 图传地址
- replay 文件路径

更完整的入口说明见：

- [isaaclab_twist2_g1/README.md](./isaaclab_twist2_g1/README.md)

### Step 4：运行 replay

常用 replay / rerecord / LeRobot 评测命令见：

- [isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md#2-replay](./isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md#2-replay)
- [isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md#3-rerecord](./isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md#3-rerecord)
- [isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md#4-lerobot--vla-评测](./isaaclab_twist2_g1/docs/COMMAND_QUICKSTART.md#4-lerobot--vla-评测)

```bash
bash isaaclab_twist2_g1/run_replay_twist2.sh
# 或
bash isaaclab_twist2_g1/run_replay_sonic.sh
```

对应数据格式说明：

- [TWIST2_DATA_FORMAT.md](./isaaclab_twist2_g1/docs/TWIST2_DATA_FORMAT.md)
- [SONIC_DATA_FORMAT.md](./isaaclab_twist2_g1/docs/SONIC_DATA_FORMAT.md)
