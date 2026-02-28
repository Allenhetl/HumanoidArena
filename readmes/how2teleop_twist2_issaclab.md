<div align="center">
  <h1>TWIST2 + IsaacLab 遥操作指南</h1>
  <p>使用 XRobot 头显在 IsaacLab 仿真中进行全身遥操作</p>
  <p><i>Taowen Wang</i></p>
</div>

---

## 目录

- [概述](#概述)
- [安装](#安装)
- [使用流程](#使用流程)
  - [Step 1：打开 XRobotToolkit](#step-1打开-xrobottoolkit)
  - [Step 2：运行 teleop.sh](#step-2运行-teleopsh)
  - [Step 3：运行 run.sh](#step-3运行-runsh)

---

## 概述

本文档介绍如何使用 TWIST2 配合 XRobot PICO 头显，在 IsaacLab 仿真环境中进行全身遥操作。

| 步骤 | 组件                    | 作用                          |
| ---- | ----------------------- | ----------------------------- |
| 1    | XRobotToolkit（Linux）  | 接收头显姿态数据并串流        |
| 2    | TWIST2 `teleop.sh`      | 姿态解算，通过 Redis 发布动作 |
| 3    | IsaacLab `run.sh`       | 仿真接收动作并驱动机器人      |

---

## 安装

运行前请先完成以下两个组件的安装：

| 组件      | 安装文档                                         |
| --------- |----------------------------------------------|
| IsaacLab  | [README.md](../isaaclab_twist2_g1/README.md) |
| TWIST2    | [TWIST2/README.md](../TWIST2/README.md)      |

---

## 使用流程

### Step 1：打开 XRobotToolkit

在 **Linux 机器**上启动 XRobotToolkit，用于接收 PICO 头显的姿态数据。

![XRobotToolkit](xrobotoolkit.png)

> 确认头显已连接并在界面中显示正常后，进入下一步。

---

### Step 2：运行 teleop.sh

进入 TWIST2 目录，启动遥操作脚本：

```bash
cd TWIST2
bash teleop.sh
```

脚本会自动激活 `gmr` conda 环境，并运行 `xrobot_teleop_to_robot_w_hand.py`。

#### 调整人体高度

在运行前，根据实际情况修改 `teleop.sh` 中的身高参数：

```bash
actual_human_height=1.79   # 单位：米，根据实际身高调整
```

> **注意**：由于 PICO 对高度估计存在误差，建议将该值设置为**略小于**实际身高。

启动后终端会以 1Hz 打印帧率，确认数据正常流动后进入下一步。

---

### Step 3：运行 run.sh

在 `isaaclab_twist2_g1` 根目录下启动仿真：

```bash
bash run.sh
```

脚本会先清空 Redis 中的历史动作数据，再启动 IsaacLab 仿真。机器人将实时跟随遥操作姿态运动。