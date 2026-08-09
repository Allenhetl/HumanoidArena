# Inspire Retarget 集成验收 (2026-08-06, 方案A 落地)

> 目标：用 AnyDexRetarget Adaptive 优化器替换现有固定三点角 hand retarget，
> 输出 a_hw_6（6 电机）对接 HumanoidArena 仿真 + DFX 实机。

## 1. 结论（决定性探针）

用真实 Pico 手部录制数据（`recording_data/HOI_pickplace_inspire/..._1786010873333529.npz`，
`inspire_raw_hand_right` 10000 帧 26x7）对比三种 retarget：

| 方法 | 手指 flex 响应 | 拇指-手指捏合 | 饱和问题 |
|---|---|---|---|
| **现三点角**（`hand_keypoints_to_a_hw`） | 敏感 ✅ | 无 ❌ | 伸展/握拳极限饱和（gt=3.14） |
| **DexPilot**（xr_teleoperate/dex-retargeting） | **响应不足，恒 1.7 饱和** ❌ | 有（投影抓取）✅ | 严重 |
| **AnyDex Adaptive**（AnyDexRetarget） | **12 关节全响应** ✅ | 有（pinch 感知）✅ | 无 |

**结论：采用 AnyDexRetarget AdaptiveOptimizerAnalytical**。

实测（有效帧，AnyDex a_hw_6 vs 三点角）：
```
f45:   [0.95 0.70 0.77 0.95 0.00 0.95]  vs  [0.73 0.70 0.73 0.73 0.14 0.78]  ✅ 手指一致
f2062: [1.08 0.28 0.47 1.47 0.30 1.31]  vs  [1.13 0.54 0.53 0.91 0.35 0.65]  ✅ flex 变化
```
拇指 flex/rotation 有系统偏移 → 需 `mediapipe_rotation`/`thumb_offset` 校准（后续迭代）。

## 2. 交付物

### 环境：`retarget`（conda）
```
conda create -n retarget python=3.10 pinocchio nlopt pyyaml -c conda-forge
conda install -n retarget -c conda-forge "pytorch=2.3.0=cpu*"
pip install pytransform3d trimesh anytree lxml numpy==1.26.4
pip install -e reference/xr_teleoperate/teleop/robot_control/dex-retargeting --no-deps
```
> **运行需 `LD_LIBRARY_PATH=/home/dreams/miniconda3/envs/retarget/lib`**（pinocchio 需要 conda 的 libstdc++ 6.0.35）。

### 模块：`tools/inspire_retarget_anydex.py`
```python
from tools.inspire_retarget_anydex import InspireRetargetAnyDex
hr = InspireRetargetAnyDex(side="right")
a_hw_6 = hr.retarget_a_hw6(kp26x7)      # rad, [index,middle,ring,pinky,thumb_pitch,thumb_yaw]
dfx = hr.a_hw6_to_dfx(a_hw_6)           # [0,1], 0=closed 1=open (DFX 硬件序)
```
- 26x7 OpenXR → MediaPipe 21 → Adaptive 12 关节 → a_hw_6（6 电机）
- 依赖 AnyDexRetarget（`reference/AnyDexRetarget`，MIT）
- 自测：`python tools/inspire_retarget_anydex.py`（合成 + 真实数据）

## 3. 与 DFX_inspire_service 对接

- 输出 a_hw_6 rad → `a_hw6_to_dfx` 归一化 [0,1]（照搬 xr_teleoperate normalize 范围）
- DFX 6 电机序：idx 0-3=index/middle/ring/pinky flex(0-1.7), 4=thumb_pitch(0-0.5), 5=thumb_yaw(-0.1-1.3)
- 硬件：发 `rt/inspire/cmd`（MotorCmds_，ID 0-5 右手 / 6-11 左手），DFX_inspire_service 订阅
  - 与 xr_teleoperate 的 `robot_hand_inspire.py` 完全同协议

## 4. 待办
- [ ] 拇指校准（AnyDex `mediapipe_rotation`/`calibrate_offset.py` 思路）消拇指偏移
- [ ] 将 `InspireRetargetAnyDex` 集成进 publisher（`twist2_teleop_server.py` 或独立进程）
- [ ] DFX 实机 DDS 发布验证
- [ ] 与现三点角做 A/B 对比（仿真观察）
