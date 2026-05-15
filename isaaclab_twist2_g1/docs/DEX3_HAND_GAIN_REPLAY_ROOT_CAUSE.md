# Dex3 Hand Gain Replay Root Cause

本文记录一次 SONIC/TWIST2 旧录制数据无法稳定 replay 的根因分析。结论是：问题不是足球对象随机化、YAML 范围、VLA 数据协议或 replay action 选择，而是 Dex3 手部 implicit actuator 的 `stiffness/damping` 被改低，破坏了确定性 replay 所要求的同一动力学链路。

## 现象

同一份旧成功的 football SONIC replay 数据，在当前代码下失败：

- `Replay env err object=football` 一直为 0，说明足球初始位置、速度和 replay 对象状态恢复正确。
- `Replay joint err` 从前几帧接近 0，随后逐渐累积，例如 frame 10 开始出现小误差，frame 100 后误差明显扩大。
- football 类任务失败更明显，因为它依赖脚和球接触瞬间的精确位姿/速度；静态对象任务可能暂时不暴露同样程度的问题。

这说明 replay 失败发生在 robot 状态转移，而不是足球对象初始化。

## 精确根因

提交 `822e4e9818758e4d05aa654ab2144662e10e4e48` (`update eval branch from server 0508`) 将 Dex3 hands actuator gain 从旧值改低：

```python
stiffness={
    ".*": 10.0,
}
damping={
    ".*": 1.0,
}
```

旧成功版本 `05c0356` 中对应值是：

```python
stiffness={
    ".*": 100.0,
}
damping={
    ".*": 10.0,
}
```

受影响位置：

- `robots/unitree.py` 中 `G129_CFG_WITH_DEX3_BASE_FIX` 的 `"hands": ImplicitActuatorCfg(...)`
- `robots/unitree.py` 中 `G129_CFG_WITH_DEX3_WHOLEBODY` 的 `"hands": ImplicitActuatorCfg(...)`

虽然 direct replay 使用的是记录下来的 29DoF body target，但仿真实际执行的是完整 robot action。Dex3 14 个手部关节仍然属于同一个 articulation，且会在每个 decimation 内通过 implicit actuator 参与动力学。手部 gain 改变后，整机惯性响应、关节跟踪和接触前姿态都会发生微小差异；这些差异在 PhysX 的确定性但伪随机/数值敏感链路中会逐帧累积，最终导致 football 接触窗口错位。

## 为什么不是 YAML / Seed

football replay 文件中已经包含 `episode_init_env_obj_football_*` 状态。当前 replay 日志显示足球对象恢复后：

- 初始足球位置与 NPZ 一致。
- replay 过程中 `Replay env err object=football` 为 0。

因此 `football_single_sonic.yaml` 的 `pose_range` 改动不会解释同一 NPZ replay 失败。范围改动只影响新 episode 的采样，不会覆盖 replay 从 NPZ 显式恢复的对象状态。

同理，`episode_object_seed` 的 32-bit/64-bit 链路也不是这次 football 失败的直接原因：对象状态已经恢复正确，漂移出现在 robot qpos。

## 为什么不是 VLA 协议

direct replay 优先读取录制文件中的 `final_body_action_29dof`，该字段等于当时实际送入 robot 的 decoder/body target。失败时并不是重新走 VLA inference，也不是使用 `vla_action_joint_pos_29` 覆盖 direct action。

因此 VLA v2/v3 的 observation/action protocol 变化不是这次 direct replay 漂移的直接原因。

## 修复

将 Dex3 hands actuator gain 恢复到旧成功版本：

```python
stiffness={
    ".*": 100.0,
}
damping={
    ".*": 10.0,
}
```

修复后，旧 football SONIC replay 可恢复成功。

## 后续守则

1. replay 兼容性要求 robot USD、robot actuator config、physics dt、decimation、headless/render 路径、对象初始化链路完全一致。
2. 修改 hand/arm/foot 这类 articulation actuator 参数，即使不是 body 29DoF target 的主要字段，也必须视为会影响 replay。
3. 如果旧数据 replay 失败，先看 `Replay env err` 和 `Replay joint err` 的分叉点：
   - env err 非 0：优先查对象恢复、seed、YAML 和 task-specific restore。
   - env err 为 0 但 joint err 累积：优先查 robot config、actuator gain、USD、physics/control stepping。
4. VLA 数据协议变更不应该影响 direct replay；如果影响了 direct replay，应优先检查 action provider 是否误用了 VLA 字段替代 `final_body_action_29dof`。

