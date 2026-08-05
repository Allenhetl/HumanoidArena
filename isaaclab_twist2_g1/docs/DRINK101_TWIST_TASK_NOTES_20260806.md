# drink101 拧盖场景工程记录 (2026-08-06)

> 记录 drink101（可拧开瓶盖的瓶子）+ SAM3D 桌子的完整接入过程、关键实验证据与最终方案。
> 配套跨会话记忆: 仓库根 `AGENTS.md`。
> 所有验证均 headless（无 GUI），不依赖 Pico 硬件。

---

## 1. 最终方案（已验证）

- **瓶身 + 瓶盖 = 两个独立 RigidObject**（`drink101_body.usd` / `drink101_cap.usd`），**无 PhysX joint**
- "拧开" = 手部对 cap 施加绕 z 轴扭矩 → `drink_state.py` 几何监控累计相对角度（sealed→twisting→armed→opened）
- "取盖" = cap 上拉分离（自由刚体一拔就起）
- cap 碰撞 mesh 在预处理时上移 0.021m，避免与瓶口碰撞咬合导致旋转卡死
- 桌子 = SAM3D 重建的 `desk0.usd`（GLB→USD），静态 kinematic，SDF 碰撞，带纹理

## 2. 关键实验证据（为什么不用 PhysX joint）

| 方案 | 实验 | 结果 |
|---|---|---|
| 运行时 UsdPhysics 关节（两个 RigidObject 间 DefinePrim RevoluteJoint） | 30N 水平力推 cap | **cap 被吹飞 3906m** —— PhysX 完全不识别运行时创建的关节 |
| 预烘焙 ArticulationRoot + RevoluteJoint（drink101_artic.usd + ArticulationCfg） | `dc.get_joint(path)` + 扭矩驱动 | joint handle=0；cap DOF 锁死（jp 恒 0.1097，施加 50N·m 也不动） |
| open_door 对照（预烘焙关节 + ArticulationCfg） | 仓库既有验证 | 可工作 —— 但仅限 USD 原生作者格式 |

**结论**: 需要"可分离/可旋转交互物"时不要依赖 PhysX joint，用独立 RigidObject + 几何监控。

## 3. 资产处理要点

### drink101（Extwin USD）
- 原始 `model_beverage13.usd`：E_body_52(瓶身, kinematic)+E_knob_58(瓶盖, dynamic)，**无 joint**
- `tools/prepare_drink101_usd.py`（gmr env pxr）:
  - `Sdf.CopySpec` 拆分单刚体（`UsdUtils.FlattenLayerStack(stage)` 收 stage）
  - 物理材质绑定 `material:binding:physics` 保留
  - cap 碰撞 mesh 上移（`UsdGeom.Xformable(mesh).AddTranslateOp().Set((0,0,0.021))`）

### SAM3D 桌子（GLB）
- `tools/convert_desk_glb_to_usd.py`（gmr env trimesh+pxr）:
  - trimesh 读 GLB 几何 + UV；glb PNG 纹理从 BIN chunk 提取（JSON/BIN 偏移易错，见脚本）
  - 顶点 y-up→z-up bake `(x,y,z)→(x,-z,y)`
  - Mesh + UsdUVTexture + PreviewSurface；RigidBodyAPI kinematic=True；`physics:approximation="sdf"`
  - **局部原点 = 桌中部**（bbox z∈[-0.5,+0.497]），桌面顶 = 原点 + 0.497

## 4. 位置计算（ipark）

```
floor_z = -1.124
桌原点 z = floor_z + 0.5          = -0.624
桌面顶 z = 桌原点 + 0.497          = -0.127
瓶身 z  = 桌面顶                   = -0.127
瓶盖 z  = 瓶身 + 0.2649            = +0.1379
桌中心   [-3.239, -3.2]（机器人前方 +Y 约 1.2m）
```
机器人 rot90° 前方是 +Y（ipark yaml 里 rot=[0.7071,0,0,0.7071]）。

## 5. 测试与验证

```bash
# 1) 离线资产校验（快，gmr env）
python tools/test_drink101_asset.py --assets-dir assets/objects/drink101
#   -> body/cap/artic/desk 全 PASS

# 2) headless 物理测试（遥操作前先跑）
rm -f /dev/shm/* 2>/dev/null
setsid /home/dreams/miniconda3/envs/unitree_sim_env_isaaclab5_0/bin/python -u \
  tools/test_drink101_twist_break.py --episodes 1 --plain > /tmp/t.log 2>&1 < /dev/null &
#   -> 瓶落桌面(body_z≈0.481≈桌面顶0.497) + 拧到2π + 取盖(cap_lift=0.78m) + 瓶身可拿起(0.037m) 全 PASS
```

测试脚本 `--plain` 用 GroundPlane 替代 ipark room（快且稳定）；位置在脚本内 hardcode（地面 z=-0.5，桌原点 z=0.0，桌面顶 0.497）。

## 6. 已知限制
- **多轮 reset 数值发散**: cap 被推远后 `env.reset()` 可能卡死（cap 拉 3.7m 后 reset 发散实测）。测试默认单轮；reset 需显式 `write_root_state_to_sim` 清零速度
- headless 需先清 `/dev/shm/*` + 磁盘充足（根盘 98% 时启动不稳）
- 遥操作真机集成（Pico 手抓瓶拧盖→取盖）尚未验证
