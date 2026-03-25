# MEMORY

## 当前项目创建场景流程（建议顺序）

1. 明确任务入口与场景绑定关系：先确认 `run.sh` 的 `TASK_NAME`，再定位对应 `*_env_cfg.py`，最后确认引用的 `base_scene_*.py`。  
2. 在 scene 文件中先定义模块级位置常量，再定义 `packing_table/object/container` 等资产。  
3. 先决定资产类型：会被抓取/推动用 `RigidObjectCfg`；仅作为放置目标或障碍用 `AssetBaseCfg`。  
4. 绑定位姿时使用“桌子绝对位置 + 相对偏移”模式，避免硬编码散落在多个对象内。  
5. 静态资产放置到桌面时，`z` 偏移要按“桌面高度 + 资产底部到原点偏移”设置，避免出现悬空或穿透。  
6. 资产路径先在 `assets/objects/...` 下核对真实文件名和后缀，再写入 `UsdFileCfg`。  
7. 每次改 scene 后先做最小校验：`python -m py_compile <scene_cfg.py>`，再跑 `run.sh` 看端到端。  
8. 日志排查先分阶段：`scene creation` 阶段问题优先查配置/资产；`simulation start` 变慢优先查渲染链路。

## 场景设计参考文件（本会话使用）

- `isaaclab_twist2_g1/run.sh`：任务入口与启动参数（相机、深度、分辨率、seed）。  
- `isaaclab_twist2_g1/sim_main.py`：环境创建与启动流程。  
- `isaaclab_twist2_g1/tasks/g1_tasks/move_pickplace_doubledesk_g1_29dof_dex3_wholebody/move_pickplace_doubledesk_g1_29dof_dex3_hw_env_cfg.py`：当前任务 env 配置与 scene 引用。  
- `isaaclab_twist2_g1/tasks/common_scene/base_scene_pickplace_doubledesk.py`：当前场景资产定义（桌子、圆柱物体、容器）。  
- `isaaclab_twist2_g1/tasks/common_scene/base_scene_pickplace_cylindercfg.py`：圆柱资产参数参考。  
- `isaaclab_twist2_g1/tasks/common_scene/base_scene_pickplace_cylindercfg_wholebody.py`：`RigidObjectCfg + UsdFileCfg` 的可用样例。  
- `isaaclab_twist2_g1/fetch_assets.sh`：资产来源与拉取方式参考。  

## 本会话关键错误与规避策略

- `SceneCfg` 类体里不要放普通列表/字典参数（如 `packing_table_l_pos = [...]`）。资产管理器会把类级字段当作资产配置解析，可能报 `Unknown asset config type`。位置常量放到模块级更安全。
- 使用 `RigidObjectCfg` 前先确认 USD 资产包含可解析的刚体能力（`RigidBodyAPI`）。如果只包含碰撞而无刚体，环境会报 `Failed to find a rigid body when resolving ...`。
- 容器/篮子类资产若仅用于“放置目标”，优先用 `AssetBaseCfg + UsdFileCfg`（静态碰撞体）。只有需要被抓取/推动时才强制走 `RigidObjectCfg`。
- 资产命名和路径要严格核对，颜色/变体后缀不同会导致引用错误。示例：`SM_Container_C04_Black_01_physics.usd` 与其他颜色是不同文件。
- 避免引用 `._*` 这类 macOS 资源分叉文件，只使用正常 `.usd` 文件。
- 启动慢不一定是新加物体导致。若 `scene creation` 时间正常但 `simulation start` 很慢，优先排查渲染链路（相机分辨率、深度图、RTX/DLSS 初始化）而非几何体本身。
- 静态容器出现悬空时，优先下调相对桌面 `z` 偏移并按资产高度校准，不要只改 `x/y`。

## 本会话可复用排查流程

1. 先看失败阶段：区分 `scene creation` 失败还是 `simulation start` 变慢。  
2. 若是对象解析失败：先查资产路径是否存在，再判断是否应使用 `RigidObjectCfg`。  
3. 若资产不可作为刚体：改为 `AssetBaseCfg` 或更换为已验证 `*_physics_rigid.usd`。  
4. 任何场景改动后先做最小校验：`python -m py_compile <scene_cfg.py>`。  
5. 再运行任务做端到端验证，按日志分离“致命错误”和“非致命 warning”。  

## 当前任务相关实践约定

- `object_l` 与 `packing_table_l` 的位置保持绑定（桌面相对偏移）。
- `container_r` 与 `packing_table_r` 的位置保持绑定（桌面相对偏移）。
- 以后新增“可交互资产”时，先确认资产物理 API 能力，再决定 `RigidObjectCfg` 或 `AssetBaseCfg`。
