# 足球紋理資產實作指南

## 一、現況分析

目前在 `move_football_g1_29dof_dex3_wholebody` 環境中，足球的定義位於：

```
tasks/common_scene/base_scene_football_cfg_wholebody.py
```

**當前實作**（約 62–96 行）：

- 使用 `sim_utils.SphereCfg` 生成** procedural 球體**
- 物理參數符合 FIFA 標準（半徑 0.11 m、質量 0.43 kg、restitution 約 0.75）
- 視覺材質：`PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.85))`，為單一灰白色
- **沒有 mesh 幾何體，也沒有貼圖 / 材質紋理**

因此，在 RGB 相機的第一人稱視角下，你看到的只是一個素色圓球，不具足球的黑白紋理，對視覺訓練的 domain 一致性不利。

---

## 二、目標方案：使用帶紋理的 USD 足球資產

要讓相機看到的足球更接近真實，需要改為使用**帶 mesh + 紋理的 USD**，並透過 `UsdFileCfg` 替換現有的 `SphereCfg`。

與專案中其他 USD 物體（如 `CubeBox_A03_21cm_PR_NVD_01_physics_rigid.usd`）一樣，需要：

- 視覺：網格 + 貼圖（含足球黑白紋理）
- 物理：rigid body、collision、mass（FIFA 足球規格）

---

## 三、資產取得來源

### GitHub 搜尋結果

針對 `soccer ball usd`、`soccer ball 3d asset` 等關鍵字搜尋後：

- 幾乎沒有專為 Isaac Sim / USD 的現成足球資產
- Isaac Sim / Omniverse 官方資產庫未發現標準足球資產
- 建議改從 OBJ/FBX/glTF 取得，再轉成 USD

### 建議的外部資源（可下載後轉換）

| 來源 | 說明 | 授權 |
|------|------|------|
| **RenderHub** | [Standard Soccer Ball - 3dinventions](https://renderhub.com/3dinventions/soccer-ball-free-asset) | Extended Use License（非商業與商業皆可） |
| **Sketchfab** | [3D Soccer Ball (OBJ) - ronildo.facanha](https://sketchfab.com/3d-models/3d-soccer-ball-obj-with-realistic-textures-f759c4370dbe4a98a36c2d73e30715a4) | 需確認各模型授權 |
| **Sketchfab** | [Soccer Ball - Mcx1m](https://sketchfab.com/3d-models/soccer-ball-11d2bc40b0cd466f9f6c62acd90a0c3d) | 同上 |

RenderHub 的免費足球模型較推薦，因為：

- OBJ 格式
- 含 color texture 與 normal map（2048x2048）
- 約 3,100 polygons，適合即時模擬
- Extended Use License 可用於模擬與訓練

---

## 四、OBJ/其他格式 → USD 轉換流程

### 方式 A：Isaac Lab Mesh Converter（指令列）

Isaac Lab 提供 `convert_mesh.py`，可將 OBJ/FBX/STL/glTF 轉成 USD，並加入物理屬性：

```bash
# 假設專案根目錄為 isaaclab_twist2_g1
./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /path/to/soccer_ball.obj \
  assets/objects/Props/soccer_ball/soccer_ball.usd \
  --make-instanceable \
  --collision-approximation convexDecomposition \
  --mass 0.43
```

- `--mass 0.43`：與 FIFA 足球規格一致  
- `--collision-approximation convexDecomposition`：用 convex decomposition 做碰撞，適合球體  
- `--make-instanceable`：適合多環境載入

**注意**：`convert_mesh.py` 可能位於 Isaac Lab 原始碼中，若專案未內建，需從 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 取得或自行實作等效轉換。

### 方式 B：Isaac Sim GUI 匯入

1. 開啟 Isaac Sim
2. **File → Import** 選擇 OBJ / FBX / glTF
3. Omniverse 會使用 ASSIMP 進行轉換，材質與紋理會轉成 NVIDIA MDL
4. 匯入後在 **Physics** 標籤加上 Rigid Body、Collision、Mass
5. **File → Export / Save As** 存成 `.usd`

### 方式 C：Omniverse Asset Converter（程式化）

Isaac Lab 的 `MeshConverter` 會呼叫 `omni.kit.asset_converter`，支援 OBJ、FBX、glTF 並自動處理材質與紋理。

---

## 五、專案配置建議

### 1. 目錄結構

建議新增：

```
assets/objects/Props/soccer_ball/
├── soccer_ball.usd          # 主資產（含 mesh、材質、物理）
└── textures/                # 若紋理需相對路徑
    ├── soccer_ball_diffuse.png
    └── soccer_ball_normal.png
```

### 2. 修改 `base_scene_football_cfg_wholebody.py`

將足球由 `SphereCfg` 改為 `UsdFileCfg`：

```python
import os
project_root = os.environ.get("PROJECT_ROOT")

# 將 object 的 spawn 從 SphereCfg 改為 UsdFileCfg
object = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[-3.3, -3.06, 0.95],
        rot=[1.0, 0.0, 0.0, 0.0],
    ),
    spawn=UsdFileCfg(
        usd_path=f"{project_root}/assets/objects/Props/soccer_ball/soccer_ball_physics_rigid.usd",
        scale=(1.0, 1.0, 1.0),  # 依模型原始尺寸調整，使直徑約 0.22 m
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            retain_accelerations=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="max",
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.75,
        ),
    ),
)
```

- `scale` 需依 OBJ/原始 USD 的尺寸調整，使視覺與物理直徑接近 0.22 m（半徑 0.11 m）
- 若 USD 已內含 physics material，可視情況省略或覆蓋 `physics_material`

### 3. 與現有 CubeBox 配置對照

可參考 `base_scene_pickplace_cylindercfg_wholebody.py` 中 box 的 `UsdFileCfg` 設定（約 84–106 行），確保 rigid body、collision、mass 等與你的需求一致。

---

## 六、檢查清單

- [ ] 下載 OBJ（或 FBX/glTF）足球模型與紋理檔
- [ ] 使用 `convert_mesh.py` 或 Isaac Sim GUI 轉成 USD
- [ ] 確認紋理在 USD 中正確顯示
- [ ] 設定 rigid body、collision、mass 符合 FIFA 規格
- [ ] 將 USD 放到 `assets/objects/Props/soccer_ball/`
- [ ] 修改 `base_scene_football_cfg_wholebody.py`，改用 `UsdFileCfg`
- [ ] 調整 `scale` 使視覺與物理尺寸正確
- [ ] 執行 `test_football_env.py` 驗證場景與物理
- [ ] 檢查 G1 RGB 相機第一人稱視角中足球外觀

---

## 七、相關連結

- [Isaac Lab - Importing a New Asset](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html)
- [Isaac Sim Mesh Formats / Asset Converter](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/formats.html)
- [Omniverse Asset Converter](https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html)
- RenderHub 足球模型: https://renderhub.com/3dinventions/soccer-ball-free-asset
- Sketchfab 足球模型: https://sketchfab.com/3d-models/soccer-ball-11d2bc40b0cd466f9f6c62acd90a0c3d
