# 從 OBJ 轉換帶物理的 USD（足球與球門）

## 問題原因

您用 OBJ 轉成的 USD 只包含**幾何和紋理**，沒有**物理資訊**（RigidBodyAPI、碰撞、質量）。

- OBJ 本身只有 mesh，沒有物理
- 轉換工具若未主動添加物理，產生的 USD 會是純視覺資產
- Isaac Lab 的 `RigidObjectCfg` 要求 prim 有 RigidBodyAPI 才能參與物理

## 目前狀態

| 資產   | 紋理       | 物理       |
|--------|------------|------------|
| 足球   | ❌ 灰色球體 | ✅ 正常     |
| 球門   | ✅ 有紋理   | ❌ 無碰撞，球會穿過 |

## 解決方案：用 Isaac Lab Mesh Converter 重新轉換 OBJ

Isaac Lab 的 `convert_mesh.py` 會：
- 保留 OBJ 的材質與紋理
- 添加 Rigid Body、Collision、Mass

### 1. 準備原始 OBJ 檔案

請將足球與球門的 OBJ 及其紋理放在專案下，例如：

```
assets/football/
  soccer_ball.obj
  soccer_ball.mtl      # 若有
  textures/            # 若有貼圖
assets/football_net/
  football_goal.obj
  football_goal.mtl
  textures/
```

### 2. 使用 convert_mesh 轉換

在 Isaac Sim 環境中執行（需與 Isaac Lab 路徑一致）：

```bash
# 進入 Isaac Lab 專案目錄（依您的路徑調整）
cd /home/hcl4070-1/Desktop/taowen/projects/IsaacLab

# 足球：加入物理 (mass 0.43 kg，convex 碰撞)
./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /path/to/soccer_ball.obj \
  /path/to/isaaclab_twist2_g1/assets/football/soccer_ball_physics.usd \
  --make-instanceable \
  --collision-approximation convexDecomposition \
  --mass 0.43 \
  --headless

# 球門：靜態碰撞 (mass 不設或設 0，kinematic 在場景配置中指定)
./isaaclab.sh -p scripts/tools/convert_mesh.py \
  /path/to/football_goal.obj \
  /path/to/isaaclab_twist2_g1/assets/football_net/football_goal_physics.usd \
  --make-instanceable \
  --collision-approximation convexDecomposition \
  --headless
```

若使用 isaaclab_twist2_g1 專案根目錄：

```bash
cd /home/hcl4070-1/Desktop/zikang/HumanoidArena/isaaclab_twist2_g1

# 假設 OBJ 在 assets 下
./isaaclab.sh -p /home/hcl4070-1/Desktop/taowen/projects/IsaacLab/scripts/tools/convert_mesh.py \
  assets/football/soccer_ball.obj \
  assets/football/soccer_ball_physics.usd \
  --make-instanceable \
  --collision-approximation convexDecomposition \
  --mass 0.43 \
  --headless
```

### 3. 調整場景配置

轉出帶物理的 USD 後，在 `base_scene_football_cfg_wholebody.py` 中：

- 足球：`object` 的 `spawn` 改為 `UsdFileCfg` 指向 `soccer_ball_physics.usd`
- 球門：`goal_net` 的 `spawn` 改為 `UsdFileCfg` 指向 `football_goal_physics.usd`，並加上 `rigid_props`（kinematic=True）

## 若無法使用 convert_mesh

可在 Isaac Sim GUI 中手動添加物理：

1. File → Open，開啟現有 USD
2. 選取 mesh → Physics → Add Collision Approximation → Convex Decomposition
3. Physics → Add → Rigid Body，設定質量（足球約 0.43 kg）
4. File → Save As 另存為新 USD

## 建議流程

1. 提供足球、球門的 OBJ（及 MTL、貼圖）
2. 用 `convert_mesh.py` 轉出帶物理的 USD
3. 更新 `base_scene_football_cfg_wholebody.py` 使用新 USD
4. 執行測試腳本驗證紋理與物理行為
