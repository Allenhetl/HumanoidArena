# 拳擊沙袋場景說明

## 一、資產轉換

拳擊沙袋 OBJ+MTL 需先轉為帶物理的 USD：

```bash
cd isaaclab_twist2_g1
conda activate unitree_sim_env
python scripts/convert_boxing_bag_assets.py --headless --device cuda
```

轉換後會產生 `assets/boxing_bag/boxing_bag_physics.usd`。

## 二、物理設定

| 項目 | 設定 | 說明 |
|------|------|------|
| **是否固定** | 否（動態剛體） | `kinematic_enabled=False`，受擊會晃動，更貼近遙操作情境 |
| **質量** | 35 kg | 接近真實沙袋，避免被擊飛 |
| **Restitution** | 0.15 | 輕微彈性，減少過度彈跳 |
| **Damping** | 0.5 | 有助晃動收斂 |
| **形變** | 無 | PhysX 剛體無形變；若需軟體形變需用 PhysX 5 deformable |

若希望沙袋完全固定（不晃動），可將 `kinematic_enabled=True`，RGB 畫面仍可採集。

## 三、可視化與遙操作

```bash
# 可視化
python scripts/test_boxing_bag_scene_env.py --enable_cameras --device cuda

# 遙操作（與 football 相同參數）
python sim_main.py --device cuda --enable_cameras \
  --task Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody \
  --robot_type g129 --enable_dex3_dds
```

## 四、佈局調整

在 `tasks/common_scene/base_scene_boxing_bag_cfg_wholebody.py` 中：

- `ROBOT_INIT_X`, `ROBOT_INIT_Y`, `ROBOT_INIT_Z`：機器人初始位置
- `BAG_DISTANCE`：沙袋與機器人的距離（沿機器人朝向）
