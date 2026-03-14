# Fabric GPU vs CPU 說明

## 什麼是 Fabric？

Isaac Sim / Isaac Lab 使用 **Fabric** 作為模擬資料傳輸層，負責將物理狀態（pose、velocity 等）在 USD 與 Python/PhysX 之間傳遞。

- **Fabric GPU**：資料存於 GPU 記憶體，適合多環境並行模擬，效能較高
- **Fabric CPU**：資料存於 CPU，透過 USD 更新，較慢但與部分除錯/編輯工具相容

## 如何讓 Fabric 使用 GPU？

Fabric 使用 GPU 或 CPU 由 **sim.device** 決定：

| sim.device | Fabric 模式 |
|------------|-------------|
| `cuda` 或 `cuda:0` | Fabric GPU |
| `cpu` | Fabric CPU |

### 測試腳本

`test_move_football_scene_env.py` 會將 `--device` 同步到 `env_cfg.sim.device`：

```bash
# 使用 GPU（Fabric GPU）
python scripts/test_move_football_scene_env.py --device cuda

# 使用 CPU（Fabric CPU）
python scripts/test_move_football_scene_env.py --device cpu
```

### sim_main.py

`sim_main.py` 透過 `parse_env_cfg(..., device=args_cli.device)` 設定，因此：

```bash
python sim_main.py --task Isaac-Move-Football-G129-Dex3-Wholebody --device cuda
```

會正確使用 Fabric GPU。

## 材質與場景載入

- **材質貼圖**：由 Omniverse RTX 渲染器在 GPU 上處理
- **場景 USD**：由 Isaac Sim 載入，與 sim.device 無直接關係
- **物理模擬**：PhysX 依 sim.device 在 GPU 或 CPU 上執行

若看到「Fabric CPU」而預期使用 GPU，請確認啟動時有傳入 `--device cuda`，且環境能偵測到 NVIDIA GPU。
