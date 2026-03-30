# Common Env Config YAML

`sim_main.py` 支持通过 `--env_config_yaml` 在 `gym.make(...)` 之前覆盖环境配置。

最简单写法：

```yaml
overrides:
  sim:
    dt: 0.005
  decimation: 4
```

也支持按路由或任务附加覆盖：

```yaml
overrides:
  sim:
    dt: 0.005
  decimation: 4

routes:
  sonic:
    decimation: 8

tasks:
  Isaac-Move-Football-Single-G129-Dex3-Wholebody:
    episode_length_s: 30.0
```

合并顺序：
`overrides` -> `routes.<gmt_backend|action_source>` -> `tasks.<task_name>` -> `route_tasks.<route>.<task>`

当前会自动同步的派生字段：
- `scene.contact_forces.update_period <- sim.dt`
- `sim.render_interval <- decimation`
