# 移动靶距离 AB V9

更新时间：2026-09-09

## 1. AB 设置

- 对照：`current` vs `rigorous + 60s warmup`
- 评估：`rollout_cpp[rigorous]` 先运行 60 s warmup，再运行 180 s 评估
- `current` 运行 180 s，无 warmup
- 距离：2 / 4 / 6 / 8 m
- seeds：20260903、20260904、20260905、20260906
- 统计：每个 seed 跑全部距离，按距离汇总四 seed 均值
- 脚本：`cpp_rollout_planner/deterministic_distance_ab.py`

## 2. valid/s 均值

| 距离 | current | rigorous+60s warmup | delta |
|---:|---:|---:|---:|
| 2 m | 4.111 | 8.225 | +4.114 |
| 4 m | 3.953 | 8.200 | +4.247 |
| 6 m | 4.200 | 7.706 | +3.506 |
| 8 m | 3.525 | 5.147 | +1.622 |

## 3. 聚合命中率

| 距离 | current | rigorous+60s warmup |
|---:|---:|---:|
| 2 m | 40.88% | 55.87% |
| 4 m | 31.50% | 54.87% |
| 6 m | 32.17% | 50.84% |
| 8 m | 26.83% | 37.89% |

## 4. 结论

在 2/4/6/8 m 四 seed AB 中，`rigorous + 60s warmup` 的 `valid/s` 均高于
`current`。收益在近中距离最大，8 m 时绝对收益和命中率都下降。

仍只代表当前确定性闪避模型和四 seed；尚未覆盖随机闪避、真实视觉噪声或
更长的不同 seed 集合。

## 5. 各 seed valid/s

距离 2 m：

```text
current:  4.106 4.122 4.072 4.144
warmup60: 8.283 8.094 8.344 8.178
```

距离 4 m：

```text
current:  4.022 4.011 3.817 3.961
warmup60: 8.156 8.072 8.433 8.139
```

距离 6 m：

```text
current:  4.150 4.150 4.300 4.200
warmup60: 7.794 7.617 7.683 7.728
```

距离 8 m：

```text
current:  3.417 3.644 3.483 3.556
warmup60: 5.222 5.017 5.156 5.194
```

## 6. 与 V7 的 6 m 差异说明

V7 的 `rigorous + 60s warmup` 6 m 四 seed
`7.894 / 7.872 / 7.761 / 8.044` 使用
`RandomEvasionSchedule(20260904)`，并通过替换
`sim.advance_optimized_evasion` 注入随机闪避轨迹，不是默认确定性 schedule。

本次 V9 使用的默认 `DeterministicEvasionSchedule` 在同一 60 s warmup 条件下
为 `7.794 / 7.617 / 7.683 / 7.728`，因此两组数据不可直接比较。

V7 的 `7.894` 组可复现命令片段：

```python
seed = 20260903
schedule = RandomEvasionSchedule(20260904)

def advance(angle, omega, direction, start_time, next_switch, dt):
    end = schedule.state_at(start_time + dt)
    return (
        end.angle,
        end.omega,
        1.0 if end.alpha >= 0.0 else -1.0,
        end.alpha,
        schedule.next_switch_at(start_time + dt),
    )

sim.advance_optimized_evasion = advance
RolloutBenchmark(
    seconds=180.0,
    seed=seed,
    dt=0.010,
    schedule=schedule,
    model_version=MODEL_VERSION_RIGOROUS,
    warmup_seconds=60.0,
).run()
```
