# 移动靶新模型：1.0 s 主动换向、无速度边界反射

更新时间：2026-09-05

## 1. 模型变化

- `EVADE_SWITCH_INTERVAL` 从 `1.5 s` 改为 `1.0 s`。
- 删除目标在 ±600°/s 处的自动反射。
- 到达 ±600°/s 后 clamp 保持，直到下一次主动换向。
- 初始角速度仍为 `+300°/s`，角加速度仍为 `±280°/s²`。
- 该模型变化全局生效；旧 V3 的 1.5 s 反射模型结果不再适用于当前代码。

代码位置：

```text
sp_vision_moving_target_visualizer_full_compare.py
EVADE_SWITCH_INTERVAL = 1.0
advance_full_acceleration()
advance_optimized_evasion()
```

## 2. 新模型下旧基线 180 s

条件：6 m、20 发/秒、阈值 0.50、四个 seed。

```text
current:
  20260903: 4.150 valid/s, 31.79%
  20260904: 4.150 valid/s, 31.79%
  20260905: 4.300 valid/s, 32.94%
  20260906: 4.200 valid/s, 32.17%
  mean=4.200

probabilistic:
  20260903: 3.300 valid/s, 49.25%
  20260904: 3.378 valid/s, 50.41%
  20260905: 3.589 valid/s, 53.57%
  20260906: 3.561 valid/s, 53.15%
  mean=3.457

window_candidate:
  20260903: 4.467 valid/s, 50.03%
  20260904: 4.433 valid/s, 49.66%
  20260905: 4.417 valid/s, 49.47%
  20260906: 4.367 valid/s, 48.91%
  mean=4.421

field_mpc:
  20260903: 4.261 valid/s, 50.39%
  20260904: 4.272 valid/s, 50.53%
  20260905: 4.039 valid/s, 47.77%
  20260906: 4.083 valid/s, 48.29%
  mean=4.164
```

旧模型中的 `window_candidate` 均值约 `4.138`；新模型下提升到约 `4.421`。

## 3. C++ 路径搜索目标

下一步在新模型上实现：

- 每 50 ms 枚举 `5^6` 条云台轨迹；
- 50 ms 段内云台恒加速度；
- 每段评分用该段 `+15 ms` 实际 gun-yaw；
- 每段使用自身 impact horizon；
- target 在路径内部按 1.0 s 主动换向、无反射动力学推进；
- C++ 返回第一个 50 ms 段命令；
- Python/基准保留 10 ms 开火资格检查。

未开始合入主程序前，新 C++ 模式暂不改变默认执行路径。

## 4. C++ 独立评估器三方向实验

独立模块：

```text
cpp_rollout_planner/
rollout_planner.cpp
rollout_planner.dll
p_hit_provider.py
rollout_benchmark.py
```

阈值 0.40、10 s：

```text
baseline 10ms:    shots=4,  valid=2, valid/s=0.200
smooth 10ms:      shots=4,  valid=2, valid/s=0.200
plan-point fire:  shots=0,  valid=0, valid/s=0.000
1ms interval:     shots=41, valid=21, valid/s=2.100
```

结论：1 ms 间隔明显改善开火机会，但仍低于主程序旧基线；平滑加速度没有变化，
规划点开火当前导致不发弹，需继续修曲线对齐。

## 5. 漏洞修复：yaw 未回绕

根因：

- 独立评估器 gimbal yaw 持续累积到 -98 rad；
- P_hit 曲线始终在车体有限弧 1.4-1.7 rad 附近；
- C++ 搜索内段间 yaw 也没有 wrap，导致所有路径得分接近 0，退回第一条命令。

修复：

- 评估器每步执行 `wrap_angle(gimbal_yaw)`；
- C++ 搜索对 +15 ms 和段间 yaw 都做周期回绕。

修复后：

```text
plan max P: 0.852
gate max P: 0.888
10 s: shots=34, valid=12, valid/s=1.200
30 s: shots=88, valid=33, valid/s=1.100
```

## 6. 后续修复：速度上限 + 未来曲线 fallback

- 云台角速度上限取自主程序 current 180 s 实测最大值：
  `1.2617046468719202 rad/s`
- Python 与 C++ 都使用同一限速积分。
- 当历史样本被高速约束全部过滤时，用零加速度确定性场景生成未来 P_hit，
  避免 6 步穷举因未来曲线为 0 而失效。

最终独立模型 180 s 四 seed：

```text
seed 20260903: 1793 发, valid=1088, 60.68%, 6.044 valid/s
seed 20260904: 1793 发, valid=1094, 61.02%, 6.078 valid/s
seed 20260905: 1793 发, valid=1094, 61.02%, 6.078 valid/s
seed 20260906: 1793 发, valid=1082, 60.35%, 6.011 valid/s
mean=6.053, hit rate≈60.77%
```

该结果高于新模型下 current 的约 4.200 和 window_candidate 的约 4.421。

## 7. C++ 零概率剪枝

- 每层 5 候选概率均 `<1e-12` 时只展开 `-50/+50`；
- 跳过 `-25/0/+25`；
- `sp_plan_rollout` 接口不变；
- 新增 `sp_plan_rollout_debug` 输出节点/edge-only/full 计数。

同 seed 180 s：

```text
剪枝前：6.044 valid/s, wall≈353.1s
剪枝后：6.044 valid/s, wall≈343.7s
```

## 8. P_hit 曲线生成移入 C++

- Python 只传入 target 状态、1.0 s 换向参数和 `(horizon, acceleration, count)` 在线历史；
- C++ 生成 6 段 impact state 和卷积 P_hit 曲线；
- Python 原曲线实现保留仅作对照。

同 seed 180 s：

```text
Python 曲线：6.044 valid/s, wall≈353.1s
C++ 曲线：  6.044 valid/s, wall≈20.9s
```

多 seed：

```text
20260903: 6.044 valid/s
20260904: 6.078 valid/s
20260905: 6.078 valid/s
20260906: 6.011 valid/s
mean=6.053
```

## 9. 随机闪避策略 A/B

策略：初始 350°/s、最大 700°/s 不触限、加速度随机 ±(280..300)°/s²、
正负持续随机 1.0-1.2s。

`random_strategy_ab.py` 中 current 与独立版共享同一目标 schedule。

180 s、策略 seed 20260904：

```text
current:     2601 发, valid=815, 31.33%, 4.528 valid/s
rollout_cpp:   45 发, valid=16, 35.56%, 0.089 valid/s
```

随机策略下 current 更稳定；独立版仍需改进对随机目标未来窗口的预测。

## 10. gate P 移入 C++ 与历史压缩

- C++ 新增 `sp_hit_probability_from_impact()`；
- Python 保留 armor 选择，P 卷积在 C++ 完成；
- 在线 history 按 horizon/加速度分桶压缩。

确定性 180 s：

```text
C++ gate: 6.039 valid/s, wall≈15.2s
```

随机策略 180 s：

```text
旧实现: valid/s≈0.089, wall≈140.8s
压缩+C++: valid/s≈0.061, wall≈92.2s
```
