# 移动靶规划 V7 交接

更新时间：2026-09-06

## 1. 当前主模型与基线

- 主模型：`rigorous`
  - 默认 `model_version=MODEL_VERSION_RIGOROUS`；
  - 不做 causal 预测加速度直接叠加；
  - 历史加速度分布来自在线样本。
- 对照基线：`rigorous + --zero-accel-history`
  - 将全部历史加速度强制为 `0`；
  - 用于判断“真实历史分布”是否带来正向收益。
- 保留 `fusion` 作为另一实现，但不再作为主模型。

## 2. 关键代码状态

正式路径已经不再使用未来 schedule：

- 预测器只用当前 `angle/omega` 和在线估计；
- C++ 不再包含旧固定 `600°/s`、`280°/s²`、固定 `1.0s` 换向接口；
- 概率曲线、gate、建弹都使用显式动态 `omega_limit/alpha_limit`；
- 历史压缩已改为按实际样本范围自适应，不再固定 `[-280,280]`；
- 命中解析按 `shot.impact_at` 真实目标状态；
- 空概率不 fallback；
- C++ yaw 网格外概率返回 0；
- 主 GUI `rollout_cpp` 也已改为 causal 预测。

## 3. 最近实验

### 3.1 pending history horizon 修复：失败并回滚

尝试将历史结算 horizon 从可变 `model.horizon` 改为固定 horizon：

```text
rigorous 180s：
回滚前 6.978 valid/s
尝试后 3.539 valid/s
```

负向，已完整回滚。不要继续用“固定 history horizon + 放宽 C++ horizon filter”的旧思路。

### 3.2 warmup 控制状态隔离：保留

warmup 已改为只推进目标、生成散布、积累运动历史，不再执行 replan/gimbal 控制。
热身结束后重置 gimbal/command/射击时间，再开始评估。

单 seed：

```text
旧 warmup：7.761 valid/s
隔离 warmup：7.772 valid/s
```

主指标正向，暂时保留。

### 3.3 zero-accel vs rigorous warmup，4 seed

zero-accel baseline（180s）：

```text
20260903: 7.300
20260904: 7.300
20260905: 7.322
20260906: 7.622
mean≈7.386
```

rigorous + 60s warmup（随后 180s）：

```text
20260903: 7.772
20260904: 7.756
20260905: 7.622
20260906: 7.961
mean≈7.778
```

差异：

```text
valid_per_s：+0.392
valid_rate：约 -0.05pp
```

### 3.4 开火阈值 0.50，4 seed AB

阈值改为 `ROLLOUT_FIRE_THRESHOLD=0.50` 后：

注意：本节的 rigorous + 60s warmup 与 zero-accel baseline 均使用
`RandomEvasionSchedule(20260904)`，并通过替换
`sim.advance_optimized_evasion` 注入；不是默认确定性 schedule。

zero-accel baseline：

```text
20260903: shots=2855, valid=1308, 45.81%, 7.267 valid/s
20260904: shots=2855, valid=1316, 46.09%, 7.311 valid/s
20260905: shots=2855, valid=1339, 46.90%, 7.439 valid/s
20260906: shots=2855, valid=1365, 47.81%, 7.583 valid/s
mean valid_per_s≈7.400
mean valid_rate≈46.65%
```

rigorous + 60s warmup：

```text
20260903: shots=2856, valid=1421, 49.75%, 7.894 valid/s
20260904: shots=2856, valid=1417, 49.61%, 7.872 valid/s
20260905: shots=2856, valid=1397, 48.91%, 7.761 valid/s
20260906: shots=2856, valid=1448, 50.70%, 8.044 valid/s
mean valid_per_s≈7.893
mean valid_rate≈49.74%
```

差异：

```text
valid_per_s：+0.493
valid_rate：+3.09pp
```

## 4. 尚未解决/需谨慎的问题

1. 6 段概率曲线仍复用同一个 `0.255s` 历史分布，并叠加到不同未来 impact 状态。
   这是当前最严重的时间语义风险，但不要盲目小改。
2. pending history 可变 horizon 仍是已知 bug；之前简单固定 horizon 修复失败，
   需要重新设计历史采样/分布，而不是直接放宽过滤。
3. 观测仍是无噪声真值 `angle/omega`，不是真实视觉链路。
4. fusion/rigorous 共用代码，但主默认已切换为 rigorous。

## 5. 运行命令

```powershell
# rigorous 主模型 180s
python cpp_rollout_planner\rollout_benchmark.py 180 --model-version rigorous

# zero-accel baseline
python cpp_rollout_planner\rollout_benchmark.py 180 --model-version rigorous --zero-accel-history

# 60s warmup + 180s eval
python cpp_rollout_planner\rollout_benchmark.py 180 --model-version rigorous --warmup 60

# fusion 保留版
python cpp_rollout_planner\rollout_benchmark.py 180 --model-version fusion
```

## 6. 代码目录

```text
sp_vision_moving_target_visualizer_full_compare.py
cpp_rollout_planner/rollout_benchmark.py
cpp_rollout_planner/random_strategy_ab.py
cpp_rollout_planner/p_hit_provider.py
cpp_rollout_planner/causal_evasion_predictor.py
cpp_rollout_planner/rollout_planner.cpp
cpp_rollout_planner/rollout_planner.py
```
