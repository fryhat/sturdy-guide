# 移动靶 C++ 修复与随机策略 A/B 交接 V6

更新时间：2026-09-05

状态：暂定最终版，2026-09-05 冻结；后续如需修改应新建 V7 或显式解除冻结。

## 1. 本次修复

### 1.1 C++ 云台候选加速度 bug

`rollout_planner.cpp` 中候选 alpha 原计算只生成了负半轴：

```text
原实际候选：-50, -37.5, -25, -12.5, 0 rad/s²
修复后：    -50, -25, 0, 25, 50 rad/s²
```

这使云台在随机闪避下只能减速或保持，无法朝概率窗口回追；确定性模型也不是最优。

### 1.2 随机 schedule 参与未来 P_hit

- `random_strategy_ab.py` 现在把共享 `RandomEvasionSchedule` 传入
  `RolloutBenchmark`；
- `p_hit_provider.py` 对每个 50 ms 段使用真实 schedule 的 impact 状态；
- gate P 使用 schedule-aware armor solution 和 impact 状态；
- C++ 新增 `sp_build_curve_from_impact()`、
  `sp_hit_probability_from_impact_limit()`；
- schedule-aware 卷积支持动态速度上限 `700°/s`、加速度上限 `300°/s²`，
  不再固定为 `600°/s`、`280°/s²` 与固定 `1.0 s` 换向。

## 2. 180 s 结果

随机策略策略 seed `20260904`、散布 seed `20260903`：

```text
current:       2601 发, valid=814,  31.30%, 4.522 valid/s
rollout_cpp[fusion]:    3090 发, valid=1604, 51.91%, 8.911 valid/s
rollout_cpp[rigorous]:  2856 发, valid=1256, 43.98%, 6.978 valid/s
```

默认 A/B 已彻底移除未来泄露：rollout 不再查询真实 schedule 的未来状态，
正式代码中不再保留 oracle 开关。

修复前：

```text
rollout_cpp: 44-45 发, valid≈0.061-0.089 valid/s
```

飞行时间求解由 12 次硬循环改为 1 次。确定性 180 s seed
`20260903` 结果 `7.967 valid/s`，wall 约 `20.7s`；随机策略 30 s
结果 `7.833 valid/s`。

注意：这不是被验证的“精度收益”。6 个 seed、30 s 重复对照中，
`max_iter=1` 均值约 `7.261 valid/s`，`max_iter=4` 均值约
`7.300 valid/s`，说明 180 s 四 seed 的进步很可能是开火时点移动后
随机弹着序列造成的样本差异。保留 `max_iter=1` 的主要收益是少一次
飞行时间预测更新，以及去除额外自洽迭代；不能把成绩提高当作依据。

确定性模型四 seed（阈值 0.40、180 s）：

```text
20260903: 7.322 valid/s
20260904: 7.161 valid/s
20260905: 7.400 valid/s
20260906: 7.433 valid/s
mean≈7.329
```

旧候选区间四 seed 均值为 `6.053 valid/s`。

最新最终版、无未来泄露、四 seed 并行、每 seed 360s：

```text
20260903: 7.533 valid/s
20260904: 7.303 valid/s
20260905: 7.481 valid/s
20260906: 7.539 valid/s
mean≈7.464
```

四个实例并行 wall time 约 `237-239s`，即每个 360s 模拟约
`1.5-1.6` 倍实时。

二次外推消融已跑：当前 causal 加 C++ 历史加速度叠加的 A 版本
`7.956 valid/s`；改为常速名义 impact 的 B 版本 `6.878 valid/s`。
最终拆成两个版本：默认 `fusion`（保留二次外推式融合配置）；
`rigorous`（常速名义 impact + 历史加速度分布，不做二次外推）。
修复命中时刻对齐后的最终结果为 `fusion=7.878`、`rigorous=6.900`。

搜索候选试验：前 3 段 7 候选、后 3 段 5 候选，全零层仍只保留最外侧
`±50`。180s 结果为 `fusion=8.522`、`rigorous=7.428`。

历史压缩范围已改为按实际样本 min/max 自适应，不再固定
`[-280,280]°/s²`；修复后 `fusion=8.911`、`rigorous=6.978`。

全链路固定目标参数清理：C++ 已删除旧 `600°/s`、`280°/s²`、
固定 `1.0s` 换向目标推进及对应旧导出接口；正式路径只使用
Python 显式传入的动态 `omega_limit/alpha_limit`。

## 3. 关键文件

```text
cpp_rollout_planner/rollout_planner.cpp
cpp_rollout_planner/rollout_planner.py
cpp_rollout_planner/p_hit_provider.py
cpp_rollout_planner/rollout_benchmark.py
cpp_rollout_planner/random_strategy_ab.py
cpp_rollout_planner/causal_evasion_predictor.py
cpp_rollout_planner/README.md
cpp_rollout_planner/rollout_planner.dll
```

## 4. 验证

```powershell
python cpp_rollout_planner\rollout_planner.py
python cpp_rollout_planner\rollout_benchmark.py 180
python cpp_rollout_planner\random_strategy_ab.py
```

`rollout_planner.py` 自测、确定性 180 s、随机 A/B 180 s 均已通过。

### 4.1 作弊点清理

- `RolloutBenchmark` 与 `run_ab()` 默认 `future_known=False`；
- oracle 分支已从正式 rollout / A/B 代码中移除；
- `CausalEvasionPredictor` 不读取 `target.alpha` 真实值，改为从最近
  0.20s 角速度历史回归估计；
- 因果预测器不再从真实 schedule 复制 `omega_limit/alpha_limit`，改为
  使用已观测历史估计值；
- 正式 rollout 在模拟结束后继续结算在途子弹，不再把它们排除在分母外。
- current 与 rollout 各用独立的同 seed schedule，不再提前生成整条轨迹后
  共享同一个含未来状态的 schedule 对象；
- 预测器 reset 只接收 `angle/omega`，不再持有完整真实目标状态；
- 空概率不再用零加速度 fallback 强制造峰，无历史支持时直接保持低概率；
- 散布随机数改为按 10ms 时间逐帧生成，开火时读取对应时间 offset，不再按
  “第几发”延迟消费随机数。
- 可视化脚本的“预测曲线”也改用 causal predictor，不再用真实 schedule 生成
  oracle 曲线；
- 命中裁判默认只按 `shot.armor_index` 判定，不再遍历四块装甲后事后择优；
  短 benchmark 消融显示该改动未改变当前结果。
- GUI `rollout_cpp` 的未来曲线改为 `CausalEvasionPredictor`，不再传真实
  `direction/next_acceleration_switch` 做未来推演；原 oracle 辅助函数
  `rollout_advance_target/rollout_target_at_time/rollout_segment_curves`
  已删除。
- 独立 benchmark 命中解析改为按 `schedule.state_at(shot.impact_at)` 取真实
  撞击时刻目标状态，不再使用当前 10ms 步末状态。
- C++ `interpolate()` 对 yaw 网格外值不再返回边界概率，改为返回 0；
  fusion/rigorous 180s 结果保持不变。

命中厚度项未按 AI 建议直接删除：当前 2D 碰撞把
`ARMOR_DRAW_THICKNESS_M` 作为板厚参与判定，直接改为零会使 current 和
rollout 全部零命中，需要先重构二维碰撞模型，不能视为纯视觉参数。

## 5. 后续注意

当前随机 A/B 中 rollout 使用共享 schedule 的未来分支作为预测源，因此结果是
“已知随机未来轨迹”的能力上限。若需要真实在线落地，下一步应把 schedule
替换为可观测的当前段状态加随机未来模型，不能依赖提前知道随机数序列。

## 6. 任意参数化 schedule 支持

`RandomEvasionSchedule` 已改为实例化参数：

```text
initial_omega / omega_limit
min_accel / max_accel
min_duration / max_duration
```

`p_hit_provider` 与 `rollout_benchmark` 从 schedule 实例读取
`omega_limit`、`alpha_limit`，不再依赖默认 `700°/s / 300°/s²`。
`random_strategy_ab.py` 可通过命令行传入不同参数。

`random_strategy_ab.py` 现在把“真实目标运动”和“rollout 预测器”分开：

- 真实目标仍由目标 schedule 推进；
- 默认 rollout 使用 `CausalEvasionPredictor`，只从当前目标状态外推；
- 正式代码中不存在未来 schedule oracle 分支；
- 默认结果可用于判断真实随机闪避下的能力。

同时删除了主动路径中无 schedule 时的隐式固定模型回退：

- `rollout_benchmark` 默认显式构造 `DeterministicEvasionSchedule`；
- `p_hit_provider` 要求传入目标 schedule，否则直接报错；
- schedule 必须暴露 `omega_limit`、`alpha_limit`，不存在 600/280/1s
  或 700/300/随机时长的内部兜底；
- 动态 C++ 接口对非正限值返回错误，不再回退固定目标常量。

两组明显不同参数的 30 s 验证：

```text
scheme A (220/520°/s, 210..260°/s², 0.60..1.30s):
  current:       4.633 valid/s
  rollout_cpp:   7.200 valid/s

scheme B (420/920°/s, 330..390°/s², 0.80..1.50s):
  current:       2.867 valid/s
  rollout_cpp:   7.133 valid/s
```

命令示例：

```powershell
python cpp_rollout_planner\random_strategy_ab.py `
  --seconds 30 --initial-omega-deg 220 --omega-limit-deg 520 `
  --min-accel-deg 210 --max-accel-deg 260 `
  --min-duration 0.60 --max-duration 1.30
```

## 7. “只要分布可学就不比 current 差”的验证边界

仅使用在线历史分布、不把未来 schedule 传给 rollout 的 30 s 对照：

```text
默认随机参数: rollout_cpp≈7.100 valid/s > current≈4.400
scheme A:     rollout_cpp≈7.600 valid/s > current≈4.633
scheme B:     rollout_cpp≈0.367 valid/s < current≈2.867
```

结论：加速度分布能学习并不自动保证不低于 current。分布本身还需要与真实的
速度/加速度边界、当前段状态及未来传播一致；scheme B 的失败来自旧 C++ 路径
仍按 `600°/s` 与 `280°/s²` 过滤/推演，而不是分布学习本身。参数化 schedule
路径修复后才达到 `7.367 valid/s`。
