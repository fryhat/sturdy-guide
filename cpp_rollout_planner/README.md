# C++ rollout planner

独立于 `sp_vision_moving_target_visualizer_full_compare.py` 的路径搜索模块。

## 已实现

- `rollout_planner.cpp`
  - 每段 50 ms，云台恒加速度；
  - 前 3 段候选 `[-50, -100/3, -50/3, 0, 50/3, 100/3, 50] rad/s²`；
  - 后 3 段候选 `[-50, -25, 0, 25, 50] rad/s²`；
  - 全零层仍只展开最外侧 `±50`；
  - 每段用 `+15 ms` 实际 gun-yaw 查询 P_hit；
  - 返回最优路径的第一段 alpha 和总分。
- `rollout_planner.py`
  - ctypes 加载 DLL；
  - 输入 yaw/omega、yaw grid、6 条 P_hit curve；
  - 输出 alpha/score。

## 接口

```c
int sp_plan_rollout(
    double yaw,
    double omega,
    const double* grid,      // N yaw grid values
    int grid_size,
    const double* curves,    // 6 * N curve values
    int curve_count,
    double* out_alpha,
    double* out_score
);
```

## 编译

```powershell
powershell -File cpp_rollout_planner\build_rollout_planner.ps1
python cpp_rollout_planner\rollout_planner.py
```

## 尚未接入

- P_hit 曲线提供器：`p_hit_provider.py`
- 独立执行评估器：`rollout_benchmark.py`

## 当前原型状态

原型可运行，但初始 `valid/s` 很低；C++ 搜索本身通过自测。路径收益与实际 10 ms
开火阈值之间的失真仍存在，尚未达到可用基线。

独立评估器开火阈值当前为 `0.40`。实测：

```text
baseline 10ms:   10 s shots=4, valid=2, valid/s=0.200
smooth 10ms:     10 s shots=4, valid=2, valid/s=0.200
plan-point fire: 10 s shots=0, valid=0, valid/s=0.000
1ms interval:    10 s shots=41, valid=21, valid/s=2.100
```

## 已修复漏洞

- 修复 C++ 候选加速度区间 bug：原代码实际只产生
  `[-50, -37.5, -25, -12.5, 0]`，现恢复为
  `[-50, -25, 0, 25, 50] rad/s²`；
- 评估器云台 yaw 每步回绕，避免漂移到概率曲线域外；
- C++ 搜索内部也按角度周期回绕段间 yaw；
- 修复前 plan-point P 全为 0；修复后 plan/gate P 都进入高概率区。

修复后 baseline 10ms 10 s：

```text
shots=34, valid=12, valid_rate=35.29%, valid/s=1.200
30 s: shots=88, valid=33, valid_rate=37.50%, valid/s=1.100
```

## 速度上限

取自主程序 current 180 s 实测最大值：

```text
ROLLOUT_MAX_GIMBAL_OMEGA = 1.2617046468719202 rad/s (72.29°/s)
```

Python 执行与 C++ 搜索都使用同一限速积分。

## 未来曲线 fallback

当历史样本在高速/换向前被约束条件全部过滤时，未来段 P_hit 曲线不能直接为 0。
现使用零等效加速度的确定性场景作为 fallback，使 6 步穷举能看到后续命中窗口。

## 修复后 180 s 四 seed

候选加速度修复 + 无未来泄露因果预测 + 飞行时间 1 次迭代后：

```text
seed 20260903: shots=2686, valid=1318, 49.07%, 7.322 valid/s
seed 20260904: shots=2686, valid=1289, 47.99%, 7.161 valid/s
seed 20260905: shots=2686, valid=1332, 49.59%, 7.400 valid/s
seed 20260906: shots=2686, valid=1338, 49.81%, 7.433 valid/s
mean≈7.329
```

旧候选区间记录：

```text
seed 20260903: shots=1793, valid=1088, 60.68%, 6.044 valid/s
seed 20260904: shots=1793, valid=1094, 61.02%, 6.078 valid/s
seed 20260905: shots=1793, valid=1094, 61.02%, 6.078 valid/s
seed 20260906: shots=1793, valid=1082, 60.35%, 6.011 valid/s
mean=6.053, hit rate≈60.77%
```

## 随机 schedule 参与未来 P_hit

- `random_strategy_ab.py` 传入共享 `RandomEvasionSchedule`；
- `p_hit_provider.py` 对每个 50 ms 段先用真实 schedule 得到 impact 状态；
- C++ 新增 `sp_build_curve_from_impact()` 和
  `sp_hit_probability_from_impact_limit()`；
- schedule-aware 曲线使用随机策略的 `700°/s` 速度上限与 `300°/s²`
  加速度上限，不再依赖 C++ 内固定 `600°/s` + 固定 `1.0 s` 换向。

## 零概率剪枝

- 每层 5 个加速度候选概率均 `<1e-12` 时，只递归 `-50/+50` 两端；
- 中间 `-25/0/+25` 不再展开；
- 最优接口和最终回退 `0.0` 不变；
- debug 接口输出节点数、edge-only 层数、full 层数。

同 seed 180 s：

```text
剪枝前：1793 发, valid=1088, 60.68%, 6.044 valid/s, wall≈353.1s
剪枝后：1793 发, valid=1088, 60.68%, 6.044 valid/s, wall≈343.7s
```

## P_hit 曲线生成移入 C++

- 在线历史样本只作为 flat `(horizon, acceleration, count)` 传入 C++；
- C++ 按真实 target 1.0 s 换向/无反射动力学生成 6 段 impact state；
- C++ 生成 6 条卷积 P_hit 曲线并返回；
- C++ 输出与 Python 原实现最大误差约 `2.2e-16`。

同 seed 180 s：

```text
Python 曲线生成：6.044 valid/s, wall≈353.1s
C++ 曲线生成：  6.044 valid/s, wall≈20.9s
```

多 seed 180 s：

```text
seed 20260903: 6.044 valid/s
seed 20260904: 6.078 valid/s
seed 20260905: 6.078 valid/s
seed 20260906: 6.011 valid/s
mean=6.053
```

## 10 ms gate P 也移入 C++

- `sp_hit_probability_from_impact(...)` 负责单点卷积；
- Python 仍用已验证的 `armor_solution_for_yaw()` 选择 impact state；
- 在线 history 传入前按 horizon/加速度分桶压缩，避免随机策略下 4550 个
  support 全量传入。

原确定性 180 s：

```text
Python gate:  6.044 valid/s, wall≈20.9s
C++ gate:     6.039 valid/s, wall≈15.2s
```

随机策略 180 s：

```text
未压缩/旧 gate：valid/s≈0.089, wall≈140.8s
压缩+C++ gate： valid/s≈0.061, wall≈92.2s
```

## 随机闪避策略 A/B

策略参数：

```text
初始角速度: 350°/s
最大角速度: 700°/s（不反射、不触限）
加速度: 随机 ±(280..300)°/s²
正/负持续: 随机 1.0-1.2s，若会触限则提前换向
```

`random_strategy_ab.py` 让 current 与独立版共享同一 schedule。

策略 seed `20260904`、散布 seed `20260903`、180 s：

```text
current:       2601 发, valid=814,  31.30%, 4.522 valid/s
rollout_cpp[fusion]:    3090 发, valid=1604, 51.91%, 8.911 valid/s
rollout_cpp[rigorous]:  2856 发, valid=1256, 43.98%, 6.978 valid/s
```

默认 A/B 已彻底移除未来泄露，正式代码不保留 oracle 分支。
`fusion` 为默认配置；`rigorous` 使用常速名义 impact，不做二次外推。
修复后独立版已明显超过 current。

## 任意参数化闪避策略

`RandomEvasionSchedule` 已把下列参数实例化，不再使用模块级固定值：

```text
initial_omega / omega_limit
min_accel / max_accel
min_duration / max_duration
```

`p_hit_provider.py` 和 `rollout_benchmark.py` 会读取 schedule 实例的
`omega_limit`、`alpha_limit`，因此换参数时不需要改 C++ 或模型常量。
`random_strategy_ab.py` 已支持命令行直接设置：

主动路径不再保留“无 schedule 时按固定模型回退”的隐藏假设：

- `rollout_benchmark` 默认显式构造 `DeterministicEvasionSchedule`；
- `p_hit_provider` 要求目标 schedule 必须存在并暴露
  `omega_limit`、`alpha_limit`；
- 动态 C++ 接口对非正 `omega_limit` / `alpha_limit` 直接返回错误；
- 没有内部兜底的 `600°/s`、`280°/s²`、固定 `1.0 s` 换向或
  `700°/s / 300°/s²` 默认值。

```powershell
python cpp_rollout_planner\random_strategy_ab.py `
  --seconds 30 `
  --initial-omega-deg 220 --omega-limit-deg 520 `
  --min-accel-deg 210 --max-accel-deg 260 `
  --min-duration 0.60 --max-duration 1.30
```

默认 rollout 使用 `CausalEvasionPredictor`，只从当前目标状态外推；真实目标
运动仍由 schedule 推进。正式代码不再提供真实未来 schedule oracle 分支。

参数明显变化后的 30 s 对照：

```text
scheme A (220/520°/s, 210..260°/s², 0.60..1.30s):
  current:       455 发, valid=139, 30.55%, 4.633 valid/s
  rollout_cpp:   429 发, valid=216, 50.35%, 7.200 valid/s

scheme B (420/920°/s, 330..390°/s², 0.80..1.50s):
  current:       326 发, valid=86,  26.38%, 2.867 valid/s
  rollout_cpp:   448 发, valid=214, 47.77%, 7.133 valid/s
```

运行：

```powershell
python cpp_rollout_planner\random_strategy_ab.py
```
