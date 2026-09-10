# 移动靶距离参数化 V8 交接

更新时间：2026-09-09

## 1. 本次改动

目标：允许 C++ rollout 和 benchmark 在非 6 m 距离下运行，同时保留默认
6 m 的原路径语义。

改动：

- C++ 概率曲线生成新增目标中心 `center_x/center_y`，装甲位置、装甲张角
  和散布角均按实际目标中心计算；
- 新增带中心参数的 `sp_build_curve_from_impact_center()` 与
  `sp_hit_probability_from_impact_limit_center()`；旧导出名保留为
  `(0, 6)` 兼容封装，不破坏原版 ABI；
- Python ctypes 包装新增默认 `center_x=0.0, center_y=6.0`，旧调用默认
  仍为 6 m；
- `p_hit_provider.build_segment_curves()` 不再写死
  `6.0 / BULLET_SPEED`，改用当前 `target.x/y` 的中心距离；
- `CausalEvasionPredictor` 与 `DeterministicEvasionSchedule` 支持目标中心；
- `rollout_benchmark.py` 新增 `--distance`，默认 `6.0`；
- `random_strategy_ab.py` 新增 `--distance`，当前基线与 rollout 同距离；
- 主程序 `--benchmark` 与 GUI 初始化新增 `--distance`；
- GUI 距离滑杆改距离时清空在线历史、概率样本、C++ rollout 决策缓存；
- UI 中“每 1.50 s”文案改为实际 `1.00 s`。

原版备份保留在：

```text
backup_distance_20260909/
```

## 2. 保留的 6 m 兼容

- C++ 默认调用仍是 `(center_x, center_y) = (0, 6)`；
- `rollout_benchmark.py` 默认 `--distance 6`；
- 主程序 benchmark 默认 `distance=6`；
- 非 6 m 诊断 CSV 使用距离后缀，默认 6 m CSV 文件名不变。

## 3. 验证

已验证：

```text
python -m py_compile ... （全部改动文件）
python cpp_rollout_planner\rollout_planner.py
python sp_vision_moving_target_visualizer_full_compare.py --self-test
```

C++ 概率曲线与同条件 Python 概率场逐点对比：

```text
distance=2.0/3.0/6.0/8.0/10.0 max_delta=0.0
```

短 benchmark：

```text
rollout_benchmark.py 10 --distance 6  -> 7.200 valid/s
rollout_benchmark.py 10 --distance 3  -> 6.600 valid/s
rollout_benchmark.py 10 --distance 8  -> 4.500 valid/s
full_compare rollout_cpp 5s distance 8 -> 3.400 valid/s
```

这些短结果只验证链路可运行和几何一致，不代表最终距离泛化性能结论。

## 4. 命令

```powershell
python cpp_rollout_planner\rollout_benchmark.py 180 `
  --model-version rigorous --distance 8

python cpp_rollout_planner\random_strategy_ab.py `
  --seconds 180 --distance 8

python sp_vision_moving_target_visualizer_full_compare.py `
  --benchmark --distance 8
```

## 5. 后续注意

- 尚未跑不同距离的 180 s 多 seed 性能结论；
- V7 中“6 段曲线复用同历史分布”和可变 history horizon 仍是未解决风险；
- `--mode` 当前只控制 GUI 模式，不会限制 `--benchmark` 的模式集合；
- 若未来让目标中心移动，C++ 曲线应进一步按每个 impact 状态传入中心，
  目前调用层传的是固定目标中心。
