# 移动靶概率/独立 C++ 规划交接 V5

更新时间：2026-09-05

## 1. 当前工作区

- 主程序：`sp_vision_moving_target_visualizer_full_compare.py`
- 独立 C++ 模块：`cpp_rollout_planner/`
- 历史交接：V2/V3/V4

不要清理以下内容：

- `programs/camera_monitor/include/camera/frame_source.hpp`
- 各 `HANDOFF_*.md`
- `cpp_rollout_planner/` 下 DLL、CSV、PNG

## 2. 模型变化

目标主动换向当前为：

```text
EVADE_SWITCH_INTERVAL = 1.0 s
初始角速度 = 300°/s
最大角速度 = 600°/s
角加速度 = ±280°/s²
无速度边界反射；若触限则 clamp 保持
```

该模型已经全局写入主程序和 rollout 文件。

## 3. 独立 C++ 模块现状

目录：

```text
cpp_rollout_planner/
rollout_planner.cpp
rollout_planner.dll
rollout_planner.py
p_hit_provider.py
rollout_benchmark.py
random_evasion_schedule.py
random_strategy_ab.py
rollout_problem.html/png
```

C++ 已实现：

- 6 段、`5^6` 全枚举；
- 每段 50 ms 恒加速度；
- `+15 ms` 实际 gun-yaw 查询；
- 每段自身 impact horizon；
- 云台速度上限：
  ```text
  1.2617046468719202 rad/s
  ```
- 零概率剪枝：5 候选全 `<1e-12` 时只展开 `-50/+50`；
- P_hit 曲线生成在 C++；
- gate P 单点卷积在 C++；
- 在线 history 传入 C++ 前按 horizon/acceleration 分桶压缩。

## 4. 确定性闪避下成绩

独立 C++ 版阈值 0.40、180 s、四 seed：

```text
20260903: 1793 发, valid=1088, 60.68%, 6.044 valid/s
20260904: 1793 发, valid=1094, 61.02%, 6.078 valid/s
20260905: 1793 发, valid=1094, 61.02%, 6.078 valid/s
20260906: 1793 发, valid=1082, 60.35%, 6.011 valid/s
mean=6.053
```

主程序新模型基线：

```text
current            ≈ 4.200 valid/s
probabilistic      ≈ 3.457 valid/s
window_candidate   ≈ 4.421 valid/s
field_mpc          ≈ 4.164 valid/s
```

## 5. 最近的 C++ 性能改动

历史版本耗时：

```text
P_hit 曲线在 Python：180 s wall≈353 s
P_hit 曲线移入 C++： 180 s wall≈20.9 s
gate P 移入 C++：    180 s wall≈15.2 s
```

对应确定性 180 s：

```text
C++ gate + 压缩后：valid/s≈6.039
```

原 Python 实现约 `6.044`，差异来自分桶压缩。

## 6. 随机闪避策略 A/B

新增策略：

```text
初始角速度 = 350°/s
最大角速度 = 700°/s
不允许反射/触限
角加速度 = 随机 ±(280..300)°/s²
正/负持续 = 随机 1.0-1.2 s
若会触限则提前换向
```

`random_strategy_ab.py` 让 current 与独立版使用同一个随机 schedule。

180 s、策略 seed 20260904、散布 seed 20260903：

```text
current:      2601 发, valid=815, 31.33%, 4.528 valid/s
rollout_cpp:    44 发, valid=11, 25.00%, 0.061 valid/s
```

随机策略下独立版大幅下降，主要原因：

- 独立版的 P_hit/路径设计仍依赖较可预测的换向模式；
- 随机未来窗口下 gate P 大多退化为 0；
- 阈值调低不能修复退化。

## 7. 阈值统计结果

原确定性独立版 18000 gate 样本：

```text
P >= 0.50 占比 = 31.34%
```

随机策略 4 seed、各 30 s：

```text
20260904: 0.33%
20260905: 2.00%
20260906: 1.77%
20260907: 1.67%
平均 ≈ 1.44%
```

随机策略 P 分布大量为 0，无法用与 31.34% 相同的比例找到 >0 阈值。

## 8. 下一步建议

1. 先修复随机策略下 gate P 全 0：需要随机策略 schedule 参与 C++ 未来 P_hit 生成。
2. 将 `random_evasion_schedule.py` 的随机未来分支与 `p_hit_provider` 结合。
3. 不再依赖“把原阈值比例直接搬过去”。
4. 保留确定性策略四 seed 与随机策略 A/B 结果。
5. 后续验证命令：

```powershell
python cpp_rollout_planner\rollout_planner.py
python cpp_rollout_planner\rollout_benchmark.py 180
python cpp_rollout_planner\random_strategy_ab.py
```
