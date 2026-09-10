# 移动靶概率规划：后续对话交接

更新时间：2026-09-04

## 1. 当前目标

保留现有方案作为基线，继续定位瓶颈并改进。所有改动必须以固定物理模型、统一随机种子和可复现实验为依据。

## 2. 工作区与主文件

- 工作区：`C:\Users\ASUS\.codex\worktrees\e8be\rm`
- 主程序：`sp_vision_moving_target_visualizer_full_compare.py`
- TinyMPC DLL：`sp_vision_tinympc.dll`
- 本交接文件：`HANDOFF_MOVING_TARGET_NEXT_V2.md`
- 旧交接：`HANDOFF_MOVING_TARGET.md`、`HANDOFF_MOVING_TARGET_NEXT.md`

不要清理、重置或覆盖无关用户文件；特别是 `programs/camera_monitor/include/camera/frame_source.hpp` 的用户修改。

## 3. 重要纠错：field_mpc 的真实状态

用户曾要求“改回原射速，引入场域 MPC”。当前代码没有实现独立的内部场域 MPC，也没有修改 TinyMPC DLL 的目标函数。

当前实际关系是：

```text
current              原始当前版基线
probabilistic        原始单峰概率方案
window_candidate     连续发射窗口候选评分实验
field_mpc            window_candidate 的别名，行为完全相同
```

`field_mpc` 不是独立算法名称；不能把它描述为已经完成的真正场域 MPC。若后续要实现真正场域 MPC，需要明确新的优化变量、轨迹概率目标和执行器动力学约束，并与现有基线做 A/B 对照。

## 4. 不得破坏的物理与比较语义

- 完整二维四装甲板模型；有限装甲线段命中判定。
- 实际发射方向使用发射时刻真实 `self.gimbal_yaw`。
- 延迟、弹速、飞行时间、散布和法向入射速度约束保持不变。
- 云台角加速度限制 `+/-50 rad/s^2`；当前无角速度上限和 jerk 约束。
- 靶车角加速度 `+/-280 deg/s^2`，角速度限制 `+/-600 deg/s`，每 1.5 s 换向。
- 基线使用 TinyMPC 原始 `plan.fire`（3 mrad）。
- 概率模式使用预测物理发射方向上的 `Pgun >= 0.50`，不叠加 3 mrad 门。
- 原射速：`SHOT_INTERVAL = 0.05 s`，即 20 发/秒。
- 主指标：`valid/s`；命中率和发射数为辅助指标。

## 5. 当前代码中的模式位置

主逻辑位于 `sp_vision_moving_target_visualizer_full_compare.py`：

- `update_planner()`：约第 939 行起，概率规划和候选轨迹处理。
- `probabilistic_yaw()`：约第 1054 行起，概率场、可达域和峰值选择。
- `field_mpc/window_candidate` 连续窗口评分：约第 974--1024 行。
- `fire_if_ready()`：约第 1192 行起，概率 gate 和 shot 建立。
- `advance_gimbal()`：约第 1219 行起，外层云台执行模型。
- `benchmark()`：约第 1734 行起，默认对比 `current/probabilistic/field_mpc`。
- CLI 模式选项：约第 1873 行。

`field_mpc` 和 `window_candidate` 在代码中都命中：

```python
if self.mode in ("field_mpc", "window_candidate"):
    raw_score = window_score(candidate_plan)
```

## 6. 已修复的真实问题

- 已发射样本的 `Pgun/Pactual` 使用同一 shot 快照统计。
- `Pgun-Pactual` 不再错误地使用 0/1 命中结果代替概率。
- 每个 shot 保存独立诊断字典，避免后续 tick 覆盖。
- blocked 分类改成互斥：`candidate_low`、`candidate_high_plan_low`、`plan_high_gun_low`、`all_high_but_blocked`。
- `Pcandidate` 不再错误复用 `Pmax`。
- 候选评分使用候选自身的 `predicted_state` 和 horizon。
- TinyMPC 中心索引已核对：`HALF_HORIZON=50` 表示当前时刻，上游 Planner 同样使用索引 50；不能改成索引 0。

## 7. 诊断字段

CSV：`moving_target_diagnostic_probabilistic.csv`

记录或汇总：

```text
Pmax, Pcandidate, Pref, Pplan, Pgun, Pactual
peak_yaw, plan_yaw, gun_yaw
gate_block, actual_result
ref_plan_launch_mrad, pred_actual_launch_mrad
accel_saturated
```

注意：`Pmax` 与最终 `Pref` 仍可能来自不同最终概率场，不能直接用 `mean(Pmax-Pref)` 判断峰值选择质量，除非先冻结同一最终场快照。

## 8. 已复现的基准结果

条件：距离 6 m、种子 `20260903`、原射速 20 发/秒、180 s、阈值 0.50。

```text
current:
  shots=2303, valid=863, valid_rate=37.47%, valid/s=4.794

probabilistic:
  shots=1253, valid=599, valid_rate=47.81%, valid/s=3.328

window_candidate:
  shots=1324, valid=693, valid_rate=52.34%, valid/s=3.850
```

同条件下 `field_mpc` 应与 `window_candidate` 完全一致；二者名称不同不是性能差异来源。

15 发/秒的旧实验结果不能与上述 20 发/秒结果直接比较。30 s 结果也不能直接当作 180 s 稳定 benchmark。

## 9. 当前瓶颈证据

180 s、20 发/秒、原始概率模式：

```text
mean(Pcandidate)=0.778
mean(Pplan)=0.501
mean(Pgun)=0.222       # 全控制周期均值
mean(Pref)=0.812       # 已发射样本均值
mean(Pgun-Pactual)=0.003  # 同一已发射样本集合
acceleration saturation=0%
|pred-actual| mean=0.62 mrad, p95=1.57 mrad
```

blocked 互斥分类：

```text
candidate_low              = 949
candidate_high_plan_low    = 4802
plan_high_gun_low          = 8636
all_high_but_blocked       = 0
```

结论：发射后 15 ms 预测链路可靠，长期加速度饱和未出现；损失主要发生在发射机会决策之前的候选/规划/执行链路。

## 10. 候选方案实验结论

### field_candidate

早期单点候选评分方案，曾在 30/60 s 中低于概率基线；不作为改进方案。

### lookahead_control

仅把外层控制目标前移到发射前 20 ms；降低部分 `Pref-Pgun`，但 valid/s 下降，不采用。

### window_candidate / field_mpc

连续窗口评分，180 s 达到 `3.850 valid/s`，相对 probabilistic 的 `3.328` 提升约 15.7%，但仍低于 current 的 `4.794`。这是当前唯一经过 180 s 验证的有效概率模式改进候选。

## 11. 推荐后续工作

1. 保持 `current` 和 `probabilistic` 不变作为基线。
2. 若实现真正场域 MPC，必须使用独立模式名和独立代码路径，不能继续把 `field_mpc` 作为 `window_candidate` 别名。
3. 新方案至少应优化连续发射窗口累计概率，并显式纳入云台执行动力学，而不是只优化一个瞬时峰值。
4. 每个候选必须用同一最终概率场快照，比较 `Pcandidate -> Pplan -> Pgun -> Pactual`。
5. 新方案先做 30/60 s 筛选，候选确认后再做 180 s；所有模式保持相同随机种子、射速、物理参数和命中规则。
6. 任何声称“超过基线”的结论都必须给出完整发射数、有效命中、命中率和 `valid/s`。

## 12. 常用验证命令

```powershell
Set-Location C:\Users\ASUS\.codex\worktrees\e8be\rm
python -m py_compile .\sp_vision_moving_target_visualizer_full_compare.py
python .\sp_vision_moving_target_visualizer_full_compare.py --self-test
python .\sp_vision_moving_target_visualizer_full_compare.py --benchmark
python .\sp_vision_moving_target_visualizer_full_compare.py --mode field_mpc
python .\sp_vision_moving_target_visualizer_full_compare.py --mode window_candidate
```

最后一次已验证：`py_compile`、`--self-test`、`git diff --check` 均通过。
