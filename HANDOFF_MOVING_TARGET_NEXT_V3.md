# 移动靶概率规划：后续对话交接 V3

更新时间：2026-09-04

## 1. 本次状态变化

V2 的纠错已经落实：`field_mpc` 不再只是 `window_candidate` 的别名。

当前模式关系：

```text
current              原始 current 基线
probabilistic        原始单峰概率基线
window_candidate     连续规划窗口候选评分实验
field_mpc            独立闭环场域候选模式（本次新增）
```

- `window_candidate` 仍使用旧代码路径和旧评分，保留作为已验证概率改进候选。
- `field_mpc` 已改为独立代码路径，不调用 `window_candidate` 的窗口评分。
- `current` 和 `probabilistic` 未改变。

## 2. 工作区与主文件

- 工作区：`C:\Users\ASUS\.codex\worktrees\e8be\rm`
- 主程序：`sp_vision_moving_target_visualizer_full_compare.py`
- TinyMPC DLL：`sp_vision_tinympc.dll`
- V2：`HANDOFF_MOVING_TARGET_NEXT_V2.md`
- 本交接：`HANDOFF_MOVING_TARGET_NEXT_V3.md`

不要清理或重置无关用户文件；特别是
`programs/camera_monitor/include/camera/frame_source.hpp` 的用户修改。

## 3. field_mpc 的独立实现

代码位置：

- `update_planner()`：约第 948 行起。
- 旧 `window_candidate/field_candidate` 候选分支：约第 981--1030 行。
- 新的 `field_mpc` 分支：约第 1031 行起。
- `_field_mpc_candidate_yaws()`：约第 1063 行起。
- `_simulate_plan_gun_yaws()`：约第 1083 行起。
- `_score_field_mpc_trajectory()`：约第 1117 行起。
- `_select_field_mpc()`：约第 1142 行起。

实现内容：

1. 对候选方向分别调用 TinyMPC，候选包括 nominal、当前概率峰以及当前峰附近正负 1/2/3 度。
2. 使用候选 `planned_yaw` 与现有加速度受限外层控制器，模拟未来 6 个 10 ms 控制步的云台执行。
3. 把模拟得到的未来枪口方向放到概率场上评分，取一个 50 ms 发射窗口内的最大概率、平均概率和可发射比例。
4. 加入候选切换代价与最小改善门限。
5. `fire_if_ready()` 仍按真实物理枪口方向 `Pgun >= 0.50` 开火；没有叠加 TinyMPC 3 mrad 门。

这是外层离散场域候选规划，没有修改 TinyMPC DLL 内部目标函数。

## 4. 180 秒基准结果

条件：距离 6 m、种子 `20260903`、20 发/秒、180 s、概率阈值 0.50。

```text
current:
  shots=2303, valid=863, valid_rate=37.47%, valid/s=4.794

probabilistic:
  shots=1253, valid=599, valid_rate=47.81%, valid/s=3.328

window_candidate:
  shots=1324, valid=693, valid_rate=52.34%, valid/s=3.850

field_mpc:
  shots=1350, valid=662, valid_rate=49.04%, valid/s=3.678
```

结论：

- 新 `field_mpc` 相对 `probabilistic` 提升约 `10.5%`。
- 新 `field_mpc` 仍低于 `window_candidate` 和 `current`。
- 本次实现证明它确实独立，但还不能声称“真实场域 MPC 已超过当前最优候选”。
- 因此正式改进候选仍保持 `window_candidate`；`field_mpc` 作为独立实验分支保留。

## 5. field_mpc 180 秒诊断

```text
all steps:
  mean(Pcandidate)=0.778
  mean(Pplan)=0.410
  mean(Pgun)=0.242

resolved shots:
  mean(Pmax)=0.719
  mean(Pref)=0.614
  mean(Pgun)=0.695
  mean(Pactual)=0.682
  mean(Pgun-Pactual)=0.013

angles:
  |ref-plan| mean=11.65 mrad, p95=28.57 mrad
  |pred-actual| mean=0.74 mrad, p95=2.77 mrad
  acceleration saturation=0%
```

blocked 互斥分类：

```text
candidate_low           = 875
candidate_high_plan_low = 5823
plan_high_gun_low       = 6965
all_high_but_blocked    = 0
```

观察：`field_mpc` 的候选/执行闭环明显减少 `plan_high_gun_low`，但增加了
`candidate_high_plan_low`。这说明“模拟执行后的候选评分”改变的是损失分配，尚未整体超过
旧 `window_candidate` 的规划窗口评分。

## 6. 后续可做方向

1. 不要删除 `window_candidate`；它仍是当前经过 180 s 验证的概率改进候选。
2. 不要把新 `field_mpc` 描述成已超过 `window_candidate`。
3. 若继续做真正场域 MPC，先实现随未来发射时刻变化的 `P_hit(yaw,t)`，再对多个 50 ms 射击块优化。
4. 候选评分需要同时覆盖：
   - 当前/下一可发射时刻的实际枪口概率；
   - 连续未来射击块的累积机会；
   - 概率场与真实命中率的校准差。
5. 所有模式保持同一随机种子、射速、物理模型、延迟、散布和命中规则。
6. 任何“超过基线”结论必须附完整发射数、有效命中、命中率和 `valid/s`。

## 7. 常用验证命令

```powershell
Set-Location C:\Users\ASUS\.codex\worktrees\e8be\rm
python -m py_compile .\sp_vision_moving_target_visualizer_full_compare.py
python .\sp_vision_moving_target_visualizer_full_compare.py --self-test
python .\sp_vision_moving_target_visualizer_full_compare.py --benchmark
python .\sp_vision_moving_target_visualizer_full_compare.py --mode field_mpc
python .\sp_vision_moving_target_visualizer_full_compare.py --mode window_candidate
```

本次验证已通过：`py_compile`、`--self-test`、`git diff --check`。

## 8. AI 静态审查后的工程修正

收到一份静态审查后，先做了一轮修正，并重跑 180 s 对照。

保留并验证的修复：

- `MovingTargetVisualizer.__init__()` 不再在 `current_plan` 初始化前读取
  `current_plan.target_yaw`，GUI 构造已验证。
- `reset_simulation()` 先重建 `current_plan`，再初始化概率峰，GUI 重置已验证。
- 不同概率模式不再覆盖同一个诊断 CSV，日志改为
  `moving_target_diagnostic_<mode>.csv`。
- `benchmark()` 的手工 runner 初始化补齐 `accel_saturated_steps`，避免后续直接读取属性时出错。
- `fire_if_ready()` 创建待发射 `Shot` 时使用门限预测 yaw，不再先存当前 yaw；
  真正发射时仍由 `resolve_shots()` 按实际 yaw 覆盖。

曾在 30 s 中尝试、但经 180 s 复跑后撤销的改动：

- 概率场改用装甲板自身距离；
- 散布改为固定弹道路径上的角度偏差；
- `field_mpc` 只统计前 5 个控制步的可开火比例；
- `field_mpc` 候选切换成本改为统一相对当前执行分支。
- `field_mpc` rollout 改用完整 `planned_acceleration` 序列。

这些改动方向分别有价值，但在当前候选评分结构下会使已稳定的模式回退：

- 装甲板距离/散布改动曾把 `window_candidate` 从 3.850 压到 3.639；
- field 评分改动曾把 `field_mpc` 从 3.678 压到 3.200。
- 使用完整 `planned_acceleration` 的 field rollout 曾把 `field_mpc` 压到 3.244。

因此当前代码恢复为：概率场半径、散布终点、field 评分均与第 4 节 180 s
结果一致。

最终 180 s 复跑：

```text
current:         2303 发, valid=863, 37.47%, 4.794 valid/s
probabilistic:   1253 发, valid=599, 47.81%, 3.328 valid/s
window_candidate: 1324 发, valid=693, 52.34%, 3.850 valid/s
field_mpc:       1350 发, valid=662, 49.04%, 3.678 valid/s
```

`window_candidate` 没有巨大下滑；之前 3.639 的结果来自被撤销的实验性概率场/
散布修改。恢复后重新复跑为 3.850，与 V2 一致。

尚未修改的问题：

- 历史加速度过滤与真实反射动力学的完全对齐；
- `P_hit(yaw,t)` 多发射时刻概率场；
- `probabilistic_model.horizon` 共享可变状态；
- `field_mpc` 候选 rollout 的滚动重规划；
- 发射门限预测是否纳入下一次控制重规划；
- 静态/斜入射概率场与真实碰撞率的一致性测试。

这些需要建模或性能重构，必须在不破坏当前 180 s 基线的前提下单独实验。

## 9. rollout_mpc 独立文件实测

收到 `sp_vision_moving_target_visualizer_rollout_mpc.py` 后，已复制到同一 worktree
并分别跑 10/30/180 s。该文件通过 `py_compile` 和自带 `--self-test`。

```text
rollout_mpc 10 s: shots=67, valid=37, 55.22%, 3.700 valid/s
rollout_mpc 30 s: shots=219, valid=103, 47.03%, 3.433 valid/s
rollout_mpc 180 s: shots=1344, valid=654, 48.66%, 3.633 valid/s
```

随后按审查意见在 rollout 文件中加入 pending-shot 队列、真实离膛角结算、
覆盖 15 ms 延迟的展开 tick，并按实际反射/换向动力学推进场景；同时修正
距离变化时 rollout 状态未重置和 UI 模式标签不一致。

修正后 rollout 180 s：

```text
rollout_mpc 180 s: shots=1347, valid=655, 48.63%, 3.639 valid/s
```

再按第二轮 AI 意见改为场景内真实 `evaluate_shot()` 命中结算：待发弹丸在
离膛时保存真实轨迹，到 impact 时用该场景推进出的目标状态与固定散点积分结算，
不再用概率场自我评分。

第二轮修正后 rollout 180 s：

```text
rollout_mpc 180 s: shots=1290, valid=660, 51.16%, 3.667 valid/s
```

相对第一轮 pending-shot 修正有提升，但仍低于 window_candidate/field_mpc。

多 seed 180 s（真实场景结算版）：

```text
seed 20260903: shots=1290, valid=660, 51.16%, 3.667 valid/s
seed 20260904: shots=1290, valid=638, 49.46%, 3.544 valid/s
seed 20260905: shots=1290, valid=637, 49.38%, 3.539 valid/s
seed 20260906: shots=1290, valid=640, 49.61%, 3.556 valid/s
mean: 3.577 valid/s, sample std ≈ 0.061
```

散射 seed 改变实际命中数，但不改变门控通过和发射数量。

当前 180 s 对照：

```text
window_candidate  3.850 valid/s
field_mpc         3.678 valid/s
rollout_mpc       3.667 valid/s（单 seed，见多 seed 表）
probabilistic     3.328 valid/s
current           4.794 valid/s
```

结论：即使改为场景内真实命中结算，rollout_mpc 仍未超过
window_candidate 或 field_mpc，未合入主程序。后续若继续，应重点处理
“第一候选分支立即被贪心策略覆盖”和候选策略身份保持问题。

## 10. current/window_candidate 多 seed 180 s

当前主程序 `full_compare.py` 已支持 benchmark 指定 seed。以下为四个 seed、
180 s、20 发/秒、阈值 0.50 的 current/window_candidate 结果：

```text
current:
  seed 20260903: 2303 发, valid=863, 37.47%, 4.794 valid/s
  seed 20260904: 2303 发, valid=832, 36.13%, 4.622 valid/s
  seed 20260905: 2303 发, valid=860, 37.34%, 4.778 valid/s
  seed 20260906: 2303 发, valid=850, 36.91%, 4.722 valid/s
  mean=4.729, std≈0.078

window_candidate:
  seed 20260903: 1324 发, valid=693, 52.34%, 3.850 valid/s
  seed 20260904: 1324 发, valid=686, 51.81%, 3.811 valid/s
  seed 20260905: 1324 发, valid=640, 48.34%, 3.556 valid/s
  seed 20260906: 1324 发, valid=669, 50.53%, 3.717 valid/s
  mean=3.734, std≈0.131
```

window_candidate 的发射数仍为固定 1324；多 seed 差异来自弹着散布。当前单 seed
（20260903）的 3.850 偏高，四个 seed 均值约为 3.734。rollout_mpc 四 seed 均值约
3.576，仍低于 window_candidate 多 seed 均值。

## 11. window_candidate 缓存化加速

主程序新增窗口候选缓存：

```text
WINDOW_CANDIDATE_REFRESH_ONLY = True
WINDOW_CANDIDATE_CACHE_PHASE = True
```

窗口候选只在概率场每 50 ms 刷新时重新枚举 nominal/当前峰/±2 度；刷新之间保持
选中的 phase，不再每个 10 ms tick 重复求解全部候选轨迹。开火、命中、射速、
延迟和散布规则不变。

加速后的 window_candidate 180 s：

```text
seed 20260903: shots=1431, valid=703, 49.13%, 3.906 valid/s
seed 20260904: shots=1431, valid=724, 50.59%, 4.022 valid/s
seed 20260905: shots=1431, valid=709, 49.55%, 3.939 valid/s
seed 20260906: shots=1431, valid=726, 50.73%, 4.033 valid/s
mean=3.975, std≈0.052
```

单进程 180 s wall time：

```text
旧候选路径实测约 478 s
缓存化路径实测约 359 s
速度提升约 25%
```

注意：V3 上文第 4 节的 window_candidate 3.850 是旧候选路径结果；当前代码默认
使用缓存化路径。若需旧行为，可临时把两个常量改为 False。

## 12. 在线历史统计器

概率模型已改为在线累计：

- 每个实际状态点只在获得未来观测时计算一次等效加速度；
- 同一 horizon/加速度支持点按计数永久累计到 `sample_entries`；
- 不再每次概率场刷新时清空样本并重新遍历最近窗口；
- `hit_probability_field()` 直接按压缩支持集和计数计算经验分布。

旧 `MOTION_HISTORY_WINDOW` 扫描路径已从概率场刷新中移除。当前代码默认不重置
`sample_entries`，因此窗口外历史不会丢失。

在线历史 + 缓存化 window_candidate 180 s：

```text
seed 20260903: shots=1387, valid=725, 52.27%, 4.028 valid/s
seed 20260904: shots=1387, valid=727, 52.42%, 4.039 valid/s
seed 20260905: shots=1387, valid=730, 52.63%, 4.056 valid/s
seed 20260906: shots=1387, valid=728, 52.49%, 4.044 valid/s
mean=4.042, std≈0.012
```

对照同样缓存化但滚动 160 点历史的版本：

```text
seed 20260903: 3.906 valid/s
seed 20260904: 4.022 valid/s
seed 20260905: 3.939 valid/s
seed 20260906: 4.033 valid/s
mean=3.975, std≈0.052
```

在线历史版本进一步提升稳定性，并显著减少发射数/门控结果随 50 ms 重建造成的波动。

## 13. 卷积曲线缓存 + 有限车体弧扫描

后续两项改进已实现：

- 概率场刷新时缓存最终卷积后的 P_hit(yaw) 曲线；
- 后续命中概率查询直接使用缓存曲线，不再保留未卷积的原始角度分布；
- 取消以云台为中心的全圆 ±180° 扫描；
- 曲线定义域覆盖固定车体中心在射手视角形成的有限弧段，并加入散布/50 ms
  移动余量；
- 只有概率峰筛选时才与云台可达区间取交集；
- 缓存曲线按 nominal 装甲 yaw 做相位运输；
- 不再使用旧的 1.5° fine search，直接使用 0.10° 曲线网格。
- MPC 的中间轨迹仍不受车体弧限制。

当前 180 s 多 seed（修正曲线域/相位运输后的最终版本）：

```text
seed 20260903: shots=1420, valid=732, 51.55%, 4.067 valid/s
seed 20260904: shots=1420, valid=755, 53.17%, 4.194 valid/s
seed 20260905: shots=1420, valid=736, 51.83%, 4.089 valid/s
seed 20260906: shots=1420, valid=756, 53.24%, 4.200 valid/s
mean=4.138 valid/s, std≈0.069
mean hit rate=52.45%
```

运行速度：

```text
曲线缓存 + 有限弧 + nominal 相位运输 180 s 单进程实测约 209 s
此前在线历史缓存化路径约 330 s
```

修正前曲线域错误和缺少 nominal 相位运输时曾降到约 3.440；修正后不仅恢复
在线历史水平，还提高到约 4.138 valid/s，命中率约 52.45%，同时比旧的
330 s 路径更快。

## 14. 开火阈值 0.44 实验

将 `PROBABILITY_FIRE_THRESHOLD` 从 0.50 临时改为 0.44 后，180 s 四个 seed：

```text
mean valid/s = 4.122
mean hit rate = 49.50%
```

对照 0.50：

```text
mean valid/s = 4.138
mean hit rate = 52.45%
```

0.44 提高发射数但降低命中率，无净收益；已恢复默认 0.50。
