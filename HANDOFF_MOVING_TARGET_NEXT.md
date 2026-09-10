# 移动靶概率规划：下个对话交接

更新时间：2026-09-04

## 1. 接续原则

这是当前状态快照。历史推导和历次 benchmark 保存在 `HANDOFF_MOVING_TARGET.md`，但其中较早的程序名、阈值和结果已经过时。

后续收到用户或 Opus 的改进建议时，必须先对照最新代码和 benchmark 验证。不要无条件接受，也不要为了提高数字而改变双方共同的物理模型或比较规则。

## 2. 当前工作区与唯一主程序

- 当前工作区：`C:\Users\ASUS\.codex\worktrees\e8be\rm`
- 主程序：`sp_vision_moving_target_visualizer_full_compare.py`
- TinyMPC：`sp_vision_tinympc.dll`
- 完整历史：`HANDOFF_MOVING_TARGET.md`
- Git 状态中大量文件未跟踪，且有用户对 `programs/camera_monitor/include/camera/frame_source.hpp` 的修改。不要清理、重置或覆盖无关内容。

常用命令：

```powershell
Set-Location C:\Users\ASUS\.codex\worktrees\e8be\rm
python -m py_compile .\sp_vision_moving_target_visualizer_full_compare.py
python .\sp_vision_moving_target_visualizer_full_compare.py --self-test
python .\sp_vision_moving_target_visualizer_full_compare.py --diagnostic
python .\sp_vision_moving_target_visualizer_full_compare.py --benchmark
python .\sp_vision_moving_target_visualizer_full_compare.py --mode probabilistic
```

`--benchmark` 当前固定运行 6 m、30 s 场景，不接受时长参数。

## 3. 不得破坏的比较语义

- 完整二维四装甲板模型，不能退化为单板、中心点或一维角度命中。
- 实际发射方向来自发射时刻的真实 `self.gimbal_yaw`，不是规划参考 yaw。
- 保留 15 ms 延迟、25 m/s 弹速、弹丸飞行时间、散布和装甲有限线段命中判定。
- 有效命中还要求装甲法向入射速度大于 12 m/s。
- 云台控制角加速度限制为 `+/-50 rad/s^2`；当前云台模型没有角速度上限和 jerk 约束。
- 靶车角加速度为 `+/-280 deg/s^2`，角速度限制为 `+/-600 deg/s`，每 1.5 s 主动换向。
- 基线只使用 TinyMPC 原始 `plan.fire`（3 mrad）。
- 概率模式不使用 3 mrad 门，使用预测物理发射方向上的 `Pgun >= 0.50`。
- 两种模式都受 `SHOT_INTERVAL=0.05 s` 限制。
- 主要指标是单位时间有效命中数 `valid/s`；命中率和发射数是辅助指标。

`Pref` 只是规划参考峰的诊断概率；`Pgun` 才是开火概率。禁止再次用 `Pref` 代替 `Pgun`。

## 4. 已完成的关键正确性修复

1. 概率缓存保存相对 baseline yaw 的 phase；缓存命中时，每 10 ms 按新绝对 yaw 重算单点概率。
2. 概率场对条件筛选后的全部经验样本做确定性平均，不再进行 64 次 bootstrap 抽样；`count` 和 `seed` 仅保留接口兼容。
3. 当前状态下不可行的历史加速度直接剔除并对剩余质量自然归一化，不再 clamp 到边界；无可行样本时概率场返回 0，不添加人为先验。
4. 概率峰选定后执行第二次真实 TinyMPC 求解，不平移旧解数组。
5. override 的中心索引已修为 `HALF_HORIZON + 1`。
6. `armor_solution_for_yaw()` 使装甲选择、径向距离、飞行时间和 impact state 自洽；改选装甲后会迭代重建撞击状态。
7. 概率模型 horizon 跟随最终 `plan.delay + plan.flight_time`，最多进行三次概率场/飞行时间收敛迭代。
8. 15 ms 延迟跨越 10 ms tick 时，在步内精确重建发射 yaw；impact 时刻也精确重建目标姿态。
9. 开火门在预测的物理 launch yaw 上计算 `Pgun`。审计误差：平均 `0.039 deg`、p95 `0.157 deg`、最大 `0.209 deg`。
10. 运动历史采用有界 `MOTION_HISTORY_WINDOW=160`；避免 80 点窗口不足一个 1.5 s 换向周期，也避免无限历史造成成本增长和旧状态稀释。
11. `scatter_sigma` 使用常量；`add_trajectory()` 已去掉循环内切片复制。
12. 可达域由错误的对称 `abs(omega)` 外包络改为有符号终点区间：

    ```text
    omega*t - 0.5*amax*t^2 <= delta_yaw <= omega*t + 0.5*amax*t^2
    ```

    coarse 候选、fine grid、平滑结果和缓存重建结果都受该区间检查或投影。
13. 概率 TinyMPC 最终重规划后，`Pref` 按最终 `target_yaw` 和最终 `predicted_state` 重新计算。

自测覆盖改选装甲后的 `flight_time * BULLET_SPEED == planned_range`、概率/基线开火门分离、缓存概率刷新、边界样本筛选和正向高速时的非对称可达域。

## 5. 已审查但不应照抄的建议

- “50 ms 缓存等于车体转过约 15 deg，所以 yaw 过期”：混淆车体姿态和视线角。实测 nominal yaw 在 50 ms 内平均变化 `1.55 deg`、p99 `3.37 deg`、最大 `3.49 deg`；缓存 phase 会叠加到最新 baseline yaw。10 ms 全场刷新 A/B 更慢且 `valid/s` 更低。
- “按历史样本自身速度判断未来加速度可行性”：条件语义错误。预测必须根据当前 omega 筛选未来可用的经验加速度。
- “0.72 平滑保留 72% 旧值”：代数错误。公式中旧值权重是 28%，新目标权重是 72%；取消平滑的 A/B 更差。
- “概率模式重新叠加 TinyMPC 3 mrad 门”：错误。该误差是优化解相对内部参考的误差，不是物理炮口误差。
- “以 gimbal 为中心的网格会漏峰”：错误。当前 241 点网格覆盖完整 `+/-180 deg`。
- “把 `phase_offset * progress**2` 改为线性”：没有充分依据。二次 phase 对应相对恒速度预测施加恒定有效角加速度后的位移；线性 phase 会在起点凭空引入速度偏移。它不是完整的随机路径模型，但不是可直接替换的 bug。
- “两个可达终点平滑后可能变成不可达”：对当前无速度/jerk 约束的标量双积分终点集合不成立，因为该集合是凸区间。代码仍在平滑后投影，以处理环绕和数值边界。
- “使用无限完整历史或短长窗口混合”：实验不支持。160 点为 `3.100 valid/s`，320 点为 `3.133` 但场计算约慢一倍；80/320 的两种混合分别只有 `2.867` 和 `2.667 valid/s`。
- “命中改为无限射线与装甲求交”：错误。当前仿真是在精确 impact 时刻比较弹丸点与有限装甲线段，不能改成无限射线。

## 6. 当前 benchmark 基线

固定 6 m、种子 20260903、30 s，最新正确性版本：

| 模式 | 有效命中/已判定 | 有效命中率 | valid/s |
|---|---:|---:|---:|
| current | 138/386 | 35.75% | 4.600 |
| probabilistic | 93/203 | 45.81% | 3.100 |

解释：概率模式单发质量更高，但 `Pgun >= 0.50` 放弃了太多射击，主要指标仍低于基线。旧版本 benchmark 使用过不同概率语义和时序，不能直接与当前数字比较。

## 7. 当前可达域的准确边界

当前流程确实使用可达域，但只用于概率峰的“撞击时刻终点 yaw”预筛选。对于当前无云台速度上限、无 jerk 约束、控制加速度可瞬时选取 `+/-50 rad/s^2` 的模型，上述有符号区间是展开角坐标上的精确终点集合。

它不是整段参考轨迹可跟踪性的严格证明，也不是 TinyMPC 全状态可达集。后续若引入云台速度上限、jerk、加速度连续性或机械边界，必须重新推导分段可达域，不能继续把当前公式称为精确模型。

## 8. 下一步建议

最高优先级不是继续改几何或增加门控，而是在最新正确性语义下重新标定 `PROBABILITY_FIRE_THRESHOLD`：

1. 扫描若干阈值，以 `valid/s` 为主指标，同时记录命中率和发射数。
2. 初筛使用多个固定随机种子；候选阈值至少运行 60 s，避免只拟合单个 30 s 周期。
3. 扫描期间保持几何、发射方向、散布、射频、执行器限制和基线规则完全不变。
4. 若阈值扫描仍无法超过基线，再分析概率参考提高了 `Pref` 却没有转化成 `Pgun` 的环节，例如执行器跟踪和峰切换，而不是回退已经修复的时序或概率语义。

当前没有证据支持直接改变 `progress**2`、恢复 3 mrad 交集门或使用无限历史。

## 9. 交接前验证状态

2026-09-04 已通过：

```text
python -m py_compile sp_vision_moving_target_visualizer_full_compare.py
python sp_vision_moving_target_visualizer_full_compare.py --self-test
git diff --check
```

最后一次 `--self-test` 输出：`self-test passed`。
