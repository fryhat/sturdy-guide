from __future__ import annotations

import html
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sp_vision_moving_target_visualizer_full_compare as sim

from causal_evasion_predictor import CausalEvasionPredictor
from p_hit_provider import (
    build_segment_curves,
    curve_grid,
)
from random_evasion_schedule import RandomEvasionSchedule

STRATEGY_SEED = 20260904
DT = 0.010
HORIZON = sim.LOW_SPEED_DELAY + 6.0 / sim.BULLET_SPEED
OUT_DIR = Path(__file__).resolve().parent / "random_strategy_diagnostics"
EDGE = Path(
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)

FONT = "Microsoft YaHei UI, Segoe UI, sans-serif"
ARMOR_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#d97706")
_SCHEDULE: RandomEvasionSchedule | None = None


@dataclass
class SnapshotData:
    time: float
    state: object
    entries: dict[tuple[float, float], int]
    grid: list[float]
    curves: list[list[float]]
    trace_plan: list[dict[str, float | bool]]
    trace_gate: list[dict[str, float | bool]]


def as_target(state) -> sim.TargetState:
    return sim.TargetState(
        0.0,
        6.0,
        0.0,
        0.0,
        state.angle,
        state.omega,
        state.alpha,
    )


def armor_normal_speed(target: sim.TargetState, yaw: float, armor) -> float:
    return (
        max(
            0.0,
            -math.cos(
                sim.angle_error(
                    yaw,
                    math.atan2(armor.normal_y, armor.normal_x),
                )
            ),
        )
        * sim.BULLET_SPEED
    )


def valid_armor_centers(target: sim.TargetState) -> list[tuple[int, float, float]]:
    output: list[tuple[int, float, float]] = []
    for armor in sim.armor_poses(target):
        yaw = math.atan2(armor.y, armor.x)
        if armor_normal_speed(target, yaw, armor) > sim.MIN_NORMAL_HIT_SPEED:
            output.append((armor.index, yaw, math.degrees(sim.angle_error(yaw, math.pi / 2))))
    return output


def nearest_valid_center(target: sim.TargetState) -> float | None:
    centers = valid_armor_centers(target)
    return min(centers, key=lambda item: abs(item[1]))[2] if centers else None


def build_entries_until(schedule: RandomEvasionSchedule, end_time: float) -> dict[tuple[float, float], int]:
    entries: dict[tuple[float, float], int] = {}
    cursor = 0.0
    while cursor + HORIZON <= end_time + 1e-12:
        start = schedule.state_at(cursor)
        finish = schedule.state_at(cursor + HORIZON)
        acceleration = 2.0 * (
            finish.angle - start.angle - start.omega * HORIZON
        ) / (HORIZON * HORIZON)
        key = (round(HORIZON, 3), round(acceleration, 4))
        entries[key] = entries.get(key, 0) + 1
        cursor += DT
    return entries


def make_snapshot_data(
    schedule: RandomEvasionSchedule,
    selected: float,
) -> SnapshotData:
    snapshot = schedule.state_at(selected)
    target = as_target(snapshot)
    grid = curve_grid(target)
    entries = build_entries_until(schedule, selected)
    direction = 1.0 if snapshot.alpha >= 0.0 else -1.0
    next_switch = schedule.next_switch_at(selected)
    model = sim.ProbabilisticAimModel(HORIZON)
    model.sample_entries = entries
    history_time = max(0.0, selected - 0.20)
    previous = schedule.state_at(history_time)
    acceleration_estimate = (
        snapshot.omega - previous.omega
    ) / max(selected - history_time, 1e-9)
    predictor = CausalEvasionPredictor()
    predictor.reset(
        snapshot.angle,
        snapshot.omega,
        selected,
        max(abs(snapshot.omega), abs(previous.omega), 1e-9),
        max(abs(acceleration_estimate), 1e-9),
        acceleration_estimate,
    )
    curves = build_segment_curves(
        model,
        target,
        direction,
        next_switch,
        selected,
        grid,
        schedule=predictor,
    )
    return SnapshotData(selected, snapshot, entries, grid, curves, [], [])


def color_for_time(fraction: float) -> str:
    stops = [
        (0.0, "#60a5fa"),
        (0.35, "#2563eb"),
        (0.7, "#9333ea"),
        (1.0, "#db2777"),
    ]
    if fraction <= 0.0:
        return stops[0][1]
    if fraction >= 1.0:
        return stops[-1][1]
    for index in range(len(stops) - 1):
        start, color_start = stops[index]
        end, color_end = stops[index + 1]
        if start <= fraction <= end:
            local = (fraction - start) / max(end - start, 1e-9)

            def channel(a: str, b: str) -> int:
                return round(int(a, 16) + (int(b, 16) - int(a, 16)) * local)

            return (
                f"#{channel(color_start[1:3], color_end[1:3]):02x}"
                f"{channel(color_start[3:5], color_end[3:5]):02x}"
                f"{channel(color_start[5:7], color_end[5:7]):02x}"
            )
    return stops[-1][1]


def local_body_panel(data: SnapshotData, x: float, y: float, width: float, height: float) -> str:
    xspan = 0.72
    yspan = 0.86
    margin = 22
    px0 = x + margin
    py0 = y + margin
    pw = width - 2 * margin
    ph = height - 2 * margin
    sx = pw / xspan
    sy = ph / yspan
    center_x = 0.0
    center_y = 6.0

    def tx(world_x: float) -> float:
        return px0 + (world_x - (center_x - xspan / 2)) * sx

    def ty(world_y: float) -> float:
        return py0 + (center_y + yspan / 2 - world_y) * sy

    def body_points(state, future: float) -> str:
        points: list[str] = []
        for local_x, local_y in (
            (-sim.BODY_WIDTH_M / 2, -sim.BODY_LENGTH_M / 2),
            (sim.BODY_WIDTH_M / 2, -sim.BODY_LENGTH_M / 2),
            (sim.BODY_WIDTH_M / 2, sim.BODY_LENGTH_M / 2),
            (-sim.BODY_WIDTH_M / 2, sim.BODY_LENGTH_M / 2),
        ):
            dx, dy = sim.rotate(local_x, local_y, state.angle)
            points.append(f"{tx(state.x + dx):.1f},{ty(state.y + dy):.1f}")
        return " ".join(points)

    parts: list[str] = []
    parts.append(
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{width:.0f}" height="{height:.0f}" '
        'rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{x + width / 2:.0f}" y="{y + 21:.0f}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="15" font-weight="700" fill="#0f172a">'
        "装甲实况：当前车体与未来装甲中心轨迹</text>"
    )
    for grid_y in (5.55, 6.0, 6.45):
        parts.append(
            f'<line x1="{px0:.0f}" y1="{ty(grid_y):.0f}" x2="{px0 + pw:.0f}" '
            f'y2="{ty(grid_y):.0f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
    for grid_x in (-0.2, 0.0, 0.2):
        parts.append(
            f'<line x1="{tx(grid_x):.0f}" y1="{py0:.0f}" x2="{tx(grid_x):.0f}" '
            f'y2="{py0 + ph:.0f}" stroke="#e2e8f0" stroke-width="1"/>'
        )

    future_points: list[list[tuple[float, float, int, float]]] = [[] for _ in range(4)]
    for index in range(4):
        for tau_index in range(0, 61):
            tau = tau_index * 0.01
            future = as_target(data.state if tau <= 0 else data.state)
            if tau > 1e-12:
                future = as_target(schedule_at(data, tau))
            armor = sim.armor_poses(future)[index]
            future_points[index].append(
                (armor.x, armor.y, tau_index, tau)
            )

    for index, points in enumerate(future_points):
        if len(points) < 2:
            continue
        color = ARMOR_COLORS[index]
        path = []
        for world_x, world_y, _tau_index, _tau in points:
            path.append(f"{tx(world_x):.1f},{ty(world_y):.1f}")
        parts.append(
            f'<polyline points="{" ".join(path)}" fill="none" stroke="{color}" '
            f'stroke-width="2" opacity="0.55" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    future_state = as_target(schedule_at(data, 0.3))
    parts.append(
        f'<polygon points="{body_points(future_state, 0.3)}" fill="none" '
        'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.7"/>'
    )

    current = as_target(data.state)
    parts.append(
        f'<polygon points="{body_points(current, 0)}" fill="#dbeafe" '
        'stroke="#1e40af" stroke-width="2.5"/>'
    )
    for armor in sim.armor_poses(current):
        half = sim.ARMOR_WIDTH_M / 2
        ax, ay = armor.x - armor.tangent_x * half, armor.y - armor.tangent_y * half
        bx, by = armor.x + armor.tangent_x * half, armor.y + armor.tangent_y * half
        parts.append(
            f'<line x1="{tx(ax):.1f}" y1="{ty(ay):.1f}" x2="{tx(bx):.1f}" '
            f'y2="{ty(by):.1f}" stroke="{ARMOR_COLORS[armor.index]}" stroke-width="7" '
            'stroke-linecap="round"/>'
        )

    for segment in range(6):
        impact_time = data.time + segment * 0.05 + HORIZON
        impact = as_target(schedule_at(data, impact_time - data.time))
        for armor in sim.armor_poses(impact):
            marker_color = ARMOR_COLORS[armor.index]
            parts.append(
                f'<circle cx="{tx(armor.x):.1f}" cy="{ty(armor.y):.1f}" r="3.2" '
                f'fill="{marker_color}" stroke="white" stroke-width="1" opacity="0.85"/>'
            )

    legend_x = x + width - 220
    for index, name in enumerate(("后侧板", "右侧板", "前侧板", "左侧板")):
        ly = y + 50 + index * 21
        parts.append(
            f'<line x1="{legend_x:.0f}" y1="{ly:.0f}" x2="{legend_x + 25:.0f}" '
            f'y2="{ly:.0f}" stroke="{ARMOR_COLORS[index]}" stroke-width="5"/>'
        )
        parts.append(
            f'<text x="{legend_x + 33:.0f}" y="{ly + 4:.0f}" font-family="{FONT}" '
            f'font-size="12" fill="#475569">{name}</text>'
        )
    parts.append(
        f'<text x="{legend_x:.0f}" y="{y + 44:.0f}" font-family="{FONT}" '
        'font-size="12" font-weight="700" fill="#0f172a">当前板/轨迹</text>'
    )
    return "\n".join(parts)


def schedule_at(data: SnapshotData, offset: float):
    # Kept as a separate helper so a visualization script can later swap in a
    # schedule-aware model without changing all drawing functions.
    global _SCHEDULE
    if _SCHEDULE is None:
        raise RuntimeError("schedule has not been initialised")
    return _SCHEDULE.state_at(data.time + offset)


def p_curve_panel(data: SnapshotData, x: float, y: float, width: float, height: float) -> str:
    margin = 42
    px0 = x + 64
    py0 = y + margin
    pw = width - 64 - margin
    ph = height - margin - 34
    min_deg = -6.0
    max_deg = 6.0

    def x_for(deg: float) -> float:
        return px0 + (deg - min_deg) / (max_deg - min_deg) * pw

    def y_for(prob: float) -> float:
        return py0 + ph * (1.0 - prob)

    parts: list[str] = []
    parts.append(
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{width:.0f}" height="{height:.0f}" '
        'rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{x + width / 2:.0f}" y="{y + 25:.0f}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="16" font-weight="700" fill="#0f172a">'
        "同一时刻 C++ 生成 P_hit(yaw) vs 真实装甲可瞄中心</text>"
    )
    for tick in range(int(min_deg), int(max_deg) + 1, 1):
        px = x_for(tick)
        parts.append(
            f'<line x1="{px:.1f}" y1="{py0:.1f}" x2="{px:.1f}" y2="{py0 + ph:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{py0 + ph + 18:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="12" fill="#64748b">{tick:+d}</text>'
        )
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        py = y_for(tick)
        parts.append(
            f'<line x1="{px0:.1f}" y1="{py:.1f}" x2="{px0 + pw:.1f}" y2="{py:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px0 - 9:.1f}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="12" fill="#64748b">{tick:.1f}</text>'
        )
    parts.append(
        f'<text x="{x + width / 2:.0f}" y="{y + height - 4:.0f}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="13" fill="#475569">枪口 yaw 相对正前方 90° 的偏差 (°)</text>'
    )

    actual_sets: list[tuple[int, float, str]] = []
    for segment in range(6):
        impact_time = data.time + segment * 0.05 + HORIZON
        impact = as_target(schedule_at(data, impact_time - data.time))
        for armor_index, yaw, relative in valid_armor_centers(impact):
            actual_sets.append((segment, relative, ARMOR_COLORS[armor_index]))

    for segment, curve in enumerate(data.curves):
        color = color_for_time(segment / 5.0)
        path: list[str] = []
        for grid_yaw, probability in zip(data.grid, curve):
            rel = math.degrees(sim.angle_error(grid_yaw, math.pi / 2))
            path.append(f"{x_for(rel):.1f},{y_for(probability):.2f}")
        parts.append(
            f'<polyline points="{" ".join(path)}" fill="none" stroke="{color}" '
            f'stroke-width="3" opacity="0.95"/>'
        )
        peak_index = max(range(len(curve)), key=curve.__getitem__)
        peak_rel = math.degrees(sim.angle_error(data.grid[peak_index], math.pi / 2))
        parts.append(
            f'<circle cx="{x_for(peak_rel):.1f}" cy="{y_for(curve[peak_index]):.2f}" r="4.2" '
            f'fill="{color}" stroke="#ffffff" stroke-width="1.2"/>'
        )

    for segment, relative, color in actual_sets:
        if -6.0 <= relative <= 6.0:
            parts.append(
                f'<line x1="{x_for(relative):.1f}" y1="{py0 - 8:.1f}" '
                f'x2="{x_for(relative):.1f}" y2="{py0 + ph + 8:.1f}" '
                f'stroke="{color}" stroke-width="2.2" opacity="0.55"/>'
                f'<circle cx="{x_for(relative):.1f}" cy="{py0 + ph + 18:.1f}" r="3.2" '
                f'fill="{color}" stroke="white" stroke-width="1"/>'
            )

    parts.append(
        f'<text x="{px0:.0f}" y="{y + 55:.0f}" font-family="{FONT}" font-size="12" '
        'fill="#7c3aed">彩色曲线/圆点：当前 C++ 6 段 P_hit</text>'
    )
    parts.append(
        f'<text x="{px0:.0f}" y="{y + 73:.0f}" font-family="{FONT}" font-size="12" '
        'fill="#475569">彩色竖线/底部点：该段真实目标板在未来命中时刻的 yaw</text>'
    )
    return "\n".join(parts)


def phase_panel(data: SnapshotData, x: float, y: float, width: float, height: float) -> str:
    margin = 40
    px0 = x + 76
    py0 = y + margin
    pw = width - 76 - margin
    ph = height - margin - 36
    min_deg = -5.0
    max_deg = 5.0
    min_segment = 0.0
    max_segment = 5.0

    def sx(segment: float) -> float:
        return px0 + (segment - min_segment) / (max_segment - min_segment) * pw

    def sy(deg: float) -> float:
        return py0 + (max_deg - deg) / (max_deg - min_deg) * ph

    parts: list[str] = []
    parts.append(
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{width:.0f}" height="{height:.0f}" '
        'rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{x + width / 2:.0f}" y="{y + 25:.0f}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="15" font-weight="700" fill="#0f172a">'
        "真实装甲相位移动 vs C++ 概率峰相位</text>"
    )
    for segment in range(6):
        px = sx(segment)
        parts.append(
            f'<line x1="{px:.1f}" y1="{py0:.1f}" x2="{px:.1f}" y2="{py0 + ph:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{py0 + ph + 18:.1f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="12" fill="#64748b">{segment}</text>'
        )
    for deg in range(-4, 5, 1):
        py = sy(deg)
        parts.append(
            f'<line x1="{px0:.1f}" y1="{py:.1f}" x2="{px0 + pw:.1f}" y2="{py:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-family="{FONT}" font-size="12" fill="#64748b">{deg:+d}</text>'
        )
    parts.append(
        f'<text x="{x + width / 2:.0f}" y="{y + height - 4:.0f}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="12" fill="#475569">规划段索引（每段 50 ms）</text>'
    )

    actual_path = []
    for offset_index in range(0, 61):
        offset = offset_index * 0.01
        state = as_target(schedule_at(data, offset))
        relative = nearest_valid_center(state)
        if relative is not None:
            segment_axis = min(5.0, offset / 0.05)
            actual_path.append((segment_axis, relative))
    if actual_path:
        path = " ".join(
            f"{sx(segment):.1f},{sy(deg):.1f}" for segment, deg in actual_path
        )
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="#0f766e" '
            'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    parts.append(
        f'<text x="{px0:.0f}" y="{py0 + 18:.0f}" font-family="{FONT}" font-size="12" '
        'fill="#0f766e">实际最近可命中装甲中心（每 10 ms）</text>'
    )

    predicted_path = []
    for segment, curve in enumerate(data.curves):
        peak = max(range(len(curve)), key=curve.__getitem__)
        predicted_path.append(
            (
                segment,
                math.degrees(sim.angle_error(data.grid[peak], math.pi / 2)),
            )
        )
    path = " ".join(
        f"{sx(segment):.1f},{sy(deg):.1f}" for segment, deg in predicted_path
    )
    parts.append(
        f'<polyline points="{path}" fill="none" stroke="#7c3aed" stroke-width="3" '
        'stroke-dasharray="7 4" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for segment, deg in predicted_path:
        parts.append(
            f'<circle cx="{sx(segment):.1f}" cy="{sy(deg):.1f}" r="4" fill="#7c3aed" '
            'stroke="white" stroke-width="1"/>'
        )
    parts.append(
        f'<text x="{px0 + 330:.0f}" y="{py0 + 18:.0f}" font-family="{FONT}" '
        'font-size="12" fill="#7c3aed">C++ P_hit 峰（虚线）</text>'
    )
    return "\n".join(parts)


def snapshot_svg(data: SnapshotData, row_stats: dict[str, str]) -> str:
    width = 1600
    height = 1120
    state = data.state
    title = (
        f"随机闪避策略 t={data.time:.2f}s  "
        f"目标角速度={math.degrees(state.omega):+7.1f}°/s  "
        f"目标加速度={math.degrees(state.alpha):+6.1f}°/s²  "
        f"C++首段最大P={max(data.curves[0]):.3f}  "
        f"样本数={row_stats.get('entries', '')}"
    )
    parts: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">',
        '<rect width="100%" height="100%" fill="#f1f5f9"/>',
        f'<text x="{width / 2:.0f}" y="34" text-anchor="middle" font-size="20" '
        'font-weight="700" fill="#0f172a">'
        + html.escape(title)
        + "</text>",
    ]
    parts.append(local_body_panel(data, 30, 58, 600, 470))
    parts.append(p_curve_panel(data, 650, 58, 920, 470))
    parts.append(phase_panel(data, 30, 548, 1540, 300))

    stats = [
        f"P非零段={row_stats.get('nonzero', '')}/6",
        f"曲线峰范围={row_stats.get('peak_range', '')}°",
        f"实际装甲相位范围={row_stats.get('actual_range', '')}°",
        f"第1段曲线峰={row_stats.get('peak0', '')}°",
        f"第1段实际命中角={row_stats.get('actual0', '')}°",
    ]
    for index, text in enumerate(stats):
        parts.append(
            f'<text x="{52 + index * 310:.0f}" y="920" font-size="14" fill="#334155">'
            f"{text}</text>"
        )
    parts.append(
        f'<text x="{width / 2:.0f}" y="972" text-anchor="middle" font-size="13" fill="#64748b">'
        "图上采用与 rollout_benchmark 相同的随机策略种子 20260904、20 发/秒与 15 ms 开火延迟；"
        "概率曲线由当前 C++ 独立模块原样生成。"
        "</text>"
    )
    parts.append(
        f'<text x="{width / 2:.0f}" y="1000" text-anchor="middle" font-size="13" fill="#64748b">'
        "真实装甲位置来自同一随机 schedule，不是 C++ 内部默认 1.0 s 换向模型的未来推演。"
        "</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def acceleration_svg(
    random_bins: dict[int, int],
    fixed_bins: dict[int, int],
) -> str:
    width = 1600
    height = 780
    left_w = 730
    right_w = 730

    def draw_hist(
        x: float,
        y: float,
        w: float,
        h: float,
        bins: dict[int, int],
        title: str,
        subtitle: str,
    ) -> list[str]:
        bin_min = -340
        bin_max = 340
        margin = 48
        px0 = x + 62
        py0 = y + margin
        pw = w - 62 - margin
        ph = h - margin - 42
        max_count = max(bins.values(), default=1)
        parts: list[str] = []

        def sx(acc: float) -> float:
            return px0 + (acc - bin_min) / (bin_max - bin_min) * pw

        def sy(count: float) -> float:
            return py0 + ph * (1.0 - count / max_count)

        parts.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="8" '
            'fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{x + w / 2:.0f}" y="{y + 24:.0f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="16" font-weight="700" fill="#0f172a">'
            f"{title}</text>"
        )
        parts.append(
            f'<text x="{x + w / 2:.0f}" y="{y + 45:.0f}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="12" fill="#64748b">{subtitle}</text>'
        )
        for acc in range(-300, 301, 100):
            px = sx(acc)
            parts.append(
                f'<line x1="{px:.1f}" y1="{py0:.1f}" x2="{px:.1f}" y2="{py0 + ph:.1f}" '
                f'stroke="#e2e8f0" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{px:.1f}" y="{py0 + ph + 17:.1f}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="11" fill="#64748b">{acc}</text>'
            )
        for label_y in (0.0, 0.5, 1.0):
            py = sy(max_count * label_y)
            parts.append(
                f'<line x1="{px0:.1f}" y1="{py:.1f}" x2="{px0 + pw:.1f}" y2="{py:.1f}" '
                f'stroke="#e2e8f0" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{px0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" '
                f'font-family="{FONT}" font-size="11" fill="#64748b">{max_count * label_y:.0f}</text>'
            )
        for acc in range(-320, 321, 10):
            count = bins.get(acc, 0)
            if count <= 0:
                continue
            color = "#2563eb"
            if acc < -275:
                color = "#0891b2"
            elif acc > 275:
                color = "#d97706"
            bar_left = sx(acc - 2.5)
            bar_right = sx(acc + 2.5)
            parts.append(
                f'<rect x="{bar_left:.1f}" y="{sy(count):.1f}" width="{max(1.0, bar_right - bar_left):.1f}" '
                f'height="{max(0.0, py0 + ph - sy(count)):.1f}" fill="{color}" opacity="0.85"/>'
            )
        for acc in (-280.0, 280.0, -300.0, 300.0):
            px = sx(acc)
            parts.append(
                f'<line x1="{px:.1f}" y1="{py0:.1f}" x2="{px:.1f}" y2="{py0 + ph:.1f}" '
                'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5 3"/>'
            )
        parts.append(
            f'<text x="{px0:.0f}" y="{py0 + 16:.0f}" font-family="{FONT}" font-size="12" '
            'fill="#475569">横轴：等效加速度 (°/s²)，峰值柱按正/负加速方向着色</text>'
        )
        return parts

    left = draw_hist(
        40,
        70,
        left_w,
        570,
        random_bins,
        "随机闪避策略真实等效加速度分布",
        f"策略 seed {STRATEGY_SEED}，实际按 10 ms 在线样本统计；h={HORIZON:.3f} s",
    )
    right = draw_hist(
        830,
        70,
        right_w,
        570,
        fixed_bins,
        "固定 1.0 s 换向策略可训练分布",
        "来自 learned_acceleration_distribution.csv 的等效历史支持集",
    )
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">',
        '<rect width="100%" height="100%" fill="#f1f5f9"/>',
        f'<text x="{width / 2:.0f}" y="40" text-anchor="middle" font-size="20" '
        'font-weight="700" fill="#0f172a">加速度分布诊断：随机策略 vs 当前模型训练分布</text>',
        *left,
        *right,
        '<text x="800" y="735" text-anchor="middle" font-size="13" fill="#64748b">'
        "固定策略主要由 -280°/s² 和少量 +280°/s² 构成；随机策略把正负加速质量拆到两侧 ±(280..300)°/s²，"
        "C++ 只按当前 600°/s 速度约束与固定换向假设使用这些样本。"
        "</text>",
        "</svg>",
    ]
    return "\n".join(parts)


def render_png(svg: str, stem: str, width: int, height: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / f"{stem}.html"
    png_path = OUT_DIR / f"{stem}.png"
    html_path.write_text(
        f"<!doctype html><meta charset='utf-8'>"
        f"<style>html,body{{margin:0;padding:0;overflow:hidden;width:{width}px;height:{height}px;background:#f1f5f9}}</style>"
        + svg,
        encoding="utf-8",
    )
    if not EDGE.exists():
        raise RuntimeError(f"Edge not found: {EDGE}")
    subprocess.run(
        [
            str(EDGE),
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width + 20},{height + 20}",
            f"--screenshot={png_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return png_path


def load_fixed_accel_bins() -> dict[int, int]:
    csv_path = ROOT / "learned_acceleration_distribution_summary.csv"
    bins: dict[int, int] = {}
    if not csv_path.exists():
        csv_path = ROOT / "learned_acceleration_distribution.csv"
    if not csv_path.exists():
        return bins
    for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split(",")
        if len(fields) < 2:
            continue
        try:
            acceleration = float(fields[0])
            probability = float(fields[1])
        except ValueError:
            continue
        bucket = round(acceleration / 10.0) * 10
        if -320 <= bucket <= 320:
            bins[bucket] = bins.get(bucket, 0) + max(1, int(probability * 10_000))
    return bins


def main() -> None:
    global _SCHEDULE
    schedule = RandomEvasionSchedule(STRATEGY_SEED)
    schedule.state_at(40.0)
    _SCHEDULE = schedule

    selected_times = (5.55, 14.50, 21.55, 25.0)
    output_paths: list[Path] = []
    for selected in selected_times:
        data = make_snapshot_data(schedule, selected)
        nonzero = sum(
            1 for curve in data.curves if any(value > 1e-12 for value in curve)
        )
        peak_rel = [
            math.degrees(
                sim.angle_error(
                    data.grid[max(range(len(curve)), key=curve.__getitem__)],
                    math.pi / 2,
                )
            )
            for curve in data.curves
        ]
        actual_offsets: list[float] = []
        for segment in range(6):
            impact_time = data.time + segment * 0.05 + HORIZON
            impact = as_target(schedule.state_at(impact_time))
            nearest = nearest_valid_center(impact)
            if nearest is not None:
                actual_offsets.append(nearest)
        actual_range = (
            f"{min(actual_offsets):+.1f}..{max(actual_offsets):+.1f}"
            if actual_offsets
            else "n/a"
        )
        stats = {
            "entries": str(sum(data.entries.values())),
            "nonzero": str(nonzero),
            "peak_range": f"{min(peak_rel):+.2f}..{max(peak_rel):+.2f}",
            "actual_range": actual_range,
            "peak0": f"{peak_rel[0]:+.2f}",
            "actual0": (
                f"{actual_offsets[0]:+.2f}" if len(actual_offsets) > 0 else "n/a"
            ),
        }
        stem = f"random_strategy_snapshot_t{int(round(selected * 100)):04d}"
        output_paths.append(render_png(snapshot_svg(data, stats), stem, 1620, 1140))
        print(f"wrote {output_paths[-1]}")

    random_bins: dict[int, int] = {}
    cursor = 0.0
    while cursor + HORIZON <= 40.0 + 1e-12:
        start = schedule.state_at(cursor)
        finish = schedule.state_at(cursor + HORIZON)
        acceleration = 2.0 * (
            finish.angle - start.angle - start.omega * HORIZON
        ) / (HORIZON * HORIZON)
        bucket = round(math.degrees(acceleration) / 10.0) * 10
        if -320 <= bucket <= 320:
            random_bins[bucket] = random_bins.get(bucket, 0) + 1
        cursor += DT
    output_paths.append(
        render_png(
            acceleration_svg(random_bins, load_fixed_accel_bins()),
            "random_strategy_acceleration_distribution",
            1620,
            800,
        )
    )
    print("generated:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
