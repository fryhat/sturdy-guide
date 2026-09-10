from __future__ import annotations

import math

import sp_vision_moving_target_visualizer_full_compare as sim
from rollout_planner import RolloutPlanner

_curve_planner: RolloutPlanner | None = None


def compress_history_entries(
    entries: list[tuple[float, float, int]],
    bins: int = 256,
) -> list[tuple[float, float, int]]:
    if not entries:
        return []
    accelerations = [acceleration for _horizon, acceleration, _count in entries]
    raw_lower = min(accelerations)
    raw_upper = max(accelerations)
    padding = max((raw_upper - raw_lower) * 1e-4, 1e-9)
    lower = raw_lower - padding
    upper = raw_upper + padding
    step = (upper - lower) / bins
    histogram: dict[tuple[float, int], int] = {}
    for horizon, acceleration, count in entries:
        bucket = max(
            0,
            min(bins - 1, int((acceleration - lower) / step)),
        )
        key = (round(horizon, 2), bucket)
        histogram[key] = histogram.get(key, 0) + count
    return [
        (
            horizon,
            lower + (bucket + 0.5) * step,
            count,
        )
        for (horizon, bucket), count in sorted(histogram.items())
    ]


def advance_target(
    target: sim.TargetState,
    direction: float,
    next_switch: float,
    start_time: float,
    dt: float,
) -> tuple[sim.TargetState, float, float]:
    angle, omega, direction, alpha, next_switch = sim.advance_optimized_evasion(
        target.angle,
        target.omega,
        direction,
        start_time,
        next_switch,
        dt,
    )
    return (
        sim.replace(
            target,
            angle=angle,
            omega=omega,
            alpha=alpha,
        ),
        direction,
        next_switch,
    )


def target_at_time(
    target: sim.TargetState,
    direction: float,
    next_switch: float,
    start_time: float,
    target_time: float,
) -> tuple[sim.TargetState, float, float]:
    return advance_target(
        target,
        direction,
        next_switch,
        start_time,
        max(0.0, target_time - start_time),
    )


def schedule_state_at(
    schedule,
    target: sim.TargetState,
    time: float,
) -> sim.TargetState:
    """Return the schedule state while preserving the target's fixed center."""
    state = schedule.state_at(time)
    return sim.replace(
        target,
        angle=state.angle,
        omega=state.omega,
        alpha=state.alpha,
    )


def schedule_armor_solution_for_yaw(
    schedule,
    target: sim.TargetState,
    launch_time: float,
    yaw: float,
) -> tuple[object, sim.TargetState, float]:
    """Armor solution that follows the random schedule during flight."""
    delayed = schedule_state_at(schedule, target, launch_time)
    solutions: list[tuple[bool, float, object, sim.TargetState, float]] = []
    for armor_index in range(4):
        flight_time = math.hypot(
            sim.armor_poses(delayed)[armor_index].x,
            sim.armor_poses(delayed)[armor_index].y,
        ) / sim.BULLET_SPEED
        for _iteration in range(sim.MAX_FLIGHT_ITERATIONS):
            impact = schedule_state_at(
                schedule,
                target,
                launch_time + flight_time,
            )
            armor = sim.armor_poses(impact)[armor_index]
            updated = math.hypot(armor.x, armor.y) / sim.BULLET_SPEED
            if abs(updated - flight_time) < sim.FLIGHT_TIME_TOLERANCE:
                flight_time = updated
                break
            flight_time = updated
        impact = schedule_state_at(schedule, target, launch_time + flight_time)
        armor = sim.armor_poses(impact)[armor_index]
        flight_time = math.hypot(armor.x, armor.y) / sim.BULLET_SPEED
        armor_yaw = math.atan2(armor.y, armor.x)
        normal_speed = max(
            0.0,
            -math.cos(sim.angle_error(yaw, math.atan2(armor.normal_y, armor.normal_x))),
        ) * sim.BULLET_SPEED
        solutions.append(
            (
                normal_speed > sim.MIN_NORMAL_HIT_SPEED,
                abs(sim.angle_error(armor_yaw, yaw)),
                armor,
                impact,
                flight_time,
            )
        )
    facing = [solution for solution in solutions if solution[0]]
    _valid, _error, armor, impact_state, flight_time = min(
        facing or solutions,
        key=lambda solution: solution[1],
    )
    return armor, impact_state, flight_time


def curve_grid(
    target: sim.TargetState,
    grid_step: float = sim.ARMOR_SCAN_STEP,
) -> list[float]:
    center_distance = max(math.hypot(target.x, target.y), 1e-6)
    center_yaw = math.atan2(target.y, target.x)
    half_span = math.asin(min(1.0, sim.BODY_MAX_RADIUS_M / center_distance))
    half_span += math.atan2(
        sim.ARMOR_WIDTH_M / 2.0 + sim.ARMOR_DRAW_THICKNESS_M / 2.0,
        center_distance,
    )
    scatter_padding = 4.0 * math.atan2(sim.SCATTER_SIGMA, center_distance)
    curve_padding = scatter_padding + sim.ARMOR_CURVE_MOVEMENT_RAD
    grid: list[float] = []
    cursor = center_yaw - half_span - curve_padding
    upper = center_yaw + half_span + curve_padding
    while cursor <= upper + 1e-12:
        grid.append(cursor)
        cursor += grid_step
    return grid


def build_segment_curves(
    model: sim.ProbabilisticAimModel,
    target: sim.TargetState,
    direction: float,
    next_switch: float,
    start_time: float,
    grid: list[float],
    segment_count: int = 6,
    schedule=None,
    omega_limit: float | None = None,
    alpha_limit: float | None = None,
    history_entries: dict[tuple[float, float], int] | None = None,
) -> list[list[float]]:
    global _curve_planner
    if _curve_planner is None:
        _curve_planner = RolloutPlanner()
    center_distance = max(math.hypot(target.x, target.y), 1e-6)
    horizon = sim.LOW_SPEED_DELAY + center_distance / sim.BULLET_SPEED
    source = (
        history_entries
        if history_entries is not None
        else model.sample_entries
    )
    entries = compress_history_entries([
        (entry_horizon, acceleration, count)
        for (entry_horizon, acceleration), count in source.items()
    ])
    if schedule is None:
        raise ValueError("target schedule is required; no fixed target assumptions")
    if not hasattr(schedule, "omega_limit") or not hasattr(
        schedule, "alpha_limit"
    ):
        raise ValueError(
            "schedule must expose omega_limit and alpha_limit; "
            "no fixed target assumptions are allowed"
        )
    limit = schedule.omega_limit if omega_limit is None else omega_limit
    alpha = schedule.alpha_limit if alpha_limit is None else alpha_limit
    curves = []
    for segment in range(segment_count):
        impact = schedule_state_at(
            schedule,
            target,
            start_time + segment * 0.05 + horizon,
        )
        curve = _curve_planner.build_curve_from_impact(
            impact.angle,
            impact.omega,
            horizon,
            limit,
            alpha,
            entries,
            grid,
            center_x=target.x,
            center_y=target.y,
        )
        curves.append(curve)
    return curves[:segment_count]
