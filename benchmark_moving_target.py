"""Headless A/B benchmark for the moving-target shooter planners."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import replace

import sp_vision_moving_target_visualizer as sim


def run(mode: str, distance: float, duration: float, seed: int) -> tuple[int, int, int, int]:
    random.seed(seed)
    planner = sim.TinyMpcPlanner()
    target = sim.TargetState(0.0, distance, 0.0, 0.0, 0.0, sim.INITIAL_OMEGA, sim.ALPHA_LIMIT)
    gimbal_yaw = math.pi / 2.0
    gimbal_vel = 0.0
    direction = 1.0
    next_switch = sim.EVADE_SWITCH_INTERVAL
    model = sim.ProbabilisticAimModel(sim.LOW_SPEED_DELAY + distance / sim.BULLET_SPEED)
    history: list[tuple[float, float, float]] = []
    previous_peak: float | None = None
    shots: list[sim.Shot] = []
    last_shot = -100.0
    valid = low = miss = fired = 0

    for step in range(round(duration / sim.PLANNER_DT)):
        now = (step + 1) * sim.PLANNER_DT
        target.angle, target.omega, direction, target.alpha, next_switch = sim.advance_optimized_evasion(
            target.angle, target.omega, direction, now - sim.PLANNER_DT, next_switch, sim.PLANNER_DT
        )
        history.append((now, target.angle, target.omega))
        plan = planner.plan(replace(target), sim.BULLET_SPEED)
        fire = plan.fire

        if mode == "probabilistic":
            model.samples.clear()
            model.add_trajectory(history[-80:])
            grid = [gimbal_yaw + math.radians(1.5 * i) for i in range(-120, 121)]
            field = model.hit_probability_field(plan.predicted_state, grid, count=64, seed=step // 5)
            reach_time = max(plan.delay + plan.flight_time, sim.PLANNER_DT)
            reach = abs(gimbal_vel) * reach_time + 0.5 * sim.GIMBAL_MAX_ACCEL * reach_time**2
            reachable = [i for i, yaw in enumerate(grid) if abs(sim.angle_error(yaw, gimbal_yaw)) <= reach]
            candidates = reachable or list(range(len(grid)))
            peak_index = max(candidates, key=field.__getitem__)
            peak = grid[peak_index]
            if previous_peak is not None:
                peak = previous_peak + sim.PROBABILITY_YAW_SMOOTHING * sim.angle_error(peak, previous_peak)
            previous_peak = peak
            phase = sim.angle_error(peak, plan.target_yaw)
            plan = replace(
                plan,
                target_yaw=plan.target_yaw + phase,
                yaw=plan.yaw + phase,
                planned_aim_x=math.hypot(plan.planned_aim_x, plan.planned_aim_y) * math.cos(peak),
                planned_aim_y=math.hypot(plan.planned_aim_x, plan.planned_aim_y) * math.sin(peak),
                reference_yaw=tuple(yaw + phase for yaw in plan.reference_yaw),
                planned_yaw=tuple(yaw + phase for yaw in plan.planned_yaw),
            )
            fire = fire and field[peak_index] >= sim.PROBABILITY_FIRE_THRESHOLD

        acceleration = sim.clamp(
            plan.yaw_acceleration + 35.0 * sim.angle_error(plan.yaw, gimbal_yaw)
            + 10.0 * (plan.yaw_velocity - gimbal_vel),
            -sim.GIMBAL_MAX_ACCEL,
            sim.GIMBAL_MAX_ACCEL,
        )
        gimbal_yaw += gimbal_vel * sim.PLANNER_DT + 0.5 * acceleration * sim.PLANNER_DT**2
        gimbal_vel += acceleration * sim.PLANNER_DT
        if fire and now - last_shot >= sim.SHOT_INTERVAL - 1e-12:
            shots.append(sim.shot_from_plan(plan, now, gimbal_yaw, scatter=True))
            last_shot = now
            fired += 1

        pending: list[sim.Shot] = []
        for shot in shots:
            if now < shot.impact_at:
                pending.append(shot)
                continue
            result = sim.evaluate_shot(shot, target)
            if result.outcome == "valid_hit":
                valid += 1
            elif result.outcome == "low_normal_speed":
                low += 1
            else:
                miss += 1
        shots = pending

    return valid, low, miss, fired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--distances", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    args = parser.parse_args()
    for mode in ("current", "probabilistic"):
        totals = [0, 0, 0, 0]
        for distance in args.distances:
            for seed in args.seeds:
                result = run(mode, distance, args.duration, seed)
                totals = [a + b for a, b in zip(totals, result)]
        resolved = sum(totals[:3])
        rate = totals[0] / resolved if resolved else 0.0
        throughput = totals[0] / (args.duration * len(args.distances) * len(args.seeds))
        print(f"{mode:14s} valid={totals[0]:4d} fired={totals[3]:4d} rate={rate * 100:6.2f}% valid/s={throughput:6.3f}")


if __name__ == "__main__":
    main()
