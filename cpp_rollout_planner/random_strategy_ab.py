from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sp_vision_moving_target_visualizer_full_compare as sim
from random_evasion_schedule import RandomEvasionSchedule
from rollout_benchmark import (
    MODEL_VERSION_FUSION,
    MODEL_VERSION_RIGOROUS,
    RolloutBenchmark,
)


def make_schedule_advance(schedule):
    def schedule_advance(
        _angle: float,
        _omega: float,
        _direction: float,
        start_time: float,
        _next_switch: float,
        _dt: float,
    ):
        end_time = start_time + _dt
        end = schedule.state_at(end_time)
        return (
            end.angle,
            end.omega,
            1.0 if end.alpha >= 0.0 else -1.0,
            end.alpha,
            schedule.next_switch_at(end_time),
        )

    return schedule_advance


def run_ab(
    schedule: RandomEvasionSchedule,
    seconds: float = 180.0,
    scatter_seed: int = 20260903,
    model_version: str = MODEL_VERSION_RIGOROUS,
    zero_acceleration_history: bool = False,
    distance: float = 6.0,
) -> None:
    original_initial_omega = sim.INITIAL_OMEGA
    original_advance = sim.advance_optimized_evasion
    sim.INITIAL_OMEGA = schedule.initial_omega
    try:
        sim.advance_optimized_evasion = make_schedule_advance(schedule)
        sim.benchmark(
            seconds=seconds,
            modes=("current",),
            seed=scatter_seed,
            distance=distance,
        )
    finally:
        sim.advance_optimized_evasion = original_advance
        sim.INITIAL_OMEGA = original_initial_omega

    rollout_schedule = RandomEvasionSchedule(
        schedule.seed,
        initial_omega=schedule.initial_omega,
        omega_limit=schedule.omega_limit,
        min_accel=schedule.min_accel,
        max_accel=schedule.max_accel,
        min_duration=schedule.min_duration,
        max_duration=schedule.max_duration,
    )
    sim.INITIAL_OMEGA = rollout_schedule.initial_omega
    try:
        sim.advance_optimized_evasion = make_schedule_advance(rollout_schedule)
        RolloutBenchmark(
            seconds=seconds,
            seed=scatter_seed,
            dt=0.010,
            schedule=rollout_schedule,
            model_version=model_version,
            zero_acceleration_history=zero_acceleration_history,
            distance=distance,
        ).run()
    finally:
        sim.advance_optimized_evasion = original_advance
        sim.INITIAL_OMEGA = original_initial_omega


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--strategy-seed", type=int, default=20260904)
    parser.add_argument("--scatter-seed", type=int, default=20260903)
    parser.add_argument(
        "--model-version",
        choices=(MODEL_VERSION_FUSION, MODEL_VERSION_RIGOROUS),
        default=MODEL_VERSION_RIGOROUS,
    )
    parser.add_argument(
        "--zero-accel-history",
        action="store_true",
        help="force predicted acceleration history to zero",
    )
    parser.add_argument("--distance", type=float, default=6.0)
    parser.add_argument("--initial-omega-deg", type=float, default=350.0)
    parser.add_argument("--omega-limit-deg", type=float, default=700.0)
    parser.add_argument("--min-accel-deg", type=float, default=280.0)
    parser.add_argument("--max-accel-deg", type=float, default=300.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=1.2)
    args = parser.parse_args()
    schedule = RandomEvasionSchedule(
        args.strategy_seed,
        initial_omega=math.radians(args.initial_omega_deg),
        omega_limit=math.radians(args.omega_limit_deg),
        min_accel=math.radians(args.min_accel_deg),
        max_accel=math.radians(args.max_accel_deg),
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    run_ab(
        schedule,
        seconds=args.seconds,
        scatter_seed=args.scatter_seed,
        model_version=args.model_version,
        zero_acceleration_history=args.zero_accel_history,
        distance=args.distance,
    )


if __name__ == "__main__":
    main()
