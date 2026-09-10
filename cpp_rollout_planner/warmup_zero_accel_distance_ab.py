from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sp_vision_moving_target_visualizer_full_compare as sim
from random_evasion_schedule import RandomEvasionSchedule
from rollout_benchmark import MODEL_VERSION_RIGOROUS, RolloutBenchmark


RESULT_RE = re.compile(
    r"rollout_cpp\[rigorous\]:\s+shots=(\d+), valid=(\d+), low=(\d+), "
    r"miss=(\d+), valid_rate=([0-9.]+)%, valid_per_s=([0-9.]+)"
)


@dataclass
class RunResult:
    shots: int
    valid: int
    low: int
    miss: int
    valid_rate: float
    valid_per_s: float


def parse_result(text: str) -> RunResult:
    match = RESULT_RE.search(text)
    if match is None:
        raise RuntimeError(f"could not parse result from: {text[-500:]}")
    shots, valid, low, miss, valid_rate, valid_per_s = match.groups()
    return RunResult(
        shots=int(shots),
        valid=int(valid),
        low=int(low),
        miss=int(miss),
        valid_rate=float(valid_rate),
        valid_per_s=float(valid_per_s),
    )


def run_case(
    *,
    seconds: float,
    warmup_seconds: float,
    zero_acceleration_history: bool,
    scatter_seed: int,
    schedule_seed: int,
    distance: float,
) -> RunResult:
    schedule = RandomEvasionSchedule(schedule_seed)

    def advance(
        _angle: float,
        _omega: float,
        _direction: float,
        start_time: float,
        _next_switch: float,
        dt: float,
    ):
        end = schedule.state_at(start_time + dt)
        return (
            end.angle,
            end.omega,
            1.0 if end.alpha >= 0.0 else -1.0,
            end.alpha,
            schedule.next_switch_at(start_time + dt),
        )

    original_advance = sim.advance_optimized_evasion
    sim.advance_optimized_evasion = advance
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            RolloutBenchmark(
                seconds=seconds,
                seed=scatter_seed,
                distance=distance,
                schedule=schedule,
                model_version=MODEL_VERSION_RIGOROUS,
                warmup_seconds=warmup_seconds,
                zero_acceleration_history=zero_acceleration_history,
            ).run()
    finally:
        sim.advance_optimized_evasion = original_advance
    return parse_result(output.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--schedule-seed", type=int, default=20260904)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260903,))
    parser.add_argument("--distances", type=float, nargs="+", default=(2.0, 4.0, 6.0, 8.0))
    args = parser.parse_args()

    for seed in args.seeds:
        rows: dict[float, tuple[RunResult, RunResult]] = {}
        for distance in args.distances:
            warmup = run_case(
                seconds=args.seconds,
                warmup_seconds=args.warmup,
                zero_acceleration_history=False,
                scatter_seed=seed,
                schedule_seed=args.schedule_seed,
                distance=distance,
            )
            zero = run_case(
                seconds=args.seconds,
                warmup_seconds=0.0,
                zero_acceleration_history=True,
                scatter_seed=seed,
                schedule_seed=args.schedule_seed,
                distance=distance,
            )
            rows[distance] = (zero, warmup)
            print(
                f"seed={seed} distance={distance:g} "
                f"zero={zero.valid_per_s:.3f}/{zero.valid_rate:.2f}% "
                f"warmup={warmup.valid_per_s:.3f}/{warmup.valid_rate:.2f}% "
                f"delta={warmup.valid_per_s - zero.valid_per_s:+.3f}"
            )
        for distance in args.distances:
            zero, warmup = rows[distance]
            print(
                f"aggregate seed={seed} distance={distance:g} "
                f"zero={zero.shots} shots, {zero.valid} valid, "
                f"{zero.valid_rate:.2f}%, {zero.valid_per_s:.3f}/s | "
                f"warmup={warmup.shots} shots, {warmup.valid} valid, "
                f"{warmup.valid_rate:.2f}%, {warmup.valid_per_s:.3f}/s"
            )


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"wall_seconds={time.perf_counter() - started:.3f}")
