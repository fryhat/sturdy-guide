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
from rollout_benchmark import MODEL_VERSION_RIGOROUS, RolloutBenchmark


CURRENT_RE = re.compile(
    r"current:\s+shots=(\d+), valid=(\d+), low=(\d+), miss=(\d+), "
    r"valid_rate=([0-9.]+)%, valid_per_s=([0-9.]+)"
)
ROLLOUT_RE = re.compile(
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


def parse_result(text: str, pattern: re.Pattern[str]) -> RunResult:
    match = pattern.search(text)
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


def run_current(seconds: float, seed: int, distance: float) -> RunResult:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        sim.benchmark(
            seconds=seconds,
            modes=("current",),
            seed=seed,
            distance=distance,
        )
    return parse_result(output.getvalue(), CURRENT_RE)


def run_rollout(
    seconds: float,
    warmup_seconds: float,
    seed: int,
    distance: float,
) -> RunResult:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        RolloutBenchmark(
            seconds=seconds,
            seed=seed,
            distance=distance,
            model_version=MODEL_VERSION_RIGOROUS,
            warmup_seconds=warmup_seconds,
        ).run()
    return parse_result(output.getvalue(), ROLLOUT_RE)


def aggregate(
    rows: dict[int, tuple[RunResult, RunResult]],
) -> tuple[RunResult, RunResult]:
    sums = [[0, 0, 0, 0], [0, 0, 0, 0]]
    rates = [[], []]
    per_seconds = [[], []]
    for current, rollout in rows.values():
        for index, result in enumerate((current, rollout)):
            sums[index][0] += result.shots
            sums[index][1] += result.valid
            sums[index][2] += result.low
            sums[index][3] += result.miss
            rates[index].append(result.valid_rate)
            per_seconds[index].append(result.valid_per_s)

    def make(index: int) -> RunResult:
        resolved = sums[index][1] + sums[index][2] + sums[index][3]
        rate = sums[index][1] / resolved if resolved else 0.0
        return RunResult(
            shots=sums[index][0],
            valid=sums[index][1],
            low=sums[index][2],
            miss=sums[index][3],
            valid_rate=rate * 100.0,
            valid_per_s=sum(per_seconds[index]) / len(per_seconds[index]),
        )

    return make(0), make(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(20260903,))
    parser.add_argument("--distances", type=float, nargs="+", default=(2.0, 4.0, 6.0, 8.0))
    args = parser.parse_args()

    for seed in args.seeds:
        rows: dict[int, tuple[RunResult, RunResult]] = {}
        for distance in args.distances:
            current = run_current(args.seconds, seed, distance)
            rollout = run_rollout(
                args.seconds,
                args.warmup,
                seed,
                distance,
            )
            rows[distance] = (current, rollout)
            print(
                f"seed={seed} distance={distance:g} "
                f"current={current.valid_per_s:.3f}/{current.valid_rate:.2f}% "
                f"warmup60={rollout.valid_per_s:.3f}/{rollout.valid_rate:.2f}% "
                f"delta={rollout.valid_per_s - current.valid_per_s:+.3f}"
            )
        for distance in args.distances:
            current, rollout = rows[distance]
            print(
                f"aggregate seed={seed} distance={distance:g} "
                f"current={current.shots} shots, {current.valid} valid, "
                f"{current.valid_rate:.2f}%, {current.valid_per_s:.3f}/s | "
                f"warmup60={rollout.shots} shots, {rollout.valid} valid, "
                f"{rollout.valid_rate:.2f}%, {rollout.valid_per_s:.3f}/s"
            )


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"wall_seconds={time.perf_counter() - started:.3f}")
