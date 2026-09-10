"""Export the learned acceleration distribution from experiment v2."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("tracking_experiment_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(r"C:\Users\ASUS\Downloads\moving_target_tracking_experiment_v2.py"))
    parser.add_argument("--output", type=Path, default=Path("learned_acceleration_distribution.csv"))
    parser.add_argument("--history-seconds", type=float, default=5.0)
    parser.add_argument("--query-omega-deg", type=float, default=300.0)
    parser.add_argument("--horizon-ms", type=float, default=255.0)
    args = parser.parse_args()
    sim = load_module(args.source)
    cfg = sim.Config(
        history_window=args.history_seconds,
        minimum_history_horizon=max(args.horizon_ms / 1000.0, 0.01),
    )
    rows: list[tuple[float, sim.State]] = []
    state = sim.State(0.0, math.radians(300.0), cfg.max_alpha)
    direction = 1.0
    now = 0.0
    while now < args.history_seconds - 1e-12:
        state, direction = sim.advance_target(state, direction, cfg.dt, cfg)
        now += cfg.dt
        rows.append((now, state))
    query = sim.State(state.angle, math.radians(args.query_omega_deg), state.alpha)
    model = sim.HistoryAccelerationModel(cfg)
    horizon = args.horizon_ms / 1000.0
    samples = model.samples_for_horizon(rows, horizon)
    weighted, ess = model.weighted_accelerations(rows, query, horizon)
    weight_by_acceleration: dict[float, float] = {}
    for acceleration, weight in weighted:
        weight_by_acceleration[acceleration] = weight_by_acceleration.get(acceleration, 0.0) + weight
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_timestamp_s", "sample_velocity_deg_s", "sample_acceleration_deg_s2", "acceleration_sign", "speed_margin_deg_s", "normalized_weight", "included"))
        lower = max(-cfg.max_alpha, (-cfg.max_omega - query.omega) / horizon)
        upper = min(cfg.max_alpha, (cfg.max_omega - query.omega) / horizon)
        for sample in samples:
            included = lower <= sample.acceleration <= upper
            velocity_weight = math.exp(-0.5 * ((sample.velocity - query.omega) / cfg.velocity_sigma) ** 2)
            margin_weight = math.exp(-0.5 * ((sample.speed_margin - (cfg.max_omega - abs(query.omega))) / cfg.margin_sigma) ** 2)
            direction_weight = 1.0 if sample.acceleration_sign == sim.sign(query.alpha) else cfg.opposite_acceleration_weight
            raw = velocity_weight * margin_weight * direction_weight if included else 0.0
            normalized = weight_by_acceleration.get(sample.acceleration, 0.0) if included else 0.0
            writer.writerow((f"{sample.timestamp:.6f}", f"{math.degrees(sample.velocity):.6f}", f"{math.degrees(sample.acceleration):.6f}", sample.acceleration_sign, f"{math.degrees(sample.speed_margin):.6f}", f"{normalized:.12f}", int(included)))
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("acceleration_deg_s2", "probability", "weighted_contribution_to_mean"))
        aggregated: dict[float, float] = {}
        for acceleration, weight in weighted:
            acceleration_deg = round(math.degrees(acceleration), 1)
            aggregated[acceleration_deg] = aggregated.get(acceleration_deg, 0.0) + weight
        for acceleration_deg, weight in sorted(aggregated.items()):
            writer.writerow((f"{acceleration_deg:.1f}", f"{weight:.12f}", f"{acceleration_deg * weight:.9f}"))
    mean = sum(acceleration * weight for acceleration, weight in weighted)
    variance = sum(weight * (acceleration - mean) ** 2 for acceleration, weight in weighted)
    print(f"source={args.source}")
    print(f"samples={len(samples)} included={len(weighted)} ESS={ess:.3f}")
    print(f"query_omega={args.query_omega_deg:.3f} deg/s horizon={args.horizon_ms:.3f} ms")
    print(f"mean_acceleration={math.degrees(mean):.6f} deg/s^2 std={math.degrees(math.sqrt(max(variance, 0.0))):.6f} deg/s^2")
    print(f"wrote {args.output.resolve()}")
    print(f"wrote {summary_path.resolve()}")


if __name__ == "__main__":
    main()
