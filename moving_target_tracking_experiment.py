"""二维移动标记点跟踪实验。

本文件与原自瞄可视化程序隔离，不包含发射、弹道、装甲或开火语义。
默认使用可复现的纯 Python 受限执行器；TinyMPC 仅作为可选后端。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path


PI2 = 2.0 * math.pi


@dataclass(frozen=True)
class Config:
    dt: float = 0.01
    warmup: float = 5.0
    duration: float = 20.0
    distance: float = 6.0
    marker_width: float = 0.14
    max_omega: float = math.radians(600.0)
    max_alpha: float = math.radians(280.0)
    executor_max_accel: float = 50.0
    executor_max_speed: float = 12.0
    delay: float = 0.015
    kp: float = 35.0
    kd: float = 10.0
    history_horizon: float = 0.255
    velocity_sigma: float = math.radians(180.0)
    scatter_sigma: float | None = None
    branch_margin: float = 0.035
    branch_confirm_frames: int = 3
    seed: int = 11

    def __post_init__(self) -> None:
        if self.scatter_sigma is None:
            object.__setattr__(self, "scatter_sigma", (self.marker_width / 2.0) / 1.6448536269514722)


@dataclass(frozen=True)
class State:
    angle: float
    omega: float
    alpha: float


@dataclass(frozen=True)
class HistorySample:
    timestamp: float
    position: float
    velocity: float
    acceleration: float
    weight: float


@dataclass
class Event:
    timestamp: float
    mode: str
    delay: float
    target_angle: float
    target_omega: float
    target_alpha: float
    reference_yaw: float
    predicted_yaw: float
    actual_yaw: float
    predicted_probability: float
    covered: bool
    angle_error: float
    branch: int
    planning_ms: float


def wrap_error(target: float, current: float) -> float:
    return (target - current + math.pi) % PI2 - math.pi


def advance_target(state: State, direction: float, dt: float, cfg: Config) -> tuple[State, float]:
    """Periodic maximum-acceleration reversal with hard speed limits."""
    alpha = direction * cfg.max_alpha
    omega = state.omega + alpha * dt
    if omega > cfg.max_omega:
        omega = cfg.max_omega
        direction = -1.0
    elif omega < -cfg.max_omega:
        omega = -cfg.max_omega
        direction = 1.0
    return State(state.angle + state.omega * dt + 0.5 * alpha * dt * dt, omega, alpha), direction


def hermite_angle(rows: list[tuple[float, float, float]], timestamp: float) -> tuple[float, float]:
    """Cubic Hermite interpolation over continuous (unwrapped) angles."""
    if not rows:
        raise ValueError("empty history")
    if timestamp <= rows[0][0]:
        return rows[0][1], rows[0][2]
    if timestamp >= rows[-1][0]:
        return rows[-1][1], rows[-1][2]
    hi = next(i for i in range(1, len(rows)) if rows[i][0] >= timestamp)
    t0, p0, v0 = rows[hi - 1]
    t1, p1, v1 = rows[hi]
    h = t1 - t0
    s = (timestamp - t0) / h
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    position = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
    dh00 = 6 * s**2 - 6 * s
    dh10 = 3 * s**2 - 4 * s + 1
    dh01 = -6 * s**2 + 6 * s
    dh11 = 3 * s**2 - 2 * s
    velocity = (dh00 * p0 + dh10 * h * v0 + dh01 * p1 + dh11 * h * v1) / h
    return position, velocity


class HistoryAccelerationModel:
    def __init__(self, cfg: Config, conditional: bool = True) -> None:
        self.cfg = cfg
        self.conditional = conditional
        self.samples: list[HistorySample] = []

    def fit(self, history: list[tuple[float, float, float]]) -> None:
        self.samples.clear()
        horizon = self.cfg.history_horizon
        for timestamp, position, velocity in history:
            future_time = timestamp + horizon
            if future_time > history[-1][0]:
                continue
            future_position, _ = hermite_angle(history, future_time)
            acceleration = 2.0 * (future_position - position - velocity * horizon) / (horizon * horizon)
            if abs(acceleration) > self.cfg.max_alpha * 1.25:
                continue
            if self.conditional:
                weight = math.exp(-0.5 * ((velocity - history[-1][2]) / self.cfg.velocity_sigma) ** 2)
            else:
                weight = 1.0
            self.samples.append(HistorySample(timestamp, position, velocity, acceleration, weight))

    def weighted_accelerations(self, velocity: float) -> list[tuple[float, float]]:
        horizon = self.cfg.history_horizon
        lower = max(-self.cfg.max_alpha, (-self.cfg.max_omega - velocity) / horizon)
        upper = min(self.cfg.max_alpha, (self.cfg.max_omega - velocity) / horizon)
        usable = [(s.acceleration, s.weight) for s in self.samples if lower <= s.acceleration <= upper]
        if not usable:
            return [(0.0, 1.0)]
        total = sum(weight for _, weight in usable)
        return [(acceleration, weight / total) for acceleration, weight in usable]

    def future_angle_distribution(self, state: State, horizon: float | None = None) -> list[tuple[float, float]]:
        horizon = self.cfg.history_horizon if horizon is None else max(0.0, horizon)
        return [
            (state.angle + state.omega * horizon + 0.5 * acceleration * horizon**2, weight)
            for acceleration, weight in self.weighted_accelerations(state.omega)
        ]


def coverage_probability(yaw: float, angle: float, distance: float, cfg: Config) -> float:
    half_angle = math.atan2(cfg.marker_width / 2.0, distance)
    sigma = max(float(cfg.scatter_sigma) / distance, 1e-9)
    scale = sigma * math.sqrt(2.0)
    delta = wrap_error(yaw, angle)
    return 0.5 * (math.erf((delta + half_angle) / scale) - math.erf((delta - half_angle) / scale))


class BranchSelector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.current: float | None = None
        self.current_branch = 0
        self.pending: float | None = None
        self.pending_frames = 0

    def choose(self, candidates: list[tuple[float, float]]) -> tuple[float, float, int]:
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
        best_yaw, best_probability = candidates[0]
        if self.current is None:
            self.current = best_yaw
            return best_yaw, best_probability, self.current_branch
        # The grid is kept on an unwrapped angle axis.  If the old branch is
        # more than half a turn away from every current candidate, it is stale
        # rather than a legitimate wrapped equivalent; re-anchor it.
        if abs(best_yaw - self.current) > math.pi:
            self.current = best_yaw
            self.current_branch += 1
            self.pending = None
            self.pending_frames = 0
            return best_yaw, best_probability, self.current_branch
        current_probability = max(
            (probability for yaw, probability in candidates if abs(wrap_error(yaw, self.current)) < math.radians(8.0)),
            default=0.0,
        )
        if best_probability > current_probability + self.cfg.branch_margin:
            if self.pending is not None and abs(wrap_error(best_yaw, self.pending)) < math.radians(4.0):
                self.pending_frames += 1
            else:
                self.pending = best_yaw
                self.pending_frames = 1
            if self.pending_frames >= self.cfg.branch_confirm_frames:
                self.current = best_yaw
                self.current_branch += 1
                self.pending = None
                self.pending_frames = 0
        else:
            self.pending = None
            self.pending_frames = 0
        current_probability = max(
            (probability for yaw, probability in candidates if abs(yaw - self.current) < math.radians(8.0)),
            default=0.0,
        )
        return self.current, current_probability, self.current_branch


class Executor:
    def __init__(self, cfg: Config) -> None:
        self.yaw = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.cfg = cfg

    def step(self, reference_yaw: float, reference_velocity: float, reference_acceleration: float) -> None:
        error = wrap_error(reference_yaw, self.yaw)
        command = reference_acceleration + self.cfg.kp * error + self.cfg.kd * (reference_velocity - self.velocity)
        self.acceleration = max(-self.cfg.executor_max_accel, min(self.cfg.executor_max_accel, command))
        self.yaw += self.velocity * self.cfg.dt + 0.5 * self.acceleration * self.cfg.dt**2
        self.velocity = max(-self.cfg.executor_max_speed, min(self.cfg.executor_max_speed, self.velocity + self.acceleration * self.cfg.dt))


def make_reference(mode: str, state: State, history: list[tuple[float, float, float]], cfg: Config, selector: BranchSelector | None) -> tuple[float, float, float, float, int]:
    horizon = cfg.delay
    nominal_yaw = state.angle + state.omega * horizon
    nominal_velocity = state.omega
    if mode == "constant_velocity":
        return nominal_yaw, nominal_velocity, 0.0, coverage_probability(nominal_yaw, nominal_yaw, cfg.distance, cfg), 0
    model = HistoryAccelerationModel(cfg, conditional=True)
    model.fit(history)
    distribution = model.future_angle_distribution(state, horizon)
    grid = [nominal_yaw + math.radians(i) for i in range(-90, 91)]
    candidates = []
    for yaw in grid:
        probability = sum(weight * coverage_probability(yaw, angle, cfg.distance, cfg) for angle, weight in distribution)
        if probability >= 0.01:
            candidates.append((yaw, probability))
    if not candidates:
        candidates = [(nominal_yaw, 0.0)]
    if selector:
        peak_yaw, peak_probability, branch = selector.choose(candidates)
    else:
        peak_yaw, peak_probability = max(candidates, key=lambda item: item[1])
        branch = 0
    # A single marker has one local geometric branch.  Reject an unwrapped
    # stale branch that is far outside the current reachable nominal window;
    # true multi-branch cases remain represented by candidates near nominal.
    if abs(peak_yaw - nominal_yaw) > math.pi / 4.0:
        peak_yaw = nominal_yaw
        peak_probability = max(
            (probability for yaw, probability in candidates if abs(yaw - nominal_yaw) <= math.pi / 12.0),
            default=0.0,
        )
    # Derivative of the selected branch is the measured angular velocity.
    return peak_yaw, state.omega, 0.0, peak_probability, branch


def run_trial(mode: str, cfg: Config, seed: int) -> list[Event]:
    rng = random.Random(seed)
    total_steps = round((cfg.warmup + cfg.duration) / cfg.dt)
    target = State(0.0, math.radians(300.0), cfg.max_alpha)
    direction = 1.0
    executor = Executor(cfg)
    history: list[tuple[float, float, float]] = []
    selector = BranchSelector(cfg) if mode == "probabilistic" else None
    target_states: list[State] = []
    pending: list[tuple[int, float, float, float, float, int, float]] = []
    events: list[Event] = []
    next_event = cfg.warmup
    reference = target.angle
    reference_velocity = target.omega
    reference_acceleration = 0.0
    probability = 0.0
    branch = 0
    planning_ms = 0.0
    for step in range(total_steps):
        now = (step + 1) * cfg.dt
        target, direction = advance_target(target, direction, cfg.dt, cfg)
        target_states.append(target)
        observed_angle = target.angle + rng.gauss(0.0, math.radians(0.15))
        observed_omega = target.omega + rng.gauss(0.0, math.radians(3.0))
        history.append((now, observed_angle, observed_omega))
        if step % max(1, round(0.05 / cfg.dt)) == 0:
            start = time.perf_counter()
            reference, reference_velocity, reference_acceleration, probability, branch = make_reference(
                mode, State(observed_angle, observed_omega, target.alpha), history[-300:], cfg, selector
            )
            planning_ms = (time.perf_counter() - start) * 1000.0
        delayed_steps = max(1, round(cfg.delay / cfg.dt))
        eval_step = step + delayed_steps
        predicted_yaw = executor.yaw + executor.velocity * cfg.delay + 0.5 * executor.acceleration * cfg.delay**2
        predicted_eval_angle = observed_angle + observed_omega * cfg.delay
        executor.step(reference, reference_velocity, reference_acceleration)
        ready = [item for item in pending if item[0] <= step]
        pending = [item for item in pending if item[0] > step]
        for ready_step, event_time, event_reference, event_predicted_yaw, event_probability, event_branch, event_planning_ms in ready:
            eval_state = target_states[min(ready_step, len(target_states) - 1)]
            actual_yaw = executor.yaw
            if event_time < cfg.warmup:
                continue
            covered_probability = coverage_probability(actual_yaw, eval_state.angle, cfg.distance, cfg)
            covered = rng.random() < covered_probability
            events.append(Event(event_time, mode, cfg.delay, eval_state.angle, eval_state.omega, eval_state.alpha, event_reference, event_predicted_yaw, actual_yaw, event_probability, covered, wrap_error(actual_yaw, eval_state.angle), event_branch, event_planning_ms))
        if now + 1e-12 < next_event:
            continue
        next_event += 0.05
        pending.append((eval_step, now, reference, predicted_yaw, probability, branch, planning_ms))
    return events


def summarize(events: list[Event]) -> dict[str, float | int | str]:
    probabilities = [event.predicted_probability for event in events]
    outcomes = [1.0 if event.covered else 0.0 for event in events]
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(events) if events else 0.0
    mean_prediction = statistics.mean(probabilities) if probabilities else 0.0
    actual_rate = statistics.mean(outcomes) if outcomes else 0.0
    bins = [[] for _ in range(10)]
    for event in events:
        bins[min(9, int(event.predicted_probability * 10))].append(event)
    ece = sum(len(bucket) / len(events) * abs(statistics.mean([e.predicted_probability for e in bucket]) - statistics.mean([float(e.covered) for e in bucket])) for bucket in bins if bucket) if events else 0.0
    errors = [abs(event.angle_error) for event in events]
    return {"mode": events[0].mode if events else "", "events": len(events), "coverage_rate": actual_rate, "mean_prediction": mean_prediction, "brier_score": brier, "ece": ece, "mae_deg": math.degrees(statistics.mean(errors)) if errors else 0.0, "rmse_deg": math.degrees(math.sqrt(statistics.mean([e * e for e in errors]))) if errors else 0.0, "p95_error_deg": math.degrees(sorted(errors)[max(0, int(0.95 * len(errors)) - 1)]) if errors else 0.0, "events_per_second": len(events) / max(1e-9, events[-1].timestamp - events[0].timestamp) if len(events) > 1 else 0.0, "branch_switches": max((event.branch for event in events), default=0), "planning_p50_ms": statistics.median([e.planning_ms for e in events]) if events else 0.0, "planning_p95_ms": sorted([e.planning_ms for e in events])[max(0, int(0.95 * len(events)) - 1)] if events else 0.0}


def write_outputs(output: Path, details: list[Event], summaries: list[dict[str, float | int | str]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(Event.__dataclass_fields__.keys())
        writer.writerows(event.__dict__.values() for event in details)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt:
        for title, attr, ylabel in (("coverage-reliability", "predicted_probability", "actual coverage"), ("tracking-error", "angle_error", "angle error (rad)"), ("planning-time", "planning_ms", "planning time (ms)")):
            plt.figure(figsize=(7, 4))
            for mode in sorted({event.mode for event in details}):
                values = [getattr(event, attr) for event in details if event.mode == mode]
                if title == "coverage-reliability":
                    bins = [[] for _ in range(10)]
                    for event in details:
                        if event.mode == mode:
                            bins[min(9, int(event.predicted_probability * 10))].append(event)
                    xs = [statistics.mean([e.predicted_probability for e in bucket]) for bucket in bins if bucket]
                    ys = [statistics.mean([float(e.covered) for e in bucket]) for bucket in bins if bucket]
                    plt.plot(xs, ys, "o-", label=mode)
                else:
                    plt.plot(values, ".", label=mode, alpha=0.5)
            if title == "coverage-reliability":
                plt.plot([0, 1], [0, 1], "k--", label="ideal")
                plt.xlabel("mean predicted coverage probability")
            else:
                plt.xlabel("evaluation event")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output / f"{title}.png", dpi=140)
            plt.close()
    else:
        def svg_plot(path: Path, title: str, series: list[tuple[str, list[tuple[float, float]]]], x_label: str, y_label: str) -> None:
            width, height = 760, 420
            all_points = [point for _, points in series for point in points]
            xmin = min((point[0] for point in all_points), default=0.0)
            xmax = max((point[0] for point in all_points), default=1.0)
            ymin = min((point[1] for point in all_points), default=0.0)
            ymax = max((point[1] for point in all_points), default=1.0)
            if xmax <= xmin: xmax = xmin + 1.0
            if ymax <= ymin: ymax = ymin + 1.0
            def point_xy(point: tuple[float, float]) -> str:
                x = 70 + (point[0] - xmin) / (xmax - xmin) * 660
                y = 370 - (point[1] - ymin) / (ymax - ymin) * 320
                return f"{x:.1f},{y:.1f}"
            colors = ("#1565c0", "#c62828", "#2e7d32")
            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', f'<text x="70" y="24" font-size="18">{title}</text>', '<line x1="70" y1="370" x2="730" y2="370" stroke="#555"/><line x1="70" y1="50" x2="70" y2="370" stroke="#555"/>', f'<text x="380" y="410" text-anchor="middle" font-size="13">{x_label}</text>', f'<text x="16" y="220" transform="rotate(-90 16 220)" text-anchor="middle" font-size="13">{y_label}</text>']
            for index, (label, points) in enumerate(series):
                if points:
                    parts.append(f'<polyline fill="none" stroke="{colors[index % len(colors)]}" stroke-width="2" points="{" ".join(point_xy(point) for point in points)}"/>')
                    parts.append(f'<text x="{590 + (index % 2) * 70}" y="{45 + (index // 2) * 18}" font-size="12" fill="{colors[index % len(colors)]}">{label}</text>')
            parts.append("</svg>")
            path.write_text("".join(parts), encoding="utf-8")
        modes = sorted({event.mode for event in details})
        reliability = []
        for mode in modes:
            buckets = [[] for _ in range(10)]
            for event in details:
                if event.mode == mode:
                    buckets[min(9, int(event.predicted_probability * 10))].append(event)
            reliability.append((mode, [(statistics.mean([e.predicted_probability for e in bucket]), statistics.mean([float(e.covered) for e in bucket])) for bucket in buckets if bucket]))
        svg_plot(output / "coverage-reliability.svg", "coverage reliability", reliability, "predicted coverage probability", "actual coverage")
        svg_plot(output / "tracking-error.svg", "tracking error", [(mode, [(i, event.angle_error) for i, event in enumerate(details) if event.mode == mode]) for mode in modes], "evaluation event", "angle error (rad)")
        svg_plot(output / "planning-time.svg", "planning time", [(mode, [(i, event.planning_ms) for i, event in enumerate(details) if event.mode == mode]) for mode in modes], "evaluation event", "planning time (ms)")
    report = output / "report.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# 2D moving-target tracking experiment\n\n")
        handle.write("This experiment excludes firing, ballistics, armor, and hit semantics. `covered` means the noisy execution direction fell inside the moving marker interval.\n\n")
        handle.write("## Summary\n\n| mode | events | coverage rate | Brier | ECE | MAE (deg) | RMSE (deg) | events/s | P50 ms | P95 ms |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for item in summaries:
            handle.write(f"| {item['mode']} | {item['events']} | {item['coverage_rate']:.4f} | {item['brier_score']:.5f} | {item['ece']:.5f} | {item['mae_deg']:.3f} | {item['rmse_deg']:.3f} | {item['events_per_second']:.3f} | {item['planning_p50_ms']:.3f} | {item['planning_p95_ms']:.3f} |\n")
        handle.write("\n## Confirmed\n\n- Both modes use the same constrained feed-forward plus PD executor.\n- Probability estimates are deterministic weighted sums over filtered historical acceleration samples.\n- History uses cubic Hermite interpolation at the non-grid horizon.\n- Warm-up events are excluded from the output statistics.\n\n## Not established\n\n- This short default run does not establish global superiority. Run at least 20 seeds per scenario before drawing a conclusion.\n")


def self_test() -> None:
    rows = [(0.0, 0.0, 1.0), (0.1, 0.1, 1.0)]
    position, velocity = hermite_angle(rows, 0.05)
    assert abs(position - 0.05) < 1e-9 and abs(velocity - 1.0) < 1e-9
    cfg = Config(history_horizon=0.255)
    model = HistoryAccelerationModel(cfg)
    history = [(i * 0.01, 0.2 * i * 0.01, 0.2) for i in range(50)]
    model.fit(history)
    weighted = model.weighted_accelerations(0.2)
    assert weighted and abs(sum(weight for _, weight in weighted) - 1.0) < 1e-12
    assert all(-cfg.max_alpha - 1e-12 <= a <= cfg.max_alpha + 1e-12 for a, _ in weighted)
    state = State(0.4, 0.2, 0.0)
    first = make_reference("probabilistic", state, history, cfg, BranchSelector(cfg))
    second = make_reference("probabilistic", state, history, cfg, BranchSelector(cfg))
    assert first == second and 0.0 <= first[3] <= 1.0
    selector = BranchSelector(Config(branch_margin=0.2, branch_confirm_frames=3))
    selector.choose([(0.0, 0.5), (1.0, 0.4)])
    selector.choose([(1.0, 0.71), (0.0, 0.5)])
    selector.choose([(1.0, 0.71), (0.0, 0.5)])
    assert selector.current_branch == 0
    selector.choose([(1.0, 0.71), (0.0, 0.5)])
    assert selector.current_branch == 1
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("tracking_experiment_output"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--delay-ms", type=float, nargs="+", default=[0, 15, 30, 50, 100])
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    all_events: list[Event] = []
    summaries: list[dict[str, float | int | str]] = []
    for delay_ms in args.delay_ms:
        for mode in ("constant_velocity", "probabilistic"):
            delay = delay_ms / 1000.0
            config = Config(duration=args.duration, delay=delay, history_horizon=0.255)
            events = [event for seed in range(args.seeds) for event in run_trial(mode, config, config.seed + seed)]
            for event in events:
                event.delay = delay_ms / 1000.0
            all_events.extend(events)
            summary = summarize(events)
            summary["delay_ms"] = delay_ms
            summary["events_per_second"] = len(events) / max(config.duration * args.seeds, 1e-9)
            summaries.append(summary)
            print(f"delay={delay_ms:5.1f} mode={mode:18s} coverage={summary['coverage_rate']:.3f} MAE={summary['mae_deg']:.3f} valid/s={summary['events_per_second']:.2f} p95_ms={summary['planning_p95_ms']:.3f}")
    write_outputs(args.output, all_events, summaries)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
