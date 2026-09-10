"""Safe 2D moving-marker tracking experiment, version 2.

This program studies prediction, probability calibration, branch selection, and
constrained rotary-executor tracking. It contains no firing, ballistics, armor,
or automatic-trigger logic.

Key additions over v1:
- exact continuous event times, including true 0 ms delay;
- target and executor state interpolation at each evaluation time;
- constant-velocity, historical-probability, blended fallback, and oracle modes;
- ideal and constrained executors;
- probability evaluated at the predicted executor direction;
- prediction, tracking, executor-forecast, and final-error decomposition;
- oracle upper-bound and internal no-regret checks;
- common random numbers across modes;
- per-seed, aggregate, calibration, and event CSV outputs.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

TAU = 2.0 * math.pi


@dataclass(frozen=True)
class Config:
    dt: float = 0.01
    plan_period: float = 0.05
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
    history_window: float = 3.0
    minimum_history_horizon: float = 0.05
    velocity_sigma: float = math.radians(120.0)
    margin_sigma: float = math.radians(100.0)
    opposite_acceleration_weight: float = 0.08
    minimum_ess: float = 8.0
    full_history_ess: float = 35.0
    observation_angle_sigma: float = math.radians(0.15)
    observation_omega_sigma: float = math.radians(3.0)
    scatter_lateral_sigma: float | None = None
    grid_half_width: float = math.radians(90.0)
    grid_step: float = math.radians(1.0)
    branch_margin: float = 0.035
    branch_confirm_frames: int = 3
    branch_match_radius: float = math.radians(8.0)
    max_prediction_correction: float = math.radians(10.0)
    executor_residual_sigma_floor: float = math.radians(0.15)
    seed: int = 11

    def __post_init__(self) -> None:
        if self.scatter_lateral_sigma is None:
            sigma = (self.marker_width / 2.0) / 1.6448536269514722
            object.__setattr__(self, "scatter_lateral_sigma", sigma)
        if self.dt <= 0 or self.plan_period <= 0:
            raise ValueError("dt and plan_period must be positive")
        if self.duration <= 0 or self.distance <= 0:
            raise ValueError("duration and distance must be positive")


@dataclass(frozen=True)
class State:
    angle: float
    omega: float
    alpha: float


@dataclass(frozen=True)
class ExecutorState:
    yaw: float
    velocity: float
    acceleration: float


@dataclass(frozen=True)
class HistoryAccelerationSample:
    timestamp: float
    velocity: float
    acceleration: float
    acceleration_sign: int
    speed_margin: float


@dataclass(frozen=True)
class DistributionPoint:
    angle: float
    weight: float


@dataclass(frozen=True)
class Plan:
    mode: str
    reference_yaw: float
    reference_velocity: float
    reference_acceleration: float
    cv_yaw: float
    model_peak_yaw: float
    peak_probability: float
    predicted_executor_yaw: float
    predicted_executor_probability: float
    cv_model_probability: float
    effective_sample_size: float
    history_blend: float
    branch: int
    hysteresis_regret: float
    planning_ms: float
    oracle_yaw: float
    oracle_best_probability: float
    oracle_cv_probability: float
    oracle_model_probability: float


@dataclass(frozen=True)
class PendingEvent:
    event_id: int
    created_at: float
    evaluate_at: float
    observed_state: State
    plan: Plan


@dataclass(frozen=True)
class Event:
    seed: int
    scenario: str
    executor_mode: str
    mode: str
    event_id: int
    created_at: float
    evaluate_at: float
    requested_delay_ms: float
    actual_delay_ms: float
    target_angle: float
    target_omega: float
    target_alpha: float
    cv_yaw: float
    reference_yaw: float
    model_peak_yaw: float
    predicted_executor_yaw: float
    actual_executor_yaw: float
    peak_probability: float
    predicted_probability: float
    actual_coverage_probability: float
    covered: bool
    cv_prediction_error: float
    model_prediction_error: float
    tracking_error: float
    executor_forecast_error: float
    final_error: float
    branch: int
    hysteresis_regret: float
    effective_sample_size: float
    history_blend: float
    planning_ms: float
    oracle_yaw: float
    oracle_best_probability: float
    oracle_cv_probability: float
    oracle_model_probability: float


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_error(target: float, current: float) -> float:
    return (target - current + math.pi) % TAU - math.pi


def sign(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def normal_cdf_interval(center_error: float, half_width: float, sigma: float) -> float:
    sigma = max(sigma, 1e-12)
    scale = sigma * math.sqrt(2.0)
    return 0.5 * (
        math.erf((center_error + half_width) / scale)
        - math.erf((center_error - half_width) / scale)
    )


def coverage_probability(
    yaw: float,
    target_angle: float,
    cfg: Config,
    extra_angular_sigma: float = 0.0,
) -> float:
    half_angle = math.atan2(cfg.marker_width / 2.0, cfg.distance)
    scatter_sigma = float(cfg.scatter_lateral_sigma) / cfg.distance
    total_sigma = math.sqrt(scatter_sigma**2 + max(0.0, extra_angular_sigma) ** 2)
    error = wrap_error(yaw, target_angle)
    return clamp(normal_cdf_interval(error, half_angle, total_sigma), 0.0, 1.0)


def advance_target(state: State, direction: float, dt: float, cfg: Config) -> tuple[State, float]:
    """Advance bang-bang motion while resolving an in-step speed-limit reflection."""
    remaining = dt
    angle = state.angle
    omega = state.omega
    direction = 1.0 if direction >= 0 else -1.0
    while remaining > 1e-12:
        boundary = cfg.max_omega if direction > 0 else -cfg.max_omega
        alpha = direction * cfg.max_alpha
        time_to_boundary = max(0.0, (boundary - omega) / alpha)
        step = min(remaining, time_to_boundary)
        angle += omega * step + 0.5 * alpha * step * step
        omega += alpha * step
        remaining -= step
        if time_to_boundary <= step + 1e-12:
            omega = boundary
            direction = -direction
        else:
            break
    return State(angle, omega, direction * cfg.max_alpha), direction


def advance_target_exact(state: State, direction: float, horizon: float, cfg: Config) -> State:
    return advance_target(state, direction, max(0.0, horizon), cfg)[0]


def interpolate_state(rows: list[tuple[float, State]], timestamp: float) -> State:
    """Cubic Hermite interpolation of continuous angle and linear alpha interpolation."""
    if not rows:
        raise ValueError("empty state history")
    if timestamp <= rows[0][0] + 1e-12:
        return rows[0][1]
    if timestamp >= rows[-1][0] - 1e-12:
        return rows[-1][1]
    lo, hi = 0, len(rows) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] < timestamp:
            lo = mid
        else:
            hi = mid
    t0, s0 = rows[lo]
    t1, s1 = rows[hi]
    h = t1 - t0
    u = (timestamp - t0) / h
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    angle = h00 * s0.angle + h10 * h * s0.omega + h01 * s1.angle + h11 * h * s1.omega
    dh00 = 6 * u**2 - 6 * u
    dh10 = 3 * u**2 - 4 * u + 1
    dh01 = -6 * u**2 + 6 * u
    dh11 = 3 * u**2 - 2 * u
    omega = (dh00 * s0.angle + dh10 * h * s0.omega + dh01 * s1.angle + dh11 * h * s1.omega) / h
    alpha = s0.alpha + u * (s1.alpha - s0.alpha)
    return State(angle, omega, alpha)


def interpolate_executor(rows: list[tuple[float, ExecutorState]], timestamp: float) -> ExecutorState:
    if not rows:
        raise ValueError("empty executor history")
    if timestamp <= rows[0][0] + 1e-12:
        return rows[0][1]
    if timestamp >= rows[-1][0] - 1e-12:
        return rows[-1][1]
    lo, hi = 0, len(rows) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if rows[mid][0] < timestamp:
            lo = mid
        else:
            hi = mid
    t0, e0 = rows[lo]
    t1, e1 = rows[hi]
    u = (timestamp - t0) / (t1 - t0)
    return ExecutorState(
        yaw=e0.yaw + u * (e1.yaw - e0.yaw),
        velocity=e0.velocity + u * (e1.velocity - e0.velocity),
        acceleration=e0.acceleration + u * (e1.acceleration - e0.acceleration),
    )


class HistoryAccelerationModel:
    """Query-conditioned empirical average-acceleration model."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def samples_for_horizon(
        self,
        history: list[tuple[float, State]],
        horizon: float,
    ) -> list[HistoryAccelerationSample]:
        if len(history) < 2:
            return []
        estimation_horizon = max(horizon, self.cfg.minimum_history_horizon)
        end_time = history[-1][0]
        start_cutoff = max(history[0][0], end_time - self.cfg.history_window)
        output: list[HistoryAccelerationSample] = []
        for timestamp, state in history:
            if timestamp < start_cutoff or timestamp + estimation_horizon > end_time + 1e-12:
                continue
            future = interpolate_state(history, timestamp + estimation_horizon)
            acceleration = 2.0 * (
                future.angle - state.angle - state.omega * estimation_horizon
            ) / estimation_horizon**2
            if abs(acceleration) > self.cfg.max_alpha * 1.05:
                continue
            margin = self.cfg.max_omega - abs(state.omega)
            output.append(
                HistoryAccelerationSample(
                    timestamp=timestamp,
                    velocity=state.omega,
                    acceleration=acceleration,
                    acceleration_sign=sign(state.alpha),
                    speed_margin=margin,
                )
            )
        return output

    def weighted_accelerations(
        self,
        history: list[tuple[float, State]],
        query: State,
        horizon: float,
    ) -> tuple[list[tuple[float, float]], float]:
        samples = self.samples_for_horizon(history, horizon)
        if not samples:
            return [(0.0, 1.0)], 1.0
        physical_horizon = max(horizon, 1e-9)
        lower = max(-self.cfg.max_alpha, (-self.cfg.max_omega - query.omega) / physical_horizon)
        upper = min(self.cfg.max_alpha, (self.cfg.max_omega - query.omega) / physical_horizon)
        query_margin = self.cfg.max_omega - abs(query.omega)
        weighted: list[tuple[float, float]] = []
        for sample in samples:
            if not lower <= sample.acceleration <= upper:
                continue
            velocity_weight = math.exp(
                -0.5 * ((sample.velocity - query.omega) / self.cfg.velocity_sigma) ** 2
            )
            margin_weight = math.exp(
                -0.5 * ((sample.speed_margin - query_margin) / self.cfg.margin_sigma) ** 2
            )
            direction_weight = (
                1.0
                if sample.acceleration_sign == sign(query.alpha)
                else self.cfg.opposite_acceleration_weight
            )
            weight = velocity_weight * margin_weight * direction_weight
            if weight > 1e-12:
                weighted.append((sample.acceleration, weight))
        if not weighted:
            return [(0.0, 1.0)], 1.0
        total = sum(weight for _, weight in weighted)
        normalized = [(a, w / total) for a, w in weighted]
        ess = 1.0 / sum(weight * weight for _, weight in normalized)
        return normalized, ess

    def distribution(
        self,
        history: list[tuple[float, State]],
        query: State,
        horizon: float,
    ) -> tuple[list[DistributionPoint], float]:
        accelerations, ess = self.weighted_accelerations(history, query, horizon)
        points = [
            DistributionPoint(
                query.angle + query.omega * horizon + 0.5 * acceleration * horizon**2,
                weight,
            )
            for acceleration, weight in accelerations
        ]
        return points, ess


class BranchSelector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.current_yaw: float | None = None
        self.current_branch = 0
        self.pending_yaw: float | None = None
        self.pending_frames = 0

    def choose(self, candidates: list[tuple[float, float]]) -> tuple[float, float, int, float]:
        ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
        best_yaw, best_probability = ordered[0]
        if self.current_yaw is None:
            self.current_yaw = best_yaw
            return best_yaw, best_probability, self.current_branch, 0.0
        current_candidate = min(
            ordered,
            key=lambda item: abs(wrap_error(item[0], self.current_yaw or item[0])),
        )
        current_yaw, current_probability = current_candidate
        if abs(wrap_error(current_yaw, self.current_yaw)) <= self.cfg.branch_match_radius:
            self.current_yaw = current_yaw
        if best_probability > current_probability + self.cfg.branch_margin:
            if self.pending_yaw is not None and abs(wrap_error(best_yaw, self.pending_yaw)) < self.cfg.branch_match_radius / 2:
                self.pending_frames += 1
            else:
                self.pending_yaw = best_yaw
                self.pending_frames = 1
            if self.pending_frames >= self.cfg.branch_confirm_frames:
                self.current_yaw = best_yaw
                self.current_branch += 1
                current_probability = best_probability
                self.pending_yaw = None
                self.pending_frames = 0
        else:
            self.pending_yaw = None
            self.pending_frames = 0
        selected = min(
            ordered,
            key=lambda item: abs(wrap_error(item[0], self.current_yaw or item[0])),
        )
        regret = max(0.0, best_probability - selected[1])
        return selected[0], selected[1], self.current_branch, regret


class Executor:
    def __init__(self, cfg: Config, mode: str) -> None:
        self.cfg = cfg
        self.mode = mode
        self.state = ExecutorState(0.0, 0.0, 0.0)

    def step(self, reference_yaw: float, reference_velocity: float, reference_acceleration: float, dt: float) -> None:
        if self.mode == "ideal":
            self.state = ExecutorState(reference_yaw, reference_velocity, reference_acceleration)
            return
        error = wrap_error(reference_yaw, self.state.yaw)
        command = (
            reference_acceleration
            + self.cfg.kp * error
            + self.cfg.kd * (reference_velocity - self.state.velocity)
        )
        acceleration = clamp(command, -self.cfg.executor_max_accel, self.cfg.executor_max_accel)
        yaw = self.state.yaw + self.state.velocity * dt + 0.5 * acceleration * dt**2
        velocity = clamp(
            self.state.velocity + acceleration * dt,
            -self.cfg.executor_max_speed,
            self.cfg.executor_max_speed,
        )
        self.state = ExecutorState(yaw, velocity, acceleration)

    def predict(
        self,
        reference_yaw: float,
        reference_velocity: float,
        reference_acceleration: float,
        horizon: float,
    ) -> float:
        # An ideal executor applies a newly issued reference immediately,
        # including at an exact 0 ms evaluation horizon.
        if self.mode == "ideal":
            return reference_yaw
        if horizon <= 0.0:
            return self.state.yaw
        clone = Executor(self.cfg, self.mode)
        clone.state = self.state
        remaining = horizon
        while remaining > 1e-12:
            step = min(self.cfg.dt, remaining)
            clone.step(reference_yaw, reference_velocity, reference_acceleration, step)
            remaining -= step
        return clone.state.yaw


def distribution_probability(
    yaw: float,
    distribution: Iterable[DistributionPoint],
    cfg: Config,
    executor_sigma: float = 0.0,
) -> float:
    return sum(
        point.weight * coverage_probability(yaw, point.angle, cfg, executor_sigma)
        for point in distribution
    )


def grid_around(center: float, cfg: Config) -> list[float]:
    count = int(round(cfg.grid_half_width / cfg.grid_step))
    return [center + i * cfg.grid_step for i in range(-count, count + 1)]


def oracle_plan(true_state: State, direction: float, horizon: float, cfg: Config, candidates: list[float]) -> tuple[float, float, float]:
    true_future = advance_target_exact(true_state, direction, horizon, cfg)
    unique_candidates = list(dict.fromkeys(candidates + [true_future.angle]))
    scored = [(yaw, coverage_probability(yaw, true_future.angle, cfg)) for yaw in unique_candidates]
    yaw, probability = max(scored, key=lambda item: item[1])
    return yaw, probability, true_future.angle


def make_plan(
    mode: str,
    observed: State,
    true_state: State,
    target_direction: float,
    history: list[tuple[float, State]],
    executor: Executor,
    selector: BranchSelector | None,
    executor_residual_sigma: float,
    cfg: Config,
) -> Plan:
    start = time.perf_counter()
    horizon = cfg.delay
    cv_yaw = observed.angle + observed.omega * horizon
    cv_distribution = [DistributionPoint(cv_yaw, 1.0)]
    history_model = HistoryAccelerationModel(cfg)
    historical_distribution, ess = history_model.distribution(history, observed, horizon)
    history_blend = clamp(
        (ess - cfg.minimum_ess) / max(cfg.full_history_ess - cfg.minimum_ess, 1e-9),
        0.0,
        1.0,
    )
    blended_distribution = [
        DistributionPoint(point.angle, point.weight * history_blend)
        for point in historical_distribution
    ] + [DistributionPoint(cv_yaw, 1.0 - history_blend)]
    if mode == "constant_velocity":
        distribution = cv_distribution
    elif mode == "historical":
        distribution = historical_distribution
    elif mode == "blend":
        distribution = blended_distribution
    elif mode == "oracle":
        true_future = advance_target_exact(true_state, target_direction, horizon, cfg)
        distribution = [DistributionPoint(true_future.angle, 1.0)]
        ess = float("inf")
        history_blend = 1.0
    else:
        raise ValueError(f"unknown mode: {mode}")

    scan_grid = grid_around(cv_yaw, cfg)
    if all(abs(wrap_error(yaw, cv_yaw)) > 1e-12 for yaw in scan_grid):
        scan_grid.append(cv_yaw)
    scored = [
        (yaw, distribution_probability(yaw, distribution, cfg, executor_residual_sigma))
        for yaw in scan_grid
    ]
    best_index = max(range(len(scored)), key=lambda index: scored[index][1])
    best_yaw, best_probability = scored[best_index]
    cv_model_probability = distribution_probability(cv_yaw, distribution, cfg, executor_residual_sigma)
    # Branch hysteresis must operate on local maxima, not on every grid point.
    # Passing the full grid caused the selector to keep a stale yaw while a
    # single peak moved continuously, producing artificial 50-60 degree lag.
    peak_candidates = [
        scored[index]
        for index in range(1, len(scored) - 1)
        if scored[index][1] >= scored[index - 1][1]
        and scored[index][1] >= scored[index + 1][1]
    ]
    peak_candidates.extend([(best_yaw, best_probability), (cv_yaw, cv_model_probability)])
    # Deduplicate numerically identical peaks. At zero horizon, every
    # acceleration hypothesis collapses to the current angle, so best and CV
    # are the same branch and must not trigger hysteresis.
    unique_peaks: list[tuple[float, float]] = []
    for candidate in sorted(peak_candidates, key=lambda item: item[1], reverse=True):
        if not any(abs(wrap_error(candidate[0], existing[0])) < 1e-8 for existing in unique_peaks):
            unique_peaks.append(candidate)
    if selector is not None and len(unique_peaks) > 1:
        selected_yaw, selected_probability, branch, regret = selector.choose(unique_peaks)
    else:
        selected_yaw, selected_probability, branch, regret = best_yaw, best_probability, 0, 0.0

    # A learned branch is a correction to the constant-velocity estimate, not
    # an unrestricted replacement.  Bound it by the reachable uncertainty
    # window so a stale or noisy historical mode cannot command a multi-turn
    # jump. The executor remains identical for every mode.
    correction = clamp(
        wrap_error(selected_yaw, cv_yaw),
        -cfg.max_prediction_correction,
        cfg.max_prediction_correction,
    )
    selected_yaw = cv_yaw + correction
    selected_probability = distribution_probability(
        selected_yaw, distribution, cfg, executor_residual_sigma
    )

    reference_velocity = observed.omega
    reference_acceleration = observed.alpha if mode == "oracle" else 0.0
    predicted_executor_yaw = executor.predict(
        selected_yaw,
        reference_velocity,
        reference_acceleration,
        horizon,
    )
    predicted_executor_probability = distribution_probability(
        predicted_executor_yaw,
        distribution,
        cfg,
        executor_residual_sigma,
    )

    oracle_candidates = scan_grid + [cv_yaw, selected_yaw, predicted_executor_yaw]
    oracle_yaw, oracle_best_probability, true_future_angle = oracle_plan(
        true_state, target_direction, horizon, cfg, oracle_candidates
    )
    oracle_cv_probability = coverage_probability(cv_yaw, true_future_angle, cfg)
    oracle_model_probability = coverage_probability(selected_yaw, true_future_angle, cfg)
    if oracle_best_probability + 1e-12 < oracle_cv_probability:
        raise AssertionError("oracle best probability is below CV candidate")
    if best_probability + 1e-12 < cv_model_probability:
        raise AssertionError("model optimum is below included CV candidate")

    planning_ms = (time.perf_counter() - start) * 1000.0
    return Plan(
        mode=mode,
        reference_yaw=selected_yaw,
        reference_velocity=reference_velocity,
        reference_acceleration=reference_acceleration,
        cv_yaw=cv_yaw,
        model_peak_yaw=best_yaw,
        peak_probability=best_probability,
        predicted_executor_yaw=predicted_executor_yaw,
        predicted_executor_probability=predicted_executor_probability,
        cv_model_probability=cv_model_probability,
        effective_sample_size=ess,
        history_blend=history_blend,
        branch=branch,
        hysteresis_regret=regret,
        planning_ms=planning_ms,
        oracle_yaw=oracle_yaw,
        oracle_best_probability=oracle_best_probability,
        oracle_cv_probability=oracle_cv_probability,
        oracle_model_probability=oracle_model_probability,
    )


def common_uniform(seed: int, event_id: int) -> float:
    return random.Random(seed * 1_000_003 + event_id * 97_409 + 17).random()


def run_trial(mode: str, executor_mode: str, cfg: Config, seed: int) -> list[Event]:
    observation_rng = random.Random(seed)
    target = State(0.0, math.radians(300.0), cfg.max_alpha)
    target_direction = 1.0
    executor = Executor(cfg, executor_mode)
    target_history: list[tuple[float, State]] = [(0.0, target)]
    observed_history: list[tuple[float, State]] = []
    executor_history: list[tuple[float, ExecutorState]] = [(0.0, executor.state)]
    selector = BranchSelector(cfg) if mode in {"historical", "blend"} else None
    pending: list[PendingEvent] = []
    events: list[Event] = []
    residuals: list[float] = []
    current_plan: Plan | None = None
    # Anchor planning and evaluation to the same absolute time lattice.
    # The v1-style `next_plan = now + period` schedule drifted by one dt and
    # compared events with plans from a different timestamp.
    next_plan_time = cfg.dt
    next_event_time = cfg.warmup + cfg.dt
    event_id = 0
    now = 0.0
    end_time = cfg.warmup + cfg.duration + cfg.delay + cfg.dt

    while now < end_time - 1e-12:
        step = min(cfg.dt, end_time - now)
        target, target_direction = advance_target(target, target_direction, step, cfg)
        now += step
        target_history.append((now, target))

        observed = State(
            target.angle + observation_rng.gauss(0.0, cfg.observation_angle_sigma),
            target.omega + observation_rng.gauss(0.0, cfg.observation_omega_sigma),
            target.alpha,
        )
        observed_history.append((now, observed))
        cutoff = now - cfg.history_window - max(cfg.delay, cfg.minimum_history_horizon) - cfg.dt
        while len(observed_history) > 2 and observed_history[1][0] < cutoff:
            observed_history.pop(0)

        if current_plan is None or now + 1e-12 >= next_plan_time:
            residual_sigma = max(
                cfg.executor_residual_sigma_floor,
                statistics.pstdev(residuals[-200:]) if len(residuals) >= 5 else 0.0,
            )
            current_plan = make_plan(
                mode,
                observed,
                target,
                target_direction,
                observed_history,
                executor,
                selector,
                residual_sigma,
                cfg,
            )
            while next_plan_time <= now + 1e-12:
                next_plan_time += cfg.plan_period

        executor.step(
            current_plan.reference_yaw,
            current_plan.reference_velocity,
            current_plan.reference_acceleration,
            step,
        )
        executor_history.append((now, executor.state))

        while now + 1e-12 >= next_event_time and next_event_time < cfg.warmup + cfg.duration - 1e-12:
            event_id += 1
            pending.append(
                PendingEvent(
                    event_id=event_id,
                    created_at=next_event_time,
                    evaluate_at=next_event_time + cfg.delay,
                    observed_state=observed,
                    plan=current_plan,
                )
            )
            next_event_time += cfg.plan_period

        ready = [item for item in pending if item.evaluate_at <= now + 1e-12]
        pending = [item for item in pending if item.evaluate_at > now + 1e-12]
        for item in ready:
            true_future = interpolate_state(target_history, item.evaluate_at)
            actual_executor = interpolate_executor(executor_history, item.evaluate_at)
            actual_probability = coverage_probability(actual_executor.yaw, true_future.angle, cfg)
            covered = common_uniform(seed, item.event_id) < actual_probability
            executor_forecast_error = wrap_error(actual_executor.yaw, item.plan.predicted_executor_yaw)
            residuals.append(executor_forecast_error)
            events.append(
                Event(
                    seed=seed,
                    scenario="periodic_reflection",
                    executor_mode=executor_mode,
                    mode=mode,
                    event_id=item.event_id,
                    created_at=item.created_at,
                    evaluate_at=item.evaluate_at,
                    requested_delay_ms=cfg.delay * 1000.0,
                    actual_delay_ms=(item.evaluate_at - item.created_at) * 1000.0,
                    target_angle=true_future.angle,
                    target_omega=true_future.omega,
                    target_alpha=true_future.alpha,
                    cv_yaw=item.plan.cv_yaw,
                    reference_yaw=item.plan.reference_yaw,
                    model_peak_yaw=item.plan.model_peak_yaw,
                    predicted_executor_yaw=item.plan.predicted_executor_yaw,
                    actual_executor_yaw=actual_executor.yaw,
                    peak_probability=item.plan.peak_probability,
                    predicted_probability=item.plan.predicted_executor_probability,
                    actual_coverage_probability=actual_probability,
                    covered=covered,
                    cv_prediction_error=wrap_error(item.plan.cv_yaw, true_future.angle),
                    model_prediction_error=wrap_error(item.plan.reference_yaw, true_future.angle),
                    tracking_error=wrap_error(actual_executor.yaw, item.plan.reference_yaw),
                    executor_forecast_error=executor_forecast_error,
                    final_error=wrap_error(actual_executor.yaw, true_future.angle),
                    branch=item.plan.branch,
                    hysteresis_regret=item.plan.hysteresis_regret,
                    effective_sample_size=item.plan.effective_sample_size,
                    history_blend=item.plan.history_blend,
                    planning_ms=item.plan.planning_ms,
                    oracle_yaw=item.plan.oracle_yaw,
                    oracle_best_probability=item.plan.oracle_best_probability,
                    oracle_cv_probability=item.plan.oracle_cv_probability,
                    oracle_model_probability=item.plan.oracle_model_probability,
                )
            )
    return events


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = clamp(q, 0.0, 1.0) * (len(ordered) - 1)
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return ordered[lo]
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(events: list[Event]) -> dict[str, float | int | str]:
    if not events:
        return {}
    outcomes = [float(event.covered) for event in events]
    predictions = [event.predicted_probability for event in events]
    actual_probabilities = [event.actual_coverage_probability for event in events]
    final_errors = [abs(event.final_error) for event in events]
    prediction_errors = [abs(event.model_prediction_error) for event in events]
    tracking_errors = [abs(event.tracking_error) for event in events]
    forecast_errors = [abs(event.executor_forecast_error) for event in events]
    bins: list[list[Event]] = [[] for _ in range(10)]
    for event in events:
        bins[min(9, int(clamp(event.predicted_probability, 0.0, 0.999999) * 10))].append(event)
    ece = sum(
        len(bucket) / len(events)
        * abs(
            statistics.mean(e.predicted_probability for e in bucket)
            - statistics.mean(float(e.covered) for e in bucket)
        )
        for bucket in bins
        if bucket
    )
    oracle_violations = sum(
        event.oracle_best_probability + 1e-12 < event.oracle_cv_probability
        for event in events
    )
    return {
        "scenario": events[0].scenario,
        "executor_mode": events[0].executor_mode,
        "mode": events[0].mode,
        "delay_ms": events[0].requested_delay_ms,
        "events": len(events),
        "coverage_rate": statistics.mean(outcomes),
        "mean_predicted_probability": statistics.mean(predictions),
        "mean_actual_probability": statistics.mean(actual_probabilities),
        "brier_score": statistics.mean((p - y) ** 2 for p, y in zip(predictions, outcomes)),
        "ece": ece,
        "final_mae_deg": math.degrees(statistics.mean(final_errors)),
        "final_rmse_deg": math.degrees(math.sqrt(statistics.mean(e * e for e in final_errors))),
        "final_p95_deg": math.degrees(percentile(final_errors, 0.95)),
        "prediction_mae_deg": math.degrees(statistics.mean(prediction_errors)),
        "tracking_mae_deg": math.degrees(statistics.mean(tracking_errors)),
        "executor_forecast_mae_deg": math.degrees(statistics.mean(forecast_errors)),
        "mean_oracle_best_probability": statistics.mean(e.oracle_best_probability for e in events),
        "mean_oracle_cv_probability": statistics.mean(e.oracle_cv_probability for e in events),
        "mean_oracle_model_probability": statistics.mean(e.oracle_model_probability for e in events),
        "oracle_violations": oracle_violations,
        "mean_hysteresis_regret": statistics.mean(e.hysteresis_regret for e in events),
        "mean_ess": statistics.mean(e.effective_sample_size for e in events if math.isfinite(e.effective_sample_size)) if any(math.isfinite(e.effective_sample_size) for e in events) else 0.0,
        "mean_history_blend": statistics.mean(e.history_blend for e in events),
        "branch_switches": max(e.branch for e in events),
        "planning_p50_ms": percentile([e.planning_ms for e in events], 0.50),
        "planning_p95_ms": percentile([e.planning_ms for e in events], 0.95),
    }


def per_seed_summaries(events: list[Event]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[int, str, str, float], list[Event]] = {}
    for event in events:
        key = (event.seed, event.executor_mode, event.mode, event.requested_delay_ms)
        groups.setdefault(key, []).append(event)
    output = []
    for key, group in sorted(groups.items()):
        item = summarize(group)
        item["seed"] = key[0]
        output.append(item)
    return output


def calibration_rows(events: list[Event]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, float, int], list[Event]] = {}
    for event in events:
        bin_index = min(9, int(clamp(event.predicted_probability, 0.0, 0.999999) * 10))
        key = (event.executor_mode, event.mode, event.requested_delay_ms, bin_index)
        groups.setdefault(key, []).append(event)
    rows = []
    for (executor_mode, mode, delay_ms, bin_index), group in sorted(groups.items()):
        rows.append({
            "executor_mode": executor_mode,
            "mode": mode,
            "delay_ms": delay_ms,
            "probability_bin": f"{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}",
            "events": len(group),
            "mean_prediction": statistics.mean(e.predicted_probability for e in group),
            "actual_coverage_rate": statistics.mean(float(e.covered) for e in group),
            "mean_actual_probability": statistics.mean(e.actual_coverage_probability for e in group),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path, summaries: list[dict[str, object]]) -> None:
    with (output / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("# 2D moving-marker tracking experiment v2\n\n")
        handle.write("This safe experiment contains no firing, ballistics, armor, or trigger logic.\n\n")
        handle.write("## Validation logic\n\n")
        handle.write("- Oracle search explicitly includes the constant-velocity candidate.\n")
        handle.write("- A runtime assertion verifies Oracle probability is never below the CV candidate.\n")
        handle.write("- Ideal and constrained executor results are reported separately.\n")
        handle.write("- Calibration uses probability at the predicted executor direction.\n")
        handle.write("- Exact requested delay is retained, including 0 ms and 15 ms.\n\n")
        handle.write("## Summary\n\n")
        columns = [
            "executor_mode", "mode", "delay_ms", "events", "coverage_rate",
            "mean_predicted_probability", "mean_actual_probability", "brier_score",
            "ece", "prediction_mae_deg", "tracking_mae_deg", "final_mae_deg",
            "mean_oracle_best_probability", "mean_oracle_cv_probability",
            "planning_p95_ms", "oracle_violations",
        ]
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        for item in summaries:
            values = []
            for column in columns:
                value = item[column]
                values.append(f"{value:.5f}" if isinstance(value, float) else str(value))
            handle.write("| " + " | ".join(values) + " |\n")
        handle.write("\n## Interpretation guide\n\n")
        handle.write("1. If Oracle violations are nonzero, the experiment framework is invalid.\n")
        handle.write("2. If historical/blend beats CV with the ideal executor but not the constrained executor, the bottleneck is control/reachability.\n")
        handle.write("3. If historical/blend loses even with the ideal executor, the distribution estimator or state conditioning is the bottleneck.\n")
        handle.write("4. Compare mean predicted probability with both empirical coverage and mean actual probability before interpreting it as calibrated.\n")


def write_outputs(output: Path, events: list[Event]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "events.csv", [asdict(event) for event in events])
    seed_rows = per_seed_summaries(events)
    write_csv(output / "per_seed_summary.csv", seed_rows)
    aggregate_groups: dict[tuple[str, str, float], list[Event]] = {}
    for event in events:
        key = (event.executor_mode, event.mode, event.requested_delay_ms)
        aggregate_groups.setdefault(key, []).append(event)
    summaries = [summarize(group) for _, group in sorted(aggregate_groups.items())]
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "calibration.csv", calibration_rows(events))
    write_report(output, summaries)


def self_test() -> None:
    cfg = Config(duration=0.5, warmup=0.2, delay=0.015)
    rows = [
        (0.0, State(0.0, 1.0, 0.0)),
        (0.1, State(0.1, 1.0, 0.0)),
    ]
    middle = interpolate_state(rows, 0.05)
    assert abs(middle.angle - 0.05) < 1e-10
    assert abs(middle.omega - 1.0) < 1e-10
    assert abs(coverage_probability(0.0, 0.0, cfg) - 0.9) < 2e-4
    model = HistoryAccelerationModel(cfg)
    history = [(i * 0.01, State(0.2 * i * 0.01, 0.2, 0.0)) for i in range(80)]
    weighted, ess = model.weighted_accelerations(history, history[-1][1], 0.05)
    assert weighted and abs(sum(weight for _, weight in weighted) - 1.0) < 1e-10
    assert ess >= 1.0
    executor = Executor(cfg, "constrained")
    predicted = executor.predict(0.1, 0.0, 0.0, 0.015)
    assert math.isfinite(predicted)
    oracle_yaw, oracle_best, true_angle = oracle_plan(
        State(0.0, 1.0, cfg.max_alpha), 1.0, 0.015, cfg, [0.0]
    )
    assert oracle_best + 1e-12 >= coverage_probability(0.0, true_angle, cfg)
    zero_delay_events = run_trial(
        "constant_velocity", "ideal", replace(cfg, delay=0.0), cfg.seed
    )
    assert zero_delay_events
    assert all(abs(event.actual_delay_ms) < 1e-9 for event in zero_delay_events)
    short_events = run_trial("blend", "constrained", cfg, cfg.seed)
    assert short_events
    assert all(0.0 <= event.predicted_probability <= 1.0 for event in short_events)
    assert all(event.oracle_best_probability + 1e-12 >= event.oracle_cv_probability for event in short_events)
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("tracking_experiment_v2_output"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--delay-ms", type=float, nargs="+", default=[0, 15, 30, 50, 100])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("constant_velocity", "historical", "blend", "oracle"),
        default=["constant_velocity", "historical", "blend", "oracle"],
    )
    parser.add_argument(
        "--executors",
        nargs="+",
        choices=("ideal", "constrained"),
        default=["ideal", "constrained"],
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    all_events: list[Event] = []
    for delay_ms in args.delay_ms:
        for executor_mode in args.executors:
            for mode in args.modes:
                cfg = Config(
                    duration=args.duration,
                    warmup=args.warmup,
                    delay=delay_ms / 1000.0,
                )
                group: list[Event] = []
                for offset in range(args.seeds):
                    group.extend(run_trial(mode, executor_mode, cfg, cfg.seed + offset))
                all_events.extend(group)
                summary = summarize(group)
                print(
                    f"delay={delay_ms:6.1f} executor={executor_mode:11s} "
                    f"mode={mode:18s} coverage={summary['coverage_rate']:.4f} "
                    f"pred_MAE={summary['prediction_mae_deg']:.3f}deg "
                    f"final_MAE={summary['final_mae_deg']:.3f}deg "
                    f"ECE={summary['ece']:.4f} p95={summary['planning_p95_ms']:.3f}ms"
                )
    write_outputs(args.output, all_events)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
