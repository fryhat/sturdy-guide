"""2D moving-target visualizer based on sp_vision_25 target prediction.

The target predictor follows TongjiSuperPower/sp_vision_25's Target model:
the center translates at the currently estimated velocity, and armor angle
advances at the currently estimated angular velocity. Future target
acceleration is not part of the state. Vertical motion, pitch, and projectile
drop are intentionally omitted.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import ctypes
import math
import random
import time
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import ttk


BODY_WIDTH_M = 0.510
BODY_LENGTH_M = 0.600
ARMOR_WIDTH_M = 0.140
ARMOR_DRAW_THICKNESS_M = 0.035
BODY_MAX_RADIUS_M = max(BODY_WIDTH_M / 2.0, BODY_LENGTH_M / 2.0)
ARMOR_SCAN_STEP = math.radians(0.10)
ARMOR_CURVE_MOVEMENT_RAD = math.radians(4.0)

OMEGA_LIMIT = math.radians(600.0)
ALPHA_LIMIT = math.radians(280.0)
INITIAL_OMEGA = math.radians(300.0)
DECISION_SPEED = 8.0  # rad/s, standard4.yaml
LOW_SPEED_DELAY = 0.015
HIGH_SPEED_DELAY = 0.015
PLANNER_DT = 0.010
PLANNER_HALF_HORIZON = 50
PLANNER_HORIZON = 100
FIRE_LOOKAHEAD = 2 * PLANNER_DT
GIMBAL_MAX_ACCEL = 50.0  # rad/s^2, standard4.yaml
FIRE_THRESHOLD = 0.003  # rad, standard4.yaml
BULLET_SPEED = 25.0
MIN_NORMAL_HIT_SPEED = 12.0

FIELD_X = (-4.0, 4.0)
FIELD_Y = (-0.45, 8.6)
SHOT_INTERVAL = 0.05
RESULT_DISPLAY_TIME = 0.6
EVADE_SWITCH_INTERVAL = 1.0
SIMULATION_TIME_SCALE = 0.1  # simulation time / wall-clock time
SCATTER_STATIC_HIT_RATE = 0.90
SCATTER_Z90 = 1.6448536269514722
SCATTER_SIGMA = (ARMOR_WIDTH_M / 2.0) / SCATTER_Z90
PROBABILITY_FIRE_THRESHOLD = 0.50
FIELD_CANDIDATE_SWITCH_PENALTY = 0.50  # probability units per radian of phase change
FIELD_CANDIDATE_MIN_GAIN = 0.02
FIELD_MPC_SWITCH_PENALTY = 0.50  # probability units per radian of branch change
FIELD_MPC_MIN_GAIN = 0.02
FIELD_MPC_TRAJECTORY_TICKS = 6
FIELD_MPC_MAX_CANDIDATES = 6
FIELD_MPC_BLOCK_SIZE = 5
WINDOW_CANDIDATE_REFRESH_ONLY = True
WINDOW_CANDIDATE_CACHE_PHASE = WINDOW_CANDIDATE_REFRESH_ONLY
PROBABILITY_FIELD_INTERVAL = 0.05
MOTION_HISTORY_WINDOW = 160
PROBABILITY_PEAK_SWITCH_MARGIN = 0.035
PROBABILITY_CONTINUITY_WEIGHT = 0.035
PROBABILITY_YAW_SMOOTHING = 0.72

ROLLOUT_SEGMENT_DT = 0.05
ROLLOUT_SEGMENT_COUNT = 6
ROLLOUT_MAX_GIMBAL_OMEGA = 1.2617046468719202
ROLLOUT_PROBABILITY_COUNT = 64
FLIGHT_TIME_TOLERANCE = 1e-6
MAX_FLIGHT_ITERATIONS = 1


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return x * cosine - y * sine, x * sine + y * cosine


def norm(x: float, y: float) -> float:
    return math.hypot(x, y)


def angle_error(target: float, current: float) -> float:
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def reachable_yaw_delta_bounds(velocity: float, horizon: float) -> tuple[float, float]:
    """Exact terminal-angle bounds for the acceleration-limited double integrator."""
    duration = max(horizon, 0.0)
    drift = velocity * duration
    control = 0.5 * GIMBAL_MAX_ACCEL * duration * duration
    return drift - control, drift + control


def project_yaw_to_reachable(
    yaw: float,
    current_yaw: float,
    velocity: float,
    horizon: float,
) -> float:
    lower, upper = reachable_yaw_delta_bounds(velocity, horizon)
    delta = angle_error(yaw, current_yaw)
    return current_yaw + clamp(delta, lower, upper)


def advance_full_acceleration(
    angle: float,
    omega: float,
    acceleration_direction: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """Advance with |alpha| fixed at ALPHA_LIMIT and no speed-boundary bounce."""
    direction = 1.0 if acceleration_direction >= 0.0 else -1.0
    remaining = dt
    alpha = direction * ALPHA_LIMIT
    while remaining > 1e-12:
        boundary = OMEGA_LIMIT if direction > 0.0 else -OMEGA_LIMIT
        time_to_boundary = max(0.0, (boundary - omega) / (direction * ALPHA_LIMIT))
        if (
            (direction > 0.0 and omega >= OMEGA_LIMIT - 1e-12)
            or (direction < 0.0 and omega <= -OMEGA_LIMIT + 1e-12)
        ):
            omega = boundary
            alpha = 0.0
            remaining = 0.0
            break
        step = min(remaining, time_to_boundary)
        alpha = direction * ALPHA_LIMIT
        angle += omega * step + 0.5 * alpha * step * step
        omega += alpha * step
        remaining -= step
        if time_to_boundary <= step + 1e-12 and remaining > 1e-12:
            omega = boundary
            alpha = 0.0
            remaining = 0.0
    return angle, omega, direction, alpha


def advance_optimized_evasion(
    angle: float,
    omega: float,
    acceleration_direction: float,
    start_time: float,
    next_switch: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    """Use maximum acceleration and reverse it every optimized interval."""
    direction = acceleration_direction
    cursor = start_time
    remaining = dt
    alpha = direction * ALPHA_LIMIT
    while remaining > 1e-12:
        until_switch = next_switch - cursor
        step = min(remaining, max(0.0, until_switch))
        angle, omega, direction, alpha = advance_full_acceleration(
            angle, omega, direction, step
        )
        cursor += step
        remaining -= step
        if until_switch <= step + 1e-12:
            direction = -direction
            alpha = direction * ALPHA_LIMIT
            next_switch += EVADE_SWITCH_INTERVAL
        elif step <= 1e-12:
            break
    return angle, omega, direction, alpha, next_switch


def advance_gimbal_limited(
    yaw: float,
    omega: float,
    alpha: float,
    dt: float,
    max_omega: float = ROLLOUT_MAX_GIMBAL_OMEGA,
) -> tuple[float, float]:
    """Advance a constant gimbal acceleration with the rollout speed cap."""
    if dt <= 0.0:
        return yaw, omega
    if abs(alpha) < 1e-12:
        return yaw + omega * dt, omega
    if alpha > 0.0 and omega >= max_omega:
        return yaw + omega * dt, omega
    if alpha < 0.0 and omega <= -max_omega:
        return yaw + omega * dt, omega
    boundary = max_omega if alpha > 0.0 else -max_omega
    time_to_boundary = (boundary - omega) / alpha
    if 0.0 <= time_to_boundary < dt:
        reached = omega + alpha * time_to_boundary
        yaw += omega * time_to_boundary + 0.5 * alpha * time_to_boundary**2
        yaw += reached * (dt - time_to_boundary)
        return yaw, reached
    yaw += omega * dt + 0.5 * alpha * dt * dt
    return yaw, omega + alpha * dt


def rollout_curve_grid(
    target: TargetState,
    grid_step: float = ARMOR_SCAN_STEP,
) -> list[float]:
    center_distance = max(norm(target.x, target.y), 1e-6)
    center_yaw = math.atan2(target.y, target.x)
    half_span = math.asin(min(1.0, BODY_MAX_RADIUS_M / center_distance))
    half_span += math.atan2(
        ARMOR_WIDTH_M / 2.0 + ARMOR_DRAW_THICKNESS_M / 2.0,
        center_distance,
    )
    scatter_padding = 4.0 * math.atan2(SCATTER_SIGMA, center_distance)
    curve_padding = scatter_padding + ARMOR_CURVE_MOVEMENT_RAD
    cursor = center_yaw - half_span - curve_padding
    upper = center_yaw + half_span + curve_padding
    grid: list[float] = []
    while cursor <= upper + 1e-12:
        grid.append(cursor)
        cursor += grid_step
    return grid


@dataclass
class TargetState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    omega: float
    alpha: float


@dataclass(frozen=True)
class MotionHistorySample:
    """One observed average-acceleration sample over a fixed horizon."""

    start_time: float
    horizon: float
    position: float
    velocity: float
    future_position: float
    average_acceleration: float


class ProbabilisticAimModel:
    """Estimate future target position from constrained historical motion."""

    def __init__(
        self,
        horizon: float,
        max_acceleration: float = ALPHA_LIMIT,
        min_velocity: float = -OMEGA_LIMIT,
        max_velocity: float = OMEGA_LIMIT,
        scatter_sigma: float | None = None,
    ) -> None:
        self.horizon = horizon
        self.max_acceleration = abs(max_acceleration)
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.samples: list[MotionHistorySample] = []
        self.sample_entries: dict[tuple[float, float], int] = {}
        # 90% of static shots fall inside one armor half-width at 6 m.
        self.scatter_sigma = scatter_sigma or SCATTER_SIGMA

    def add_sample(
        self,
        start_time: float,
        position: float,
        velocity: float,
        future_position: float,
        horizon: float | None = None,
    ) -> MotionHistorySample:
        tau = self.horizon if horizon is None else horizon
        if tau <= 0.0:
            raise ValueError("horizon must be positive")
        acceleration = 2.0 * (future_position - position - velocity * tau) / (tau * tau)
        sample = MotionHistorySample(
            start_time, tau, position, velocity, future_position, acceleration
        )
        self.samples.append(sample)
        return sample

    def accumulate_sample(
        self,
        start_time: float,
        position: float,
        velocity: float,
        future_position: float,
        horizon: float,
    ) -> None:
        """Add one newly observed equivalent-acceleration observation exactly once."""
        if horizon <= 0.0:
            raise ValueError("horizon must be positive")
        acceleration = 2.0 * (
            future_position - position - velocity * horizon
        ) / (horizon * horizon)
        key = (round(horizon, 3), round(acceleration, 4))
        self.sample_entries[key] = self.sample_entries.get(key, 0) + 1

    def _weighted_constrained_entries(
        self,
        velocity: float,
    ) -> list[tuple[float, float, int]]:
        """Return (horizon, acceleration, count) entries for one model evaluation."""
        lower, upper = self._allowed_acceleration(velocity)
        if self.sample_entries:
            return [
                (horizon, acceleration, count)
                for (horizon, acceleration), count in self.sample_entries.items()
                if abs(horizon - self.horizon) <= PLANNER_DT
                and lower - 1e-12 <= acceleration <= upper + 1e-12
            ]
        return [
            (sample.horizon, sample.average_acceleration, 1)
            for sample in self.samples
            if abs(sample.horizon - self.horizon) <= PLANNER_DT
            and lower - 1e-12 <= sample.average_acceleration <= upper + 1e-12
        ]

    def add_trajectory(
        self,
        history: list[tuple[float, float, float]],
    ) -> int:
        """Add (timestamp, position, velocity) records spaced by the model horizon."""
        added = 0
        future_index = 1
        for index, (timestamp, position, velocity) in enumerate(history):
            end_time = timestamp + self.horizon
            future_index = max(future_index, index + 1)
            while future_index < len(history) and history[future_index][0] < end_time:
                future_index += 1
            if future_index >= len(history):
                continue
            future = history[future_index]
            self.add_sample(
                timestamp,
                position,
                velocity,
                future[1],
                future[0] - timestamp,
            )
            added += 1
        return added

    def _allowed_acceleration(self, velocity: float) -> tuple[float, float]:
        tau = self.horizon
        # v(t)=v+a*t must stay within both velocity limits.
        lower = (self.min_velocity - velocity) / tau
        upper = (self.max_velocity - velocity) / tau
        return max(-self.max_acceleration, lower), min(self.max_acceleration, upper)

    def constrained_accelerations(self, velocity: float) -> list[float]:
        return [
            sample.average_acceleration
            for sample in self.constrained_samples(velocity)
        ]

    def constrained_samples(self, velocity: float) -> list[MotionHistorySample]:
        lower, upper = self._allowed_acceleration(velocity)
        return [
            sample
            for sample in self.samples
            if abs(sample.horizon - self.horizon) <= PLANNER_DT
            and lower - 1e-12 <= sample.average_acceleration <= upper + 1e-12
        ]

    def future_position_samples(
        self,
        position: float,
        velocity: float,
        seed: int = 0,
        count: int = 512,
    ) -> list[float]:
        samples = self.constrained_samples(velocity)
        if not samples:
            return []
        rng = random.Random(seed)
        values = [sample.average_acceleration for sample in samples]
        return [
            position + velocity * self.horizon + 0.5 * acceleration * self.horizon**2
            for acceleration in (rng.choice(values) for _ in range(max(1, count)))
        ]

    def hit_probability(self, aim_position: float, target_position: float) -> float:
        """One-dimensional Gaussian scatter probability over an armor width."""
        half = ARMOR_WIDTH_M / 2.0
        scale = self.scatter_sigma * math.sqrt(2.0)
        upper = (aim_position - target_position + half) / scale
        lower = (aim_position - target_position - half) / scale
        return 0.5 * (math.erf(upper) - math.erf(lower))

    def expected_hit_curve(
        self,
        future_positions: list[float],
        aim_grid: list[float],
    ) -> list[float]:
        if not future_positions:
            return [0.0 for _ in aim_grid]
        return [
            sum(self.hit_probability(aim, target) for target in future_positions)
            / len(future_positions)
            for aim in aim_grid
        ]

    def optimal_aim(
        self,
        position: float,
        velocity: float,
        aim_grid: list[float],
        seed: int = 0,
        count: int = 512,
    ) -> tuple[float, float]:
        future = self.future_position_samples(position, velocity, seed, count)
        curve = self.expected_hit_curve(future, aim_grid)
        index = max(range(len(aim_grid)), key=curve.__getitem__)
        return aim_grid[index], curve[index]

    def hit_probability_field(
        self,
        target: TargetState,
        yaw_grid: list[float],
        count: int = 256,
        seed: int = 0,
    ) -> list[float]:
        """Evaluate the exact empirical multimodal yaw hit field."""
        # ``target`` is already evaluated at the predicted impact time by the
        # planner. Add only the acceleration-model uncertainty over this same
        # horizon; do not propagate the nominal state a second time.
        entries = self._weighted_constrained_entries(target.omega)
        if not entries:
            return [0.0 for _ in yaw_grid]
        # Compressed online observations keep empirical frequency without
        # storing or rescanning every 10 ms history sample.
        weighted_angles: list[tuple[float, int]] = [
            (
                target.angle + 0.5 * acceleration * self.horizon**2,
                count,
            )
            for _horizon, acceleration, count in entries
        ]
        total_weight = sum(count for _angle, count in weighted_angles)
        if total_weight <= 0:
            return [0.0 for _ in yaw_grid]
        radius = max(norm(target.x, target.y), 1e-6)
        # Armor geometry depends only on each sampled target angle.  Build it
        # once per sample; rebuilding it inside the yaw scan multiplied the
        # trigonometric work by the number of grid points.
        sample_armors = [
            (armor_poses(replace(target, angle=angle)), count)
            for angle, count in weighted_angles
        ]
        values: list[float] = []
        for yaw in yaw_grid:
            total = 0.0
            for armors, count in sample_armors:
                best = 0.0
                for armor in armors:
                    armor_normal_yaw = math.atan2(armor.normal_y, armor.normal_x)
                    angular_sigma = self.scatter_sigma / radius
                    delta = angle_error(yaw, math.atan2(armor.y, armor.x))
                    # Integrate the calibrated Gaussian over the finite armor
                    # interval instead of using its unnormalised peak value.
                    # At zero error this reproduces the 90% static-hit
                    # calibration (up to the small angular approximation).
                    half_angle = math.atan2(ARMOR_WIDTH_M / 2.0, radius)
                    scale = angular_sigma * math.sqrt(2.0)
                    probability = 0.5 * (
                        math.erf((delta + half_angle) / scale)
                        - math.erf((delta - half_angle) / scale)
                    )
                    # Use the actual armor normal, not its center radial angle.
                    # The front plate normal points toward the shooter, opposite
                    # to the radial center vector in this coordinate convention.
                    normal_alignment = max(0.0, -math.cos(angle_error(yaw, armor_normal_yaw)))
                    if normal_alignment * BULLET_SPEED <= MIN_NORMAL_HIT_SPEED:
                        probability = 0.0
                    best = max(best, probability)
                total += count * best
            values.append(total / total_weight)
        return values


@dataclass(frozen=True)
class ArmorPose:
    index: int
    x: float
    y: float
    tangent_x: float
    tangent_y: float
    normal_x: float
    normal_y: float


@dataclass
class Shot:
    created_at: float
    launch_at: float
    impact_at: float
    launch_x: float
    launch_y: float
    aim_x: float
    aim_y: float
    planned_aim_x: float
    planned_aim_y: float
    predicted_state: TargetState
    armor_index: int
    launched: bool = False
    launch_angle: float | None = None
    gate_yaw: float | None = None
    launch_prediction_error: float | None = None
    scatter_offset: float = 0.0
    resolved: bool = False
    hit: bool = False
    outcome: str = "pending"
    normal_speed: float | None = None
    resolved_at: float = 0.0
    diagnostic: dict[str, float | str | bool | None] | None = None


@dataclass(frozen=True)
class HitResult:
    outcome: str
    normal_speed: float | None = None
    armor_index: int | None = None


@dataclass(frozen=True)
class PlannerOutput:
    target_yaw: float
    yaw: float
    yaw_velocity: float
    yaw_acceleration: float
    fire: bool
    fire_error: float
    delay: float
    flight_time: float
    planned_aim_x: float
    planned_aim_y: float
    predicted_state: TargetState
    armor_index: int
    reference_yaw: tuple[float, ...]
    planned_yaw: tuple[float, ...]


def armor_poses(state: TargetState) -> list[ArmorPose]:
    local = (
        (0.0, -BODY_LENGTH_M / 2, 1.0, 0.0, 0.0, -1.0),
        (BODY_WIDTH_M / 2, 0.0, 0.0, 1.0, 1.0, 0.0),
        (0.0, BODY_LENGTH_M / 2, -1.0, 0.0, 0.0, 1.0),
        (-BODY_WIDTH_M / 2, 0.0, 0.0, -1.0, -1.0, 0.0),
    )
    output: list[ArmorPose] = []
    for index, (cx, cy, tx, ty, nx, ny) in enumerate(local):
        rcx, rcy = rotate(cx, cy, state.angle)
        rtx, rty = rotate(tx, ty, state.angle)
        rnx, rny = rotate(nx, ny, state.angle)
        output.append(
            ArmorPose(
                index,
                state.x + rcx,
                state.y + rcy,
                rtx,
                rty,
                rnx,
                rny,
            )
        )
    return output


def predict_constant_velocity(state: TargetState, dt: float) -> TargetState:
    """The sp_vision_25 target transition: x+=vx*dt, angle+=omega*dt."""
    return TargetState(
        x=state.x + state.vx * dt,
        y=state.y + state.vy * dt,
        vx=state.vx,
        vy=state.vy,
        angle=state.angle + state.omega * dt,
        omega=state.omega,
        alpha=state.alpha,
    )


def nearest_armor(state: TargetState) -> ArmorPose:
    return min(armor_poses(state), key=lambda armor: norm(armor.x, armor.y))


def armor_nearest_yaw(state: TargetState, yaw: float) -> ArmorPose:
    """Return the plate whose center bearing best matches a selected yaw peak."""
    armors = armor_poses(state)
    valid_facing = [
        armor
        for armor in armors
        if max(
            0.0,
            -math.cos(
                angle_error(yaw, math.atan2(armor.normal_y, armor.normal_x))
            ),
        )
        * BULLET_SPEED
        > MIN_NORMAL_HIT_SPEED
    ]
    return min(
        valid_facing or armors,
        key=lambda armor: abs(angle_error(math.atan2(armor.y, armor.x), yaw)),
    )


def armor_solution_for_yaw(
    delayed: TargetState,
    yaw: float,
    bullet_speed: float,
) -> tuple[ArmorPose, TargetState, float]:
    """Choose a plate with a self-consistent pose, range, and flight time."""
    solutions: list[tuple[bool, float, ArmorPose, TargetState, float]] = []
    for armor_index in range(4):
        flight_time = norm(
            armor_poses(delayed)[armor_index].x,
            armor_poses(delayed)[armor_index].y,
        ) / bullet_speed
        for _ in range(MAX_FLIGHT_ITERATIONS):
            impact_state = predict_constant_velocity(delayed, flight_time)
            armor = armor_poses(impact_state)[armor_index]
            updated = norm(armor.x, armor.y) / bullet_speed
            if abs(updated - flight_time) < FLIGHT_TIME_TOLERANCE:
                flight_time = updated
                break
            flight_time = updated
        impact_state = predict_constant_velocity(delayed, flight_time)
        armor = armor_poses(impact_state)[armor_index]
        # Make the returned range/time identity exact after convergence.
        flight_time = norm(armor.x, armor.y) / bullet_speed
        armor_yaw = math.atan2(armor.y, armor.x)
        normal_yaw = math.atan2(armor.normal_y, armor.normal_x)
        normal_speed = max(0.0, -math.cos(angle_error(yaw, normal_yaw))) * bullet_speed
        solutions.append(
            (
                normal_speed > MIN_NORMAL_HIT_SPEED,
                abs(angle_error(armor_yaw, yaw)),
                armor,
                impact_state,
                flight_time,
            )
        )
    facing = [solution for solution in solutions if solution[0]]
    _valid, _error, armor, impact_state, flight_time = min(
        facing or solutions,
        key=lambda solution: solution[1],
    )
    return armor, impact_state, flight_time


class TinyMpcPlanner:
    """2D port of Planner::plan that calls the repository's TinyMPC solver."""

    def __init__(self) -> None:
        dll_path = Path(__file__).with_name("sp_vision_tinympc.dll")
        if not dll_path.exists():
            raise RuntimeError(f"TinyMPC bridge not found: {dll_path}")
        self._dll = ctypes.CDLL(str(dll_path))
        self._solve = self._dll.sp_solve_yaw
        double_pointer = ctypes.POINTER(ctypes.c_double)
        self._solve.argtypes = (
            double_pointer,
            double_pointer,
            ctypes.c_double,
            ctypes.c_double,
            double_pointer,
            double_pointer,
            double_pointer,
        )
        self._solve.restype = ctypes.c_int

    def plan(
        self,
        state: TargetState,
        bullet_speed: float,
        aim_yaw_override: float | None = None,
        probability_phase_profile: bool = False,
    ) -> PlannerOutput:
        delay = HIGH_SPEED_DELAY if abs(state.omega) > DECISION_SPEED else LOW_SPEED_DELAY
        delayed = predict_constant_velocity(state, delay)
        initial_armor = nearest_armor(delayed)
        flight_time = norm(initial_armor.x, initial_armor.y) / bullet_speed
        impact_state = predict_constant_velocity(delayed, flight_time)
        selected = nearest_armor(impact_state)
        if aim_yaw_override is not None:
            selected, impact_state, flight_time = armor_solution_for_yaw(
                delayed,
                aim_yaw_override,
                bullet_speed,
            )
        nominal_yaw = math.atan2(selected.y, selected.x)

        absolute_yaws: list[float] = []
        for sample in range(-PLANNER_HALF_HORIZON - 1, PLANNER_HALF_HORIZON + 1):
            sample_state = predict_constant_velocity(impact_state, sample * PLANNER_DT)
            armor = nearest_armor(sample_state)
            absolute_yaws.append(math.atan2(armor.y, armor.x))

        # A probabilistic peak changes the phase, while the nominal MPC
        # trajectory still supplies the physically correct angular velocity.
        # This keeps multi-modal selection without replacing MPC by a slow PD
        # loop with zero velocity reference.
        yaw0 = nominal_yaw
        if aim_yaw_override is not None:
            # reference_yaw[i] consumes absolute_yaws[i + 1], therefore the
            # impact-time center is at HALF_HORIZON + 1 in this padded array.
            center_yaw = absolute_yaws[PLANNER_HALF_HORIZON + 1]
            phase_offset = angle_error(aim_yaw_override, center_yaw)
            if probability_phase_profile:
                impact_horizon = max(delay + flight_time, PLANNER_DT)
                profiled_yaws: list[float] = []
                for index, yaw in enumerate(absolute_yaws):
                    sample_offset = (
                        index - (PLANNER_HALF_HORIZON + 1)
                    ) * PLANNER_DT
                    relative_time = impact_horizon + sample_offset
                    # The empirical acceleration distribution is calibrated
                    # only from now to the impact horizon. Do not extrapolate
                    # that short-horizon acceleration through the remaining
                    # 0.5 s MPC tail, where target reversals are unmodelled.
                    progress = clamp(relative_time / impact_horizon, 0.0, 1.0)
                    phase = phase_offset * progress**2
                    profiled_yaws.append(yaw + phase)
                absolute_yaws = profiled_yaws
            else:
                absolute_yaws = [yaw + phase_offset for yaw in absolute_yaws]
            yaw0 = nominal_yaw + phase_offset

        reference_yaw = [
            angle_error(absolute_yaws[i + 1], yaw0) for i in range(PLANNER_HORIZON)
        ]
        reference_velocity = [
            angle_error(absolute_yaws[i + 2], absolute_yaws[i]) / (2 * PLANNER_DT)
            for i in range(PLANNER_HORIZON)
        ]

        reference_array = (ctypes.c_double * PLANNER_HORIZON)(*reference_yaw)
        velocity_array = (ctypes.c_double * PLANNER_HORIZON)(*reference_velocity)
        output_yaw = (ctypes.c_double * PLANNER_HORIZON)()
        output_velocity = (ctypes.c_double * PLANNER_HORIZON)()
        output_acceleration = (ctypes.c_double * (PLANNER_HORIZON - 1))()
        status = self._solve(
            reference_array,
            velocity_array,
            reference_yaw[0],
            reference_velocity[0],
            output_yaw,
            output_velocity,
            output_acceleration,
        )
        # The upstream planner also uses work->x/u after its fixed 10 ADMM
        # iterations; TinyMPC returns 1 when that budget ends before tolerance.
        if status not in (0, 1):
            raise RuntimeError(f"TinyMPC solve failed with status {status}")

        center = PLANNER_HALF_HORIZON
        fire_index = center + int(round(FIRE_LOOKAHEAD / PLANNER_DT))
        fire_error = angle_error(reference_yaw[fire_index], output_yaw[fire_index])
        planned_range = norm(selected.x, selected.y)
        planned_aim_x = planned_range * math.cos(aim_yaw_override) if aim_yaw_override is not None else selected.x
        planned_aim_y = planned_range * math.sin(aim_yaw_override) if aim_yaw_override is not None else selected.y
        return PlannerOutput(
            target_yaw=yaw0 + reference_yaw[center],
            yaw=yaw0 + output_yaw[center],
            yaw_velocity=output_velocity[center],
            yaw_acceleration=output_acceleration[center],
            fire=abs(fire_error) < FIRE_THRESHOLD,
            fire_error=fire_error,
            delay=delay,
            flight_time=flight_time,
            planned_aim_x=planned_aim_x,
            planned_aim_y=planned_aim_y,
            predicted_state=impact_state,
            armor_index=selected.index,
            reference_yaw=tuple(yaw0 + value for value in reference_yaw),
            planned_yaw=tuple(yaw0 + value for value in output_yaw),
        )


class CppRolloutPlanner:
    """ctypes wrapper around the 5^6 acceleration path enumerator."""

    def __init__(self, dll_path: Path | None = None) -> None:
        dll_path = dll_path or (
            Path(__file__).resolve().parent
            / "cpp_rollout_planner"
            / "rollout_planner.dll"
        )
        if not dll_path.exists():
            raise RuntimeError(f"rollout planner not found: {dll_path}")
        self._dll = ctypes.CDLL(str(dll_path))
        self._plan = self._dll.sp_plan_rollout
        double_pointer = ctypes.POINTER(ctypes.c_double)
        self._plan.argtypes = (
            ctypes.c_double,
            ctypes.c_double,
            double_pointer,
            ctypes.c_int,
            double_pointer,
            ctypes.c_int,
            double_pointer,
            double_pointer,
        )
        self._plan.restype = ctypes.c_int

    def plan(
        self,
        yaw: float,
        omega: float,
        grid: list[float],
        curves: list[list[float]],
    ) -> tuple[float, float]:
        if not grid or any(len(curve) != len(grid) for curve in curves):
            raise ValueError("grid/curve shape mismatch")
        if len(curves) != ROLLOUT_SEGMENT_COUNT:
            raise ValueError(
                f"expected {ROLLOUT_SEGMENT_COUNT} curves, got {len(curves)}"
            )
        grid_array = (ctypes.c_double * len(grid))(*grid)
        curve_array = (
            ctypes.c_double * (len(curves) * len(grid))
        )(*(value for curve in curves for value in curve))
        alpha = ctypes.c_double()
        score = ctypes.c_double()
        status = self._plan(
            yaw,
            omega,
            grid_array,
            len(grid),
            curve_array,
            len(curves),
            ctypes.byref(alpha),
            ctypes.byref(score),
        )
        if status != 0:
            raise RuntimeError(f"rollout planner failed with status {status}")
        return alpha.value, score.value


def shot_from_plan(
    plan: PlannerOutput,
    now: float,
    launch_yaw: float | None = None,
    scatter: bool = False,
) -> Shot:
    if launch_yaw is None:
        launch_yaw = plan.yaw
    planned_range = norm(plan.planned_aim_x, plan.planned_aim_y)
    aim_x = planned_range * math.cos(launch_yaw)
    aim_y = planned_range * math.sin(launch_yaw)
    scatter_offset = 0.0
    if scatter:
        # One-dimensional lateral Gaussian spread calibrated by ProbabilisticAimModel.
        scatter_offset = random.gauss(0.0, SCATTER_SIGMA)
    return Shot(
        created_at=now,
        launch_at=now + plan.delay,
        impact_at=now + plan.delay + plan.flight_time,
        launch_x=0.0,
        launch_y=0.0,
        aim_x=aim_x,
        aim_y=aim_y,
        planned_aim_x=plan.planned_aim_x,
        planned_aim_y=plan.planned_aim_y,
        predicted_state=plan.predicted_state,
        armor_index=plan.armor_index,
        launch_angle=None,
        scatter_offset=scatter_offset,
    )


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx, aby = bx - ax, by - ay
    length_squared = abx * abx + aby * aby
    if length_squared <= 1e-12:
        return norm(px - ax, py - ay)
    t = clamp(((px - ax) * abx + (py - ay) * aby) / length_squared, 0.0, 1.0)
    return norm(px - (ax + t * abx), py - (ay + t * aby))


def evaluate_shot(shot: Shot, actual: TargetState) -> HitResult:
    direction_x = shot.aim_x - shot.launch_x
    direction_y = shot.aim_y - shot.launch_y
    path_length = norm(direction_x, direction_y)
    if path_length <= 1e-12:
        return HitResult("miss")
    velocity_x = BULLET_SPEED * direction_x / path_length
    velocity_y = BULLET_SPEED * direction_y / path_length

    covered: list[HitResult] = []
    armors = armor_poses(actual)
    if shot.armor_index >= 0:
        armor_indices = (shot.armor_index % len(armors),)
    else:
        armor_indices = range(len(armors))
    for armor_index in armor_indices:
        armor = armors[armor_index]
        half = ARMOR_WIDTH_M / 2
        ax = armor.x - armor.tangent_x * half
        ay = armor.y - armor.tangent_y * half
        bx = armor.x + armor.tangent_x * half
        by = armor.y + armor.tangent_y * half
        # This is a time-slice collision test: shot.aim is the projectile
        # position at impact_at and `actual` is the armor pose at that same
        # time. An infinite ray/segment intersection would ignore travel time
        # and incorrectly count plates crossed earlier or later as hits.
        distance = point_segment_distance(shot.aim_x, shot.aim_y, ax, ay, bx, by)
        if distance <= ARMOR_DRAW_THICKNESS_M / 2:
            normal_speed = max(
                0.0,
                -(velocity_x * armor.normal_x + velocity_y * armor.normal_y),
            )
            outcome = "valid_hit" if normal_speed > MIN_NORMAL_HIT_SPEED else "low_normal_speed"
            covered.append(HitResult(outcome, normal_speed, armor.index))

    if not covered:
        return HitResult("miss")
    if len(covered) == 1:
        return covered[0]
    return max(covered, key=lambda result: result.normal_speed or 0.0)


def shot_hits(shot: Shot, actual: TargetState) -> bool:
    return evaluate_shot(shot, actual).outcome == "valid_hit"


class MovingTargetVisualizer(tk.Tk):
    def __init__(
        self,
        mode: str = "current",
        distance: float = 6.0,
    ) -> None:
        super().__init__()
        random.seed(20260903)
        if distance <= 0.0:
            raise ValueError("distance must be positive")
        self.mode = mode
        mode_label = "C++路径枚举版" if mode == "rollout_cpp" else ("当前版：实际云台跟踪" if mode == "current" else ("场域MPC版" if mode == "field_mpc" else ("连续窗口候选版" if mode == "window_candidate" else ("执行感知候选评分版" if mode == "field_candidate" else ("发射前瞻控制版" if mode == "lookahead_control" else "加速度分布预测版")))))
        self.title(f"sp_vision_25 二维自瞄规划可视化 - {mode_label}")
        self.geometry("1180x820")
        self.minsize(820, 650)

        self.target_distance = tk.DoubleVar(value=float(distance))
        self.acceleration_direction = 1.0
        self.next_acceleration_switch = EVADE_SWITCH_INTERVAL
        self.gimbal_yaw = math.pi / 2
        self.gimbal_yaw_vel = 0.0
        self.gimbal_yaw_acc = 0.0
        self.running = True
        self.sim_time = 0.0
        self.last_clock: float | None = None
        self.simulation_accumulator = 0.0
        self.last_shot_time = -100.0
        self.valid_hits = 0
        self.low_speed_hits = 0
        self.misses = 0
        self.is_rollout_cpp = mode == "rollout_cpp"
        self.planner = None if self.is_rollout_cpp else TinyMpcPlanner()
        self.rollout_planner = CppRolloutPlanner() if self.is_rollout_cpp else None
        self._rollout_command_alpha = 0.0
        self._rollout_decision_time = -1.0
        self._rollout_path_score = 0.0
        self.target = self.initial_target()
        self._causal_observations: list[tuple[float, float]] = [
            (0.0, self.target.omega)
        ]
        self._causal_accel_history: list[tuple[float, float]] = []
        self.motion_history: list[tuple[float, float, float]] = []
        self._history_pending: list[tuple[float, float, float]] = []
        self.probabilistic_model = ProbabilisticAimModel(
            LOW_SPEED_DELAY + self.target_distance.get() / BULLET_SPEED
        )
        self._probabilistic_cache_time = -1.0
        self._probabilistic_cache: tuple[float, float] = (0.0, 0.0)
        self._probabilistic_last_phase: float | None = None
        self._probabilistic_reachable_peaks: list[tuple[float, float]] = []
        self._hit_curve_cache_time = -1.0
        self._hit_curve_grid: list[float] = []
        self._hit_curve_values: list[float] = []
        self._hit_curve_anchor_nominal_yaw = 0.0
        self._current_nominal_yaw = 0.0
        # Diagnostic only: this is the probability at the planner's reference
        # peak. Never use it as a fire gate; shots leave along self.gimbal_yaw.
        self.current_aim_probability = 0.0
        self.current_fire_probability = 0.0
        self.current_plan = (
            self._rollout_nominal_plan()
            if self.is_rollout_cpp
            else self.planner.plan(replace(self.target), BULLET_SPEED)
        )
        self.probabilistic_model.horizon = (
            self.current_plan.delay + self.current_plan.flight_time
        )
        self._probabilistic_peak_yaw = self.current_plan.target_yaw
        self._probabilistic_pmax = 0.0
        self._probabilistic_candidate_probability = 0.0
        self.diagnostic_log: list[dict[str, float | str | bool | None]] = []
        self.resolved_diagnostics: list[dict[str, float | str | bool | None]] = []
        self.gate_block_count = 0
        self.accel_saturated_steps = 0
        self._diag_current: dict[str, float | str | bool | None] | None = None
        self.shots: list[Shot] = []
        self._build_ui()
        self.bind("<space>", lambda _event: self.toggle_running())
        self.bind("<r>", lambda _event: self.reset_simulation())
        self.after(10, self.tick)

    def initial_target(self) -> TargetState:
        return TargetState(
            x=0.0,
            y=self.target_distance.get(),
            vx=0.0,
            vy=0.0,
            angle=0.0,
            omega=INITIAL_OMEGA,
            alpha=self.acceleration_direction * ALPHA_LIMIT,
        )

    def _build_ui(self) -> None:
        mode_label = "C++路径枚举版" if self.mode == "rollout_cpp" else ("当前版：实际云台跟踪" if self.mode == "current" else ("场域MPC版" if self.mode == "field_mpc" else ("连续窗口候选版" if self.mode == "window_candidate" else "加速度分布预测版")))
        ttk.Label(
            self,
            text=f"sp_vision_25 当前速度外推：二维自瞄规划（{mode_label}）",
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(anchor="w", padx=16, pady=(12, 2))
        ttk.Label(
            self,
            wraplength=1130,
            justify="left",
            text=(
                "二维化显示：实线车体为实际状态，虚线为预测状态；"
                "紫线为规划射击轨迹，橙线为受加速度限制后的实际云台方向。"
            ),
        ).pack(anchor="w", padx=16, pady=(0, 8))

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=16)
        ttk.Label(
            controls,
            text="闪避：±280°/s²，每 1.00 s 反向",
        ).pack(side="left", padx=(0, 14))
        ttk.Label(controls, text="目标中心：正前方固定").pack(side="left", padx=(0, 14))

        self.pause_button = ttk.Button(controls, text="暂停", command=self.toggle_running)
        self.pause_button.pack(side="left", padx=4)
        ttk.Button(controls, text="重置", command=self.reset_simulation).pack(side="left", padx=4)

        distance_controls = ttk.Frame(self)
        distance_controls.pack(fill="x", padx=16, pady=(7, 0))
        ttk.Label(distance_controls, text="靶车中心距离：").pack(side="left")
        self.distance_scale = ttk.Scale(
            distance_controls,
            from_=2.0,
            to=8.0,
            variable=self.target_distance,
            command=self.change_distance,
            length=360,
        )
        self.distance_scale.pack(side="left", padx=(0, 10))
        self.distance_label = ttk.Label(distance_controls, text="6.0 m", width=7)
        self.distance_label.pack(side="left")

        self.status = ttk.Label(self, text="", font=("Consolas", 10))
        self.status.pack(anchor="w", padx=16, pady=(7, 1))
        self.hit_rate_status = ttk.Label(self, text="", font=("Microsoft YaHei UI", 10, "bold"))
        self.hit_rate_status.pack(anchor="w", padx=16, pady=(0, 7))

        self.canvas = tk.Canvas(
            self,
            background="#f4f6f8",
            highlightthickness=1,
            highlightbackground="#aeb6bf",
        )
        self.canvas.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _event: self.draw())

    def change_distance(self, value: str) -> None:
        distance = round(float(value), 1)
        distance_changed = distance != self.target.y
        self.target_distance.set(distance)
        self.target.y = distance
        self._probabilistic_cache_time = -1.0
        self.distance_label.configure(text=f"{distance:.1f} m")
        if distance_changed:
            self.valid_hits = 0
            self.low_speed_hits = 0
            self.misses = 0
            self.shots.clear()
            self.last_shot_time = -100.0
            self.motion_history.clear()
            self._history_pending.clear()
            self.probabilistic_model.samples.clear()
            self.probabilistic_model.sample_entries.clear()
            self._causal_observations = [(0.0, self.target.omega)]
            self._causal_accel_history.clear()
            self._rollout_decision_time = -1.0
        self.draw()

    def toggle_running(self) -> None:
        self.running = not self.running
        self.pause_button.configure(text="暂停" if self.running else "继续")
        self.last_clock = None
        self.simulation_accumulator = 0.0

    def reset_simulation(self) -> None:
        self.sim_time = 0.0
        self.last_shot_time = -100.0
        self.acceleration_direction = 1.0
        self.next_acceleration_switch = EVADE_SWITCH_INTERVAL
        self.gimbal_yaw = math.pi / 2
        self.gimbal_yaw_vel = 0.0
        self.gimbal_yaw_acc = 0.0
        self.valid_hits = 0
        self.low_speed_hits = 0
        self.misses = 0
        self._rollout_command_alpha = 0.0
        self._rollout_decision_time = -1.0
        self._rollout_path_score = 0.0
        self.target = self.initial_target()
        self._causal_observations = [(0.0, self.target.omega)]
        self._causal_accel_history = []
        self.motion_history.clear()
        self._history_pending.clear()
        self.probabilistic_model.samples.clear()
        self.probabilistic_model.sample_entries.clear()
        self._probabilistic_cache_time = -1.0
        self._probabilistic_last_phase = None
        self._probabilistic_reachable_peaks = []
        self._hit_curve_cache_time = -1.0
        self._hit_curve_grid = []
        self._hit_curve_values = []
        self._hit_curve_anchor_nominal_yaw = 0.0
        self._current_nominal_yaw = 0.0
        self.current_aim_probability = 0.0
        self.current_fire_probability = 0.0
        self.current_plan = (
            self._rollout_nominal_plan()
            if self.is_rollout_cpp
            else self.planner.plan(replace(self.target), BULLET_SPEED)
        )
        self.probabilistic_model.horizon = (
            self.current_plan.delay + self.current_plan.flight_time
        )
        self._probabilistic_peak_yaw = self.current_plan.target_yaw
        self._probabilistic_pmax = 0.0
        self._probabilistic_candidate_probability = 0.0
        self.diagnostic_log.clear()
        self.resolved_diagnostics.clear()
        self.gate_block_count = 0
        self.accel_saturated_steps = 0
        self._diag_current = None
        self.shots.clear()
        self.running = True
        self.pause_button.configure(text="暂停")
        self.last_clock = None
        self.simulation_accumulator = 0.0
        self.draw()

    def _rollout_delay(self) -> float:
        return (
            HIGH_SPEED_DELAY
            if abs(self.target.omega) > DECISION_SPEED
            else LOW_SPEED_DELAY
        )

    def _rollout_predicted_launch_yaw(
        self,
        alpha: float | None = None,
    ) -> float:
        command = (
            self._rollout_command_alpha
            if alpha is None
            else alpha
        )
        launch_yaw, _velocity = advance_gimbal_limited(
            self.gimbal_yaw,
            self.gimbal_yaw_vel,
            command,
            self._rollout_delay(),
        )
        return wrap_angle(launch_yaw)

    def _rollout_nominal_plan(
        self,
        launch_yaw: float | None = None,
        alpha: float | None = None,
    ) -> PlannerOutput:
        command = self._rollout_command_alpha if alpha is None else alpha
        if launch_yaw is None:
            launch_yaw = self._rollout_predicted_launch_yaw(command)
        delay = self._rollout_delay()
        delayed = predict_constant_velocity(self.target, delay)
        armor, impact_state, flight_time = armor_solution_for_yaw(
            delayed,
            launch_yaw,
            BULLET_SPEED,
        )
        yaw = wrap_angle(self.gimbal_yaw)
        repeated = (wrap_angle(launch_yaw),) * PLANNER_HORIZON
        return PlannerOutput(
            target_yaw=wrap_angle(math.atan2(armor.y, armor.x)),
            yaw=yaw,
            yaw_velocity=self.gimbal_yaw_vel,
            yaw_acceleration=command,
            fire=False,
            fire_error=0.0,
            delay=delay,
            flight_time=flight_time,
            planned_aim_x=armor.x,
            planned_aim_y=armor.y,
            predicted_state=impact_state,
            armor_index=armor.index,
            reference_yaw=repeated,
            planned_yaw=repeated,
        )

    def _update_rollout_cpp(self) -> None:
        decision_time = (
            math.floor(self.sim_time / ROLLOUT_SEGMENT_DT)
            * ROLLOUT_SEGMENT_DT
        )
        cache_time = (
            math.floor(self.sim_time / PROBABILITY_FIELD_INTERVAL)
            * PROBABILITY_FIELD_INTERVAL
        )
        if self._rollout_decision_time != decision_time:
            grid = rollout_curve_grid(self.target)
            curves = self._causal_rollout_curves(grid)
            alpha, path_score = self.rollout_planner.plan(
                wrap_angle(self.gimbal_yaw),
                self.gimbal_yaw_vel,
                grid,
                curves,
            )
            self._rollout_command_alpha = alpha
            self._rollout_decision_time = decision_time
            self._rollout_path_score = path_score
            self._probabilistic_cache_time = cache_time
            self._probabilistic_cache = (0.0, max(curves[0]) if curves else 0.0)
            self._probabilistic_reachable_peaks = []
            anchor_plan = self._rollout_nominal_plan(alpha=alpha)
            self._cache_hit_curve(
                cache_time,
                grid,
                curves[0],
                anchor_plan.target_yaw,
            )
        plan = self._rollout_nominal_plan(
            alpha=self._rollout_command_alpha,
        )
        self.current_plan = plan
        self.probabilistic_model.horizon = plan.delay + plan.flight_time
        self._current_nominal_yaw = plan.target_yaw
        if (
            self._hit_curve_cache_time == cache_time
            and self._hit_curve_grid
            and len(self._hit_curve_grid) == len(self._hit_curve_values)
        ):
            self.current_aim_probability = self.probability_for_yaw(
                plan.target_yaw
            )
        else:
            self.current_aim_probability = 0.0

    def update_planner(self) -> None:
        if self.mode == "rollout_cpp":
            self._update_rollout_cpp()
            return
        base_plan = self.planner.plan(replace(self.target), BULLET_SPEED)
        self._current_nominal_yaw = base_plan.target_yaw
        self.current_plan = base_plan
        if self.mode in ("probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control"):
            cache_time = (
                math.floor(self.sim_time / PROBABILITY_FIELD_INTERVAL)
                * PROBABILITY_FIELD_INTERVAL
            )
            refreshing_field = self._probabilistic_cache_time != cache_time
            attempts = 3 if refreshing_field else 1
            for _ in range(attempts):
                target_yaw, self.current_aim_probability = self.probabilistic_yaw(
                    base_plan.target_yaw
                )
                self._probabilistic_candidate_probability = self.current_aim_probability
                # Rebuild the complete reference trajectory around the selected
                # probability-field branch and solve TinyMPC again.
                replanned = self.planner.plan(
                    replace(self.target),
                    BULLET_SPEED,
                    aim_yaw_override=target_yaw,
                    probability_phase_profile=True,
                )
                self.current_plan = replanned
                if not refreshing_field:
                    break
                plan_horizon = replanned.delay + replanned.flight_time
                if abs(plan_horizon - self.probabilistic_model.horizon) < 1e-5:
                    break
                # The selected plate changed the flight time. Rebuild the field
                # at that horizon so yaw, branch, range, and impact time converge
                # together instead of retaining a peak from the old timestamp.
                self._probabilistic_cache_time = -1.0
            if self.mode in ("field_candidate", "window_candidate") and (
                self.mode == "field_candidate"
                or (not WINDOW_CANDIDATE_REFRESH_ONLY)
                or refreshing_field
            ):
                nominal = base_plan.target_yaw
                selected = self.current_plan.target_yaw
                candidate_yaws = [
                    nominal,
                    selected,
                    selected + math.radians(2.0),
                    selected - math.radians(2.0),
                ]
                best_plan = self.current_plan
                baseline_launch = (
                    self.gimbal_yaw + self.gimbal_yaw_vel * self.current_plan.delay
                    + 0.5 * self.gimbal_yaw_acc * self.current_plan.delay**2
                )
                self.probabilistic_model.horizon = self.current_plan.delay + self.current_plan.flight_time
                def window_score(plan):
                    self.probabilistic_model.horizon = plan.delay + plan.flight_time
                    center = PLANNER_HALF_HORIZON + int(round(FIRE_LOOKAHEAD / PLANNER_DT))
                    indices = range(max(0, center - 2), min(len(plan.planned_yaw), center + 3))
                    query_yaws = [plan.planned_yaw[i] for i in indices]
                    if self._hit_curve_cache_time == cache_time:
                        values = self._cached_curve_values(query_yaws)
                    else:
                        values = self.probabilistic_model.hit_probability_field(
                            plan.predicted_state,
                            query_yaws,
                            count=64,
                            seed=int(cache_time * 100),
                        )
                    return 0.7 * sum(values) / max(1, len(values)) + 0.3 * sum(
                        value >= PROBABILITY_FIRE_THRESHOLD for value in values
                    ) / max(1, len(values))
                best_score = window_score(self.current_plan)
                for candidate_yaw in candidate_yaws:
                    candidate_plan = self.planner.plan(
                        replace(self.target), BULLET_SPEED,
                        aim_yaw_override=candidate_yaw,
                        probability_phase_profile=True,
                    )
                    launch_delay = candidate_plan.delay
                    predicted_launch_yaw = (
                        self.gimbal_yaw + self.gimbal_yaw_vel * launch_delay
                        + 0.5 * self.gimbal_yaw_acc * launch_delay * launch_delay
                    )
                    self.probabilistic_model.horizon = candidate_plan.delay + candidate_plan.flight_time
                    raw_score = window_score(candidate_plan) if self.mode == "window_candidate" else self._cached_curve_value(
                        predicted_launch_yaw
                    )
                    score = raw_score - FIELD_CANDIDATE_SWITCH_PENALTY * abs(
                        angle_error(candidate_yaw, selected)
                    )
                    if score > best_score + FIELD_CANDIDATE_MIN_GAIN:
                        best_score, best_plan = score, candidate_plan
                self.current_plan = best_plan
                self.probabilistic_model.horizon = best_plan.delay + best_plan.flight_time
                self.current_aim_probability = self.probability_for_yaw(best_plan.target_yaw)
                if self.mode == "window_candidate" and WINDOW_CANDIDATE_CACHE_PHASE:
                    cached_phase = angle_error(best_plan.target_yaw, nominal)
                    self._probabilistic_cache = (
                        cached_phase,
                        self.current_aim_probability,
                    )
                    self._probabilistic_last_phase = cached_phase
            elif self.mode == "field_mpc":
                self._select_field_mpc(base_plan)
            # Pref is diagnostic, but it must still describe the final plan's
            # target and impact state rather than the pre-replan base state.
            self.current_aim_probability = self.probability_for_yaw(
                self.current_plan.target_yaw
            )
            self._diag_current = {
                "time": self.sim_time,
                "pmax": self._probabilistic_pmax,
                "pref": self.current_aim_probability,
                "pcandidate": self._probabilistic_candidate_probability,
                "pplan": None,
                "pgun": None,
                "pactual": None,
                "peak_yaw": self._probabilistic_peak_yaw,
                "plan_yaw": self.current_plan.yaw,
                "gun_yaw": self.gimbal_yaw,
                "gate_block": False,
                "ref_plan_launch_mrad": None,
                "pred_actual_launch_mrad": None,
                "accel_saturated": False,
                "actual_result": None,
            }
            fire_index = PLANNER_HALF_HORIZON + int(round(FIRE_LOOKAHEAD / PLANNER_DT))
            if fire_index < len(self.current_plan.planned_yaw):
                ref_launch = self.current_plan.reference_yaw[fire_index]
                plan_launch = self.current_plan.planned_yaw[fire_index]
                self._diag_current["pplan"] = self.probability_for_yaw(plan_launch)
                self._diag_current["ref_plan_launch_mrad"] = angle_error(ref_launch, plan_launch) * 1000.0
            self.diagnostic_log.append(self._diag_current)

    def _field_mpc_candidate_yaws(self, base_plan: PlannerOutput) -> list[float]:
        """Collect reachable probability branches for one field-MPC outer loop."""
        selected_yaw = self.current_plan.target_yaw
        candidate_yaws: list[float] = []

        def add(yaw: float) -> None:
            if all(abs(angle_error(yaw, seen)) > math.radians(1.0) for seen in candidate_yaws):
                candidate_yaws.append(yaw)

        add(base_plan.target_yaw)
        add(selected_yaw)
        for offset_degrees in (1.0, 2.0, 3.0):
            for direction in (1.0, -1.0):
                if len(candidate_yaws) >= FIELD_MPC_MAX_CANDIDATES:
                    break
                add(selected_yaw + direction * math.radians(offset_degrees))
            if len(candidate_yaws) >= FIELD_MPC_MAX_CANDIDATES:
                break
        return candidate_yaws[:FIELD_MPC_MAX_CANDIDATES]

    def _simulate_plan_gun_yaws(self, plan: PlannerOutput) -> list[float]:
        """Project gun yaws if this MPC trajectory is followed for a few steps."""
        yaw = self.gimbal_yaw
        velocity = self.gimbal_yaw_vel
        acceleration = self.gimbal_yaw_acc
        output: list[float] = []
        horizon = len(plan.planned_yaw)
        for step in range(FIELD_MPC_TRAJECTORY_TICKS):
            index = min(PLANNER_HALF_HORIZON + step, horizon - 1)
            target_yaw = plan.planned_yaw[index]
            before = max(0, index - 1)
            after = min(horizon - 1, index + 1)
            target_velocity = angle_error(
                plan.planned_yaw[after],
                plan.planned_yaw[before],
            ) / (2.0 * PLANNER_DT)
            target_acceleration = plan.yaw_acceleration
            command_acc = clamp(
                target_acceleration
                + 35.0 * angle_error(target_yaw, yaw)
                + 10.0 * (target_velocity - velocity),
                -GIMBAL_MAX_ACCEL,
                GIMBAL_MAX_ACCEL,
            )
            yaw += velocity * PLANNER_DT + 0.5 * command_acc * PLANNER_DT * PLANNER_DT
            velocity += command_acc * PLANNER_DT
            acceleration = command_acc
            output.append(
                yaw
                + velocity * plan.delay
                + 0.5 * acceleration * plan.delay * plan.delay
            )
        return output

    def _score_field_mpc_trajectory(self, plan: PlannerOutput) -> float:
        """Score closed-loop gun probabilities across one 50 ms firing block."""
        self.probabilistic_model.horizon = plan.delay + plan.flight_time
        cache_time = (
            math.floor(self.sim_time / PROBABILITY_FIELD_INTERVAL)
            * PROBABILITY_FIELD_INTERVAL
        )
        values = self.probabilistic_model.hit_probability_field(
            plan.predicted_state,
            self._simulate_plan_gun_yaws(plan),
            count=64,
            seed=int(cache_time * 100),
        )
        if not values:
            return 0.0
        block = values[:FIELD_MPC_BLOCK_SIZE]
        if not block:
            return 0.0
        maximum = max(block)
        mean_value = sum(block) / len(block)
        ready_fraction = sum(
            value >= PROBABILITY_FIRE_THRESHOLD for value in values
        ) / len(values)
        return 0.50 * maximum + 0.30 * mean_value + 0.20 * ready_fraction

    def _select_field_mpc(self, base_plan: PlannerOutput) -> None:
        """Select the best candidate MPC trajectory under simulated execution."""
        best_plan = self.current_plan
        best_score = self._score_field_mpc_trajectory(best_plan)
        selected_yaw = best_plan.target_yaw
        for candidate_yaw in self._field_mpc_candidate_yaws(base_plan):
            if abs(angle_error(candidate_yaw, selected_yaw)) < math.radians(0.5):
                continue
            candidate_plan = self.planner.plan(
                replace(self.target),
                BULLET_SPEED,
                aim_yaw_override=candidate_yaw,
                probability_phase_profile=True,
            )
            raw_score = self._score_field_mpc_trajectory(candidate_plan)
            score = raw_score - FIELD_MPC_SWITCH_PENALTY * abs(
                angle_error(candidate_yaw, selected_yaw)
            )
            if score > best_score + FIELD_MPC_MIN_GAIN:
                best_score = score
                best_plan = candidate_plan
        self.current_plan = best_plan
        self.probabilistic_model.horizon = (
            best_plan.delay + best_plan.flight_time
        )

    def probabilistic_yaw(
        self,
        phase_reference_yaw: float | None = None,
    ) -> tuple[float, float]:
        """Select the best reachable peak of the future hit-probability field."""
        nominal_yaw = (
            self.current_plan.target_yaw
            if phase_reference_yaw is None
            else phase_reference_yaw
        )
        cache_time = (
            math.floor(self.sim_time / PROBABILITY_FIELD_INTERVAL)
            * PROBABILITY_FIELD_INTERVAL
        )
        if self._probabilistic_cache_time == cache_time:
            phase, _stale_probability = self._probabilistic_cache
            selected_yaw = project_yaw_to_reachable(
                nominal_yaw + phase,
                self.gimbal_yaw,
                self.gimbal_yaw_vel,
                self.current_plan.delay + self.current_plan.flight_time,
            )
            # The cached phase moves with the current nominal target yaw. Its
            # old probability belongs to the previous absolute yaw/state, so
            # recompute this cheap one-point field evaluation every 10 ms.
            probability = self.probability_for_yaw(selected_yaw)
            self._probabilistic_cache = (phase, probability)
            return selected_yaw, probability
        self.probabilistic_model.horizon = (
            self.current_plan.delay + self.current_plan.flight_time
        )
        predicted = self.current_plan.predicted_state
        # The curve domain is separate from the reachable search domain. Cache
        # the full armor-bearing arc plus padding; only peak selection applies
        # the gimbal reachability interval afterward.
        center_distance = max(norm(self.target.x, self.target.y), 1e-6)
        center_yaw = math.atan2(self.target.y, self.target.x)
        half_span = math.asin(
            min(1.0, BODY_MAX_RADIUS_M / center_distance)
        )
        half_span += math.atan2(
            ARMOR_WIDTH_M / 2.0 + ARMOR_DRAW_THICKNESS_M / 2.0,
            center_distance,
        )
        scatter_padding = 4.0 * math.atan2(SCATTER_SIGMA, center_distance)
        curve_padding = scatter_padding + ARMOR_CURVE_MOVEMENT_RAD
        curve_lower = center_yaw - half_span - curve_padding
        curve_upper = center_yaw + half_span + curve_padding
        reach_time = max(
            self.current_plan.delay + self.current_plan.flight_time,
            PLANNER_DT,
        )
        reach_lower, reach_upper = reachable_yaw_delta_bounds(
            self.gimbal_yaw_vel,
            reach_time,
        )
        grid: list[float] = []
        cursor = curve_lower
        while cursor <= curve_upper + 1e-12:
            grid.append(cursor)
            cursor += ARMOR_SCAN_STEP
        if not grid:
            grid = [curve_lower]
        field = self.probabilistic_model.hit_probability_field(
            predicted, grid, count=64, seed=int(cache_time * 100)
        )
        self._cache_hit_curve(cache_time, grid, field, nominal_yaw)
        # Reject peaks that cannot be reached before impact under the gimbal
        # acceleration limit. This is the one-step reachable-set approximation
        # used by chance-constrained MPC outer loops.
        reachable = [
            index for index, yaw in enumerate(grid)
            if reach_lower <= yaw - self.gimbal_yaw <= reach_upper
        ]
        body_edge = math.asin(
            min(1.0, BODY_MAX_RADIUS_M / center_distance)
        )
        body_reachable = [
            index
            for index in reachable
            if abs(angle_error(grid[index], center_yaw)) < body_edge - 1e-12
        ]
        self._probabilistic_pmax = max(
            (field[i] for i in body_reachable),
            default=0.0,
        )
        self._probabilistic_peak_yaw = (
            grid[max(body_reachable, key=lambda i: field[i])]
            if body_reachable
            else nominal_yaw
        )
        # Keep several local maxima as branches.  A small continuity cost
        # prevents stochastic resampling from jumping between opposite armor
        # peaks unless the new branch is materially better.
        candidates = [
            i
            for i in body_reachable
            if i == 0
            or i == len(field) - 1
            or field[i] >= field[i - 1]
            and field[i] >= field[i + 1]
        ]
        if not candidates:
            if not body_reachable:
                return (
                    project_yaw_to_reachable(
                        nominal_yaw,
                        self.gimbal_yaw,
                        self.gimbal_yaw_vel,
                        max(
                            self.current_plan.delay
                            + self.current_plan.flight_time,
                            PLANNER_DT,
                        ),
                    ),
                    0.0,
                )
            candidates = body_reachable
        self._probabilistic_reachable_peaks = sorted(
            (
                (grid[index], field[index])
                for index in candidates
                if field[index] > 0.0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[:10]
        previous = self._probabilistic_last_phase
        def branch_score(index: int) -> float:
            probability = field[index]
            reference = nominal_yaw if previous is None else nominal_yaw + previous
            delta = angle_error(grid[index], reference)
            return probability - PROBABILITY_CONTINUITY_WEIGHT * (delta / math.pi) ** 2
        best = max(candidates, key=branch_score)
        if previous is not None:
            best_probability = field[best]
            previous_index = min(range(len(grid)), key=lambda i: abs(angle_error(grid[i], nominal_yaw + previous)))
            previous_probability = field[previous_index]
            if best_probability < previous_probability + PROBABILITY_PEAK_SWITCH_MARGIN:
                best = previous_index if previous_index in reachable else best
        selected_yaw = grid[best]
        selected_phase = angle_error(selected_yaw, nominal_yaw)
        if previous is not None:
            selected_phase = previous + PROBABILITY_YAW_SMOOTHING * angle_error(selected_phase, previous)
        selected_yaw = project_yaw_to_reachable(
            nominal_yaw + selected_phase,
            self.gimbal_yaw,
            self.gimbal_yaw_vel,
            reach_time,
        )
        selected_phase = angle_error(selected_yaw, nominal_yaw)
        selected_probability = self._cached_curve_value(
            nominal_yaw + selected_phase
        )
        self._probabilistic_last_phase = selected_phase
        self._probabilistic_cache = (selected_phase, selected_probability)
        self._probabilistic_cache_time = cache_time
        return nominal_yaw + selected_phase, selected_probability

    def probability_for_yaw(self, yaw: float) -> float:
        """Probability that a shot along this actual yaw hits at impact time."""
        if self.__dict__.get("mode") == "rollout_cpp":
            delay = self._rollout_delay()
            delayed = predict_constant_velocity(self.target, delay)
            _armor, impact_state, flight_time = armor_solution_for_yaw(
                delayed,
                yaw,
                BULLET_SPEED,
            )
            self.probabilistic_model.horizon = delay + flight_time
            return self.probabilistic_model.hit_probability_field(
                impact_state,
                [yaw],
                count=ROLLOUT_PROBABILITY_COUNT,
                seed=int(self.sim_time * 100),
            )[0]
        cache_time = (
            math.floor(self.sim_time / PROBABILITY_FIELD_INTERVAL)
            * PROBABILITY_FIELD_INTERVAL
        )
        if self.__dict__.get("_hit_curve_cache_time", -1.0) == cache_time:
            return self._cached_curve_value(yaw)
        return self.probabilistic_model.hit_probability_field(
            self.current_plan.predicted_state,
            [yaw],
            count=64,
            seed=int(cache_time * 100),
        )[0]

    def _cache_hit_curve(
        self,
        cache_time: float,
        grid: list[float],
        values: list[float],
        nominal_yaw: float,
    ) -> None:
        self._hit_curve_cache_time = cache_time
        self._hit_curve_grid = list(grid)
        self._hit_curve_values = list(values)
        self._hit_curve_anchor_nominal_yaw = nominal_yaw

    def _cached_curve_value(self, yaw: float) -> float:
        grid = self._hit_curve_grid
        values = self._hit_curve_values
        if len(grid) != len(values) or len(grid) < 2:
            return 0.0
        anchor = self.__dict__.get("_hit_curve_anchor_nominal_yaw")
        if anchor is not None:
            current_nominal = self.__dict__.get(
                "_current_nominal_yaw",
                self.current_plan.target_yaw,
            )
            yaw -= angle_error(current_nominal, anchor)
        if yaw < grid[0] or yaw > grid[-1]:
            return 0.0
        index = bisect_left(grid, yaw)
        if index <= 0:
            return values[0]
        before = grid[index - 1]
        after = grid[index]
        fraction = (yaw - before) / (after - before)
        return values[index - 1] + fraction * (values[index] - values[index - 1])

    def _cached_curve_values(self, yaws: list[float]) -> list[float]:
        return [self._cached_curve_value(yaw) for yaw in yaws]

    def fire_if_ready(self) -> None:
        launch_delay = self.current_plan.delay
        if self.sim_time - self.last_shot_time < SHOT_INTERVAL:
            return
        if self.mode == "rollout_cpp":
            predicted_launch_yaw = self._rollout_predicted_launch_yaw(
                self._rollout_command_alpha
            )
        else:
            predicted_launch_yaw = (
                self.gimbal_yaw
                + self.gimbal_yaw_vel * launch_delay
                + 0.5 * self.gimbal_yaw_acc * launch_delay * launch_delay
            )
        if self.mode in ("probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control", "rollout_cpp"):
            # Gate the predicted physical muzzle direction at launch, not the
            # reference peak or the current pre-delay direction. Do not replace
            # this with current_aim_probability: that recurring bug authorizes
            # shots based on where the planner wants to point.
            self.current_fire_probability = self.probability_for_yaw(
                predicted_launch_yaw
            )
            diag_current = self.__dict__.get("_diag_current")
            if diag_current is not None:
                diag_current["pgun"] = self.current_fire_probability
                diag_current["gun_yaw"] = self.gimbal_yaw
            if self.current_fire_probability < PROBABILITY_FIRE_THRESHOLD:
                self.gate_block_count = self.__dict__.get("gate_block_count", 0) + 1
                if diag_current is not None:
                    diag_current["gate_block"] = True
                return
        elif not self.current_plan.fire:
            return
        shot = shot_from_plan(
            self.current_plan,
            self.sim_time,
            predicted_launch_yaw,
            scatter=True,
        )
        if self.mode in ("probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control", "rollout_cpp"):
            shot.gate_yaw = predicted_launch_yaw
            current_diag = self.__dict__.get("_diag_current")
            shot.diagnostic = dict(current_diag) if current_diag is not None else None
            if shot.diagnostic is not None:
                self.resolved_diagnostics.append(shot.diagnostic)
        self.shots.append(shot)
        self.last_shot_time = self.sim_time

    def advance_gimbal(self, dt: float) -> None:
        """Track the absolute yaw/velocity command with the configured acceleration limit."""
        start_yaw = self.gimbal_yaw
        start_velocity = self.gimbal_yaw_vel
        if self.mode == "rollout_cpp":
            command_acc = self._rollout_command_alpha
            next_yaw, next_vel = advance_gimbal_limited(
                start_yaw,
                start_velocity,
                command_acc,
                dt,
            )
            self.gimbal_yaw = wrap_angle(next_yaw)
            self.gimbal_yaw_vel = next_vel
            self.gimbal_yaw_acc = command_acc
            if abs(command_acc) >= GIMBAL_MAX_ACCEL - 1e-9:
                self.accel_saturated_steps = (
                    self.__dict__.get("accel_saturated_steps", 0) + 1
                )
            self._gimbal_step = (
                self.sim_time - dt,
                start_yaw,
                start_velocity,
                command_acc,
            )
            return
        control_yaw = self.current_plan.yaw
        control_velocity = self.current_plan.yaw_velocity
        control_acceleration = self.current_plan.yaw_acceleration
        if self.mode == "lookahead_control":
            lookahead_index = PLANNER_HALF_HORIZON + int(round(FIRE_LOOKAHEAD / PLANNER_DT))
            control_yaw = self.current_plan.planned_yaw[lookahead_index]
            if lookahead_index < len(self.current_plan.planned_yaw) - 1:
                control_velocity = angle_error(
                    self.current_plan.planned_yaw[lookahead_index + 1],
                    self.current_plan.planned_yaw[lookahead_index - 1],
                ) / (2 * PLANNER_DT)
                control_acceleration = self.current_plan.yaw_acceleration
        yaw_error = angle_error(control_yaw, self.gimbal_yaw)
        velocity_error = control_velocity - self.gimbal_yaw_vel
        command_acc = clamp(
            control_acceleration + 35.0 * yaw_error + 10.0 * velocity_error,
            -GIMBAL_MAX_ACCEL,
            GIMBAL_MAX_ACCEL,
        )
        next_vel = self.gimbal_yaw_vel + command_acc * dt
        self.gimbal_yaw += self.gimbal_yaw_vel * dt + 0.5 * command_acc * dt * dt
        self.gimbal_yaw_vel = next_vel
        self.gimbal_yaw_acc = command_acc
        if abs(command_acc) >= GIMBAL_MAX_ACCEL - 1e-9:
            self.accel_saturated_steps = self.__dict__.get("accel_saturated_steps", 0) + 1
            diag_current = self.__dict__.get("_diag_current")
            if diag_current is not None:
                diag_current["accel_saturated"] = True
        self._gimbal_step = (
            self.sim_time - dt,
            start_yaw,
            start_velocity,
            command_acc,
        )

    def resolve_shots(self) -> None:
        for shot in self.shots:
            if not shot.launched and self.sim_time >= shot.launch_at:
                shot.launched = True
                shot.launch_angle = self.gimbal_yaw
                if hasattr(self, "_gimbal_step"):
                    start_time, start_yaw, start_velocity, acceleration = self._gimbal_step
                    if start_time <= shot.launch_at <= self.sim_time:
                        tau = shot.launch_at - start_time
                        if self.__dict__.get("mode") == "rollout_cpp":
                            shot.launch_angle = advance_gimbal_limited(
                                start_yaw,
                                start_velocity,
                                acceleration,
                                tau,
                            )[0]
                        else:
                            shot.launch_angle = (
                                start_yaw
                                + start_velocity * tau
                                + 0.5 * acceleration * tau * tau
                            )
                        if self.__dict__.get("mode") == "rollout_cpp":
                            shot.launch_angle = wrap_angle(shot.launch_angle)
                if shot.gate_yaw is not None:
                    shot.launch_prediction_error = angle_error(
                        shot.launch_angle,
                        shot.gate_yaw,
                    )
                    if shot.diagnostic is not None:
                        shot.diagnostic["pred_actual_launch_mrad"] = shot.launch_prediction_error * 1000.0
                if shot.diagnostic is not None:
                    pactual = self.probabilistic_model.hit_probability_field(
                        shot.predicted_state, [shot.launch_angle], count=64, seed=int(shot.created_at * 100)
                    )[0]
                    shot.diagnostic["pactual"] = pactual
                planned_range = norm(shot.planned_aim_x, shot.planned_aim_y)
                shot.aim_x = planned_range * math.cos(shot.launch_angle)
                shot.aim_y = planned_range * math.sin(shot.launch_angle)
                shot.aim_x -= shot.scatter_offset * math.sin(shot.launch_angle)
                shot.aim_y += shot.scatter_offset * math.cos(shot.launch_angle)
            if not shot.resolved and self.sim_time >= shot.impact_at:
                shot.resolved = True
                impact_target = self.target
                if hasattr(self, "_target_step"):
                    (
                        start_time,
                        start_target,
                        start_direction,
                        start_next_switch,
                    ) = self._target_step
                    if start_time <= shot.impact_at <= self.sim_time:
                        tau = shot.impact_at - start_time
                        angle, omega, _direction, alpha, _next_switch = (
                            advance_optimized_evasion(
                                start_target.angle,
                                start_target.omega,
                                start_direction,
                                start_time,
                                start_next_switch,
                                tau,
                            )
                        )
                        impact_target = replace(
                            start_target,
                            angle=angle,
                            omega=omega,
                            alpha=alpha,
                        )
                result = evaluate_shot(shot, impact_target)
                shot.outcome = result.outcome
                if shot.diagnostic is not None:
                    shot.diagnostic["actual_result"] = result.outcome == "valid_hit"
                shot.normal_speed = result.normal_speed
                shot.hit = result.outcome == "valid_hit"
                shot.resolved_at = self.sim_time
                if result.outcome == "valid_hit":
                    self.valid_hits += 1
                elif result.outcome == "low_normal_speed":
                    self.low_speed_hits += 1
                else:
                    self.misses += 1
        self.shots = [
            shot
            for shot in self.shots
            if not shot.resolved or self.sim_time - shot.resolved_at < RESULT_DISPLAY_TIME
        ]

    def advance_actual(self, dt: float) -> None:
        self._target_step = (
            self.sim_time - dt,
            replace(self.target),
            self.acceleration_direction,
            self.next_acceleration_switch,
        )
        (
            self.target.angle,
            self.target.omega,
            self.acceleration_direction,
            self.target.alpha,
            self.next_acceleration_switch,
        ) = advance_optimized_evasion(
            self.target.angle,
            self.target.omega,
            self.acceleration_direction,
            self.sim_time - dt,
            self.next_acceleration_switch,
            dt,
        )
        self.target.x = 0.0
        self.target.vx = 0.0
        self._causal_observations.append((self.sim_time, self.target.omega))
        self._causal_accel_history.append(
            (self.sim_time, self._estimate_causal_acceleration())
        )
        self._ingest_online_history()

    def _estimate_causal_acceleration(self) -> float:
        cutoff = self.sim_time - 0.20
        points = [
            (time, omega)
            for time, omega in self._causal_observations
            if time >= cutoff
        ]
        if len(points) < 3:
            return 0.0
        count = float(len(points))
        mean_time = sum(point[0] for point in points) / count
        mean_omega = sum(point[1] for point in points) / count
        numerator = sum(
            (point[0] - mean_time) * (point[1] - mean_omega)
            for point in points
        )
        denominator = sum(
            (point[0] - mean_time) ** 2
            for point in points
        )
        return numerator / denominator if abs(denominator) > 1e-18 else 0.0

    def _causal_rollout_limits(self) -> tuple[float, float]:
        omega_limit = max(
            (abs(omega) for _time, omega in self._causal_observations),
            default=1e-9,
        )
        alpha_limit = max(
            (
                abs(alpha)
                for _time, alpha in self._causal_accel_history
            ),
            default=1e-9,
        )
        return max(omega_limit, 1e-9), max(alpha_limit, 1e-9)

    def _causal_rollout_curves(self, grid: list[float]) -> list[list[float]]:
        import sys as _sys

        planner_dir = Path(__file__).resolve().parent / "cpp_rollout_planner"
        if str(planner_dir) not in _sys.path:
            _sys.path.insert(0, str(planner_dir))
        from causal_evasion_predictor import CausalEvasionPredictor
        from p_hit_provider import build_segment_curves

        omega_limit, alpha_limit = self._causal_rollout_limits()
        predictor = CausalEvasionPredictor()
        predictor.reset(
            self.target.angle,
            self.target.omega,
            self.sim_time,
            omega_limit,
            alpha_limit,
            self._estimate_causal_acceleration(),
            center_x=self.target.x,
            center_y=self.target.y,
        )
        return build_segment_curves(
            self.probabilistic_model,
            self.target,
            self.acceleration_direction,
            self.next_acceleration_switch,
            self.sim_time,
            grid,
            schedule=predictor,
        )

    def _ingest_online_history(self) -> None:
        """Append each observed start once when it gains a future observation."""
        horizon = self.probabilistic_model.horizon
        now = self.sim_time
        pending = self.__dict__.setdefault("_history_pending", [])
        current = (now, self.target.angle, self.target.omega)
        pending.append(current)
        retained: list[tuple[float, float, float]] = []
        for start_time, start_angle, start_omega in pending:
            if start_time + horizon <= now + 1e-12:
                tau = now - start_time
                self.probabilistic_model.accumulate_sample(
                    start_time,
                    start_angle,
                    start_omega,
                    self.target.angle,
                    tau,
                )
            else:
                retained.append((start_time, start_angle, start_omega))
        self._history_pending = retained

    def tick(self) -> None:
        now = time.perf_counter()
        if self.last_clock is None:
            self.last_clock = now
        dt = min(max(now - self.last_clock, 0.0), 0.04)
        self.last_clock = now
        if self.running:
            self.simulation_accumulator += dt * SIMULATION_TIME_SCALE
            # Preserve the original 10 ms discrete dynamics and planner cadence;
            # wall-clock scaling only controls how often those fixed steps run.
            while self.simulation_accumulator >= PLANNER_DT:
                self.simulation_accumulator -= PLANNER_DT
                self.sim_time += PLANNER_DT
                self.advance_actual(PLANNER_DT)
                self.update_planner()
                self.advance_gimbal(PLANNER_DT)
                self.fire_if_ready()
                self.resolve_shots()
            self.draw()
        self.after(10, self.tick)

    def draw(self) -> None:
        canvas = self.canvas
        width = max(canvas.winfo_width(), 760)
        height = max(canvas.winfo_height(), 440)
        canvas.delete("all")
        margin = 34.0

        def world_to_screen(x: float, y: float) -> tuple[float, float]:
            sx = margin + (x - FIELD_X[0]) / (FIELD_X[1] - FIELD_X[0]) * (width - 2 * margin)
            sy = height - margin - (y - FIELD_Y[0]) / (FIELD_Y[1] - FIELD_Y[0]) * (height - 2 * margin)
            return sx, sy

        def polygon_for_body(state: TargetState) -> list[float]:
            points = (
                (-BODY_WIDTH_M / 2, -BODY_LENGTH_M / 2),
                (BODY_WIDTH_M / 2, -BODY_LENGTH_M / 2),
                (BODY_WIDTH_M / 2, BODY_LENGTH_M / 2),
                (-BODY_WIDTH_M / 2, BODY_LENGTH_M / 2),
            )
            output: list[float] = []
            for local_x, local_y in points:
                dx, dy = rotate(local_x, local_y, state.angle)
                output.extend(world_to_screen(state.x + dx, state.y + dy))
            return output

        canvas.create_rectangle(margin, margin, width - margin, height - margin, outline="#9aa3aa")
        for meter in range(-3, 4):
            x1, y1 = world_to_screen(meter, FIELD_Y[0])
            x2, y2 = world_to_screen(meter, FIELD_Y[1])
            canvas.create_line(x1, y1, x2, y2, fill="#d8dde1", dash=(3, 5))
        for meter in range(0, 9):
            x1, y1 = world_to_screen(FIELD_X[0], meter)
            x2, y2 = world_to_screen(FIELD_X[1], meter)
            canvas.create_line(x1, y1, x2, y2, fill="#d8dde1", dash=(3, 5))
            canvas.create_text(margin + 4, y1 - 2, anchor="sw", text=f"{meter} m", fill="#687078")

        shooter_x, shooter_y = world_to_screen(0.0, 0.0)
        canvas.create_polygon(
            shooter_x,
            shooter_y - 12,
            shooter_x - 11,
            shooter_y + 10,
            shooter_x + 11,
            shooter_y + 10,
            fill="#455a64",
        )
        canvas.create_text(shooter_x + 16, shooter_y, anchor="w", text="射手", fill="#263238")

        if self.current_plan is not None:
            predicted = self.current_plan.predicted_state
            canvas.create_polygon(
                polygon_for_body(predicted),
                fill="",
                outline="#7b1fa2",
                width=2,
                dash=(6, 4),
            )
            px, py = world_to_screen(
                self.current_plan.planned_aim_x, self.current_plan.planned_aim_y
            )
            canvas.create_line(px - 8, py, px + 8, py, fill="#7b1fa2", width=2)
            canvas.create_line(px, py - 8, px, py + 8, fill="#7b1fa2", width=2)
            canvas.create_line(shooter_x, shooter_y, px, py, fill="#7b1fa2", dash=(5, 5))

        gimbal_length = 8.2
        gx, gy = world_to_screen(
            gimbal_length * math.cos(self.gimbal_yaw),
            gimbal_length * math.sin(self.gimbal_yaw),
        )
        canvas.create_line(shooter_x, shooter_y, gx, gy, fill="#ef6c00", width=3)
        canvas.create_polygon(
            polygon_for_body(self.target), fill="#dfe5e9", outline="#263238", width=2
        )
        scale_x = (width - 2 * margin) / (FIELD_X[1] - FIELD_X[0])
        scale_y = (height - 2 * margin) / (FIELD_Y[1] - FIELD_Y[0])
        for armor in armor_poses(self.target):
            half = ARMOR_WIDTH_M / 2
            ax, ay = world_to_screen(
                armor.x - armor.tangent_x * half,
                armor.y - armor.tangent_y * half,
            )
            bx, by = world_to_screen(
                armor.x + armor.tangent_x * half,
                armor.y + armor.tangent_y * half,
            )
            canvas.create_line(ax, ay, bx, by, fill="#1976d2", width=max(5, int(ARMOR_DRAW_THICKNESS_M * min(scale_x, scale_y))))

        latest_resolved = next((shot for shot in reversed(self.shots) if shot.resolved), None)
        for shot in self.shots:
            if shot.resolved:
                if shot is latest_resolved:
                    ix, iy = world_to_screen(shot.aim_x, shot.aim_y)
                    if shot.outcome == "valid_hit":
                        color = "#2e7d32"
                        result_text = f"命中：法向 {shot.normal_speed:.1f} > 12 m/s"
                    elif shot.outcome == "low_normal_speed":
                        color = "#ef6c00"
                        result_text = f"无效：法向 {shot.normal_speed:.1f} ≤ 12 m/s"
                    else:
                        color = "#c62828"
                        result_text = "未命中"
                    canvas.create_oval(ix - 6, iy - 6, ix + 6, iy + 6, outline=color, width=3)
                    canvas.create_text(
                        ix + 10,
                        iy - 10,
                        anchor="sw",
                        text=result_text,
                        fill=color,
                        font=("Microsoft YaHei UI", 10, "bold"),
                    )
                continue
            if not shot.launched:
                continue
            progress = clamp(
                (self.sim_time - shot.launch_at) / max(shot.impact_at - shot.launch_at, 1e-6),
                0.0,
                1.0,
            )
            bullet_x = shot.launch_x + (shot.aim_x - shot.launch_x) * progress
            bullet_y = shot.launch_y + (shot.aim_y - shot.launch_y) * progress
            bx, by = world_to_screen(bullet_x, bullet_y)
            canvas.create_oval(bx - 4, by - 4, bx + 4, by + 4, fill="#f57c00", outline="")

        fire_index = PLANNER_HALF_HORIZON + int(round(FIRE_LOOKAHEAD / PLANNER_DT))
        predicted_fire_yaw = self.current_plan.reference_yaw[fire_index]
        predicted_gimbal_yaw = (
            self.gimbal_yaw
            + self.gimbal_yaw_vel * FIRE_LOOKAHEAD
            + 0.5 * self.gimbal_yaw_acc * FIRE_LOOKAHEAD * FIRE_LOOKAHEAD
        )
        predicted_gimbal_error = angle_error(predicted_fire_yaw, predicted_gimbal_yaw)
        if self.mode in (
            "probabilistic",
            "field_candidate",
            "field_mpc",
            "window_candidate",
            "lookahead_control",
            "rollout_cpp",
        ):
            fire_allowed = self.current_fire_probability >= PROBABILITY_FIRE_THRESHOLD
            fire_state = (
                f"{'允许' if fire_allowed else '抑制'}"
                f"(Pgun={self.current_fire_probability:.2f}>={PROBABILITY_FIRE_THRESHOLD:.2f}, "
                f"Pref={self.current_aim_probability:.2f})"
            )
        else:
            fire_state = "允许(3mrad)" if self.current_plan.fire else "抑制(3mrad)"
        self.status.configure(
            text=(
                f"t={self.sim_time:6.2f}s   ω={math.degrees(self.target.omega):+6.1f}°/s   "
                f"αtarget={math.degrees(self.target.alpha):+4.0f}°/s²   "
                f"云台={math.degrees(wrap_angle(self.gimbal_yaw)):+6.1f}°  "
                f"云台ω={self.gimbal_yaw_vel:+5.2f}rad/s  "
                f"αcmd={self.gimbal_yaw_acc:+5.1f}rad/s²   "
                f"预测误差={predicted_gimbal_error * 1000:+5.2f}mrad  "
                f"fire={fire_state}   射频上限=20发/s"
            )
        )
        resolved_count = self.valid_hits + self.low_speed_hits + self.misses
        hit_rate = 100.0 * self.valid_hits / resolved_count if resolved_count else 0.0
        self.hit_rate_status.configure(
            text=(
                f"有效命中率={hit_rate:5.1f}%  "
                f"有效命中={self.valid_hits}  低速无效={self.low_speed_hits}  "
                f"脱靶={self.misses}  已判定={resolved_count}"
            )
        )


def self_test() -> None:
    reach_lower, reach_upper = reachable_yaw_delta_bounds(OMEGA_LIMIT, 0.10)
    assert reach_lower > 0.0
    assert reach_upper > reach_lower
    projected = project_yaw_to_reachable(
        -1.0,
        0.0,
        OMEGA_LIMIT,
        0.10,
    )
    assert abs(projected - reach_lower) < 1e-12

    angle, omega, direction, alpha = advance_full_acceleration(
        0.0, OMEGA_LIMIT - math.radians(1.0), 1.0, 0.02
    )
    assert -OMEGA_LIMIT <= omega <= OMEGA_LIMIT
    assert direction == 1.0
    assert alpha == 0.0
    assert abs(omega - OMEGA_LIMIT) < 1e-12
    assert angle > 0.0

    angle, omega, direction, alpha, next_switch = advance_optimized_evasion(
        0.0, INITIAL_OMEGA, 1.0, 1.49, 1.50, 0.02
    )
    assert direction == -1.0
    assert alpha == -ALPHA_LIMIT
    assert abs(next_switch - 2.5) < 1e-12
    assert -OMEGA_LIMIT <= omega <= OMEGA_LIMIT

    planner = TinyMpcPlanner()
    state = TargetState(0.0, 6.0, 1.0, 0.0, 0.0, 2.0, 5.0)
    predicted = predict_constant_velocity(state, 0.25)
    assert abs(predicted.x - 0.25) < 1e-12
    assert abs(predicted.y - 6.0) < 1e-12
    assert abs(predicted.angle - 0.5) < 1e-12
    assert abs(predicted.omega - 2.0) < 1e-12
    plan = planner.plan(state, BULLET_SPEED)
    shot = shot_from_plan(plan, 1.0)
    assert abs(shot.launch_at - (1.0 + HIGH_SPEED_DELAY)) < 1e-12
    assert shot.launch_angle is None
    assert shot.impact_at > shot.launch_at
    assert 0 <= shot.armor_index < 4
    assert len(plan.reference_yaw) == PLANNER_HORIZON
    assert len(plan.planned_yaw) == PLANNER_HORIZON
    assert abs(plan.yaw_acceleration) <= GIMBAL_MAX_ACCEL + 1e-6
    static = TargetState(0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    static_plan = planner.plan(static, BULLET_SPEED)
    static_shot = shot_from_plan(static_plan, 0.0)
    actual_at_impact = predict_constant_velocity(static, static_shot.impact_at)
    straight_result = evaluate_shot(static_shot, actual_at_impact)
    assert straight_result.outcome == "valid_hit"
    assert abs((straight_result.normal_speed or 0.0) - BULLET_SPEED) < 0.1

    oblique = TargetState(0.0, 6.0, 0.0, 0.0, math.radians(70.0), 0.0, 0.0)
    oblique_armor = armor_poses(oblique)[0]
    oblique_shot = replace(
        static_shot,
        aim_x=oblique_armor.x,
        aim_y=oblique_armor.y,
    )
    oblique_result = evaluate_shot(oblique_shot, oblique)
    assert oblique_result.outcome == "low_normal_speed"
    assert (oblique_result.normal_speed or 0.0) <= MIN_NORMAL_HIT_SPEED

    missed_shot = replace(static_shot, aim_x=static_shot.aim_x + 1.0)
    assert evaluate_shot(missed_shot, static).outcome == "miss"
    # A static infinite ray crosses the front plate, but a projectile point at
    # twice that radius is not at the plate at impact time.
    beyond_plate = replace(
        static_shot,
        aim_x=2.0 * static_shot.aim_x,
        aim_y=2.0 * static_shot.aim_y,
    )
    assert evaluate_shot(beyond_plate, static).outcome == "miss"

    angled = replace(static, angle=math.radians(45.0))
    alternate = armor_poses(angled)[3]
    alternate_yaw = math.atan2(alternate.y, alternate.x)
    alternate_plan = planner.plan(angled, BULLET_SPEED, aim_yaw_override=alternate_yaw)
    assert alternate_plan.armor_index == alternate.index
    assert abs(
        norm(alternate_plan.planned_aim_x, alternate_plan.planned_aim_y)
        - norm(alternate.x, alternate.y)
    ) < 0.05
    assert abs(
        alternate_plan.flight_time * BULLET_SPEED
        - norm(alternate_plan.planned_aim_x, alternate_plan.planned_aim_y)
    ) < 1e-9

    launch_runner = MovingTargetVisualizer.__new__(MovingTargetVisualizer)
    launch_runner.sim_time = 0.02
    launch_runner.gimbal_yaw = 9.0
    launch_runner._gimbal_step = (0.01, 1.0, 2.0, 4.0)
    launch_runner.shots = [replace(static_shot, launch_at=0.015, impact_at=1.0)]
    launch_runner.resolve_shots()
    expected_launch_yaw = 1.0 + 2.0 * 0.005 + 0.5 * 4.0 * 0.005**2
    assert abs((launch_runner.shots[0].launch_angle or 0.0) - expected_launch_yaw) < 1e-12

    assert static_plan.fire
    assert abs(static_plan.fire_error) < FIRE_THRESHOLD
    fire_runner = MovingTargetVisualizer.__new__(MovingTargetVisualizer)
    fire_runner.mode = "probabilistic"
    fire_runner.current_plan = replace(static_plan, fire=False)
    fire_runner.current_aim_probability = 1.0
    fire_runner.current_fire_probability = 0.0
    fire_runner.probability_for_yaw = lambda _yaw: PROBABILITY_FIRE_THRESHOLD - 1e-6
    fire_runner.sim_time = 1.0
    fire_runner.last_shot_time = -100.0
    fire_runner.gimbal_yaw = static_plan.yaw
    fire_runner.gimbal_yaw_vel = 0.0
    fire_runner.gimbal_yaw_acc = 0.0
    fire_runner.shots = []
    fire_runner.fire_if_ready()
    assert not fire_runner.shots
    fire_runner.probability_for_yaw = lambda _yaw: PROBABILITY_FIRE_THRESHOLD
    fire_runner.fire_if_ready()
    assert len(fire_runner.shots) == 1
    fire_runner.mode = "current"
    fire_runner.sim_time = 2.0
    fire_runner.last_shot_time = -100.0
    fire_runner.shots = []
    fire_runner.fire_if_ready()
    assert not fire_runner.shots
    shifted_plan = planner.plan(static, BULLET_SPEED, aim_yaw_override=static_plan.target_yaw + 0.1)
    assert abs(angle_error(shifted_plan.target_yaw, static_plan.target_yaw + 0.1)) < 1e-9
    assert abs(shifted_plan.yaw_velocity - static_plan.yaw_velocity) < 1e-3
    profiled_plan = planner.plan(
        static,
        BULLET_SPEED,
        aim_yaw_override=static_plan.target_yaw + 0.1,
        probability_phase_profile=True,
    )
    assert abs(angle_error(profiled_plan.target_yaw, static_plan.target_yaw + 0.1)) < 1e-9
    assert abs(profiled_plan.yaw_velocity - static_plan.yaw_velocity) > 1e-3

    probabilistic = ProbabilisticAimModel(0.20)
    sample = probabilistic.add_sample(0.0, 0.0, 2.0, 0.5)
    assert abs(sample.average_acceleration - 5.0) < 1e-12
    probabilistic.add_sample(0.2, 0.5, 2.0, 0.9)
    constrained = probabilistic.constrained_accelerations(OMEGA_LIMIT - 0.01)
    assert all(value <= (OMEGA_LIMIT - (OMEGA_LIMIT - 0.01)) / 0.20 + 1e-12 for value in constrained)
    boundary_model = ProbabilisticAimModel(0.20)
    for acceleration in (-ALPHA_LIMIT, ALPHA_LIMIT, ALPHA_LIMIT):
        boundary_model.add_sample(
            0.0,
            0.0,
            0.0,
            0.5 * acceleration * boundary_model.horizon**2,
        )
    boundary_values = boundary_model.constrained_accelerations(OMEGA_LIMIT - 0.01)
    assert boundary_values == [-ALPHA_LIMIT]
    assert (OMEGA_LIMIT - (OMEGA_LIMIT - 0.01)) / 0.20 not in boundary_values
    # The calibration is defined at the armor half-width for a static target.
    sigma = probabilistic.scatter_sigma
    z = (ARMOR_WIDTH_M / 2.0) / (sigma * math.sqrt(2.0))
    assert abs(math.erf(z) - SCATTER_STATIC_HIT_RATE) < 1e-12
    aim, expected = probabilistic.optimal_aim(0.0, 0.0, [-0.1, 0.0, 0.1], count=32)
    assert aim == 0.0
    assert 0.0 < expected <= 1.0
    field = probabilistic.hit_probability_field(static, [-math.pi / 2, 0.0, math.pi / 2], count=16)
    assert len(field) == 3 and all(0.0 <= value <= 1.0 for value in field)
    deterministic_field = probabilistic.hit_probability_field(
        static,
        [-math.pi / 2, 0.0, math.pi / 2],
        count=512,
        seed=999,
    )
    assert deterministic_field == field
    online_model = ProbabilisticAimModel(0.20)
    online_model.accumulate_sample(0.0, 0.0, 0.0, 0.09, 0.20)
    online_model.accumulate_sample(0.0, 0.0, 0.0, 0.09, 0.20)
    online_entries = online_model._weighted_constrained_entries(0.0)
    assert sum(count for _h, _a, count in online_entries) == 2
    cache_runner = MovingTargetVisualizer.__new__(MovingTargetVisualizer)
    cache_runner.sim_time = 0.01
    cache_runner._probabilistic_cache_time = 0.0
    cache_runner._probabilistic_cache = (0.2, 0.99)
    cache_runner.current_plan = replace(static_plan, target_yaw=1.0)
    cache_runner.gimbal_yaw = 1.0
    cache_runner.gimbal_yaw_vel = 0.0
    evaluated_yaws: list[float] = []
    cache_runner.probability_for_yaw = lambda yaw: evaluated_yaws.append(yaw) or 0.37
    cached_yaw, cached_probability = cache_runner.probabilistic_yaw()
    assert abs(cached_yaw - 1.2) < 1e-12
    assert evaluated_yaws == [cached_yaw]
    assert cached_probability == 0.37

    trajectory_runner = MovingTargetVisualizer.__new__(MovingTargetVisualizer)
    trajectory_runner.gimbal_yaw = 0.0
    trajectory_runner.gimbal_yaw_vel = 0.0
    trajectory_runner.gimbal_yaw_acc = 0.0
    simulated_yaws = trajectory_runner._simulate_plan_gun_yaws(static_plan)
    assert len(simulated_yaws) == FIELD_MPC_TRAJECTORY_TICKS
    assert all(math.isfinite(value) for value in simulated_yaws)
    print("self-test passed")


def benchmark(
    seconds: float = 30.0,
    modes: tuple[str, ...] = ("current", "probabilistic", "field_mpc"),
    seed: int = 20260903,
    distance: float = 6.0,
) -> None:
    """Run the exact GUI dynamics without Tk so displayed metrics are auditable."""
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    for mode in modes:
        random.seed(seed)
        if seed != 20260903:
            print(f"seed={seed}")
        runner = MovingTargetVisualizer.__new__(MovingTargetVisualizer)
        runner.mode = mode
        runner.target_distance = type(
            "V",
            (),
            {"get": lambda self, value=distance: value},
        )()
        runner.acceleration_direction = 1.0
        runner.next_acceleration_switch = EVADE_SWITCH_INTERVAL
        runner.gimbal_yaw = math.pi / 2
        runner.gimbal_yaw_vel = 0.0
        runner.gimbal_yaw_acc = 0.0
        runner.sim_time = 0.0
        runner.last_shot_time = -100.0
        runner.valid_hits = runner.low_speed_hits = runner.misses = 0
        runner.is_rollout_cpp = mode == "rollout_cpp"
        runner.planner = None if runner.is_rollout_cpp else TinyMpcPlanner()
        runner.rollout_planner = (
            CppRolloutPlanner() if runner.is_rollout_cpp else None
        )
        runner._rollout_command_alpha = 0.0
        runner._rollout_decision_time = -1.0
        runner._rollout_path_score = 0.0
        runner.target = TargetState(
            0.0,
            distance,
            0.0,
            0.0,
            0.0,
            INITIAL_OMEGA,
            ALPHA_LIMIT,
        )
        runner._causal_observations = [(0.0, runner.target.omega)]
        runner._causal_accel_history = []
        runner.motion_history = []
        runner._history_pending = []
        runner.probabilistic_model = ProbabilisticAimModel(
            LOW_SPEED_DELAY + distance / BULLET_SPEED
        )
        runner._probabilistic_cache_time = -1.0
        runner._probabilistic_cache = (0.0, 0.0)
        runner._probabilistic_last_phase = None
        runner._hit_curve_cache_time = -1.0
        runner._hit_curve_grid = []
        runner._hit_curve_values = []
        runner._hit_curve_anchor_nominal_yaw = 0.0
        runner._current_nominal_yaw = 0.0
        runner.current_aim_probability = 0.0
        runner.current_fire_probability = 0.0
        runner.current_plan = (
            runner._rollout_nominal_plan()
            if runner.is_rollout_cpp
            else runner.planner.plan(replace(runner.target), BULLET_SPEED)
        )
        runner._probabilistic_peak_yaw = runner.current_plan.target_yaw
        runner._probabilistic_pmax = 0.0
        runner._probabilistic_candidate_probability = 0.0
        runner.diagnostic_log = []
        runner.resolved_diagnostics = []
        runner.gate_block_count = 0
        runner.accel_saturated_steps = 0
        runner._probabilistic_reachable_peaks = []
        runner._diag_current = None
        runner.shots = []
        steps = int(seconds / PLANNER_DT)
        for _ in range(steps):
            runner.sim_time += PLANNER_DT
            runner.advance_actual(PLANNER_DT)
            runner.update_planner()
            runner.advance_gimbal(PLANNER_DT)
            runner.fire_if_ready()
            runner.resolve_shots()
        resolved = runner.valid_hits + runner.low_speed_hits + runner.misses
        rate = runner.valid_hits / resolved if resolved else 0.0
        print(f"{mode}: shots={resolved}, valid={runner.valid_hits}, low={runner.low_speed_hits}, miss={runner.misses}, valid_rate={rate*100:.2f}%, valid_per_s={runner.valid_hits/seconds:.3f}")
        if mode in ("probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control") and runner.diagnostic_log:
            rows = runner.diagnostic_log
            shot_rows = runner.resolved_diagnostics
            if distance == 6.0:
                log_name = (
                    f"moving_target_diagnostic_{mode}.csv"
                    if seed == 20260903
                    else f"moving_target_diagnostic_{mode}_{seed}.csv"
                )
            else:
                distance_suffix = f"{distance:g}m"
                log_name = (
                    f"moving_target_diagnostic_{mode}_{distance_suffix}.csv"
                    if seed == 20260903
                    else (
                        f"moving_target_diagnostic_{mode}_"
                        f"{distance_suffix}_{seed}.csv"
                    )
                )
            log_path = Path(log_name)
            with log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "time", "pmax", "pref", "pcandidate", "pplan", "pgun", "pactual", "peak_yaw",
                    "plan_yaw", "gun_yaw", "gate_block", "actual_result",
                    "ref_plan_launch_mrad", "pred_actual_launch_mrad", "accel_saturated",
                ))
                writer.writeheader()
                writer.writerows(rows)
                # Resolved-shot rows carry the same field names plus the
                # actual launch probability/result; they are emitted after
                # control-step rows for unambiguous post-processing.
                writer.writerows(shot_rows)
            def mean(key: str, source=rows) -> float:
                values = [float(row[key]) for row in source if row.get(key) is not None]
                return sum(values) / len(values) if values else 0.0
            # Pmax/Pref are control-step diagnostics.  Pgun/Pactual must be
            # compared only on the same resolved-shot rows; otherwise the
            # means mix thousands of gated cycles with a few fired shots.
            shot_rows = [row for row in shot_rows if row.get("pactual") is not None and row.get("actual_result") is not None]
            pmax_pref = mean("pmax", shot_rows) - mean("pref", shot_rows)
            pref_pgun = mean("pref", shot_rows) - mean("pgun", shot_rows)
            pgun_actual = mean("pgun", shot_rows) - mean("pactual", shot_rows)
            blocked_rows = [row for row in rows if row.get("gate_block")]
            high = lambda row, key: row.get(key) is not None and float(row[key]) >= PROBABILITY_FIRE_THRESHOLD
            blocked_type3 = sum(1 for r in blocked_rows if not high(r, "pcandidate"))
            blocked_type1 = sum(1 for r in blocked_rows if high(r, "pcandidate") and not high(r, "pplan"))
            blocked_type2 = sum(1 for r in blocked_rows if high(r, "pcandidate") and high(r, "pplan") and not high(r, "pgun"))
            blocked_type4 = sum(1 for r in blocked_rows if high(r, "pcandidate") and high(r, "pplan") and high(r, "pgun"))
            angle_rows = [row for row in shot_rows if row.get("ref_plan_launch_mrad") is not None]
            pred_angle_rows = [row for row in shot_rows if row.get("pred_actual_launch_mrad") is not None]
            abs_ref_plan = [abs(float(row["ref_plan_launch_mrad"])) for row in angle_rows]
            abs_pred_actual = [abs(float(row["pred_actual_launch_mrad"])) for row in pred_angle_rows]
            sat_steps = sum(bool(row.get("accel_saturated")) for row in rows)
            print(
                "diagnostic: "
                f"all_steps: mean(Pcandidate)={mean('pcandidate'):.3f}, mean(Pplan)={mean('pplan'):.3f}, mean(Pgun)={mean('pgun'):.3f}; "
                f"shots: mean(Pmax)={mean('pmax', shot_rows):.3f}, mean(Pref)={mean('pref', shot_rows):.3f}, "
                f"mean(Pgun)={mean('pgun', shot_rows):.3f}, mean(Pactual)={mean('pactual', shot_rows):.3f}, "
                f"mean(Pmax-Pref)={pmax_pref:.3f}, mean(Pref-Pgun)={pref_pgun:.3f}, "
                f"mean(Pgun-actual)={pgun_actual:.3f}, gate_block={runner.gate_block_count}"
            )
            if abs_ref_plan:
                print(f"angles: |ref-plan| mean={sum(abs_ref_plan)/len(abs_ref_plan):.2f}mrad, p95={sorted(abs_ref_plan)[int(.95*(len(abs_ref_plan)-1))]:.2f}mrad")
            if abs_pred_actual:
                print(f"angles: |pred-actual| mean={sum(abs_pred_actual)/len(abs_pred_actual):.2f}mrad, p95={sorted(abs_pred_actual)[int(.95*(len(abs_pred_actual)-1))]:.2f}mrad")
            print(f"accel_saturation={sat_steps}/{len(rows)} ({100.0*sat_steps/max(1,len(rows)):.2f}%)")
            print(f"resolved_shot_rows={len(shot_rows)}")
            print(f"blocked_rows={len(blocked_rows)}; exclusive: candidate_low={blocked_type3}, candidate_high_plan_low={blocked_type1}, plan_high_gun_low={blocked_type2}, all_high_but_blocked={blocked_type4}; total={blocked_type1+blocked_type2+blocked_type3+blocked_type4}")
            print(f"diagnostic_log={log_path.resolve()}")


def probability_alignment_diagnostic() -> None:
    """Check the historical acceleration-to-absolute-angle mapping in isolation."""
    horizon = LOW_SPEED_DELAY + 6.0 / BULLET_SPEED
    position = 0.0
    velocity = math.radians(300.0)
    alpha_neg = math.radians(-280.0)
    alpha_pos = math.radians(280.0)
    cv = position + velocity * horizon
    expected_neg = cv + 0.5 * alpha_neg * horizon * horizon
    expected_pos = cv + 0.5 * alpha_pos * horizon * horizon

    model = ProbabilisticAimModel(horizon)
    # Synthetic history whose inferred accelerations are exactly the two
    # physically realizable branches.  The 96/4 weights mirror the reported
    # learned distribution and make the expected peak unambiguous.
    for index in range(96):
        model.add_sample(index * PLANNER_DT, position, velocity,
                         cv + 0.5 * alpha_neg * horizon * horizon, horizon)
    for index in range(4):
        model.add_sample((96 + index) * PLANNER_DT, position, velocity,
                         cv + 0.5 * alpha_pos * horizon * horizon, horizon)
    samples = model.future_position_samples(position, velocity, seed=7, count=10000)
    mean = sum(samples) / len(samples)
    # Evaluate only the two branch locations; this is a direct probability
    # mass check, independent of gimbal reachability or TinyMPC timing.
    p_neg = sum(abs(value - expected_neg) < math.radians(0.01) for value in samples) / len(samples)
    p_pos = sum(abs(value - expected_pos) < math.radians(0.01) for value in samples) / len(samples)
    print("probability alignment diagnostic")
    print(f"horizon={horizon*1000:.1f} ms")
    print(f"constant_velocity={math.degrees(cv):.6f} deg")
    print(f"negative_branch={math.degrees(expected_neg):.6f} deg (relative {math.degrees(expected_neg-cv):+.6f} deg, mass {p_neg:.4f})")
    print(f"positive_branch={math.degrees(expected_pos):.6f} deg (relative {math.degrees(expected_pos-cv):+.6f} deg, mass {p_pos:.4f})")
    print(f"weighted_mean={math.degrees(mean):.6f} deg (error to negative branch {math.degrees(mean-expected_neg):+.6f} deg)")
    assert abs(math.degrees(expected_neg) - 67.3965) < 0.02
    assert p_neg > 0.94 and p_pos < 0.08


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("current", "probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control", "rollout_cpp"), default="current")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="run fixed-state probability alignment diagnostic")
    parser.add_argument("--distance", type=float, default=6.0)
    arguments = parser.parse_args()
    if arguments.diagnostic:
        probability_alignment_diagnostic()
    elif arguments.benchmark:
        benchmark(distance=arguments.distance)
    elif arguments.self_test:
        self_test()
    else:
        MovingTargetVisualizer(arguments.mode, arguments.distance).mainloop()
