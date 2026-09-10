"""2D moving-target visualizer based on sp_vision_25 target prediction.

The target predictor follows TongjiSuperPower/sp_vision_25's Target model:
the center translates at the currently estimated velocity, and armor angle
advances at the currently estimated angular velocity. Future target
acceleration is not part of the state. Vertical motion, pitch, and projectile
drop are intentionally omitted.
"""

from __future__ import annotations

import argparse
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
# --- True receding-horizon rollout (mode "rollout_mpc") ---------------------
# Unlike FIELD_MPC_*, these govern a closed-loop rollout that re-solves TinyMPC
# at every simulated tick, propagates the target under sampled accelerations,
# and re-selects the probability branch with the same policy used online.
ROLLOUT_TICKS = 6                  # simulated control steps per rollout
ROLLOUT_LAUNCH_TICKS = 9           # planner steps needed to resolve 15 ms delays
ROLLOUT_TOTAL_TICKS = 37           # enough to settle flight-time windows
ROLLOUT_SCENARIOS = 3              # quantile representatives of the empirical accel law
ROLLOUT_MAX_CANDIDATES = 4         # first-move branches compared per decision
ROLLOUT_DECISION_INTERVAL = PLANNER_DT  # replan cadence; raise to trade CPU for latency
ROLLOUT_DISCOUNT = 0.92            # per-tick discount on expected hits
ROLLOUT_READINESS_WEIGHT = 0.15    # tie-breaker on mean gate probability
ROLLOUT_SWITCH_PENALTY = 0.50      # expected-hit units per radian of branch change
ROLLOUT_MIN_GAIN = 0.02
ROLLOUT_SAMPLE_LIMIT = 48          # strided empirical samples used inside rollout
ROLLOUT_POLICY_STEP = math.radians(1.5)
ROLLOUT_POLICY_SPAN = 4            # local re-selection scan half-width, in policy steps
# Flight time varies with the selected plate radius, so a rollout tick may sit a
# few milliseconds away from the horizon at which the history samples were
# recorded. This is an explicit, bounded approximation used only during rollout.
ROLLOUT_HORIZON_TOLERANCE = 0.03
ROLLOUT_SCATTER_POINTS = (
    (-1.5 * SCATTER_SIGMA, 0.0668),
    (-0.5 * SCATTER_SIGMA, 0.2417),
    (0.0, 0.3829),
    (0.5 * SCATTER_SIGMA, 0.2417),
    (1.5 * SCATTER_SIGMA, 0.0668),
)
MODE_LABELS = {
    "current": "当前版：实际云台跟踪",
    "probabilistic": "加速度分布预测版",
    "field_candidate": "执行感知候选评分版",
    "field_mpc": "场域MPC版",
    "window_candidate": "连续窗口候选版",
    "lookahead_control": "发射前瞻控制版",
    "rollout_mpc": "滚动重规划版",
}
PROBABILITY_FIELD_INTERVAL = 0.05
MOTION_HISTORY_WINDOW = 160
PROBABILITY_PEAK_SWITCH_MARGIN = 0.035
PROBABILITY_CONTINUITY_WEIGHT = 0.035
PROBABILITY_YAW_SMOOTHING = 0.72


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
        # 90% of static shots fall inside one armor half-width at 6 m.
        self.scatter_sigma = scatter_sigma or SCATTER_SIGMA
        # Samples are accepted when their recording horizon is within this
        # tolerance of the current horizon. Rollout widens it temporarily.
        self.horizon_tolerance = PLANNER_DT

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
            if abs(sample.horizon - self.horizon) <= self.horizon_tolerance
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
        sample_limit: int | None = None,
    ) -> list[float]:
        """Evaluate the exact empirical multimodal yaw hit field."""
        # ``target`` is already evaluated at the predicted impact time by the
        # planner. Add only the acceleration-model uncertainty over this same
        # horizon; do not propagate the nominal state a second time.
        samples = self.constrained_samples(target.omega)
        if not samples:
            return [0.0 for _ in yaw_grid]
        # Use every retained history sample exactly once. Repeated values keep
        # their empirical frequency, so this is the normalized empirical
        # expectation without bootstrap noise. `count` and `seed` remain in
        # the signature for compatibility with diagnostic callers.
        if sample_limit is not None and len(samples) > sample_limit > 0:
            # Deterministic stride keeps the empirical shape (including both
            # bang-bang branches) while bounding rollout cost. It is a uniform
            # subsample, not a re-weighting.
            stride = len(samples) / float(sample_limit)
            samples = [samples[int(index * stride)] for index in range(sample_limit)]
        angle_samples = [
            target.angle + 0.5 * sample.average_acceleration * self.horizon**2
            for sample in samples
        ]
        radius = max(norm(target.x, target.y), 1e-6)
        # Armor geometry depends only on each sampled target angle.  Build it
        # once per sample; rebuilding it inside the yaw scan multiplied the
        # trigonometric work by the number of grid points.
        sample_armors = [armor_poses(replace(target, angle=angle)) for angle in angle_samples]
        values: list[float] = []
        for yaw in yaw_grid:
            total = 0.0
            for armors in sample_armors:
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
                total += best
            values.append(total / len(angle_samples))
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


@dataclass
class PendingRolloutShot:
    launch_at: float
    impact_at: float
    planned_range: float
    discount: float
    launch_angle: float | None = None
    aim_x: float | None = None
    aim_y: float | None = None


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
        for _ in range(12):
            impact_state = predict_constant_velocity(delayed, flight_time)
            armor = armor_poses(impact_state)[armor_index]
            updated = norm(armor.x, armor.y) / bullet_speed
            if abs(updated - flight_time) < 1e-12:
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
    for armor in armor_poses(actual):
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
    return max(covered, key=lambda result: result.normal_speed or 0.0)


def shot_hits(shot: Shot, actual: TargetState) -> bool:
    return evaluate_shot(shot, actual).outcome == "valid_hit"


class MovingTargetVisualizer(tk.Tk):
    def __init__(self, mode: str = "current") -> None:
        super().__init__()
        random.seed(20260903)
        self.mode = mode
        mode_label = MODE_LABELS[mode]
        self.title(f"sp_vision_25 二维自瞄规划可视化 - {mode_label}")
        self.geometry("1180x820")
        self.minsize(820, 650)

        self.target_distance = tk.DoubleVar(value=6.0)
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
        self.planner = TinyMpcPlanner()
        self.target = self.initial_target()
        self.motion_history: list[tuple[float, float, float]] = []
        self.probabilistic_model = ProbabilisticAimModel(
            LOW_SPEED_DELAY + self.target_distance.get() / BULLET_SPEED
        )
        self._probabilistic_cache_time = -1.0
        self._probabilistic_cache: tuple[float, float] = (0.0, 0.0)
        self._probabilistic_last_phase: float | None = None
        self._probabilistic_reachable_peaks: list[tuple[float, float]] = []
        self._rollout_decision_time = -1.0
        self._rollout_phase: float | None = None
        self._rollout_score = 0.0
        # Diagnostic only: this is the probability at the planner's reference
        # peak. Never use it as a fire gate; shots leave along self.gimbal_yaw.
        self.current_aim_probability = 0.0
        self.current_fire_probability = 0.0
        self.current_plan = self.planner.plan(replace(self.target), BULLET_SPEED)
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
        mode_label = MODE_LABELS[self.mode]
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
            text="闪避：±280°/s²，每 1.50 s 反向",
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
            self.probabilistic_model.samples.clear()
            self._probabilistic_last_phase = None
            self._probabilistic_reachable_peaks = []
            self._rollout_decision_time = -1.0
            self._rollout_phase = None
            self._rollout_score = 0.0
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
        self.target = self.initial_target()
        self.motion_history.clear()
        self.probabilistic_model.samples.clear()
        self._probabilistic_cache_time = -1.0
        self._probabilistic_last_phase = None
        self._probabilistic_reachable_peaks = []
        self._rollout_decision_time = -1.0
        self._rollout_phase = None
        self._rollout_score = 0.0
        self.current_aim_probability = 0.0
        self.current_fire_probability = 0.0
        self.current_plan = self.planner.plan(replace(self.target), BULLET_SPEED)
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

    def update_planner(self) -> None:
        base_plan = self.planner.plan(replace(self.target), BULLET_SPEED)
        self.current_plan = base_plan
        if self.mode in ("probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control", "rollout_mpc"):
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
            if self.mode in ("field_candidate", "window_candidate"):
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
                    values = self.probabilistic_model.hit_probability_field(
                        plan.predicted_state, [plan.planned_yaw[i] for i in indices],
                        count=64, seed=int(cache_time * 100),
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
                    raw_score = window_score(candidate_plan) if self.mode == "window_candidate" else self.probabilistic_model.hit_probability_field(
                        candidate_plan.predicted_state, [predicted_launch_yaw], count=64, seed=int(cache_time * 100)
                    )[0]
                    score = raw_score - FIELD_CANDIDATE_SWITCH_PENALTY * abs(
                        angle_error(candidate_yaw, selected)
                    )
                    if score > best_score + FIELD_CANDIDATE_MIN_GAIN:
                        best_score, best_plan = score, candidate_plan
                self.current_plan = best_plan
                self.probabilistic_model.horizon = best_plan.delay + best_plan.flight_time
                self.current_aim_probability = self.probability_for_yaw(best_plan.target_yaw)
            elif self.mode == "field_mpc":
                self._select_field_mpc(base_plan)
            elif self.mode == "rollout_mpc":
                self._select_rollout_mpc(base_plan)
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

    # ------------------------------------------------------------------
    # True receding-horizon rollout ("rollout_mpc")
    # ------------------------------------------------------------------
    # Difference from ``field_mpc``: that mode tracks one frozen TinyMPC
    # trajectory against one frozen impact state. Here every simulated tick
    # re-solves TinyMPC from the rolled-forward target state, the target is
    # propagated under sampled accelerations, the probability branch is
    # re-selected by the same policy used online, and the objective is the
    # discounted expected number of valid hits under the real fire gate and
    # rate limit. Only the first move of the winning rollout is applied.
    def _rollout_scenarios(self) -> list[tuple[float, float]]:
        """Weighted empirical acceleration branches, not equal quantiles."""
        samples = self.probabilistic_model.constrained_samples(self.target.omega)
        if not samples:
            return [(self.target.alpha, 1.0)]
        negative = [
            sample.average_acceleration
            for sample in samples
            if sample.average_acceleration < 0.0
        ]
        positive = [
            sample.average_acceleration
            for sample in samples
            if sample.average_acceleration > 0.0
        ]
        zero = [
            sample.average_acceleration
            for sample in samples
            if abs(sample.average_acceleration) <= 1e-9
        ]
        groups = (negative, zero, positive)
        scenarios = []
        total = len(samples)
        for group in groups:
            if group:
                scenarios.append((sum(group) / len(group), len(group) / total))
        return scenarios

    @staticmethod
    def _advance_scenario_target(
        state: TargetState,
        acceleration: float,
        dt: float,
    ) -> TargetState:
        """Propagate the rollout target under one sampled average acceleration."""
        omega = clamp(state.omega + acceleration * dt, -OMEGA_LIMIT, OMEGA_LIMIT)
        # Integrate the realized average rate so the angle stays consistent
        # with the clamped velocity instead of an unreachable acceleration.
        average_omega = 0.5 * (state.omega + omega)
        return replace(
            state,
            angle=state.angle + average_omega * dt,
            omega=omega,
            alpha=acceleration,
        )

    @staticmethod
    def _step_gimbal_open(
        yaw: float,
        velocity: float,
        plan: PlannerOutput,
        dt: float,
    ) -> tuple[float, float, float]:
        """One control tick under exactly the same law as advance_gimbal."""
        command_acc = clamp(
            plan.yaw_acceleration
            + 35.0 * angle_error(plan.yaw, yaw)
            + 10.0 * (plan.yaw_velocity - velocity),
            -GIMBAL_MAX_ACCEL,
            GIMBAL_MAX_ACCEL,
        )
        next_yaw = yaw + velocity * dt + 0.5 * command_acc * dt * dt
        return next_yaw, velocity + command_acc * dt, command_acc

    def _rollout_once(
        self,
        first_yaw: float,
        scenario_acceleration: float,
        seed: int,
    ) -> tuple[float, float]:
        """Roll out approvals and settle them against this scenario's actual state."""
        model = self.probabilistic_model
        yaw = self.gimbal_yaw
        velocity = self.gimbal_yaw_vel
        acceleration = self.gimbal_yaw_acc
        state = replace(self.target)
        aim_yaw = first_yaw
        clock = self.sim_time
        last_shot = self.last_shot_time
        expected_hits = 0.0
        gate_total = 0.0
        pending: list[PendingRolloutShot] = []
        shot_discount = 1.0
        zero_scenario = abs(scenario_acceleration) <= 1e-9
        direction = (
            0.0
            if zero_scenario
            else (1.0 if scenario_acceleration > 0.0 else -1.0)
        )
        next_switch = self.__dict__.get(
            "next_acceleration_switch", EVADE_SWITCH_INTERVAL
        )
        plan: PlannerOutput | None = None

        for tick in range(ROLLOUT_TOTAL_TICKS):
            if plan is None or tick < ROLLOUT_LAUNCH_TICKS:
                plan = self.planner.plan(
                    replace(state),
                    BULLET_SPEED,
                    aim_yaw_override=aim_yaw,
                    probability_phase_profile=True,
                )
                model.horizon = plan.delay + plan.flight_time

            if tick < ROLLOUT_TICKS:
                launch_yaw = (
                    yaw
                    + velocity * plan.delay
                    + 0.5 * acceleration * plan.delay * plan.delay
                )
                reach_lower, reach_upper = reachable_yaw_delta_bounds(
                    velocity, max(model.horizon, PLANNER_DT)
                )
                grid = [launch_yaw]
                for offset in range(-ROLLOUT_POLICY_SPAN, ROLLOUT_POLICY_SPAN + 1):
                    candidate = aim_yaw + offset * ROLLOUT_POLICY_STEP
                    if reach_lower <= angle_error(candidate, yaw) <= reach_upper:
                        grid.append(candidate)
                values = model.hit_probability_field(
                    plan.predicted_state,
                    grid,
                    count=64,
                    seed=seed,
                    sample_limit=ROLLOUT_SAMPLE_LIMIT,
                )
                gate_probability = values[0]
                gate_total += gate_probability
                if (
                    gate_probability >= PROBABILITY_FIRE_THRESHOLD
                    and clock - last_shot >= SHOT_INTERVAL - 1e-12
                ):
                    pending.append(
                        PendingRolloutShot(
                            launch_at=clock + plan.delay,
                            impact_at=clock + plan.delay + plan.flight_time,
                            planned_range=norm(
                                plan.planned_aim_x,
                                plan.planned_aim_y,
                            ),
                            discount=shot_discount,
                        )
                    )
                    shot_discount *= ROLLOUT_DISCOUNT
                    last_shot = clock
                if len(values) > 1:
                    best = max(range(1, len(values)), key=values.__getitem__)
                    aim_yaw = grid[best]

            start_yaw = yaw
            start_velocity = velocity
            yaw, velocity, acceleration = self._step_gimbal_open(
                yaw, velocity, plan, PLANNER_DT
            )
            target_start = replace(state)
            direction_start = direction
            next_switch_start = next_switch
            if zero_scenario:
                state = predict_constant_velocity(state, PLANNER_DT)
            else:
                target_angle, target_omega, direction, target_alpha, next_switch = (
                    advance_optimized_evasion(
                        state.angle,
                        state.omega,
                        direction,
                        clock,
                        next_switch,
                        PLANNER_DT,
                    )
                )
                state = replace(
                    state,
                    angle=target_angle,
                    omega=target_omega,
                    alpha=target_alpha,
                )

            for shot in pending:
                if (
                    shot.launch_angle is None
                    and clock <= shot.launch_at <= clock + PLANNER_DT
                ):
                    tau = shot.launch_at - clock
                    actual_yaw = (
                        start_yaw
                        + start_velocity * tau
                        + 0.5 * acceleration * tau * tau
                    )
                    shot.launch_angle = actual_yaw
                    shot.aim_x = shot.planned_range * math.cos(actual_yaw)
                    shot.aim_y = shot.planned_range * math.sin(actual_yaw)

            remaining_after_step: list[PendingRolloutShot] = []
            for shot in pending:
                if shot.impact_at > clock + PLANNER_DT + 1e-12:
                    remaining_after_step.append(shot)
                    continue
                if shot.launch_angle is None:
                    remaining_after_step.append(shot)
                    continue
                tau = shot.impact_at - clock
                if zero_scenario:
                    impact_state = predict_constant_velocity(target_start, tau)
                else:
                    impact_angle, impact_omega, _d, impact_alpha, _ns = (
                        advance_optimized_evasion(
                            target_start.angle,
                            target_start.omega,
                            direction_start,
                            clock,
                            next_switch_start,
                            tau,
                        )
                    )
                    impact_state = replace(
                        target_start,
                        angle=impact_angle,
                        omega=impact_omega,
                        alpha=impact_alpha,
                    )
                expected_hits += shot.discount * self._rollout_scatter_probability(
                    shot, impact_state
                )
            pending = remaining_after_step

            clock += PLANNER_DT

        # Bounded horizon: any remaining shot was already launched and has a
        # real trajectory, so settle it at the horizon state rather than using
        # the probability model as a backstop.
        for shot in pending:
            if shot.launch_angle is not None:
                expected_hits += shot.discount * self._rollout_scatter_probability(
                    shot, state
                )
        return expected_hits, gate_total / max(1, ROLLOUT_TICKS)

    @staticmethod
    def _rollout_scatter_probability(
        shot: PendingRolloutShot,
        impact_state: TargetState,
    ) -> float:
        """Integrate finite scatter offsets against this scenario's impact pose."""
        if shot.launch_angle is None or shot.aim_x is None or shot.aim_y is None:
            return 0.0
        probability = 0.0
        for lateral_offset, weight in ROLLOUT_SCATTER_POINTS:
            aim_x = shot.planned_range * math.cos(shot.launch_angle)
            aim_y = shot.planned_range * math.sin(shot.launch_angle)
            aim_x -= lateral_offset * math.sin(shot.launch_angle)
            aim_y += lateral_offset * math.cos(shot.launch_angle)
            rollout_shot = Shot(
                created_at=0.0,
                launch_at=shot.launch_at,
                impact_at=shot.impact_at,
                launch_x=0.0,
                launch_y=0.0,
                aim_x=aim_x,
                aim_y=aim_y,
                planned_aim_x=shot.aim_x,
                planned_aim_y=shot.aim_y,
                predicted_state=impact_state,
                armor_index=0,
                launched=True,
                launch_angle=shot.launch_angle,
            )
            if evaluate_shot(rollout_shot, impact_state).outcome == "valid_hit":
                probability += weight
        return probability

    def _score_rollout(
        self,
        candidate_yaw: float,
        scenarios: list[tuple[float, float]],
        seed: int,
    ) -> float:
        hits = 0.0
        readiness = 0.0
        for acceleration, weight in scenarios:
            expected, ready = self._rollout_once(candidate_yaw, acceleration, seed)
            hits += weight * expected
            readiness += weight * ready
        return hits + ROLLOUT_READINESS_WEIGHT * readiness

    def _rollout_candidate_yaws(self, base_plan: PlannerOutput) -> list[float]:
        """First moves to compare: incumbent, nominal, and real field branches."""
        yaws: list[float] = []

        def add(yaw: float) -> None:
            if all(abs(angle_error(yaw, seen)) > math.radians(1.0) for seen in yaws):
                yaws.append(yaw)

        add(base_plan.target_yaw)
        for peak_yaw, _probability in self.__dict__.get(
            "_probabilistic_reachable_peaks", []
        ):
            if len(yaws) >= ROLLOUT_MAX_CANDIDATES:
                break
            add(peak_yaw)
        return yaws[:ROLLOUT_MAX_CANDIDATES]

    def _select_rollout_mpc(self, base_plan: PlannerOutput) -> None:
        model = self.probabilistic_model
        decision_time = (
            math.floor(self.sim_time / ROLLOUT_DECISION_INTERVAL)
            * ROLLOUT_DECISION_INTERVAL
        )
        incumbent = self.current_plan.target_yaw
        held_phase = self.__dict__.get("_rollout_phase")
        if (
            self.__dict__.get("_rollout_decision_time") == decision_time
            and held_phase is not None
        ):
            # Decision held between rollouts: hold the phase relative to the
            # nominal yaw, not an absolute yaw that ages as the target turns.
            chosen = base_plan.target_yaw + held_phase
        else:
            saved_horizon = model.horizon
            saved_tolerance = model.horizon_tolerance
            model.horizon_tolerance = ROLLOUT_HORIZON_TOLERANCE
            seed = int(decision_time * 100)
            try:
                scenarios = self._rollout_scenarios()
                chosen = incumbent
                best_score = self._score_rollout(incumbent, scenarios, seed)
                for candidate_yaw in self._rollout_candidate_yaws(base_plan):
                    if abs(angle_error(candidate_yaw, incumbent)) < math.radians(0.5):
                        continue
                    score = self._score_rollout(
                        candidate_yaw, scenarios, seed
                    ) - ROLLOUT_SWITCH_PENALTY * abs(
                        angle_error(candidate_yaw, incumbent)
                    )
                    if score > best_score + ROLLOUT_MIN_GAIN:
                        best_score = score
                        chosen = candidate_yaw
            finally:
                model.horizon_tolerance = saved_tolerance
                model.horizon = saved_horizon
            self._rollout_decision_time = decision_time
            self._rollout_phase = angle_error(chosen, base_plan.target_yaw)
            self._rollout_score = best_score
        # Receding horizon: discard the rest of the rollout and re-solve the
        # applied plan from the true current state around the winning branch.
        self.current_plan = self.planner.plan(
            replace(self.target),
            BULLET_SPEED,
            aim_yaw_override=chosen,
            probability_phase_profile=True,
        )
        model.horizon = self.current_plan.delay + self.current_plan.flight_time

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
        # Build historical acceleration samples for the same absolute impact
        # horizon used by the current plan, not for the target-center range.
        self.probabilistic_model.horizon = (
            self.current_plan.delay + self.current_plan.flight_time
        )
        if len(self.motion_history) >= 2:
            # Rebuild the short rolling window instead of appending the same
            # observations on every cache refresh.  This keeps the empirical
            # distribution stationary in size and avoids quadratic growth.
            self.probabilistic_model.samples.clear()
            self.probabilistic_model.add_trajectory(
                self.motion_history[-MOTION_HISTORY_WINDOW:]
            )
        predicted = self.current_plan.predicted_state
        # Coarse global scan plus a narrow local refinement gives nearly the
        # same peak location at a fraction of the Python field-evaluation cost.
        grid = [self.gimbal_yaw + math.radians(1.5 * index) for index in range(-120, 121)]
        field = self.probabilistic_model.hit_probability_field(
            predicted, grid, count=64, seed=int(cache_time * 100)
        )
        # Reject peaks that cannot be reached before impact under the gimbal
        # acceleration limit. This is the one-step reachable-set approximation
        # used by chance-constrained MPC outer loops.
        reach_time = max(self.current_plan.delay + self.current_plan.flight_time, PLANNER_DT)
        reach_lower, reach_upper = reachable_yaw_delta_bounds(
            self.gimbal_yaw_vel,
            reach_time,
        )
        reachable = [
            index for index, yaw in enumerate(grid)
            if reach_lower <= yaw - self.gimbal_yaw <= reach_upper
        ]
        self._probabilistic_pmax = max((field[i] for i in reachable), default=0.0)
        self._probabilistic_peak_yaw = grid[max(reachable, key=lambda i: field[i])] if reachable else nominal_yaw
        # Keep several local maxima as branches.  A small continuity cost
        # prevents stochastic resampling from jumping between opposite armor
        # peaks unless the new branch is materially better.
        candidates = [i for i in reachable if i == 0 or i == len(field) - 1 or
                      field[i] >= field[i - 1] and field[i] >= field[i + 1]]
        if not candidates:
            candidates = reachable or list(range(len(grid)))
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
        if 0 < best < len(grid) - 1:
            step = math.radians(1.5)
            fine_grid = [
                yaw
                for offset in range(-4, 5)
                if reach_lower
                <= (yaw := selected_yaw + step * offset / 4.0) - self.gimbal_yaw
                <= reach_upper
            ]
            fine_field = self.probabilistic_model.hit_probability_field(
                predicted, fine_grid, count=64, seed=int(cache_time * 100)
            )
            fine_index = max(range(len(fine_grid)), key=fine_field.__getitem__)
            selected_yaw = fine_grid[fine_index]
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
        selected_probability = self.probabilistic_model.hit_probability_field(
            predicted,
            [nominal_yaw + selected_phase],
            count=64,
            seed=int(cache_time * 100),
        )[0]
        self._probabilistic_last_phase = selected_phase
        self._probabilistic_cache = (selected_phase, selected_probability)
        self._probabilistic_cache_time = cache_time
        return nominal_yaw + selected_phase, selected_probability

    def probability_for_yaw(self, yaw: float) -> float:
        """Probability that a shot along this actual yaw hits at impact time."""
        cache_time = (
            math.floor(self.sim_time / PROBABILITY_FIELD_INTERVAL)
            * PROBABILITY_FIELD_INTERVAL
        )
        return self.probabilistic_model.hit_probability_field(
            self.current_plan.predicted_state,
            [yaw],
            count=64,
            seed=int(cache_time * 100),
        )[0]

    def fire_if_ready(self) -> None:
        launch_delay = self.current_plan.delay
        predicted_launch_yaw = (
            self.gimbal_yaw
            + self.gimbal_yaw_vel * launch_delay
            + 0.5 * self.gimbal_yaw_acc * launch_delay * launch_delay
        )
        if self.mode in ("probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control", "rollout_mpc"):
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
        if self.sim_time - self.last_shot_time < SHOT_INTERVAL:
            return
        shot = shot_from_plan(
            self.current_plan,
            self.sim_time,
            predicted_launch_yaw,
            scatter=True,
        )
        if self.mode in ("probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control", "rollout_mpc"):
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
                        shot.launch_angle = (
                            start_yaw
                            + start_velocity * tau
                            + 0.5 * acceleration * tau * tau
                        )
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
        self.motion_history.append((self.sim_time, self.target.angle, self.target.omega))
        del self.motion_history[:-MOTION_HISTORY_WINDOW]

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
            "rollout_mpc",
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

    # --- receding-horizon rollout ------------------------------------------
    limited = probabilistic.hit_probability_field(
        static, [0.0], count=16, sample_limit=1
    )
    assert 0.0 <= limited[0] <= 1.0

    clamped = MovingTargetVisualizer._advance_scenario_target(
        TargetState(0.0, 6.0, 0.0, 0.0, 0.0, OMEGA_LIMIT, 0.0),
        ALPHA_LIMIT,
        PLANNER_DT,
    )
    assert abs(clamped.omega - OMEGA_LIMIT) < 1e-12
    assert abs(clamped.angle - OMEGA_LIMIT * PLANNER_DT) < 1e-12

    # The rollout must step the gimbal with exactly the executed control law.
    step_runner = MovingTargetVisualizer.__new__(MovingTargetVisualizer)
    step_runner.mode = "current"
    step_runner.sim_time = 0.0
    step_runner.current_plan = static_plan
    step_runner.gimbal_yaw = 1.0
    step_runner.gimbal_yaw_vel = 0.3
    step_runner.gimbal_yaw_acc = 0.0
    step_runner.advance_gimbal(PLANNER_DT)
    open_yaw, open_velocity, _open_acc = MovingTargetVisualizer._step_gimbal_open(
        1.0, 0.3, static_plan, PLANNER_DT
    )
    assert abs(step_runner.gimbal_yaw - open_yaw) < 1e-12
    assert abs(step_runner.gimbal_yaw_vel - open_velocity) < 1e-12

    rollout_runner = MovingTargetVisualizer.__new__(MovingTargetVisualizer)
    rollout_runner.mode = "rollout_mpc"
    rollout_runner.planner = planner
    rollout_runner.sim_time = 1.0
    rollout_runner.last_shot_time = -100.0
    rollout_runner.gimbal_yaw = math.pi / 2
    rollout_runner.gimbal_yaw_vel = 0.0
    rollout_runner.gimbal_yaw_acc = 0.0
    rollout_runner.target = TargetState(
        0.0, 6.0, 0.0, 0.0, 0.0, INITIAL_OMEGA, ALPHA_LIMIT
    )
    rollout_runner.current_plan = planner.plan(
        replace(rollout_runner.target), BULLET_SPEED
    )
    rollout_horizon = (
        rollout_runner.current_plan.delay + rollout_runner.current_plan.flight_time
    )
    rollout_runner.probabilistic_model = ProbabilisticAimModel(rollout_horizon)
    for index in range(60):
        branch = ALPHA_LIMIT if index % 2 else -ALPHA_LIMIT
        rollout_runner.probabilistic_model.add_sample(
            index * PLANNER_DT,
            0.0,
            INITIAL_OMEGA,
            INITIAL_OMEGA * rollout_horizon + 0.5 * branch * rollout_horizon**2,
            rollout_horizon,
        )
    scenarios = rollout_runner._rollout_scenarios()
    assert 1 <= len(scenarios) <= ROLLOUT_SCENARIOS
    assert abs(sum(weight for _a, weight in scenarios) - 1.0) < 1e-12
    # A bimodal bang-bang history must keep both branches as scenarios.
    assert min(a for a, _w in scenarios) < 0.0 < max(a for a, _w in scenarios)

    expected_hits, mean_gate = rollout_runner._rollout_once(
        rollout_runner.current_plan.target_yaw, 0.0, 0
    )
    assert expected_hits >= 0.0
    assert 0.0 <= mean_gate <= 1.0
    # The fire-rate limit caps how many shots one rollout can credit.
    assert expected_hits <= math.ceil(ROLLOUT_TICKS * PLANNER_DT / SHOT_INTERVAL) + 1

    rollout_runner._probabilistic_reachable_peaks = []
    rollout_runner._rollout_decision_time = -1.0
    rollout_runner._rollout_phase = None
    rollout_base = planner.plan(replace(rollout_runner.target), BULLET_SPEED)
    rollout_runner._select_rollout_mpc(rollout_base)
    assert rollout_runner._rollout_phase is not None
    # The widened horizon tolerance must be restored after the rollout.
    assert rollout_runner.probabilistic_model.horizon_tolerance == PLANNER_DT
    assert abs(
        angle_error(
            rollout_runner.current_plan.target_yaw,
            rollout_base.target_yaw + rollout_runner._rollout_phase,
        )
    ) < 1e-9
    # A held decision must re-anchor its phase to the fresh nominal yaw.
    held_phase = rollout_runner._rollout_phase
    rollout_runner.target = replace(
        rollout_runner.target, angle=rollout_runner.target.angle + 0.05
    )
    shifted_base = planner.plan(replace(rollout_runner.target), BULLET_SPEED)
    rollout_runner._select_rollout_mpc(shifted_base)
    assert rollout_runner._rollout_phase == held_phase
    assert abs(
        angle_error(
            rollout_runner.current_plan.target_yaw,
            shifted_base.target_yaw + held_phase,
        )
    ) < 1e-9

    print("self-test passed")


def benchmark(
    seconds: float = 30.0,
    modes: tuple[str, ...] = ("current", "probabilistic", "field_mpc", "rollout_mpc"),
    seed: int = 20260903,
) -> None:
    """Run the exact GUI dynamics without Tk so displayed metrics are auditable."""
    for mode in modes:
        random.seed(seed)
        if seed != 20260903:
            print(f"seed={seed}")
        runner = MovingTargetVisualizer.__new__(MovingTargetVisualizer)
        runner.mode = mode
        runner.target_distance = type("V", (), {"get": lambda self: 6.0})()
        runner.acceleration_direction = 1.0
        runner.next_acceleration_switch = EVADE_SWITCH_INTERVAL
        runner.gimbal_yaw = math.pi / 2
        runner.gimbal_yaw_vel = 0.0
        runner.gimbal_yaw_acc = 0.0
        runner.sim_time = 0.0
        runner.last_shot_time = -100.0
        runner.valid_hits = runner.low_speed_hits = runner.misses = 0
        runner.planner = TinyMpcPlanner()
        runner.target = TargetState(0.0, 6.0, 0.0, 0.0, 0.0, INITIAL_OMEGA, ALPHA_LIMIT)
        runner.motion_history = []
        runner.probabilistic_model = ProbabilisticAimModel(LOW_SPEED_DELAY + 6.0 / BULLET_SPEED)
        runner._probabilistic_cache_time = -1.0
        runner._probabilistic_cache = (0.0, 0.0)
        runner._probabilistic_last_phase = None
        runner.current_aim_probability = 0.0
        runner.current_fire_probability = 0.0
        runner.current_plan = runner.planner.plan(replace(runner.target), BULLET_SPEED)
        runner._probabilistic_peak_yaw = runner.current_plan.target_yaw
        runner._probabilistic_pmax = 0.0
        runner._probabilistic_candidate_probability = 0.0
        runner.diagnostic_log = []
        runner.resolved_diagnostics = []
        runner.gate_block_count = 0
        runner.accel_saturated_steps = 0
        runner._probabilistic_reachable_peaks = []
        runner._rollout_decision_time = -1.0
        runner._rollout_phase = None
        runner._rollout_score = 0.0
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
        if mode in ("probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control", "rollout_mpc") and runner.diagnostic_log:
            rows = runner.diagnostic_log
            shot_rows = runner.resolved_diagnostics
            log_name = (
                f"moving_target_diagnostic_{mode}.csv"
                if seed == 20260903
                else f"moving_target_diagnostic_{mode}_{seed}.csv"
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
    parser.add_argument("--mode", choices=("current", "probabilistic", "field_candidate", "field_mpc", "window_candidate", "lookahead_control", "rollout_mpc"), default="current")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="run fixed-state probability alignment diagnostic")
    arguments = parser.parse_args()
    if arguments.diagnostic:
        probability_alignment_diagnostic()
    elif arguments.benchmark:
        benchmark()
    elif arguments.self_test:
        self_test()
    else:
        MovingTargetVisualizer(arguments.mode).mainloop()
