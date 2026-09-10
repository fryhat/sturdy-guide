"""2D moving-target visualizer based on sp_vision_25 target prediction.

The target predictor follows TongjiSuperPower/sp_vision_25's Target model:
the center translates at the currently estimated velocity, and armor angle
advances at the currently estimated angular velocity. Future target
acceleration is not part of the state. Vertical motion, pitch, and projectile
drop are intentionally omitted.
"""

from __future__ import annotations

import argparse
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
EVADE_SWITCH_INTERVAL = 1.5
SIMULATION_TIME_SCALE = 0.1  # simulation time / wall-clock time
SCATTER_STATIC_HIT_RATE = 0.90
SCATTER_Z90 = 1.6448536269514722
PROBABILITY_FIRE_THRESHOLD = 0.15
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


def advance_full_acceleration(
    angle: float,
    omega: float,
    acceleration_direction: float,
    dt: float,
) -> tuple[float, float, float, float]:
    """Advance with |alpha| fixed at ALPHA_LIMIT, reflecting at speed limits."""
    direction = 1.0 if acceleration_direction >= 0.0 else -1.0
    remaining = dt
    while remaining > 1e-12:
        boundary = OMEGA_LIMIT if direction > 0.0 else -OMEGA_LIMIT
        time_to_boundary = max(0.0, (boundary - omega) / (direction * ALPHA_LIMIT))
        step = min(remaining, time_to_boundary)
        alpha = direction * ALPHA_LIMIT
        angle += omega * step + 0.5 * alpha * step * step
        omega += alpha * step
        remaining -= step
        if time_to_boundary <= step + 1e-12:
            omega = boundary
            direction = -direction
        else:
            break
    return angle, omega, direction, direction * ALPHA_LIMIT


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
    return angle, omega, direction, direction * ALPHA_LIMIT, next_switch


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
        half_width = ARMOR_WIDTH_M / 2.0
        self.scatter_sigma = scatter_sigma or half_width / SCATTER_Z90

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
        for index, (timestamp, position, velocity) in enumerate(history):
            end_time = timestamp + self.horizon
            future = next((row for row in history[index + 1 :] if row[0] >= end_time), None)
            if future is None:
                continue
            self.add_sample(timestamp, position, velocity, future[1], future[0] - timestamp)
            added += 1
        return added

    def _allowed_acceleration(self, velocity: float) -> tuple[float, float]:
        tau = self.horizon
        # v(t)=v+a*t must stay within both velocity limits.
        lower = (self.min_velocity - velocity) / tau
        upper = (self.max_velocity - velocity) / tau
        return max(-self.max_acceleration, lower), min(self.max_acceleration, upper)

    def constrained_accelerations(self, velocity: float) -> list[float]:
        lower, upper = self._allowed_acceleration(velocity)
        values = [
            clamp(sample.average_acceleration, lower, upper)
            for sample in self.samples
            if abs(sample.horizon - self.horizon) <= PLANNER_DT
        ]
        if not values:
            values = [clamp(0.0, lower, upper)]
        return values

    def future_position_samples(
        self,
        position: float,
        velocity: float,
        seed: int = 0,
        count: int = 512,
    ) -> list[float]:
        values = self.constrained_accelerations(velocity)
        rng = random.Random(seed)
        return [
            position + velocity * self.horizon + 0.5 * rng.choice(values) * self.horizon**2
            for _ in range(max(1, count))
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
        """Evaluate a multimodal yaw hit field for a predicted target ensemble."""
        # ``target`` is already evaluated at the predicted impact time by the
        # planner. Add only the acceleration-model uncertainty over this same
        # horizon; do not propagate the nominal state a second time.
        accelerations = self.constrained_accelerations(target.omega)
        rng = random.Random(seed)
        angle_samples = [
            target.angle + 0.5 * rng.choice(accelerations) * self.horizon**2
            for _ in range(max(1, count))
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
    resolved: bool = False
    hit: bool = False
    outcome: str = "pending"
    normal_speed: float | None = None
    resolved_at: float = 0.0


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
    ) -> PlannerOutput:
        delay = HIGH_SPEED_DELAY if abs(state.omega) > DECISION_SPEED else LOW_SPEED_DELAY
        delayed = predict_constant_velocity(state, delay)
        initial_armor = nearest_armor(delayed)
        flight_time = norm(initial_armor.x, initial_armor.y) / bullet_speed
        impact_state = predict_constant_velocity(delayed, flight_time)
        selected = nearest_armor(impact_state)
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
            center_yaw = absolute_yaws[PLANNER_HALF_HORIZON]
            phase_offset = angle_error(aim_yaw_override, center_yaw)
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
    if scatter:
        # One-dimensional lateral Gaussian spread calibrated by ProbabilisticAimModel.
        offset = random.gauss(0.0, ProbabilisticAimModel(0.2).scatter_sigma)
        aim_x -= offset * math.sin(launch_yaw)
        aim_y += offset * math.cos(launch_yaw)
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
        launch_angle=launch_yaw,
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
        self.mode = mode
        mode_label = "当前版：实际云台跟踪" if mode == "current" else "加速度分布预测版"
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
        self._probabilistic_cache: tuple[float, float] = (self.gimbal_yaw, 0.0)
        self._probabilistic_last_yaw: float | None = None
        self.current_plan = self.planner.plan(replace(self.target), BULLET_SPEED)
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
        mode_label = "当前版：实际云台跟踪" if self.mode == "current" else "加速度分布预测版"
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
        self.probabilistic_model.horizon = LOW_SPEED_DELAY + distance / BULLET_SPEED
        self.distance_label.configure(text=f"{distance:.1f} m")
        if distance_changed:
            self.valid_hits = 0
            self.low_speed_hits = 0
            self.misses = 0
            self.shots.clear()
            self.last_shot_time = -100.0
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
        self.probabilistic_model.horizon = LOW_SPEED_DELAY + self.target_distance.get() / BULLET_SPEED
        self.motion_history.clear()
        self.probabilistic_model.samples.clear()
        self._probabilistic_cache_time = -1.0
        self._probabilistic_last_yaw = None
        self.current_plan = self.planner.plan(replace(self.target), BULLET_SPEED)
        self.shots.clear()
        self.running = True
        self.pause_button.configure(text="暂停")
        self.last_clock = None
        self.simulation_accumulator = 0.0
        self.draw()

    def update_planner(self) -> None:
        base_plan = self.planner.plan(replace(self.target), BULLET_SPEED)
        self.current_plan = base_plan
        if self.mode == "probabilistic":
            target_yaw, probability = self.probabilistic_yaw()
            # Apply only a phase shift to the already solved MPC trajectory.
            # This preserves its velocity/acceleration profile and avoids a
            # second DLL solve every 10 ms, which was the source of animation
            # stalls in the probability mode.
            phase = angle_error(target_yaw, base_plan.target_yaw)
            planned_range = norm(base_plan.planned_aim_x, base_plan.planned_aim_y)
            probabilistic_plan = replace(
                base_plan,
                target_yaw=base_plan.target_yaw + phase,
                yaw=base_plan.yaw + phase,
                planned_aim_x=planned_range * math.cos(target_yaw),
                planned_aim_y=planned_range * math.sin(target_yaw),
                reference_yaw=tuple(yaw + phase for yaw in base_plan.reference_yaw),
                planned_yaw=tuple(yaw + phase for yaw in base_plan.planned_yaw),
            )
            self.current_plan = replace(
                probabilistic_plan,
                fire=probabilistic_plan.fire and probability >= PROBABILITY_FIRE_THRESHOLD,
            )

    def probabilistic_yaw(self) -> tuple[float, float]:
        """Select the best reachable peak of the future hit-probability field."""
        cache_time = math.floor(self.sim_time * 20.0) / 20.0
        if self._probabilistic_cache_time == cache_time:
            return self._probabilistic_cache
        if len(self.motion_history) >= 2:
            # Rebuild the short rolling window instead of appending the same
            # observations on every cache refresh.  This keeps the empirical
            # distribution stationary in size and avoids quadratic growth.
            self.probabilistic_model.samples.clear()
            self.probabilistic_model.add_trajectory(self.motion_history[-80:])
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
        reach = abs(self.gimbal_yaw_vel) * reach_time + 0.5 * GIMBAL_MAX_ACCEL * reach_time**2
        reachable = [
            index for index, yaw in enumerate(grid)
            if abs(angle_error(yaw, self.gimbal_yaw)) <= reach
        ]
        # Keep several local maxima as branches.  A small continuity cost
        # prevents stochastic resampling from jumping between opposite armor
        # peaks unless the new branch is materially better.
        candidates = [i for i in reachable if i == 0 or i == len(field) - 1 or
                      field[i] >= field[i - 1] and field[i] >= field[i + 1]]
        if not candidates:
            candidates = reachable or list(range(len(grid)))
        previous = self._probabilistic_last_yaw
        def branch_score(index: int) -> float:
            probability = field[index]
            if previous is None:
                return probability
            delta = angle_error(grid[index], previous)
            return probability - PROBABILITY_CONTINUITY_WEIGHT * (delta / math.pi) ** 2
        best = max(candidates, key=branch_score)
        if previous is not None:
            best_probability = field[best]
            previous_index = min(range(len(grid)), key=lambda i: abs(angle_error(grid[i], previous)))
            previous_probability = field[previous_index]
            if best_probability < previous_probability + PROBABILITY_PEAK_SWITCH_MARGIN:
                best = previous_index if previous_index in reachable else best
        selected_yaw = grid[best]
        if 0 < best < len(grid) - 1:
            step = math.radians(1.5)
            fine_grid = [selected_yaw + step * offset / 4.0 for offset in range(-4, 5)]
            fine_field = self.probabilistic_model.hit_probability_field(
                predicted, fine_grid, count=64, seed=int(cache_time * 100)
            )
            fine_index = max(range(len(fine_grid)), key=fine_field.__getitem__)
            selected_yaw = fine_grid[fine_index]
        if previous is not None:
            selected_yaw = previous + PROBABILITY_YAW_SMOOTHING * angle_error(selected_yaw, previous)
        self._probabilistic_last_yaw = selected_yaw
        self._probabilistic_cache = (selected_yaw, field[best])
        nominal_index = min(
            range(len(grid)),
            key=lambda i: abs(angle_error(grid[i], self.current_plan.target_yaw)),
        )
        self._probabilistic_cache_time = cache_time
        return self._probabilistic_cache

    def fire_if_ready(self) -> None:
        fire_index = PLANNER_HALF_HORIZON + int(round(FIRE_LOOKAHEAD / PLANNER_DT))
        predicted_fire_yaw = self.current_plan.reference_yaw[fire_index]
        predicted_gimbal_yaw = (
            self.gimbal_yaw
            + self.gimbal_yaw_vel * FIRE_LOOKAHEAD
            + 0.5 * self.gimbal_yaw_acc * FIRE_LOOKAHEAD * FIRE_LOOKAHEAD
        )
        predicted_gimbal_error = angle_error(predicted_fire_yaw, predicted_gimbal_yaw)
        # Keep the upstream planner's fire permission.  The simulated projectile
        # still uses the actual gimbal direction at launch, so tracking error is
        # visible in the hit result instead of silently suppressing the fire rate.
        if not self.current_plan.fire:
            return
        if self.sim_time - self.last_shot_time < SHOT_INTERVAL:
            return
        self.shots.append(
            shot_from_plan(self.current_plan, self.sim_time, self.gimbal_yaw, scatter=True)
        )
        self.last_shot_time = self.sim_time

    def advance_gimbal(self, dt: float) -> None:
        """Track the absolute yaw/velocity command with the configured acceleration limit."""
        if self.mode == "probabilistic":
            target_yaw, _hit_probability = self.probabilistic_yaw()
            yaw_error = angle_error(target_yaw, self.gimbal_yaw)
            command_acc = clamp(35.0 * yaw_error - 10.0 * self.gimbal_yaw_vel,
                                -GIMBAL_MAX_ACCEL, GIMBAL_MAX_ACCEL)
            self.gimbal_yaw += self.gimbal_yaw_vel * dt + 0.5 * command_acc * dt * dt
            self.gimbal_yaw_vel += command_acc * dt
            self.gimbal_yaw_acc = command_acc
            return
        yaw_error = angle_error(self.current_plan.yaw, self.gimbal_yaw)
        velocity_error = self.current_plan.yaw_velocity - self.gimbal_yaw_vel
        command_acc = clamp(
            self.current_plan.yaw_acceleration + 35.0 * yaw_error + 10.0 * velocity_error,
            -GIMBAL_MAX_ACCEL,
            GIMBAL_MAX_ACCEL,
        )
        next_vel = self.gimbal_yaw_vel + command_acc * dt
        self.gimbal_yaw += self.gimbal_yaw_vel * dt + 0.5 * command_acc * dt * dt
        self.gimbal_yaw_vel = next_vel
        self.gimbal_yaw_acc = command_acc

    def resolve_shots(self) -> None:
        for shot in self.shots:
            if not shot.launched and self.sim_time >= shot.launch_at:
                shot.launched = True
            if not shot.resolved and self.sim_time >= shot.impact_at:
                shot.resolved = True
                result = evaluate_shot(shot, self.target)
                shot.outcome = result.outcome
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
        fire_state = "允许" if self.current_plan.fire else "抑制"
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
    angle, omega, direction, alpha = advance_full_acceleration(
        0.0, OMEGA_LIMIT - math.radians(1.0), 1.0, 0.02
    )
    assert -OMEGA_LIMIT <= omega <= OMEGA_LIMIT
    assert direction == -1.0
    assert alpha == -ALPHA_LIMIT
    assert angle > 0.0

    angle, omega, direction, alpha, next_switch = advance_optimized_evasion(
        0.0, INITIAL_OMEGA, 1.0, 1.49, 1.50, 0.02
    )
    assert direction == -1.0
    assert alpha == -ALPHA_LIMIT
    assert abs(next_switch - 3.0) < 1e-12
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

    assert static_plan.fire
    assert abs(static_plan.fire_error) < FIRE_THRESHOLD
    shifted_plan = planner.plan(static, BULLET_SPEED, aim_yaw_override=static_plan.target_yaw + 0.1)
    assert abs(angle_error(shifted_plan.target_yaw, static_plan.target_yaw + 0.1)) < 1e-9
    assert abs(shifted_plan.yaw_velocity - static_plan.yaw_velocity) < 1e-3

    probabilistic = ProbabilisticAimModel(0.20)
    sample = probabilistic.add_sample(0.0, 0.0, 2.0, 0.5)
    assert abs(sample.average_acceleration - 5.0) < 1e-12
    probabilistic.add_sample(0.2, 0.5, 2.0, 0.9)
    constrained = probabilistic.constrained_accelerations(OMEGA_LIMIT - 0.01)
    assert all(value <= (OMEGA_LIMIT - (OMEGA_LIMIT - 0.01)) / 0.20 + 1e-12 for value in constrained)
    # The calibration is defined at the armor half-width for a static target.
    sigma = probabilistic.scatter_sigma
    z = (ARMOR_WIDTH_M / 2.0) / (sigma * math.sqrt(2.0))
    assert abs(math.erf(z) - SCATTER_STATIC_HIT_RATE) < 1e-12
    aim, expected = probabilistic.optimal_aim(0.0, 0.0, [-0.1, 0.0, 0.1], count=32)
    assert aim == 0.0
    assert 0.0 < expected <= 1.0
    field = probabilistic.hit_probability_field(static, [-math.pi / 2, 0.0, math.pi / 2], count=16)
    assert len(field) == 3 and all(0.0 <= value <= 1.0 for value in field)
    print("self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("current", "probabilistic"), default="current")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        MovingTargetVisualizer(arguments.mode).mainloop()
