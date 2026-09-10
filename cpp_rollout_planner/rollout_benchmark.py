from __future__ import annotations

import math
import random
import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sp_vision_moving_target_visualizer_full_compare as sim

from causal_evasion_predictor import CausalEvasionPredictor
from deterministic_evasion_schedule import DeterministicEvasionSchedule
from p_hit_provider import (
    advance_target,
    build_segment_curves,
    compress_history_entries,
    curve_grid,
    schedule_armor_solution_for_yaw,
    schedule_state_at,
)
from rollout_planner import RolloutPlanner

ROLLOUT_FIRE_THRESHOLD = 0.50
ROLLOUT_MAX_GIMBAL_OMEGA = 1.2617046468719202
MODEL_VERSION_FUSION = "fusion"
MODEL_VERSION_RIGOROUS = "rigorous"


def advance_gimbal_limited(
    yaw: float,
    omega: float,
    alpha: float,
    dt: float,
) -> tuple[float, float]:
    limit = ROLLOUT_MAX_GIMBAL_OMEGA
    if abs(alpha) < 1e-12:
        return yaw + omega * dt, omega
    if alpha > 0.0 and omega >= limit:
        return yaw + omega * dt, omega
    if alpha < 0.0 and omega <= -limit:
        return yaw + omega * dt, omega
    boundary = limit if alpha > 0.0 else -limit
    time_to_boundary = (boundary - omega) / alpha
    if 0.0 <= time_to_boundary < dt:
        reached = omega + alpha * time_to_boundary
        yaw += omega * time_to_boundary + 0.5 * alpha * time_to_boundary**2
        yaw += reached * (dt - time_to_boundary)
        return yaw, reached
    yaw += omega * dt + 0.5 * alpha * dt * dt
    return yaw, omega + alpha * dt


@dataclass
class ShotEvent:
    launch_at: float
    impact_at: float
    predicted_state: sim.TargetState
    planned_range: float
    armor_index: int = 0
    launch_angle: float | None = None
    aim_x: float = 0.0
    aim_y: float = 0.0
    scatter_offset: float = 0.0
    resolved: bool = False
    valid: bool = False


class RolloutBenchmark:
    def __init__(
        self,
        seconds: float = 10.0,
        seed: int = 20260903,
        distance: float = 6.0,
        dt: float = 0.010,
        smooth_acceleration: bool = False,
        fire_at_plan_point: bool = False,
        schedule=None,
        model_version: str = MODEL_VERSION_RIGOROUS,
        warmup_seconds: float = 0.0,
        zero_acceleration_history: bool = False,
    ) -> None:
        self.seconds = seconds
        self.seed = seed
        self.distance = float(distance)
        if self.distance <= 0.0:
            raise ValueError("distance must be positive")
        self.dt = dt
        self.smooth_acceleration = smooth_acceleration
        self.fire_at_plan_point = fire_at_plan_point
        if model_version not in (MODEL_VERSION_FUSION, MODEL_VERSION_RIGOROUS):
            raise ValueError(f"unknown model_version: {model_version}")
        self.model_version = model_version
        self.warmup_seconds = max(0.0, warmup_seconds)
        self.zero_acceleration_history = zero_acceleration_history
        self.target = sim.TargetState(
            0.0, self.distance, 0.0, 0.0, 0.0, sim.INITIAL_OMEGA, sim.ALPHA_LIMIT
        )
        self.schedule = (
            schedule
            if schedule is not None
            else DeterministicEvasionSchedule(
                center_x=self.target.x,
                center_y=self.target.y,
            )
        )
        self._prediction_schedule = CausalEvasionPredictor(
            center_x=self.target.x,
            center_y=self.target.y,
        )
        random.seed(seed)
        self.planner = RolloutPlanner()
        self.acceleration_direction = 1.0
        self.next_acceleration_switch = sim.EVADE_SWITCH_INTERVAL
        self.sim_time = 0.0
        self.gimbal_yaw = math.pi / 2
        self.gimbal_yaw_vel = 0.0
        self.gimbal_yaw_acc = 0.0
        self.command_time = -1.0
        self.command_alpha = 0.0
        self.point_launch_yaw: float | None = None
        self.point_launch_probability = 0.0
        self.model = sim.ProbabilisticAimModel(
            sim.LOW_SPEED_DELAY + self._center_radius() / sim.BULLET_SPEED
        )
        self.pending_starts: list[tuple[float, float, float]] = []
        self.shots: list[ShotEvent] = []
        self._shot_times: list[float] = []
        self.trace: list[dict[str, float | bool]] = []
        self._motion_observations: list[tuple[float, float]] = [
            (0.0, self.target.omega)
        ]
        self._acceleration_estimates: list[tuple[float, float]] = []
        self._scatter_rng = random.Random(seed)
        self._time_scatter: dict[float, float] = {}
        self.valid_hits = 0
        self.low_speed_hits = 0
        self.misses = 0
        self.resolved_shots = 0

    def _ingest_history(self) -> None:
        now = self.sim_time
        horizon = self.model.horizon
        self.pending_starts.append((now, self.target.angle, self.target.omega))
        retained: list[tuple[float, float, float]] = []
        for start_time, start_angle, start_omega in self.pending_starts:
            if start_time + horizon <= now + 1e-12:
                tau = now - start_time
                self.model.accumulate_sample(
                    start_time,
                    start_angle,
                    start_omega,
                    self.target.angle,
                    tau,
                )
            else:
                retained.append((start_time, start_angle, start_omega))
        self.pending_starts = retained

    def _append_motion_observation(self) -> None:
        self._motion_observations.append((self.sim_time, self.target.omega))

    def _estimate_acceleration(self) -> float:
        cutoff = self.sim_time - 0.20
        points = [
            (time, omega)
            for time, omega in self._motion_observations
            if time >= cutoff
        ]
        if len(points) < 3:
            return 0.0
        n = float(len(points))
        mean_time = sum(point[0] for point in points) / n
        mean_omega = sum(point[1] for point in points) / n
        numerator = sum(
            (point[0] - mean_time) * (point[1] - mean_omega)
            for point in points
        )
        denominator = sum(
            (point[0] - mean_time) ** 2
            for point in points
        )
        return numerator / denominator if abs(denominator) > 1e-18 else 0.0

    def _estimated_limits(self) -> tuple[float, float]:
        omega_limit = max(
            (abs(omega) for _time, omega in self._motion_observations),
            default=1e-9,
        )
        alpha_limit = max(
            (
                abs(alpha)
                for _time, alpha in self._acceleration_estimates
            ),
            default=1e-9,
        )
        return max(omega_limit, 1e-9), max(alpha_limit, 1e-9)

    def _history_source(self) -> dict[tuple[float, float], int]:
        if not self.zero_acceleration_history:
            return self.model.sample_entries
        zeroed: dict[tuple[float, float], int] = {}
        for (horizon, _acceleration), count in self.model.sample_entries.items():
            key = (round(horizon, 3), 0.0)
            zeroed[key] = zeroed.get(key, 0) + count
        return zeroed

    def _reset_predictor(self) -> None:
        omega_limit, alpha_limit = self._estimated_limits()
        self._prediction_schedule.reset(
            self.target.angle,
            self.target.omega,
            self.sim_time,
            omega_limit,
            alpha_limit,
            (
                0.0
                if self.model_version == MODEL_VERSION_RIGOROUS
                else self._estimate_acceleration()
            ),
            center_x=self.target.x,
            center_y=self.target.y,
        )

    def _center_radius(self) -> float:
        return math.hypot(self.target.x, self.target.y)

    def _advance_target(self, dt: float) -> None:
        self.target, self.acceleration_direction, self.next_acceleration_switch = (
            advance_target(
                self.target,
                self.acceleration_direction,
                self.next_acceleration_switch,
                self.sim_time - dt,
                dt,
            )
        )

    def _replan(self) -> None:
        self._reset_predictor()
        self.model.horizon = (
            sim.LOW_SPEED_DELAY + self._center_radius() / sim.BULLET_SPEED
        )
        grid = curve_grid(self.target)
        curves = build_segment_curves(
            self.model,
            self.target,
            self.acceleration_direction,
            self.next_acceleration_switch,
            self.sim_time,
            grid,
            schedule=self._prediction_schedule,
            history_entries=self._history_source(),
        )
        raw_alpha, raw_score = self.planner.plan(
            self.gimbal_yaw,
            self.gimbal_yaw_vel,
            grid,
            curves,
        )
        if self.smooth_acceleration:
            previous = self.command_alpha if self.command_time >= 0.0 else raw_alpha
            self.command_alpha = 0.5 * previous + 0.5 * raw_alpha
        else:
            self.command_alpha = raw_alpha
        self.command_time = self.sim_time
        self.point_launch_yaw = sim.wrap_angle(
            advance_gimbal_limited(
                self.gimbal_yaw,
                self.gimbal_yaw_vel,
                self.command_alpha,
                sim.LOW_SPEED_DELAY,
            )[0]
        )
        self.point_launch_probability = self._probability_at_launch(
            self.point_launch_yaw
        )
        self.trace.append(
            {
                "time": self.sim_time,
                "kind": "plan",
                "gimbal_yaw": self.gimbal_yaw,
                "gimbal_vel": self.gimbal_yaw_vel,
                "command_alpha": self.command_alpha,
                "path_score": raw_score,
                "launch_yaw": self.point_launch_yaw,
                "probability": self.point_launch_probability,
            }
        )

    def _probability_at_launch(self, launch_yaw: float) -> float:
        self._reset_predictor()
        entries = compress_history_entries([
            (horizon, acceleration, count)
            for (horizon, acceleration), count in self._history_source().items()
        ])
        schedule_omega_limit = self._prediction_schedule.omega_limit
        schedule_alpha_limit = self._prediction_schedule.alpha_limit
        launch_time = self.sim_time + sim.LOW_SPEED_DELAY
        _armor, impact_state, flight_time = schedule_armor_solution_for_yaw(
            self._prediction_schedule,
            self.target,
            launch_time,
            launch_yaw,
        )
        self.model.horizon = sim.LOW_SPEED_DELAY + flight_time
        return self.planner.hit_probability_from_impact(
            impact_state.angle,
            impact_state.omega,
            self.model.horizon,
            launch_yaw,
            entries,
            schedule_omega_limit,
            schedule_alpha_limit,
            center_x=impact_state.x,
            center_y=impact_state.y,
        )

    def _create_shot(self, launch_yaw: float) -> None:
        scatter_key = round(self.sim_time, 6)
        if scatter_key not in self._time_scatter:
            raise RuntimeError(f"scatter offset missing for time {scatter_key}")
        self._reset_predictor()
        launch_time = self.sim_time + sim.LOW_SPEED_DELAY
        armor, impact_state, flight_time = schedule_armor_solution_for_yaw(
            self._prediction_schedule,
            self.target,
            launch_time,
            launch_yaw,
        )
        planned_range = math.hypot(armor.x, armor.y)
        self.shots.append(
            ShotEvent(
                launch_at=self.sim_time + sim.LOW_SPEED_DELAY,
                impact_at=self.sim_time
                + sim.LOW_SPEED_DELAY
                + flight_time,
                predicted_state=impact_state,
                planned_range=planned_range,
                armor_index=armor.index,
                scatter_offset=self._time_scatter[scatter_key],
            )
        )

    def _ensure_time_scatter(self) -> None:
        key = round(self.sim_time, 6)
        if key not in self._time_scatter:
            self._time_scatter[key] = self._scatter_rng.gauss(
                0.0,
                sim.SCATTER_SIGMA,
            )

    def _step(self, dt: float, allow_fire: bool = True) -> None:
        self.sim_time += dt
        self._advance_target(dt)
        self._ensure_time_scatter()
        self._append_motion_observation()
        self._acceleration_estimates.append(
            (self.sim_time, self._estimate_acceleration())
        )
        self._ingest_history()
        should_replan = (
            self.command_time < 0.0
            or self.sim_time - self.command_time >= 0.05 - 1e-12
        )
        if should_replan:
            self._replan()

        start_yaw = self.gimbal_yaw
        start_velocity = self.gimbal_yaw_vel
        self.gimbal_yaw, self.gimbal_yaw_vel = advance_gimbal_limited(
            start_yaw,
            start_velocity,
            self.command_alpha,
            dt,
        )
        self.gimbal_yaw_acc = self.command_alpha
        self.gimbal_yaw = sim.wrap_angle(self.gimbal_yaw)
        gimbal_step = (
            self.sim_time - dt,
            start_yaw,
            start_velocity,
            self.command_alpha,
        )

        if allow_fire and self.fire_at_plan_point:
            if (
                should_replan
                and self.sim_time - self._last_shot_time()
                >= sim.SHOT_INTERVAL - 1e-12
                and self.point_launch_yaw is not None
                and self.point_launch_probability >= ROLLOUT_FIRE_THRESHOLD
            ):
                self._create_shot(self.point_launch_yaw)
                self._shot_times.append(self.sim_time)
        elif allow_fire:
            if (
                self.sim_time - self._last_shot_time()
                >= sim.SHOT_INTERVAL - 1e-12
            ):
                launch_yaw = sim.wrap_angle(
                    advance_gimbal_limited(
                        self.gimbal_yaw,
                        self.gimbal_yaw_vel,
                        self.gimbal_yaw_acc,
                        sim.LOW_SPEED_DELAY,
                    )[0]
                )
                gate_probability = self._probability_at_launch(launch_yaw)
                self.trace.append(
                    {
                        "time": self.sim_time,
                        "kind": "gate",
                        "gimbal_yaw": self.gimbal_yaw,
                        "gimbal_vel": self.gimbal_yaw_vel,
                        "command_alpha": self.command_alpha,
                        "launch_yaw": launch_yaw,
                        "probability": gate_probability,
                    }
                )
                if gate_probability >= ROLLOUT_FIRE_THRESHOLD:
                    self._create_shot(launch_yaw)
                    self._shot_times.append(self.sim_time)

        for shot in self.shots:
            if shot.launch_angle is None and self.sim_time >= shot.launch_at:
                start_time, yaw0, omega0, alpha = gimbal_step
                if start_time <= shot.launch_at <= self.sim_time:
                    tau = shot.launch_at - start_time
                    shot.launch_angle = (
                        yaw0 + omega0 * tau + 0.5 * alpha * tau * tau
                    )
                else:
                    shot.launch_angle = self.gimbal_yaw
                shot.aim_x = shot.planned_range * math.cos(shot.launch_angle)
                shot.aim_y = shot.planned_range * math.sin(shot.launch_angle)
                shot.aim_x -= shot.scatter_offset * math.sin(shot.launch_angle)
                shot.aim_y += shot.scatter_offset * math.cos(shot.launch_angle)
            if not shot.resolved and self.sim_time >= shot.impact_at:
                shot.resolved = True
                result = sim.evaluate_shot(
                    sim.Shot(
                        created_at=0.0,
                        launch_at=shot.launch_at,
                        impact_at=shot.impact_at,
                        launch_x=0.0,
                        launch_y=0.0,
                        aim_x=shot.aim_x,
                        aim_y=shot.aim_y,
                        planned_aim_x=shot.aim_x,
                        planned_aim_y=shot.aim_y,
                        predicted_state=shot.predicted_state,
                        armor_index=shot.armor_index,
                        launched=True,
                        launch_angle=shot.launch_angle,
                    ),
                    self._actual_target_at(shot.impact_at),
                )
                shot.valid = result.outcome == "valid_hit"
                self.resolved_shots += 1
                if result.outcome == "valid_hit":
                    self.valid_hits += 1
                elif result.outcome == "low_normal_speed":
                    self.low_speed_hits += 1
                else:
                    self.misses += 1
        self.shots = [shot for shot in self.shots if not shot.resolved]

    def _last_shot_time(self) -> float:
        return self._shot_times[-1] if self._shot_times else -100.0

    def _actual_target_at(self, time: float) -> sim.TargetState:
        return schedule_state_at(self.schedule, self.target, time)

    def run(self) -> None:
        warmup_steps = int(self.warmup_seconds / self.dt)
        for _ in range(warmup_steps):
            self.sim_time += self.dt
            self._advance_target(self.dt)
            self._ensure_time_scatter()
            self._append_motion_observation()
            self._acceleration_estimates.append(
                (self.sim_time, self._estimate_acceleration())
            )
            self._ingest_history()
        self.gimbal_yaw = math.pi / 2
        self.gimbal_yaw_vel = 0.0
        self.gimbal_yaw_acc = 0.0
        self.command_time = -1.0
        self.command_alpha = 0.0
        self.point_launch_yaw = None
        self.point_launch_probability = 0.0
        self._shot_times.clear()
        self.last_shot_time = -100.0
        self.trace.clear()
        for _ in range(int(self.seconds / self.dt)):
            self._step(self.dt)
        while self.shots:
            self._step(self.dt, allow_fire=False)
        rate = (
            self.valid_hits / self.resolved_shots
            if self.resolved_shots
            else 0.0
        )
        print(
            f"rollout_cpp[{self.model_version}]: "
            f"shots={self.resolved_shots}, valid={self.valid_hits}, "
            f"low={self.low_speed_hits}, miss={self.misses}, "
            f"valid_rate={rate * 100:.2f}%, "
            f"valid_per_s={self.valid_hits / self.seconds:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seconds", type=float, nargs="?", default=10.0)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--point-fire", action="store_true")
    parser.add_argument("--dt", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--distance", type=float, default=6.0)
    parser.add_argument(
        "--model-version",
        choices=(MODEL_VERSION_FUSION, MODEL_VERSION_RIGOROUS),
        default=MODEL_VERSION_RIGOROUS,
    )
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument(
        "--zero-accel-history",
        action="store_true",
        help="force all predicted acceleration history samples to zero",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    RolloutBenchmark(
        seconds=args.seconds,
        smooth_acceleration=args.smooth,
        fire_at_plan_point=args.point_fire,
        dt=args.dt,
        seed=args.seed,
        distance=args.distance,
        model_version=args.model_version,
        warmup_seconds=args.warmup,
        zero_acceleration_history=args.zero_accel_history,
    ).run()
    print(f"wall_seconds={time.perf_counter() - started:.3f}")
