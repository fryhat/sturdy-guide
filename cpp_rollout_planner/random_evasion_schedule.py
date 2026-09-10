from __future__ import annotations

import math
import random
from dataclasses import dataclass


OMEGA_LIMIT = math.radians(700.0)
INITIAL_OMEGA = math.radians(350.0)
MIN_ACCEL = math.radians(280.0)
MAX_ACCEL = math.radians(300.0)
MIN_DURATION = 1.0
MAX_DURATION = 1.2


@dataclass
class TargetStateSnapshot:
    time: float
    angle: float
    omega: float
    alpha: float


@dataclass
class EvasionSegment:
    start_time: float
    duration: float
    alpha: float
    start_angle: float
    start_omega: float


class RandomEvasionSchedule:
    def __init__(
        self,
        seed: int,
        initial_omega: float = INITIAL_OMEGA,
        omega_limit: float = OMEGA_LIMIT,
        min_accel: float = MIN_ACCEL,
        max_accel: float = MAX_ACCEL,
        min_duration: float = MIN_DURATION,
        max_duration: float = MAX_DURATION,
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.initial_omega = initial_omega
        self.omega_limit = abs(omega_limit)
        self.min_accel = abs(min_accel)
        self.max_accel = abs(max_accel)
        self.alpha_limit = abs(max_accel)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.segments: list[EvasionSegment] = []
        self.time = 0.0
        self.angle = 0.0
        self.omega = initial_omega
        self.alpha = 0.0

    def _sample_alpha(self) -> float:
        if self.rng.random() < 0.5:
            return -self.rng.uniform(self.min_accel, self.max_accel)
        return self.rng.uniform(self.min_accel, self.max_accel)

    def _generate_until(self, target_time: float) -> None:
        while self.time < target_time:
            alpha = self._sample_alpha()
            duration = self.rng.uniform(self.min_duration, self.max_duration)
            boundary = self.omega_limit if alpha > 0.0 else -self.omega_limit
            time_to_boundary = (
                (boundary - self.omega) / alpha
                if abs(alpha) > 1e-12
                else math.inf
            )
            if 0.0 <= time_to_boundary < duration:
                duration = 0.98 * time_to_boundary
            segment = EvasionSegment(
                start_time=self.time,
                duration=duration,
                alpha=alpha,
                start_angle=self.angle,
                start_omega=self.omega,
            )
            self.segments.append(segment)
            self.angle += self.omega * duration + 0.5 * alpha * duration**2
            self.omega += alpha * duration
            self.alpha = alpha
            self.time += duration

    def state_at(self, time: float) -> TargetStateSnapshot:
        self._generate_until(time)
        if not self.segments or time < self.segments[0].start_time:
            return TargetStateSnapshot(time, 0.0, self.initial_omega, self.alpha)
        lo = 0
        hi = len(self.segments) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.segments[mid].start_time <= time:
                lo = mid
            else:
                hi = mid - 1
        segment = self.segments[lo]
        tau = max(0.0, min(time - segment.start_time, segment.duration))
        angle = (
            segment.start_angle
            + segment.start_omega * tau
            + 0.5 * segment.alpha * tau * tau
        )
        omega = segment.start_omega + segment.alpha * tau
        return TargetStateSnapshot(time, angle, omega, segment.alpha)

    def next_switch_at(self, time: float) -> float:
        self._generate_until(time)
        if not self.segments:
            return 1.0
        lo = 0
        hi = len(self.segments) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.segments[mid].start_time <= time:
                lo = mid
            else:
                hi = mid - 1
        segment = self.segments[lo]
        return segment.start_time + segment.duration
