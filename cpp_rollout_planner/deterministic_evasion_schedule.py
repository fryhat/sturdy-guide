from __future__ import annotations

import math

import sp_vision_moving_target_visualizer_full_compare as sim


class DeterministicEvasionSchedule:
    """Explicit deterministic target trajectory used by rollout benchmarks."""

    def __init__(
        self,
        initial_omega: float = sim.INITIAL_OMEGA,
        omega_limit: float = sim.OMEGA_LIMIT,
        alpha_limit: float = sim.ALPHA_LIMIT,
        switch_interval: float = sim.EVADE_SWITCH_INTERVAL,
        center_x: float = 0.0,
        center_y: float = 6.0,
    ) -> None:
        self.initial_omega = initial_omega
        self.omega_limit = abs(omega_limit)
        self.alpha_limit = abs(alpha_limit)
        self.switch_interval = switch_interval
        self.center_x = center_x
        self.center_y = center_y

    def _advance(
        self,
        angle: float,
        omega: float,
        direction: float,
        dt: float,
    ) -> tuple[float, float, float]:
        limit = self.omega_limit
        alpha = direction * self.alpha_limit
        if direction > 0.0 and omega >= limit:
            return angle + omega * dt, omega, 0.0
        if direction < 0.0 and omega <= -limit:
            return angle + omega * dt, omega, 0.0
        boundary = limit if direction > 0.0 else -limit
        time_to_boundary = (boundary - omega) / alpha
        if 0.0 <= time_to_boundary < dt:
            reached = omega + alpha * time_to_boundary
            angle += (
                omega * time_to_boundary
                + 0.5 * alpha * time_to_boundary * time_to_boundary
            )
            angle += reached * (dt - time_to_boundary)
            return angle, reached, 0.0
        angle += omega * dt + 0.5 * alpha * dt * dt
        return angle, omega + alpha * dt, alpha

    def state_at(self, time: float) -> sim.TargetState:
        if time <= 0.0:
            return sim.TargetState(
                self.center_x,
                self.center_y,
                0.0,
                0.0,
                0.0,
                self.initial_omega,
                self.alpha_limit,
            )
        angle = 0.0
        omega = self.initial_omega
        direction = 1.0
        next_switch = self.switch_interval
        cursor = 0.0
        remaining = time
        while remaining > 1e-12:
            until_switch = max(0.0, next_switch - cursor)
            step = min(remaining, until_switch)
            if step <= 0.0:
                direction = -direction
                next_switch += self.switch_interval
                continue
            angle, omega, alpha = self._advance(angle, omega, direction, step)
            cursor += step
            remaining -= step
            if until_switch <= step + 1e-12:
                direction = -direction
                alpha = direction * self.alpha_limit
                next_switch += self.switch_interval
        return sim.TargetState(
            self.center_x,
            self.center_y,
            0.0,
            0.0,
            angle,
            omega,
            alpha,
        )
