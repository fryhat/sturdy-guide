from __future__ import annotations

import sp_vision_moving_target_visualizer_full_compare as sim


class CausalEvasionPredictor:
    """Forecast target states from observations available at decision time."""

    def __init__(
        self,
        omega_limit: float = 1e-9,
        alpha_limit: float = 1e-9,
        center_x: float = 0.0,
        center_y: float = 6.0,
    ) -> None:
        self.omega_limit = abs(omega_limit)
        self.alpha_limit = abs(alpha_limit)
        self.state = sim.TargetState(
            center_x,
            center_y,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.acceleration_estimate = 0.0
        self.base_time = 0.0

    def reset(
        self,
        angle: float,
        omega: float,
        time: float,
        omega_limit: float,
        alpha_limit: float,
        acceleration_estimate: float,
        center_x: float | None = None,
        center_y: float | None = None,
    ) -> None:
        resolved_center_x = self.state.x if center_x is None else center_x
        resolved_center_y = self.state.y if center_y is None else center_y
        self.state = sim.TargetState(
            resolved_center_x,
            resolved_center_y,
            0.0,
            0.0,
            angle,
            omega,
            acceleration_estimate,
        )
        self.base_time = time
        self.omega_limit = max(abs(omega_limit), 1e-9)
        self.alpha_limit = max(abs(alpha_limit), 1e-9)
        self.acceleration_estimate = acceleration_estimate

    def _forecast(self, dt: float) -> sim.TargetState:
        if dt <= 0.0:
            return self.state
        alpha = self.acceleration_estimate
        omega = self.state.omega
        angle = self.state.angle

        if abs(alpha) < 1e-12:
            return sim.replace(
                self.state,
                angle=angle + omega * dt,
                alpha=0.0,
            )

        direction = 1.0 if alpha >= 0.0 else -1.0
        limit = self.omega_limit
        if direction > 0.0 and omega >= limit:
            return sim.replace(
                self.state,
                angle=angle + omega * dt,
                alpha=0.0,
            )
        if direction < 0.0 and omega <= -limit:
            return sim.replace(
                self.state,
                angle=angle + omega * dt,
                alpha=0.0,
            )

        boundary = limit if direction > 0.0 else -limit
        time_to_boundary = (boundary - omega) / alpha
        if 0.0 <= time_to_boundary < dt:
            reached = omega + alpha * time_to_boundary
            angle += (
                omega * time_to_boundary
                + 0.5 * alpha * time_to_boundary * time_to_boundary
            )
            angle += reached * (dt - time_to_boundary)
            return sim.replace(
                self.state,
                angle=angle,
                omega=reached,
                alpha=0.0,
            )

        angle += omega * dt + 0.5 * alpha * dt * dt
        return sim.replace(
            self.state,
            angle=angle,
            omega=omega + alpha * dt,
            alpha=alpha,
        )

    def state_at(self, time: float) -> sim.TargetState:
        return self._forecast(time - self.base_time)
