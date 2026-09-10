from __future__ import annotations

import math

import sp_vision_moving_target_visualizer as m


def phase_intervals(distance: float, yaw: float = math.pi / 2.0) -> list[float]:
    n = 36000
    values: list[bool] = []
    for i in range(n):
        angle = -math.pi + 2.0 * math.pi * i / n
        state = m.TargetState(0.0, distance, 0.0, 0.0, angle, 0.0, 0.0)
        shot = m.Shot(
            0.0,
            0.0,
            distance / m.BULLET_SPEED,
            0.0,
            0.0,
            (distance - 0.3) * math.cos(yaw),
            (distance - 0.3) * math.sin(yaw),
            0.0,
            0.0,
            state,
            0,
            True,
            yaw,
        )
        values.append(m.evaluate_shot(shot, state).outcome == "valid_hit")
    widths: list[float] = []
    start = None
    for i, value in enumerate(values + [values[0]]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            widths.append((i - start) * 360.0 / n)
            start = None
    return sorted(widths, reverse=True)


def main() -> None:
    incidence = math.degrees(math.acos(m.MIN_NORMAL_HIT_SPEED / m.BULLET_SPEED))
    print(f"incidence half-angle from normal = {incidence:.6f} deg")
    print(f"phase-window widths at 2/4/6/8 m = {[phase_intervals(d) for d in (2,4,6,8)]}")
    for distance in (2.0, 6.0):
        for omega_deg in (300.0, 600.0):
            h = 1e-5
            vals = []
            for i in range(10000):
                angle = 2.0 * math.pi * i / 10000.0
                def los(t: float) -> float:
                    a = angle + math.radians(omega_deg) * t
                    x, y = m.rotate(0.0, -m.BODY_LENGTH_M / 2.0, a)
                    return math.atan2(distance + y, x)
                acc = (los(h) - 2.0 * los(0.0) + los(-h)) / (h * h)
                vals.append(abs(acc))
            print(f"continuous front-armor LOS max d={distance:g} omega={omega_deg:g}: {max(vals):.3f} rad/s2")
    print(f"sqrt(30.699/280) relation gives r* = {25*(math.sqrt(30.699/280)-0.015):.6f} m")
    for distance in (2.0, 4.0, 6.0, 8.0):
        target = m.TargetState(0.0, distance, 0.0, 0.0, 0.0, m.INITIAL_OMEGA, m.ALPHA_LIMIT)
        controller_dir = 1.0
        next_switch = m.EVADE_SWITCH_INTERVAL
        valid = total = 0
        for index in range(6000):
            now = (index + 1) * 0.01
            target.angle, target.omega, controller_dir, target.alpha, next_switch = m.advance_optimized_evasion(
                target.angle, target.omega, controller_dir, now - 0.01, next_switch, 0.01
            )
            if index % 10:
                continue
            range_to_armor = distance - 0.3
            shot = m.Shot(0.0, now, now + range_to_armor / m.BULLET_SPEED, 0.0, 0.0,
                          0.0, range_to_armor, 0.0, 0.0, target, 0, True, math.pi / 2)
            result = m.evaluate_shot(shot, target)
            total += 1
            valid += result.outcome == "valid_hit"
        print(f"static gimbal d={distance:g}: hit rate={100*valid/total:.2f}% ({valid}/{total})")


if __name__ == "__main__":
    main()
