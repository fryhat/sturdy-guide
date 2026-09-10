"""Search bang-bang rotation policies against the local sp_vision_25 simulation."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import sp_vision_moving_target_visualizer as sim


@dataclass(frozen=True)
class Policy:
    name: str
    dwell_min: float | None
    dwell_max: float | None


@dataclass(frozen=True)
class Score:
    valid: int
    low_speed: int
    miss: int

    @property
    def resolved(self) -> int:
        return self.valid + self.low_speed + self.miss

    @property
    def hit_rate(self) -> float:
        return self.valid / self.resolved if self.resolved else 0.0


class BangBangController:
    def __init__(self, policy: Policy, seed: int) -> None:
        self.policy = policy
        self.rng = random.Random(seed)
        self.direction = 1.0
        self.next_switch = self._next_dwell()

    def _next_dwell(self) -> float:
        if self.policy.dwell_min is None or self.policy.dwell_max is None:
            return math.inf
        return self.rng.uniform(self.policy.dwell_min, self.policy.dwell_max)

    def advance(self, state: sim.TargetState, now: float, dt: float) -> None:
        remaining = dt
        cursor = now
        while remaining > 1e-12:
            until_switch = self.next_switch - cursor
            step = min(remaining, max(0.0, until_switch))
            state.angle, state.omega, self.direction, state.alpha = sim.advance_full_acceleration(
                state.angle, state.omega, self.direction, step
            )
            cursor += step
            remaining -= step
            if until_switch <= step + 1e-12:
                self.direction = -self.direction
                state.alpha = self.direction * sim.ALPHA_LIMIT
                self.next_switch = cursor + self._next_dwell()
            elif step <= 1e-12:
                break


def run(policy: Policy, distance: float, duration: float, seed: int) -> Score:
    planner = sim.TinyMpcPlanner()
    controller = BangBangController(policy, seed)
    target = sim.TargetState(
        x=0.0,
        y=distance,
        vx=0.0,
        vy=0.0,
        angle=0.0,
        omega=sim.INITIAL_OMEGA,
        alpha=sim.ALPHA_LIMIT,
    )
    shots: list[sim.Shot] = []
    last_shot_time = -100.0
    valid = low_speed = miss = 0
    steps = round(duration / sim.PLANNER_DT)

    for step_index in range(steps):
        now = (step_index + 1) * sim.PLANNER_DT
        controller.advance(target, now - sim.PLANNER_DT, sim.PLANNER_DT)
        plan = planner.plan(target, sim.BULLET_SPEED)
        if plan.fire and now - last_shot_time >= sim.SHOT_INTERVAL - 1e-12:
            shots.append(sim.shot_from_plan(plan, now))
            last_shot_time = now
        pending: list[sim.Shot] = []
        for shot in shots:
            if now < shot.impact_at:
                pending.append(shot)
                continue
            result = sim.evaluate_shot(shot, target)
            if result.outcome == "valid_hit":
                valid += 1
            elif result.outcome == "low_normal_speed":
                low_speed += 1
            else:
                miss += 1
        shots = pending
    return Score(valid, low_speed, miss)


def aggregate(policy: Policy, distances: list[float], duration: float, seeds: list[int]) -> Score:
    scores = [run(policy, distance, duration, seed) for distance in distances for seed in seeds]
    return Score(
        sum(score.valid for score in scores),
        sum(score.low_speed for score in scores),
        sum(score.miss for score in scores),
    )


def candidate_policies() -> list[Policy]:
    policies = [Policy("speed-limit reversal", None, None)]
    for dwell in (0.95, 1.00, 1.04, 1.07, 1.10, 1.14, 1.20, 1.30, 1.50, 1.80, 2.20, 2.60):
        policies.append(Policy(f"fixed {dwell:.2f}s", dwell, dwell))
    for low, high in (
        (1.25, 1.75),
        (1.35, 1.65),
        (1.40, 1.60),
        (1.45, 1.55),
        (1.30, 1.90),
    ):
        policies.append(Policy(f"random {low:.2f}-{high:.2f}s", low, high))
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--distances", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    args = parser.parse_args()

    results = []
    for policy in candidate_policies():
        seeds = [0] if policy.dwell_min == policy.dwell_max else args.seeds
        score = aggregate(policy, args.distances, args.duration, seeds)
        results.append((score.hit_rate, policy, score))
        print(
            f"{policy.name:24s} hit={score.hit_rate * 100:6.2f}% "
            f"valid={score.valid:4d} low={score.low_speed:4d} miss={score.miss:4d}"
        )
    best = min(results, key=lambda item: item[0])
    print(f"BEST {best[1].name}: {best[0] * 100:.2f}% ({best[2].resolved} resolved)")


if __name__ == "__main__":
    main()
