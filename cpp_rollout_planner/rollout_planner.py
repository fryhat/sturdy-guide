from __future__ import annotations

import ctypes
import math
from pathlib import Path


class RolloutPlanner:
    def __init__(self, dll_path: Path | None = None) -> None:
        dll_path = dll_path or Path(__file__).with_name("rollout_planner.dll")
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
        self._plan_debug = self._dll.sp_plan_rollout_debug
        self._plan_debug.argtypes = (
            ctypes.c_double,
            ctypes.c_double,
            double_pointer,
            ctypes.c_int,
            double_pointer,
            ctypes.c_int,
            double_pointer,
            double_pointer,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        )
        self._plan_debug.restype = ctypes.c_int
        self._build_curve_from_impact = self._dll.sp_build_curve_from_impact_center
        self._build_curve_from_impact.argtypes = (
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            double_pointer,
            ctypes.c_int,
            double_pointer,
            ctypes.c_int,
            double_pointer,
        )
        self._build_curve_from_impact.restype = ctypes.c_int
        self._hit_from_impact_limit = self._dll.sp_hit_probability_from_impact_limit_center
        self._hit_from_impact_limit.argtypes = (
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            double_pointer,
            ctypes.c_int,
            double_pointer,
        )
        self._hit_from_impact_limit.restype = ctypes.c_int

    def plan(
        self,
        yaw: float,
        omega: float,
        grid: list[float],
        curves: list[list[float]],
    ) -> tuple[float, float]:
        if len(grid) == 0 or any(len(curve) != len(grid) for curve in curves):
            raise ValueError("grid/curve shape mismatch")
        grid_array = (ctypes.c_double * len(grid))(*grid)
        curve_array = (ctypes.c_double * (len(curves) * len(grid)))(
            *(value for curve in curves for value in curve)
        )
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

    def plan_debug(
        self,
        yaw: float,
        omega: float,
        grid: list[float],
        curves: list[list[float]],
    ) -> tuple[float, float, int, int, int]:
        if len(grid) == 0 or any(len(curve) != len(grid) for curve in curves):
            raise ValueError("grid/curve shape mismatch")
        grid_array = (ctypes.c_double * len(grid))(*grid)
        curve_array = (ctypes.c_double * (len(curves) * len(grid)))(
            *(value for curve in curves for value in curve)
        )
        alpha = ctypes.c_double()
        score = ctypes.c_double()
        nodes = ctypes.c_int()
        edge_only = ctypes.c_int()
        full = ctypes.c_int()
        status = self._plan_debug(
            yaw,
            omega,
            grid_array,
            len(grid),
            curve_array,
            len(curves),
            ctypes.byref(alpha),
            ctypes.byref(score),
            ctypes.byref(nodes),
            ctypes.byref(edge_only),
            ctypes.byref(full),
        )
        if status != 0:
            raise RuntimeError(f"rollout debug planner failed with status {status}")
        return alpha.value, score.value, nodes.value, edge_only.value, full.value

    def build_curve_from_impact(
        self,
        impact_angle: float,
        impact_omega: float,
        horizon: float,
        omega_limit: float,
        alpha_limit: float,
        entries: list[tuple[float, float, int]],
        grid: list[float],
        center_x: float = 0.0,
        center_y: float = 6.0,
    ) -> list[float]:
        if len(grid) == 0:
            raise ValueError("empty yaw grid")
        entry_array = (ctypes.c_double * (len(entries) * 3))(
            *(value for entry in entries for value in entry)
        )
        grid_array = (ctypes.c_double * len(grid))(*grid)
        curve = (ctypes.c_double * len(grid))()
        status = self._build_curve_from_impact(
            impact_angle,
            impact_omega,
            center_x,
            center_y,
            horizon,
            omega_limit,
            alpha_limit,
            entry_array,
            len(entries),
            grid_array,
            len(grid),
            curve,
        )
        if status != 0:
            raise RuntimeError(f"impact curve builder failed with status {status}")
        return [curve[i] for i in range(len(grid))]

    def hit_probability_from_impact(
        self,
        impact_angle: float,
        impact_omega: float,
        horizon: float,
        launch_yaw: float,
        entries: list[tuple[float, float, int]],
        omega_limit: float,
        alpha_limit: float,
        center_x: float = 0.0,
        center_y: float = 6.0,
    ) -> float:
        entry_array = (ctypes.c_double * (len(entries) * 3))(
            *(value for entry in entries for value in entry)
        )
        probability = ctypes.c_double()
        status = self._hit_from_impact_limit(
            impact_angle,
            impact_omega,
            center_x,
            center_y,
            horizon,
            omega_limit,
            alpha_limit,
            launch_yaw,
            entry_array,
            len(entries),
            ctypes.byref(probability),
        )
        if status != 0:
            raise RuntimeError(f"impact hit query failed with status {status}")
        return probability.value


def synthetic_self_test() -> None:
    grid = [math.radians(88.0 + 0.1 * i) for i in range(41)]
    curves: list[list[float]] = []
    for step in range(6):
        center = math.radians(90.0 - 0.8 * step)
        sigma = math.radians(0.2)
        curves.append(
            [
                math.exp(-0.5 * ((yaw - center) / sigma) ** 2)
                for yaw in grid
            ]
        )
    planner = RolloutPlanner()
    alpha, score = planner.plan(math.radians(89.0), 0.0, grid, curves)
    assert abs(alpha) <= 50.0
    assert score > 0.0
    print(f"synthetic self-test ok: alpha={alpha:.3f}, score={score:.3f}")
    zeros = [[0.0 for _ in grid] for _ in range(6)]
    d_alpha, d_score, nodes, edge_only, full = planner.plan_debug(
        math.radians(89.0), 0.0, grid, zeros
    )
    assert d_alpha == 0.0 and d_score == 0.0
    assert nodes == 63
    assert edge_only == 63
    assert full == 0
    d_alpha2, d_score2, nodes2, edge_only2, full2 = planner.plan_debug(
        math.radians(89.0), 0.0, grid, curves
    )
    assert (d_alpha2, d_score2) == (alpha, score)
    assert nodes2 > nodes
    assert full2 >= 1
    print("zero-pruning self-test ok")
    recovery_grid = [
        math.pi / 2 + math.radians(-8.0 + 0.20 * i)
        for i in range(81)
    ]
    recovery_curves: list[list[float]] = []
    for _step in range(6):
        sigma = math.radians(1.0)
        recovery_curves.append(
            [
                math.exp(-0.5 * ((yaw - math.pi / 2) / sigma) ** 2)
                for yaw in recovery_grid
            ]
        )
    recovery_alpha, recovery_score = planner.plan(
        math.pi / 2 + math.radians(-4.8),
        0.0,
        recovery_grid,
        recovery_curves,
    )
    assert recovery_alpha >= 25.0
    assert recovery_score > 1.0
    print("positive recovery regression ok")


if __name__ == "__main__":
    synthetic_self_test()
