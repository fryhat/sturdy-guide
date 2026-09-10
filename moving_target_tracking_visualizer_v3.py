"""Live A/B view for the realistic v3 tracking experiment.

This visualizer has no firing or ballistic logic. It compares marker coverage
for the constant-velocity baseline and the v3 bounded historical predictor.
"""

from __future__ import annotations

import math
import random
import tkinter as tk
from dataclasses import replace

import moving_target_tracking_experiment_v3 as model

DISPLAY_YAW_OFFSET = -math.pi / 2.0


class Panel:
    def __init__(self, canvas: tk.Canvas, mode: str, cfg: model.Config) -> None:
        self.canvas = canvas
        self.mode = mode
        self.cfg = cfg
        self.executor = model.Executor(cfg, "constrained")
        self.selector = model.BranchSelector(cfg) if mode == "historical" else None
        self.plan: model.Plan | None = None
        self.history: list[tuple[float, model.State]] = []
        self.events: list[tuple[float, float, model.State]] = []
        self.next_event = cfg.warmup
        self.covered = 0
        self.total = 0
        self.last_plan = -1.0

    def plan_step(self, now: float, observed: model.State, true_state: model.State, direction: float) -> None:
        if self.plan is not None and now - self.last_plan < self.cfg.plan_period - 1e-12:
            return
        residual = 0.0
        self.plan = model.make_plan(
            self.mode, observed, true_state, direction, self.history,
            self.executor, self.selector, residual, self.cfg,
        )
        self.last_plan = now

    def step_executor(self, dt: float) -> None:
        if self.plan is not None:
            self.executor.step(self.plan.reference_yaw, self.plan.reference_velocity, self.plan.reference_acceleration, dt)

    def evaluate(self, now: float, target_history: list[tuple[float, model.State]], rng: random.Random) -> None:
        if self.plan is None:
            return
        if now >= self.next_event:
            self.events.append((self.next_event + self.cfg.delay, self.plan.reference_yaw, self.plan))
            self.next_event += self.cfg.plan_period
        ready = [event for event in self.events if event[0] <= now + 1e-12]
        self.events = [event for event in self.events if event[0] > now + 1e-12]
        for evaluate_at, _, _ in ready:
            future = model.interpolate_state(target_history, evaluate_at)
            probability = model.coverage_probability(self.executor.state.yaw, future.angle, self.cfg)
            self.total += 1
            self.covered += rng.random() < probability

    def draw(self, target: model.State, width: int, height: int) -> None:
        c = self.canvas
        c.delete("all")
        cx, cy = width // 2, height // 2
        radius = min(width, height) * 0.34
        c.create_rectangle(28, 28, width - 28, height - 28, outline="#9aa3aa")
        for grid in range(-3, 4):
            gx = cx + grid * radius / 3.0
            gy = cy - grid * radius / 3.0
            c.create_line(gx, 28, gx, height - 28, fill="#d8dde1", dash=(3, 5))
            c.create_line(28, gy, width - 28, gy, fill="#d8dde1", dash=(3, 5))
        # The target center is fixed in front of the observer.  ``target.angle``
        # is its self-rotation phase, not an orbital position around the
        # observer.
        tx, ty = cx, cy - radius
        body_w, body_h = 34.0, 46.0
        body_points = []
        for lx, ly in ((-body_w / 2, -body_h / 2), (body_w / 2, -body_h / 2), (body_w / 2, body_h / 2), (-body_w / 2, body_h / 2)):
            dx = lx * math.cos(target.angle) - ly * math.sin(target.angle)
            dy = lx * math.sin(target.angle) + ly * math.cos(target.angle)
            body_points.extend((tx + dx, ty - dy))
        c.create_polygon(body_points, fill="#dfe5e9", outline="#263238", width=2)
        marker_dx = (body_h / 2.0) * math.sin(target.angle)
        marker_dy = (body_h / 2.0) * math.cos(target.angle)
        c.create_oval(tx + marker_dx - 6, ty + marker_dy - 6, tx + marker_dx + 6, ty + marker_dy + 6, fill="#1565c0", outline="")
        display_executor_yaw = self.executor.state.yaw + DISPLAY_YAW_OFFSET
        ex = cx + radius * math.cos(display_executor_yaw)
        ey = cy - radius * math.sin(display_executor_yaw)
        c.create_line(cx, cy, ex, ey, fill="#ef6c00", width=3)
        if self.plan is not None:
            display_reference_yaw = self.plan.reference_yaw + DISPLAY_YAW_OFFSET
            px = cx + radius * math.cos(display_reference_yaw)
            py = cy - radius * math.sin(display_reference_yaw)
            c.create_line(cx, cy, px, py, fill="#7b1fa2", width=2, dash=(5, 3))
        rate = self.covered / self.total if self.total else 0.0
        c.create_text(12, 14, anchor="nw", text=f"覆盖率 {rate * 100:5.1f}%   覆盖 {self.covered}/{self.total}   覆盖/s {self.covered / max(1e-9, self.cfg.duration):.2f}", font=("Consolas", 11))
        c.create_text(12, 34, anchor="nw", text=f"实际云台 {math.degrees(model.wrap_error(self.executor.state.yaw, 0.0)):7.2f}°", fill="#ef6c00")


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("v3 真实跟踪对比：恒速度基线 vs 历史概率 ±10°")
        self.geometry("1280x650")
        self.cfg = model.Config(delay=0.015, duration=60.0, warmup=5.0, max_prediction_correction=math.radians(10.0))
        self.running = True
        self.sim_time = 0.0
        self.last_clock: float | None = None
        self.accumulator = 0.0
        self.target = model.State(0.0, math.radians(300.0), self.cfg.max_alpha)
        self.direction = 1.0
        self.target_history: list[tuple[float, model.State]] = [(0.0, self.target)]
        self.rng = random.Random(self.cfg.seed)
        top = tk.Frame(self)
        top.pack(fill="x", padx=12, pady=8)
        tk.Label(top, text="蓝点=移动标记，橙线=实际执行器，紫虚线=参考方向；两侧共用同一目标轨迹和噪声", font=("Microsoft YaHei UI", 11)).pack(side="left")
        self.status = tk.Label(top, text="", font=("Consolas", 10))
        self.status.pack(side="right")
        body = tk.Frame(self)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        left = tk.Frame(body); left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = tk.Frame(body); right.pack(side="left", fill="both", expand=True, padx=(6, 0))
        tk.Label(left, text="恒速度外推基线").pack()
        tk.Label(right, text="历史加速度概率预测（修正限制 ±10°）").pack()
        self.left_canvas = tk.Canvas(left, background="#f4f6f8", highlightthickness=1, highlightbackground="#aeb6bf")
        self.right_canvas = tk.Canvas(right, background="#f4f6f8", highlightthickness=1, highlightbackground="#aeb6bf")
        self.left_canvas.pack(fill="both", expand=True); self.right_canvas.pack(fill="both", expand=True)
        self.panels = [Panel(self.left_canvas, "constant_velocity", self.cfg), Panel(self.right_canvas, "historical", self.cfg)]
        self.after(10, self.tick)

    def tick(self) -> None:
        self.accumulator += 0.01 * 0.1
        while self.accumulator >= self.cfg.dt:
            self.accumulator -= self.cfg.dt
            self.sim_time += self.cfg.dt
            self.target, self.direction = model.advance_target(self.target, self.direction, self.cfg.dt, self.cfg)
            self.target_history.append((self.sim_time, self.target))
            observed = model.State(self.target.angle + self.rng.gauss(0, self.cfg.observation_angle_sigma), self.target.omega + self.rng.gauss(0, self.cfg.observation_omega_sigma), self.target.alpha)
            for panel in self.panels:
                panel.history.append((self.sim_time, observed))
                panel.plan_step(self.sim_time, observed, self.target, self.direction)
                panel.step_executor(self.cfg.dt)
                panel.evaluate(self.sim_time, self.target_history, self.rng)
        self.left_canvas.update_idletasks(); self.right_canvas.update_idletasks()
        self.panels[0].draw(self.target, max(self.left_canvas.winfo_width(), 500), max(self.left_canvas.winfo_height(), 400))
        self.panels[1].draw(self.target, max(self.right_canvas.winfo_width(), 500), max(self.right_canvas.winfo_height(), 400))
        self.status.configure(text=f"t={self.sim_time:6.2f}s   延迟={self.cfg.delay * 1000:.0f}ms   评估频率=20/s")
        self.after(10, self.tick)


if __name__ == "__main__":
    App().mainloop()
