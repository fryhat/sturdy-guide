"""RM2022 balance-infantry armor rotation visualizer.

Geometry comes from the RM2022 open-source package in this project. Missing
rotation limits are explicitly estimated from the published size and motion
figures; they are not presented as measured robot-controller limits.
"""

from __future__ import annotations

import math
import tkinter as tk
import time
from tkinter import ttk


BODY_WIDTH_MM = 510.0
BODY_LENGTH_MM = 600.0
BODY_HEIGHT_MM = 490.0
EXPANDED_HEIGHT_MM = 725.0
ARMOR_WIDTH_MM = 140.0
ARMOR_HEIGHT_MM = 125.0
ARMOR_DRAW_THICKNESS_MM = 16.0  # Only for a legible top-view drawing.

MAX_LINEAR_SPEED_MPS = 3.0
TIME_TO_2_MPS_S = 1.6
EFFECTIVE_RADIUS_RANGE_M = (BODY_WIDTH_MM / 2000.0, BODY_LENGTH_MM / 2000.0)
OMEGA_ESTIMATE_RANGE_DPS = tuple(
    math.degrees(MAX_LINEAR_SPEED_MPS / radius)
    for radius in reversed(EFFECTIVE_RADIUS_RANGE_M)
)
ALPHA_ESTIMATE_RANGE_DPS2 = tuple(
    math.degrees((2.0 / TIME_TO_2_MPS_S) / radius)
    for radius in reversed(EFFECTIVE_RADIUS_RANGE_M)
)

OMEGA_LIMIT_DPS = 600.0
ALPHA_LIMIT_DPS2 = 280.0
INITIAL_OMEGA_DPS = 300.0


def rotate_point(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return x * cosine - y * sine, x * sine + y * cosine


class ArmorVisualizer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RM2022 平衡步兵 - 装甲板旋转可视化")
        self.geometry("980x760")
        self.minsize(720, 620)

        self.acceleration = tk.DoubleVar(value=-ALPHA_LIMIT_DPS2)
        self.running = True
        self.omega = INITIAL_OMEGA_DPS
        self.theta = 0.0
        self.elapsed = 0.0
        self.last_tick: float | None = None

        self._build_ui()
        self.bind("<space>", lambda _event: self.toggle_running())
        self.bind("<r>", lambda _event: self.reset_motion())
        self.after(16, self._tick)

    def _build_ui(self) -> None:
        heading = ttk.Label(
            self,
            text="RM2022 平衡步兵装甲板旋转演示",
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        heading.pack(anchor="w", padx=18, pady=(14, 3))

        source = ttk.Label(
            self,
            wraplength=930,
            justify="left",
            text=(
                "文件数据：初始尺寸 510×600×490 mm，展开高度 725 mm，"
                "最大移动速度 3.0 m/s，2 m/s 加速时间 1.6 s"
            ),
        )
        source.pack(anchor="w", padx=18, pady=(0, 9))

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=18)
        ttk.Label(controls, text="角加速度：").pack(side="left")
        acceleration_box = ttk.Combobox(
            controls,
            state="readonly",
            width=25,
            values=(
                "-280°/s²（减速后反向）",
                "0°/s²（匀速）",
                "+280°/s²（加速）",
            ),
        )
        acceleration_box.current(0)
        acceleration_box.pack(side="left", padx=(0, 12))
        acceleration_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._select_acceleration(acceleration_box.current()),
        )
        self.pause_button = ttk.Button(
            controls, text="暂停", command=self.toggle_running
        )
        self.pause_button.pack(side="left", padx=4)
        ttk.Button(controls, text="重置", command=self.reset_motion).pack(
            side="left", padx=4
        )

        self.status = ttk.Label(
            self,
            text="",
            font=("Consolas", 11),
        )
        self.status.pack(anchor="w", padx=18, pady=8)

        self.canvas = tk.Canvas(
            self,
            background="#f4f6f8",
            highlightthickness=1,
            highlightbackground="#aeb6bf",
        )
        self.canvas.pack(fill="both", expand=True, padx=18, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _event: self.draw_scene())

        evidence = ttk.Label(
            self,
            wraplength=930,
            justify="left",
            text=(
                "官方小装甲 AM01：140×125 mm。估算：有效转动半径 0.255–0.300 m，"
                f"角速度 {OMEGA_ESTIMATE_RANGE_DPS[0]:.0f}–{OMEGA_ESTIMATE_RANGE_DPS[1]:.0f}°/s，"
                f"角加速度 {ALPHA_ESTIMATE_RANGE_DPS2[0]:.0f}–{ALPHA_ESTIMATE_RANGE_DPS2[1]:.0f}°/s²；"
                "程序采用 |ω|≤600°/s、|α|≤280°/s²。角速度允许穿过 0 反向。"
            ),
        )
        evidence.pack(fill="x", padx=18, pady=(0, 12))

    def _select_acceleration(self, selection: int) -> None:
        self.acceleration.set((-ALPHA_LIMIT_DPS2, 0.0, ALPHA_LIMIT_DPS2)[selection])
        self.reset_motion()

    def toggle_running(self) -> None:
        self.running = not self.running
        self.pause_button.configure(text="暂停" if self.running else "继续")
        self.last_tick = None

    def reset_motion(self) -> None:
        self.omega = INITIAL_OMEGA_DPS
        self.theta = 0.0
        self.elapsed = 0.0
        self.last_tick = None
        self.running = True
        self.pause_button.configure(text="暂停")
        self.draw_scene()

    def _tick(self) -> None:
        now = time.perf_counter()
        if self.last_tick is None:
            self.last_tick = now
        dt = min(max(now - self.last_tick, 0.0), 0.05)
        self.last_tick = now

        if self.running:
            acceleration = self.acceleration.get()
            next_omega = max(
                -OMEGA_LIMIT_DPS,
                min(OMEGA_LIMIT_DPS, self.omega + acceleration * dt),
            )
            self.theta += (self.omega + next_omega) * 0.5 * dt
            self.omega = next_omega
            self.elapsed += dt
            self.draw_scene()
        self.after(16, self._tick)

    def draw_scene(self) -> None:
        canvas = self.canvas
        width = max(canvas.winfo_width(), 640)
        height = max(canvas.winfo_height(), 390)
        canvas.delete("all")

        center_x = width * 0.5
        center_y = height * 0.50
        scale = min((width - 210) / 760.0, (height - 100) / 760.0)
        body_w = BODY_WIDTH_MM * scale
        body_h = BODY_LENGTH_MM * scale
        armor_w = ARMOR_WIDTH_MM * scale
        armor_t = max(8.0, ARMOR_DRAW_THICKNESS_MM * scale)
        theta_rad = math.radians(self.theta)

        def screen_polygon(points: list[tuple[float, float]]) -> list[float]:
            output: list[float] = []
            for x, y in points:
                rotated_x, rotated_y = rotate_point(x, y, theta_rad)
                output.extend((center_x + rotated_x, center_y + rotated_y))
            return output

        canvas.create_line(center_x, 12, center_x, height - 12, fill="#c5cbd1", dash=(4, 4))
        canvas.create_line(12, center_y, width - 12, center_y, fill="#c5cbd1", dash=(4, 4))

        body = [
            (-body_w / 2, -body_h / 2),
            (body_w / 2, -body_h / 2),
            (body_w / 2, body_h / 2),
            (-body_w / 2, body_h / 2),
        ]
        canvas.create_polygon(
            screen_polygon(body), fill="#dfe5e9", outline="#263238", width=2
        )

        armor_rectangles = [
            [(-armor_w / 2, -body_h / 2 - armor_t / 2), (armor_w / 2, -body_h / 2 - armor_t / 2), (armor_w / 2, -body_h / 2 + armor_t / 2), (-armor_w / 2, -body_h / 2 + armor_t / 2)],
            [(body_w / 2 - armor_t / 2, -armor_w / 2), (body_w / 2 + armor_t / 2, -armor_w / 2), (body_w / 2 + armor_t / 2, armor_w / 2), (body_w / 2 - armor_t / 2, armor_w / 2)],
            [(-armor_w / 2, body_h / 2 - armor_t / 2), (armor_w / 2, body_h / 2 - armor_t / 2), (armor_w / 2, body_h / 2 + armor_t / 2), (-armor_w / 2, body_h / 2 + armor_t / 2)],
            [(-body_w / 2 - armor_t / 2, -armor_w / 2), (-body_w / 2 + armor_t / 2, -armor_w / 2), (-body_w / 2 + armor_t / 2, armor_w / 2), (-body_w / 2 - armor_t / 2, armor_w / 2)],
        ]
        for armor in armor_rectangles:
            canvas.create_polygon(
                screen_polygon(armor), fill="#1976d2", outline="#0d47a1", width=1
            )

        front_tip = screen_polygon([(0, -body_h / 2 + 12), (-9, -body_h / 2 + 30), (9, -body_h / 2 + 30)])
        canvas.create_polygon(front_tip, fill="#f57c00", outline="")
        canvas.create_text(center_x, center_y, text="车体中心", fill="#263238")

        direction = "逆时针" if self.omega > 0 else "顺时针" if self.omega < 0 else "静止"
        arrow_radius = max(body_w, body_h) * 0.56
        start = 25 if self.omega >= 0 else 205
        extent = 125 if self.omega >= 0 else -125
        canvas.create_arc(
            center_x - arrow_radius,
            center_y - arrow_radius,
            center_x + arrow_radius,
            center_y + arrow_radius,
            start=start,
            extent=extent,
            style="arc",
            outline="#f57c00",
            width=3,
        )
        arrow_angle = math.radians(start + extent)
        arrow_x = center_x + arrow_radius * math.cos(arrow_angle)
        arrow_y = center_y - arrow_radius * math.sin(arrow_angle)
        tangent_angle = arrow_angle + (-math.pi / 2 if extent > 0 else math.pi / 2)
        arrow_size = 11.0
        arrow_points = [
            (arrow_x, arrow_y),
            (
                arrow_x - arrow_size * math.cos(tangent_angle - 0.55),
                arrow_y + arrow_size * math.sin(tangent_angle - 0.55),
            ),
            (
                arrow_x - arrow_size * math.cos(tangent_angle + 0.55),
                arrow_y + arrow_size * math.sin(tangent_angle + 0.55),
            ),
        ]
        canvas.create_polygon(
            [coordinate for point in arrow_points for coordinate in point],
            fill="#f57c00",
            outline="",
        )
        canvas.create_text(
            12,
            12,
            anchor="nw",
            text=f"旋转方向：{direction}",
            fill="#263238",
            font=("Microsoft YaHei UI", 10, "bold"),
        )

        self.status.configure(
            text=(
                f"t = {self.elapsed:6.2f} s    "
                f"ω = {self.omega:+7.1f}°/s    "
                f"α = {self.acceleration.get():+6.0f}°/s²    "
                f"θ = {self.theta:+9.1f}°"
            )
        )


if __name__ == "__main__":
    ArmorVisualizer().mainloop()
