from __future__ import annotations

import html
import math
import subprocess
from pathlib import Path

from rollout_benchmark import RolloutBenchmark


def run_trace() -> list[dict[str, float | bool]]:
    runner = RolloutBenchmark(seconds=3.0, dt=0.010)
    for _ in range(int(3.0 / 0.010)):
        runner._step(0.010)
    return runner.trace


def make_svg(trace: list[dict[str, float | bool]]) -> str:
    min_time = 0.0
    max_time = max(row["time"] for row in trace) if trace else 1.0
    width = 1200
    prob_height = 300
    yaw_height = 180
    top = 80
    left = 90
    plot_width = width - left - 50

    plan_rows = [row for row in trace if row["kind"] == "plan"]
    gate_rows = [row for row in trace if row["kind"] == "gate"]
    fire_rows = [
        row
        for row in trace
        if row["kind"] == "gate" and row["probability"] >= 0.40
    ]

    def x_for(value: float) -> float:
        return left + plot_width * (value - min_time) / max(1e-9, max_time - min_time)

    def prob_y(value: float) -> float:
        return top + prob_height * (1.0 - value)

    def yaw_y(value: float, center: float) -> float:
        return top + prob_height + 40 + yaw_height * (
            0.5 - (value - center) / 0.2
        )

    points_plan = " ".join(
        f"{x_for(row['time']):.1f},{prob_y(float(row['probability'])):.1f}"
        for row in plan_rows
    )
    points_gate = " ".join(
        f"{x_for(row['time']):.1f},{prob_y(float(row['probability'])):.1f}"
        for row in gate_rows
    )
    gimbal_points = " ".join(
        f"{x_for(row['time']):.1f},{yaw_y(float(row['launch_yaw']), math.pi / 2):.1f}"
        for row in gate_rows
    )
    plan_yaw_points = " ".join(
        f"{x_for(row['time']):.1f},{yaw_y(float(row['launch_yaw']), math.pi / 2):.1f}"
        for row in plan_rows
    )
    fire_marks = "".join(
        f'<circle cx="{x_for(row["time"]):.1f}" cy="{prob_y(float(row["probability"])):.1f}" r="5" fill="#16a34a"/>'
        for row in fire_rows
    )

    return f"""<svg width="{width}" height="650" viewBox="0 0 {width} 650" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="650" fill="#ffffff"/>
  <text x="24" y="30" font-family="Consolas" font-size="18" fill="#111">Independent rollout problem trace</text>
  <line x1="{left}" y1="{prob_y(0.40)}" x2="{width - 50}" y2="{prob_y(0.40)}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4"/>
  <text x="{width - 110}" y="{prob_y(0.40) - 6}" font-family="Consolas" font-size="12" fill="#dc2626">threshold 0.40</text>
  <polyline points="{points_plan}" fill="none" stroke="#7c3aed" stroke-width="2"/>
  <polyline points="{points_gate}" fill="none" stroke="#2563eb" stroke-width="2"/>
  {fire_marks}
  <text x="{left}" y="{prob_y(1.0) - 6}" font-family="Consolas" font-size="13" fill="#7c3aed">plan-point P</text>
  <text x="{left + 110}" y="{prob_y(1.0) - 6}" font-family="Consolas" font-size="13" fill="#2563eb">gate P</text>
  <text x="{left}" y="{prob_y(0.0) - 6}" font-family="Consolas" font-size="13" fill="#16a34a">green = fired</text>
  <line x1="{left}" y1="{top + prob_height}" x2="{width - 50}" y2="{top + prob_height}" stroke="#333"/>
  <text x="{left}" y="{top + prob_height + 45}" font-family="Consolas" font-size="14" fill="#333">yaw trace around pi/2</text>
  <polyline points="{gimbal_points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  <polyline points="{plan_yaw_points}" fill="none" stroke="#7c3aed" stroke-width="2"/>
  <text x="{left}" y="{top + prob_height + 70}" font-family="Consolas" font-size="13" fill="#7c3aed">plan launch yaw</text>
  <text x="{left + 130}" y="{top + prob_height + 70}" font-family="Consolas" font-size="13" fill="#2563eb">actual gate launch yaw</text>
  <text x="{left}" y="{top + prob_height + 170}" font-family="Consolas" font-size="12" fill="#666">Trace summary: plan rows={len(plan_rows)}, gate rows={len(gate_rows)}, P>=0.40 rows={len(fire_rows)}</text>
</svg>"""


def main() -> None:
    trace = run_trace()
    svg = make_svg(trace)
    html_path = Path(__file__).with_name("rollout_problem.html")
    png_path = Path(__file__).with_name("rollout_problem.png")
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'>" + svg,
        encoding="utf-8",
    )
    edge = Path(
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    )
    if edge.exists():
        subprocess.run(
            [
                str(edge),
                "--headless",
                "--disable-gpu",
                f"--screenshot={png_path}",
                "--window-size=1300,700",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    print(png_path)


if __name__ == "__main__":
    main()
