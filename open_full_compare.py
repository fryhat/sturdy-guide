"""Open the two complete, physically shared-model visualizers side by side."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROGRAM = Path(__file__).with_name("sp_vision_moving_target_visualizer_full_compare.py")


def main() -> None:
    children = [
        subprocess.Popen([sys.executable, str(PROGRAM), "--mode", "current"]),
        subprocess.Popen([sys.executable, str(PROGRAM), "--mode", "probabilistic"]),
    ]
    try:
        for child in children:
            child.wait()
    except KeyboardInterrupt:
        for child in children:
            child.terminate()


if __name__ == "__main__":
    main()
