from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "3.trace_length_performance_cliff"
    / "src"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
