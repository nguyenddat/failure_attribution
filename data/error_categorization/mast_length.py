"""Trace length distribution for MAST.

MAST logs are unstructured: a sample carries the whole trace as one
``raw_trajectory`` string. The optional ``trajectory`` field is filled in later
by ``build_agent_behaviors.py``, so the step count is unavailable until that has
run and the step plot is skipped.
"""

from pathlib import Path
from typing import List, Optional

from data.error_categorization.mast import json_dir
from data.trace_length import run

figures_dir = Path(__file__).resolve().parent / "figures"


def contents(data: dict) -> List[str]:
    return [data["raw_trajectory"]]


def steps(data: dict) -> Optional[int]:
    trajectory = data.get("trajectory")
    return len(trajectory) if trajectory else None


if __name__ == "__main__":
    run("mast", json_dir, contents, steps, figures_dir)
