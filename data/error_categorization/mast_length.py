"""Trace length adapter for MAST.

MAST logs are unstructured: a sample carries the whole trace as one
``raw_trajectory`` string. The optional ``trajectory`` field is filled in later
by ``build_agent_behaviors.py``, so until that has run the step count is
unavailable and MAST is left out of the step plot.
"""

from typing import List, Optional

from data.error_categorization.mast import json_dir
from data.trace_length import Dataset


def contents(data: dict) -> List[str]:
    return [data["raw_trajectory"]]


def steps(data: dict) -> Optional[int]:
    trajectory = data.get("trajectory")
    return len(trajectory) if trajectory else None


DATASET = Dataset("mast", json_dir, contents, steps)
