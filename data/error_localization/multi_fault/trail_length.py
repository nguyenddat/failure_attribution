"""Trace length distribution for TRAIL.

TRAIL spans carry their text inside ``attributes`` (``llm.*``, ``tool.*``,
``input.*``, ``output.*``), ``logs`` and ``events`` with no single content
field, so those three containers are serialized per span. The span tree is
nested, so both the content and the step count walk ``child_spans``.
"""

import json
from pathlib import Path
from typing import List

from data.error_localization.multi_fault.trail import dataset_path, flatten_spans
from data.trace_length import run

figures_dir = Path(__file__).resolve().parent / "figures"


def contents(data: dict) -> List[str]:
    parts = [data["question"]]
    for span in flatten_spans(data["spans"]):
        parts.append(
            json.dumps(
                [span["attributes"], span["logs"], span["events"]],
                ensure_ascii=False,
            )
        )
    return parts


def steps(data: dict) -> int:
    return sum(1 for _ in flatten_spans(data["spans"]))


if __name__ == "__main__":
    run("trail", dataset_path, contents, steps, figures_dir)
