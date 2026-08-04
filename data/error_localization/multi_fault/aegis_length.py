"""Trace length adapter for AEGIS."""

from typing import List

from data.error_localization.multi_fault.aegis import dataset_path
from data.trace_length import Dataset


def contents(data: dict) -> List[str]:
    return [
        data["question"],
        *(step["content"] for step in data["trajectory"]),
        data.get("final_output", ""),
    ]


def steps(data: dict) -> int:
    return len(data["trajectory"])


DATASET = Dataset("aegis", dataset_path, contents, steps)
