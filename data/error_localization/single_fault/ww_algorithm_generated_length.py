"""Trace length adapter for who&when algorithm-generated."""

from typing import List

from data.error_localization.single_fault.ww_algorithm_generated import dataset_path
from data.trace_length import Dataset


def contents(data: dict) -> List[str]:
    return [data["question"], *(step["content"] for step in data["trajectory"])]


def steps(data: dict) -> int:
    return len(data["trajectory"])


DATASET = Dataset("who&when algorithm-generated", dataset_path, contents, steps)
