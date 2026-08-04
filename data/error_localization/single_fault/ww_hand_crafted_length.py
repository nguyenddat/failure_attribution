"""Trace length adapter for who&when hand-crafted."""

from typing import List

from data.error_localization.single_fault.ww_hand_crafted import dataset_path
from data.trace_length import Dataset


def contents(data: dict) -> List[str]:
    return [data["question"], *(step["content"] for step in data["trajectory"])]


def steps(data: dict) -> int:
    return len(data["trajectory"])


DATASET = Dataset("who&when hand-crafted", dataset_path, contents, steps)
