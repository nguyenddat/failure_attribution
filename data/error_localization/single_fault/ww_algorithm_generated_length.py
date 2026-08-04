"""Trace length distribution for who&when algorithm-generated."""

from pathlib import Path
from typing import List

from data.error_localization.single_fault.ww_algorithm_generated import dataset_path
from data.trace_length import run

figures_dir = Path(__file__).resolve().parent / "figures"


def contents(data: dict) -> List[str]:
    return [data["question"], *(step["content"] for step in data["trajectory"])]


def steps(data: dict) -> int:
    return len(data["trajectory"])


if __name__ == "__main__":
    run("who_and_when__algorithm-generated", dataset_path, contents, steps, figures_dir)
