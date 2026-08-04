"""Trace length distribution for AEGIS."""

from pathlib import Path
from typing import List

from data.error_localization.multi_fault.aegis import dataset_path
from data.trace_length import run

figures_dir = Path(__file__).resolve().parent / "figures"


def contents(data: dict) -> List[str]:
    return [
        data["question"],
        *(step["content"] for step in data["trajectory"]),
        data.get("final_output", ""),
    ]


def steps(data: dict) -> int:
    return len(data["trajectory"])


if __name__ == "__main__":
    run("aegis", dataset_path, contents, steps, figures_dir)
