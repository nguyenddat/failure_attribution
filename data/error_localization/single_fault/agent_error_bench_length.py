"""Trace length distribution for AgentErrorBench.

Each step holds an environment observation and the agent action that answered
it, so both halves count toward the trace content.
"""

from pathlib import Path
from typing import List

from data.error_localization.single_fault.agent_error_bench import dataset_path
from data.trace_length import run

figures_dir = Path(__file__).resolve().parent / "figures"


def contents(data: dict) -> List[str]:
    parts = [data["question"]]
    for step in data["trajectory"]:
        parts.append(step["observation"])
        parts.append(step["action"])
    return parts


def steps(data: dict) -> int:
    return len(data["trajectory"])


if __name__ == "__main__":
    run("agent_error_bench", dataset_path, contents, steps, figures_dir)
