"""Build the AgentErrorBench dataset from the local files under ``data/``.

Source is the Google Drive release linked from ulab-uiuc/AgentDebug
(not on HF/GitHub — downloaded manually, no reliable scripted bulk-download
path: Google Drive rate-limits/blocks automated folder downloads). Expects
``data/Label/<env>_labels.json`` and
``data/Original_Failure_Trajectory/<Env>/<trajectory_id>.json`` to already
exist (unzip the 2 Drive folder exports into ``data/`` first).

Joins on ``trajectory_id`` (the trace file's stem). Keeps every raw field
(schemas/agent_error_bench.py) — nothing is dropped.

Usage: python -m data.error_localization.single_fault.agent_error_bench
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator

from schemas.agent_error_bench import Data, Label, Message, Metadata

base_dir = Path(__file__).resolve().parent
source_dir = base_dir.parent.parent  # data/

dataset_name = "agent_error_bench"
dataset_path = base_dir / dataset_name

label_dir = source_dir / "Label"
trajectory_dir = source_dir / "Original_Failure_Trajectory"

# Label file stem -> trace subfolder.
TASK_DIRS = {
    "alfworld": "ALFWorld",
    "gaia": "GAIA",
    "webshop": "WebShop",
}


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_data(label: Dict[str, Any], trace: Dict[str, Any]) -> Data:
    return Data(
        trajectory_id=label["trajectory_id"],
        messages=[Message(role=m["role"], content=m["content"]) for m in trace["messages"]],
        metadata=Metadata(**trace["metadata"]),
        label=Label(
            trajectory_id=label["trajectory_id"],
            LLM=label["LLM"],
            task_type=label["task_type"],
            critical_failure_step=label["critical_failure_step"],
            critical_failure_module=label["critical_failure_module"],
            step_annotations=label["step_annotations"],
        ),
    )


def iter_samples() -> Iterator[Data]:
    for task_type, folder in TASK_DIRS.items():
        labels = read_json(label_dir / f"{task_type}_labels.json")

        for label in labels:
            trace = read_json(trajectory_dir / folder / f"{label['trajectory_id']}.json")
            yield build_data(label, trace)


def load_data_path() -> Path:
    dataset_path.mkdir(parents=True, exist_ok=True)

    for index, data in enumerate(iter_samples()):
        file_path = dataset_path / f"{index}.json"
        if file_path.exists():
            continue

        file_path.write_text(json.dumps(data.model_dump(), ensure_ascii=False), encoding="utf-8")

    return dataset_path


if __name__ == "__main__":
    path = load_data_path()
    print(f"{len(list(path.glob('*.json')))} trajectories in {path}")
