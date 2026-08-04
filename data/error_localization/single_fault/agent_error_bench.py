"""Build the AgentErrorBench dataset from the local files shipped in ``data/``.

Unlike the other loaders this one downloads nothing: it joins the annotations in
``data/Label/*_labels.json`` with the raw traces in
``data/Original_Failure_Trajectory/<Environment>/`` on ``trajectory_id``, which
is the trace file's stem.

Only the fields needed for error localization are written out; see
``schemas/agent_error_bench.py`` for the resulting shape and what gets dropped.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List

from schemas.agent_error_bench import (
    Data,
    Failure,
    Step,
    normalize_failure_type,
    normalize_module,
)

base_dir = Path(__file__).resolve().parent
source_dir = base_dir.parent.parent

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

# The task statement is embedded in the opening prompt; the wording differs per
# environment ("Your task is:" for ALFWorld/WebShop, "Task:" for GAIA).
QUESTION = re.compile(
    r"(?:Your task is:|Task:)\s*(.+?)(?:\n\n|\nYour current observation|\Z)",
    re.S,
)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_question(prompt: str) -> str:
    match = QUESTION.search(prompt)
    return match.group(1).strip() if match else ""


def build_trajectory(messages: List[Dict[str, str]]) -> List[Step]:
    """Pair each environment observation with the action that answered it.

    Roles alternate user/assistant on every trajectory, so pairing by position
    is exact and the resulting index matches the annotated step number.
    """
    return [
        Step(
            step=index,
            observation=messages[2 * (index - 1)]["content"],
            action=messages[2 * (index - 1) + 1]["content"],
        )
        for index in range(1, len(messages) // 2 + 1)
    ]


def build_failure(annotation: Dict[str, Any]) -> Failure:
    module = next(key for key in annotation if key != "step")
    detail = annotation[module]
    return Failure(
        step=annotation["step"],
        module=normalize_module(module),
        failure_type=normalize_failure_type(detail.get("failure_type") or ""),
        reasoning=(detail.get("reasoning") or "").strip(),
    )


def iter_samples() -> Iterator[Data]:
    for task_type, folder in TASK_DIRS.items():
        labels = read_json(label_dir / f"{task_type}_labels.json")

        for label in labels:
            trace = read_json(trajectory_dir / folder / f"{label['trajectory_id']}.json")
            messages = trace["messages"]

            yield Data(
                question=extract_question(messages[0]["content"]),
                task_type=task_type,
                model=label["LLM"],
                trajectory=build_trajectory(messages),
                failure=build_failure(label["step_annotations"][0]),
            )


def load_data_path() -> Path:
    dataset_path.mkdir(parents=True, exist_ok=True)

    for index, data in enumerate(iter_samples()):
        file_path = dataset_path / f"{index}.json"
        if file_path.exists():
            continue

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data.model_dump(), file, ensure_ascii=False)

    return dataset_path


if __name__ == "__main__":
    path = load_data_path()
    print(f"{len(list(path.glob('*.json')))} trajectories in {path}")
