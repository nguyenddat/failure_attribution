"""Download the AEGIS dataset (Fancylalala/AEGIS) into a single local folder.

All three splits are written to the same output directory, numbered
consecutively; the originating split is kept on each sample.

Only the fields needed for error localization are written out; see
``schemas/aegis.py`` for the resulting shape and what gets dropped.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from schemas.aegis import AgentBehavior, Data, FaultyAgent

base_dir = Path(__file__).resolve().parent

dataset_name = "aegis"
dataset_repo = "Fancylalala/AEGIS"
dataset_path = base_dir / dataset_name


def build_trajectory(history: List[Dict[str, Any]]) -> List[AgentBehavior]:
    return [
        AgentBehavior(
            step=item["step"],
            agent_name=item["agent_name"],
            content=item["content"] or "",
            phase=item["phase"] or "",
        )
        for item in history
    ]


def build_faulty_agents(injected: List[Dict[str, Any]]) -> List[FaultyAgent]:
    return [
        FaultyAgent(
            agent_name=agent["agent_name"],
            error_type=agent["error_type"],
            injection_strategy=agent["injection_strategy"],
            description=agent["malicious_action_description"] or "",
        )
        for agent in injected
    ]


def row_to_data(row: pd.Series) -> Data:
    metadata = row["metadata"]
    input_ = row["input"]
    ground_truth = row["ground_truth"]

    return Data(
        question=input_["query"],
        framework=metadata["framework"],
        benchmark=metadata["benchmark"],
        task_type=metadata["task_type"],
        split=row["split"],
        trajectory=build_trajectory(input_["conversation_history"]),
        final_output=input_["final_output"] or "",
        correct_answer=ground_truth["correct_answer"],
        faulty_agents=build_faulty_agents(ground_truth["injected_agents"]),
    )


def load_dataframe() -> pd.DataFrame:
    ds = load_dataset(dataset_repo)

    frames = []
    for split_name, split_ds in ds.items():
        frame = split_ds.to_pandas()
        frame["split"] = split_name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def load_data_path() -> Path:
    dataset_path.mkdir(parents=True, exist_ok=True)

    for index, row in load_dataframe().iterrows():
        file_path = dataset_path / f"{index}.json"
        if file_path.exists():
            continue

        data = row_to_data(row)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data.model_dump(), file, ensure_ascii=False)

    return dataset_path


if __name__ == "__main__":
    path = load_data_path()
    print(f"{len(list(path.glob('*.json')))} traces in {path}")
