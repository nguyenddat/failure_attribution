"""Download the AEGIS dataset (Fancylalala/AEGIS) into a single local folder.

All three splits are written to the same output directory, numbered
consecutively; the originating split is kept on each sample.

Keeps every raw field (schemas/aegis.py) — nothing is dropped, including
known data-quality quirks (num_agents/num_injected_agents mismatch,
per-step is_injected noise, is_injection_successful).

Usage: python -m data.error_localization.multi_fault.aegis
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from schemas.aegis import (
    AgentBehavior,
    Data,
    GroundTruth,
    InjectedAgent,
    Input,
    Metadata,
    Output,
    OutputFaultyAgent,
)

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
            is_injected=item["is_injected"],
        )
        for item in history
    ]


def build_injected_agents(injected: List[Dict[str, Any]]) -> List[InjectedAgent]:
    return [
        InjectedAgent(
            agent_name=agent["agent_name"],
            error_type=agent["error_type"],
            injection_strategy=agent["injection_strategy"],
            malicious_action_description=agent["malicious_action_description"] or "",
        )
        for agent in injected
    ]


def build_output_faulty_agents(faulty: List[Dict[str, Any]]) -> List[OutputFaultyAgent]:
    return [
        OutputFaultyAgent(
            agent_name=agent["agent_name"],
            error_type=agent["error_type"],
            injection_strategy=agent["injection_strategy"],
        )
        for agent in faulty
    ]


def row_to_data(row: pd.Series) -> Data:
    metadata = row["metadata"]
    input_ = row["input"]
    output = row["output"]
    ground_truth = row["ground_truth"]

    return Data(
        id=row["id"],
        split=row["split"],
        metadata=Metadata(
            framework=metadata["framework"],
            benchmark=metadata["benchmark"],
            model=metadata["model"],
            num_agents=metadata["num_agents"],
            num_injected_agents=metadata["num_injected_agents"],
            task_type=metadata["task_type"],
        ),
        input=Input(
            query=input_["query"],
            conversation_history=build_trajectory(input_["conversation_history"]),
            final_output=input_["final_output"] or "",
        ),
        output=Output(faulty_agents=build_output_faulty_agents(output["faulty_agents"])),
        ground_truth=GroundTruth(
            correct_answer=ground_truth["correct_answer"],
            injected_agents=build_injected_agents(ground_truth["injected_agents"]),
            is_injection_successful=ground_truth["is_injection_successful"],
        ),
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
