"""Loader for the Who&When Hand-Crafted split (Kevin355/Who_and_When).

Keeps every raw column — nothing is dropped (see schemas/who_and_when.py
HandCraftedData). Raw ``history`` items use key "role" (not "name" like
Algorithm-Generated) for the agent identifier.

Usage: python -m data.error_localization.single_fault.ww_hand_crafted
"""

import json
from pathlib import Path
from typing import Any, List

import pandas as pd

from schemas.who_and_when import AgentBehavior, HandCraftedData

base_dir = Path(__file__).resolve().parent

dataset_name = "who_and_when__hand-crafted"
dataset_link = "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet"
dataset_path = base_dir / dataset_name


def history_to_trajectory(history: Any) -> List[AgentBehavior]:
    return [
        AgentBehavior(step=i, agent_name=item.get("role"), content=item.get("content"))
        for i, item in enumerate(history)
    ]


def load_dataframe() -> pd.DataFrame:
    return pd.read_parquet(dataset_link)


def row_to_data(row: pd.Series) -> HandCraftedData:
    mistake_type = row["mistake_type"]
    return HandCraftedData(
        question=row["question"],
        question_id=row["question_ID"],
        groundtruth=row["groundtruth"],
        is_corrected=bool(row["is_corrected"]),
        trajectory=history_to_trajectory(row["history"]),
        mistake_step=int(row["mistake_step"]),
        mistake_agent=row["mistake_agent"],
        mistake_reason=row["mistake_reason"] or "",
        mistake_type=None if pd.isna(mistake_type) else mistake_type,
    )


def load_data_path() -> Path:
    dataset_path.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()

    for i, row in df.iterrows():
        file_path = dataset_path / f"{i}.json"
        if file_path.exists():
            continue

        data = row_to_data(row)
        file_path.write_text(json.dumps(data.model_dump(), ensure_ascii=False), encoding="utf-8")

    return dataset_path


if __name__ == "__main__":
    path = load_data_path()
    print(f"{len(list(path.glob('*.json')))} traces in {path}")
