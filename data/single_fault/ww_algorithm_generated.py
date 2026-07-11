import os
import json
from pathlib import Path
from typing import Any, List

import pandas as pd
from utils import AgentBehavior, Data
from utils import dataset_name_to_filename

base_dir = Path(__file__).resolve().parent

dataset_name = "who&when/algorithm-generated"
dataset_link = "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet"
dataset_path = base_dir / "json" / dataset_name_to_filename(dataset_name).replace(".json", "")
dataset_path.mkdir(parents=True, exist_ok=True)

problem_fields = ["question"]
trajectory_fields = ["history", "mistake_agent", "mistake_step"]
selected_fields = problem_fields + trajectory_fields

output_dir = Path(__file__).resolve().parent / "json"

# load dataframe
df = pd.read_parquet(dataset_link)
df = df[selected_fields]

def history_to_trajectory(history: Any) -> List[AgentBehavior]:
    trajectory = []
    for i, item in enumerate(history):
        trajectory.append(AgentBehavior(
            step=i,
            agent_name=item.get("name"),
            content=item.get("content")
        ))
    return trajectory

def load_data_path() -> Path:
    for i, row in df.iterrows():
        file_path = dataset_path / f"{i}.json"
        if os.path.exists(file_path):
            continue
        
        data = Data(
            question=row["question"],
            trajectory=history_to_trajectory(row["history"]),
            mistake_agent=row["mistake_agent"],
            mistake_step=int(row["mistake_step"])
        )
        
        with open(file_path, "w") as file:
            json.dump(data.model_dump(), file)
    return dataset_path

load_data_path()