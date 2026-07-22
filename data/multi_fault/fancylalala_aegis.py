import os
import json
from pathlib import Path
from typing import Any


import pandas as pd
from datasets import load_dataset

from data.multi_fault.utils import AgentBehavior, Data, FaultyAgent, dataset_name_to_filename

base_dir = Path(__file__).resolve().parent

dataset_name = "fancylalala/aegis"
dataset_path = base_dir / "json" / dataset_name_to_filename(dataset_name).replace(".json", "")
dataset_path.mkdir(parents=True, exist_ok=True)

def input_to_trajectory(input: Any):
    question = input["query"]
    
    trajectory = []
    for i, item in enumerate(input["conversation_history"]):
        if item["content"] == None:
            item["content"] = "None"
            
        trajectory.append(AgentBehavior(
            step=item["step"],
            agent_name=item["agent_name"],
            content=item["content"]
        ))
        
    return question, trajectory

def output_to_gt(output: Any):
    faulty_agents = []
    for a in output["injected_agents"]:
        faulty_agents.append(FaultyAgent(
            agent_name=a["agent_name"],
            error_type=a["error_type"]
        ))
    return faulty_agents


def load_dataframe() -> pd.DataFrame:
    ds = load_dataset("Fancylalala/AEGIS")
    dfs = []
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        df["split"] = split_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_data_path() -> Path:
    df = load_dataframe()

    for i, row in df.iterrows():
        file_path = dataset_path / f"{i}.json"
        if os.path.exists(file_path):
            continue
        
        question, trajectory = input_to_trajectory(row["input"])
        faulty_agents = output_to_gt(row["ground_truth"])
        data = Data(
          question=question,
          trajectory=trajectory,
          faulty_agents=faulty_agents
        )

        with open(file_path, "w") as file:
          json.dump(data.model_dump(), file)

    return dataset_path
  
if __name__ == "__main__":
    load_data_path()
