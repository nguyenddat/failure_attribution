import os
import json
from pathlib import Path
from typing import Any, List, Optional


import pandas as pd
from datasets import load_dataset

from data.multi_fault.utils import AgentBehavior, Data, FaultyAgent, dataset_name_to_filename

base_dir = Path(__file__).resolve().parent

dataset_name = "mcemri/mast-data"
dataset_path = base_dir / "json" / dataset_name_to_filename(dataset_name).replace(".json", "")
dataset_path.mkdir(parents=True, exist_ok=True)


def trace_to_trajectory(trace):
    raw_trajectory = trace["trajectory"]
    print(raw_trajectory)
    return


def load_dataframe() -> pd.DataFrame:
    ds = load_dataset(
        "json",
        data_files={"train": "hf://datasets/mcemri/MAST-Data/MAD_full_dataset.json"},
    )
    return ds["train"].to_pandas()

def load_data_path() -> Path:
    df = load_dataframe()

    for i, row in df.iterrows():
        file_path = dataset_path / f"{i}.json"
        if os.path.exists(file_path):
            continue
        
        trace_to_trajectory(row["trace"])
        break

    return dataset_path

if __name__ == "__main__":
    load_data_path()
