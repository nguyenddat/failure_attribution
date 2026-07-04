import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.file import load_json
from src.utils.schema import Metadata, AccuracyMetrics, CostMetrics
from src.binary_search import binary_search_single_file
from src.all_at_once import all_at_once_single_file
from src.step_by_step import step_by_step_single_file

project_root = Path(__file__).resolve().parents[1]
out_dir = project_root / "outputs"
accuracy_dir = out_dir / "accuracy"
cost_dir = out_dir / "cost"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(accuracy_dir, exist_ok=True)
os.makedirs(cost_dir, exist_ok=True)

def _run_single(metadata: Metadata, data: dict) -> Tuple[AccuracyMetrics, CostMetrics]:
    if metadata.method == "all_at_once":
        return all_at_once_single_file(metadata=metadata, data=data)

    if metadata.method == "step_by_step":
        return step_by_step_single_file(metadata=metadata, data=data, current_step=0, total_steps=len(data["history"]))
    
    if metadata.method == "binary_search":
        return binary_search_single_file(metadata=metadata, data=data, start_step=0, end_step=len(data["history"]))
    
    raise ValueError(f"Unsupported method: {metadata.method}")

def _run_data_dir(metadata: Metadata, data_dir: Path):
    accuracy_df = pd.DataFrame(columns=["file name", "gt_agent", "gt_step", "pred_agent", "pred_step", "agent_accuracy", "step_accuracy"])
    cost_df = pd.DataFrame(columns=["file name", "latency", "input_tokens", "output_tokens", "num_input_steps"])

    accuracy_csv = accuracy_dir / f"{data_dir.name}_{metadata.method}_accuracy.csv"
    cost_csv = cost_dir / f"{data_dir.name}_{metadata.method}_cost.csv"
    if accuracy_csv.exists() and cost_csv.exists():
        accuracy_df = pd.read_csv(accuracy_csv)
        cost_df = pd.read_csv(cost_csv)
        
    for file_name in tqdm(os.listdir(data_dir)):
        file = data_dir / file_name
        if not file.suffix == ".json":
            continue
        
        # Xem file đã có kết quả chưa?
        if accuracy_df[accuracy_df["file name"] == file.name].shape[0] > 0:
            if cost_df[cost_df["file name"] == file.name].shape[0] > 0:
                continue
        
        data = load_json(file)
        file_name = file.name
        
        accuracy_metrics, cost_metrics = _run_single(metadata, data)
        
        # Chạy đến đâu lưu đến đấy
        accuracy_df.loc[len(accuracy_df)] = {
            "file name": file_name,
            "gt_agent": accuracy_metrics.gt_agent,
            "gt_step": accuracy_metrics.gt_step,
            "pred_agent": accuracy_metrics.pred_agent,
            "pred_step": accuracy_metrics.pred_step,
            "agent_accuracy": accuracy_metrics.agent_accuracy,
            "step_accuracy": accuracy_metrics.step_accuracy,
        }

        cost_df.loc[len(cost_df)] = {
            "file name": file_name,
            "latency": cost_metrics.latency,
            "input_tokens": cost_metrics.input_tokens,
            "output_tokens": cost_metrics.output_tokens,
            "num_input_steps": cost_metrics.num_input_steps,
        }

        accuracy_df.to_csv(accuracy_csv, index=False)
        cost_df.to_csv(cost_csv, index=False)

def run_benchmark(data_dir: Path):
    methods = ["all_at_once", "step_by_step", "binary_search"]
    model = "gpt-4o-mini"
    with_gt = True
    
    metadata_list = [Metadata(model_name=model, method=method, with_gt=with_gt) for method in methods]
    for metadata in metadata_list:
        _run_data_dir(metadata, data_dir)

if __name__ == "__main__":
    data_dir = project_root / "Who&When" / "Algorithm-Generated"
    print(data_dir)
    run_benchmark(data_dir)
