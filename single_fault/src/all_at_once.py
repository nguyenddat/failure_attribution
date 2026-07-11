from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from single_fault.utils.accuracy import agent_names_match
    from single_fault.utils.file import load_json, format_agent_behaviors
    from single_fault.utils.get_chat_completion import get_chat_completion
    from single_fault.utils.schema import AccuracyMetrics, AllAtOnceInput, CostMetrics, Metadata
    from single_fault.utils.datasets import DATASET_DIRS, OUTPUT_DIR
    from single_fault.utils.results import (
        has_complete_method_cost,
        has_complete_method_result,
        load_or_init_dataset_results,
        sort_results,
        update_method_cost,
        update_method_result,
        upsert_base_row,
    )
else:
    from ..utils.accuracy import agent_names_match
    from ..utils.file import load_json
    from ..utils.get_chat_completion import get_chat_completion
    from ..utils.schema import AccuracyMetrics, AllAtOnceInput, CostMetrics, Metadata
    from ..utils.datasets import DATASET_DIRS, OUTPUT_DIR
    from ..utils.results import (
        has_complete_method_cost,
        has_complete_method_result,
        load_or_init_dataset_results,
        sort_results,
        update_method_cost,
        update_method_result,
        upsert_base_row,
    )



def all_at_once_single_file(
    data: dict,
    metadata: Metadata,
    accuracy_metrics: AccuracyMetrics | None = None,
    cost_metrics: CostMetrics | None = None,
) -> Tuple[AccuracyMetrics, CostMetrics]:
    trajectory = data.get("trajectory", [])
    step_to_agent_name = {}
    for a in trajectory:
        step_to_agent_name[int(a["step"])] = a["agent_name"]

    if not cost_metrics:
        cost_metrics = CostMetrics(
            num_input_steps=len(trajectory),
            latency=0.0,
            input_tokens=0,
            output_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
        )

    if not accuracy_metrics:
        accuracy_metrics = AccuracyMetrics(
            gt_agent=data.get("mistake_agent"),
            gt_step=int(data.get("mistake_step", -1)),
            pred_agent="Not Found",
            pred_step=-1,
            agent_accuracy=0.0,
            step_accuracy=0.0,
        )

    method_input = AllAtOnceInput(
        problem=data["question"],
        chat_content=format_agent_behaviors(trajectory),
    )

    result, metrics = get_chat_completion(metadata, method_input)
    pred_step = int(result["step_number"])
    
    accuracy_metrics.pred_step = pred_step
    accuracy_metrics.pred_agent = step_to_agent_name[pred_step]
    accuracy_metrics.step_accuracy = float(accuracy_metrics.gt_step == accuracy_metrics.pred_step)
    accuracy_metrics.agent_accuracy = float(
        agent_names_match(accuracy_metrics.gt_agent, accuracy_metrics.pred_agent)
    )

    cost_metrics.latency += metrics["latency"]
    cost_metrics.input_tokens += metrics["input_tokens"]
    cost_metrics.output_tokens += metrics["output_tokens"]
    return accuracy_metrics, cost_metrics


def run_dataset(dataset_key: str, data_dir: Path, metadata: Metadata) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    accuracy_csv_path = OUTPUT_DIR / f"{dataset_key}.csv"
    cost_csv_path = OUTPUT_DIR / f"{dataset_key}_cost.csv"
    results_df = load_or_init_dataset_results(accuracy_csv_path)
    cost_df = load_or_init_dataset_results(cost_csv_path)

    file_paths = sorted(data_dir.glob("*.json"), key=lambda path: int(path.stem))
    progress = tqdm(file_paths, desc=f"{dataset_key}:{metadata.method}", unit="file")

    for file_path in progress:
        data = load_json(file_path)
        gt_agent = data["mistake_agent"]
        gt_step = int(data["mistake_step"])

        results_df = upsert_base_row(
            df=results_df,
            file_name=file_path.name,
            gt_agent=gt_agent,
            gt_step=gt_step,
        )
        cost_df = upsert_base_row(
            df=cost_df,
            file_name=file_path.name,
            gt_agent=gt_agent,
            gt_step=gt_step,
        )

        if has_complete_method_result(results_df, file_path.name, metadata.method) and has_complete_method_cost(
            cost_df, file_path.name, metadata.method
        ):
            continue

        accuracy_metrics, cost_metrics = all_at_once_single_file(
            data=data,
            metadata=metadata,
        )
        results_df = update_method_result(
            df=results_df,
            file_name=file_path.name,
            method_name=metadata.method,
            agent_accuracy=accuracy_metrics.agent_accuracy,
            step_accuracy=accuracy_metrics.step_accuracy,
        )
        cost_df = update_method_cost(
            df=cost_df,
            file_name=file_path.name,
            method_name=metadata.method,
            latency=cost_metrics.latency,
            input_tokens=cost_metrics.input_tokens,
            output_tokens=cost_metrics.output_tokens,
        )

        results_df = sort_results(results_df)
        cost_df = sort_results(cost_df)
        results_df.to_csv(accuracy_csv_path, index=False)
        cost_df.to_csv(cost_csv_path, index=False)

    results_df = sort_results(results_df)
    cost_df = sort_results(cost_df)
    results_df.to_csv(accuracy_csv_path, index=False)
    cost_df.to_csv(cost_csv_path, index=False)

    return accuracy_csv_path, cost_csv_path


if __name__ == "__main__":
    metadata = Metadata(model_name="gpt-4o-mini", method="all_at_once")

    for dataset_key, data_dir in DATASET_DIRS.items():
        accuracy_output_path, cost_output_path = run_dataset(
            dataset_key=dataset_key,
            data_dir=data_dir,
            metadata=metadata,
        )
        print(f"Saved accuracy results to: {accuracy_output_path}")
        print(f"Saved cost results to: {cost_output_path}")
