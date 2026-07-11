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
    from single_fault.utils.schema import AccuracyMetrics, CostMetrics, Metadata, StepByStepInput
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
    from ..utils.file import load_json, format_agent_behaviors
    from ..utils.get_chat_completion import get_chat_completion
    from ..utils.schema import AccuracyMetrics, CostMetrics, Metadata, StepByStepInput
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


def step_by_step_single_file(
    data: dict,
    current_step: int,
    total_steps: int,
    metadata: Metadata,
    accuracy_metrics: AccuracyMetrics | None = None,
    cost_metrics: CostMetrics | None = None,
) -> Tuple[AccuracyMetrics, CostMetrics]:
    trajectory = data.get("trajectory", [])

    if not cost_metrics:
        cost_metrics = CostMetrics(
            num_input_steps=0,
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

    if current_step >= total_steps:
        accuracy_metrics.pred_agent = "Not Found"
        accuracy_metrics.pred_step = -1
        accuracy_metrics.step_accuracy = 0.0
        accuracy_metrics.agent_accuracy = 0.0
        return accuracy_metrics, cost_metrics

    agent_name = trajectory[current_step].get("agent_name", "Unknown Agent")

    method_input = StepByStepInput(
        problem=data["question"],
        current_step_content=format_agent_behaviors([trajectory[current_step]]),
        chat_content=format_agent_behaviors(trajectory[:current_step+1]),
    )

    result, metrics = get_chat_completion(metadata, method_input)

    cost_metrics.latency += metrics["latency"]
    cost_metrics.input_tokens += metrics["input_tokens"]
    cost_metrics.output_tokens += metrics["output_tokens"]
    cost_metrics.num_input_steps += 1 + current_step

    if result["error_found"]:
        accuracy_metrics.pred_agent = agent_name
        accuracy_metrics.pred_step = current_step
        accuracy_metrics.step_accuracy = float(accuracy_metrics.gt_step == current_step)
        accuracy_metrics.agent_accuracy = float(agent_names_match(accuracy_metrics.gt_agent, agent_name))
        return accuracy_metrics, cost_metrics

    return step_by_step_single_file(
        data=data,
        current_step=current_step + 1,
        total_steps=total_steps,
        metadata=metadata,
        accuracy_metrics=accuracy_metrics,
        cost_metrics=cost_metrics,
    )


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

        total_steps = len(data.get("trajectory", []))
        accuracy_metrics, cost_metrics = step_by_step_single_file(
            data=data,
            current_step=0,
            total_steps=total_steps,
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
    metadata = Metadata(model_name="gpt-4o-mini", method="step_by_step")

    for dataset_key, data_dir in DATASET_DIRS.items():
        accuracy_output_path, cost_output_path = run_dataset(
            dataset_key=dataset_key,
            data_dir=data_dir,
            metadata=metadata,
        )
        print(f"Saved accuracy results to: {accuracy_output_path}")
        print(f"Saved cost results to: {cost_output_path}")
