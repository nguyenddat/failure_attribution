from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from single_fault.utils.file import load_json
from single_fault.utils.results import (
    has_complete_method_cost,
    has_complete_method_result,
    load_or_init_dataset_results,
    sort_results,
    update_method_cost,
    update_method_prediction,
    update_method_result,
    upsert_base_row,
)
from single_fault.utils.schema import AccuracyMetrics, CostMetrics, Metadata


SingleFileRunner = Callable[[dict, Metadata], tuple[AccuracyMetrics, CostMetrics]]


@dataclass(frozen=True)
class MethodConfig:
    method_name: str
    metadata_method: str
    run_single_file: SingleFileRunner
    progress_fields: dict[str, Any] = field(default_factory=dict)
    include_predictions: bool = False


def dataset_file_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.json"), key=lambda path: int(path.stem))


def run_method_configs_for_dataset(
    dataset_key: str,
    data_dir: Path,
    model_name: str,
    output_dir: Path,
    accuracy_file_name: str,
    cost_file_name: str,
    experiment_name: str,
    method_configs: list[MethodConfig],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracy_csv_path = output_dir / accuracy_file_name
    cost_csv_path = output_dir / cost_file_name

    accuracy_df = load_or_init_dataset_results(accuracy_csv_path)
    cost_df = load_or_init_dataset_results(cost_csv_path)

    file_paths = dataset_file_paths(data_dir)
    total_runs = len(file_paths) * len(method_configs)
    progress = tqdm(total=total_runs, desc=f"{dataset_key}:{experiment_name}", unit="run")

    try:
        for config in method_configs:
            metadata = Metadata(model_name=model_name, method=config.metadata_method)

            for file_path in file_paths:
                data = load_json(file_path)
                gt_agent = data["mistake_agent"]
                gt_step = int(data["mistake_step"])

                accuracy_df = upsert_base_row(
                    df=accuracy_df,
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

                postfix = {"method": config.method_name, "file": file_path.name}
                postfix.update(config.progress_fields)
                progress.set_postfix(postfix)

                if has_complete_method_result(accuracy_df, file_path.name, config.method_name) and has_complete_method_cost(
                    cost_df, file_path.name, config.method_name
                ):
                    progress.update(1)
                    continue

                accuracy_metrics, cost_metrics = config.run_single_file(data, metadata)
                accuracy_df = update_method_result(
                    df=accuracy_df,
                    file_name=file_path.name,
                    method_name=config.method_name,
                    agent_accuracy=accuracy_metrics.agent_accuracy,
                    step_accuracy=accuracy_metrics.step_accuracy,
                )
                if config.include_predictions:
                    accuracy_df = update_method_prediction(
                        df=accuracy_df,
                        file_name=file_path.name,
                        method_name=config.method_name,
                        pred_agent=accuracy_metrics.pred_agent,
                        pred_step=accuracy_metrics.pred_step,
                    )
                cost_df = update_method_cost(
                    df=cost_df,
                    file_name=file_path.name,
                    method_name=config.method_name,
                    latency=cost_metrics.latency,
                    input_tokens=cost_metrics.input_tokens,
                    output_tokens=cost_metrics.output_tokens,
                )

                accuracy_df = sort_results(accuracy_df)
                cost_df = sort_results(cost_df)
                accuracy_df.to_csv(accuracy_csv_path, index=False)
                cost_df.to_csv(cost_csv_path, index=False)
                progress.update(1)
    finally:
        progress.close()

    accuracy_df = sort_results(accuracy_df)
    cost_df = sort_results(cost_df)
    accuracy_df.to_csv(accuracy_csv_path, index=False)
    cost_df.to_csv(cost_csv_path, index=False)
    return accuracy_csv_path, cost_csv_path
