from __future__ import annotations

from pathlib import Path

from experiments.single_fault.experiments.shared import (
    MethodConfig,
    run_method_configs_for_dataset,
)
from experiments.single_fault.methods.baselines.all_at_once import (
    all_at_once_single_file,
)
from experiments.single_fault.methods.baselines.step_by_step import (
    step_by_step_single_file,
)
from experiments.single_fault.utils.datasets import DATASET_DIRS
from experiments.single_fault.utils.experiment_paths import BASELINE_OUTPUT_DIR
from experiments.single_fault.utils.schema import Metadata


EXPERIMENT_NAME = "baseline"
DEFAULT_MODEL_NAME = "gpt-4o-mini"


def run_all_at_once(data: dict, metadata: Metadata):
    return all_at_once_single_file(data=data, metadata=metadata)


def run_step_by_step(data: dict, metadata: Metadata):
    return step_by_step_single_file(
        data=data,
        current_step=0,
        total_steps=len(data.get("trajectory", [])),
        metadata=metadata,
    )


def build_method_configs() -> list[MethodConfig]:
    return [
        MethodConfig(
            method_name="all_at_once",
            metadata_method="all_at_once",
            run_single_file=run_all_at_once,
        ),
        MethodConfig(
            method_name="step_by_step",
            metadata_method="step_by_step",
            run_single_file=run_step_by_step,
        ),
    ]


def run_dataset(
    dataset_key: str, data_dir: Path, model_name: str = DEFAULT_MODEL_NAME
) -> tuple[Path, Path]:
    return run_method_configs_for_dataset(
        dataset_key=dataset_key,
        data_dir=data_dir,
        model_name=model_name,
        output_dir=BASELINE_OUTPUT_DIR,
        accuracy_file_name=f"{dataset_key}.csv",
        cost_file_name=f"{dataset_key}_cost.csv",
        experiment_name=EXPERIMENT_NAME,
        method_configs=build_method_configs(),
    )


def main() -> None:
    for dataset_key, data_dir in DATASET_DIRS.items():
        accuracy_output_path, cost_output_path = run_dataset(
            dataset_key=dataset_key, data_dir=data_dir
        )
        print(f"Saved baseline accuracy results to: {accuracy_output_path}")
        print(f"Saved baseline cost results to: {cost_output_path}")


if __name__ == "__main__":
    main()
