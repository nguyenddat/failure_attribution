from __future__ import annotations

from pathlib import Path

from single_fault.experiments.shared import MethodConfig, run_method_configs_for_dataset
from single_fault.methods.token_based_multi_step import (
    DEFAULT_MODEL_NAME,
    TOKEN_LEVELS,
    build_method_name,
    token_based_multi_step_single_file,
    token_budget_from_ratio,
)
from single_fault.utils.datasets import DATASET_DIRS
from single_fault.utils.experiment_paths import TOKEN_BASED_SEGMENTATION_OUTPUT_DIR
from single_fault.utils.schema import Metadata


EXPERIMENT_NAME = "token_based_segmentation"


def build_runner(token_budget: int):
    def run_single_file(data: dict, metadata: Metadata):
        return token_based_multi_step_single_file(
            data=data,
            max_chat_tokens=token_budget,
            metadata=metadata,
        )

    return run_single_file


def build_method_configs() -> list[MethodConfig]:
    configs: list[MethodConfig] = []
    for level_ratio in TOKEN_LEVELS:
        configs.append(
            MethodConfig(
                method_name=build_method_name(level_ratio),
                metadata_method="step_by_step",
                run_single_file=build_runner(token_budget_from_ratio(level_ratio)),
                progress_fields={
                    "level": f"{int(level_ratio * 100)}%",
                },
            )
        )
    return configs


def run_dataset(dataset_key: str, data_dir: Path, model_name: str = DEFAULT_MODEL_NAME) -> tuple[Path, Path]:
    return run_method_configs_for_dataset(
        dataset_key=dataset_key,
        data_dir=data_dir,
        model_name=model_name,
        output_dir=TOKEN_BASED_SEGMENTATION_OUTPUT_DIR,
        accuracy_file_name=f"{dataset_key}_token_based_multi_step.csv",
        cost_file_name=f"{dataset_key}_token_based_multi_step_cost.csv",
        experiment_name=EXPERIMENT_NAME,
        method_configs=build_method_configs(),
    )


def main() -> None:
    for dataset_key, data_dir in DATASET_DIRS.items():
        accuracy_output_path, cost_output_path = run_dataset(dataset_key=dataset_key, data_dir=data_dir)
        print(f"Saved token-based segmentation accuracy results to: {accuracy_output_path}")
        print(f"Saved token-based segmentation cost results to: {cost_output_path}")


if __name__ == "__main__":
    main()
