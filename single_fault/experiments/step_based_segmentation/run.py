from __future__ import annotations

from pathlib import Path

from single_fault.experiments.shared import MethodConfig, run_method_configs_for_dataset
from single_fault.methods.step_based_multi_step import (
    CONTEXT_MODES,
    DEFAULT_MODEL_NAME,
    WINDOW_SIZES,
    ContextMode,
    build_method_name,
    step_based_multi_step_single_file,
)
from single_fault.utils.datasets import DATASET_DIRS
from single_fault.utils.experiment_paths import STEP_BASED_SEGMENTATION_OUTPUT_DIR
from single_fault.utils.schema import Metadata


EXPERIMENT_NAME = "step_based_segmentation"


def build_runner(num_steps: int, context_mode: ContextMode):
    def run_single_file(data: dict, metadata: Metadata):
        return step_based_multi_step_single_file(
            data=data,
            num_steps=num_steps,
            context_mode=context_mode,
            metadata=metadata,
        )

    return run_single_file


def build_method_configs() -> list[MethodConfig]:
    configs: list[MethodConfig] = []
    for num_steps in WINDOW_SIZES:
        for context_mode in CONTEXT_MODES:
            configs.append(
                MethodConfig(
                    method_name=build_method_name(num_steps, context_mode),
                    metadata_method="step_by_step",
                    run_single_file=build_runner(num_steps, context_mode),
                    progress_fields={
                        "window": num_steps,
                        "mode": context_mode.value,
                    },
                )
            )
    return configs


def run_dataset(dataset_key: str, data_dir: Path, model_name: str = DEFAULT_MODEL_NAME) -> tuple[Path, Path]:
    return run_method_configs_for_dataset(
        dataset_key=dataset_key,
        data_dir=data_dir,
        model_name=model_name,
        output_dir=STEP_BASED_SEGMENTATION_OUTPUT_DIR,
        accuracy_file_name=f"{dataset_key}_step_based_multi_step.csv",
        cost_file_name=f"{dataset_key}_step_based_multi_step_cost.csv",
        experiment_name=EXPERIMENT_NAME,
        method_configs=build_method_configs(),
    )


def main() -> None:
    for dataset_key, data_dir in DATASET_DIRS.items():
        accuracy_output_path, cost_output_path = run_dataset(dataset_key=dataset_key, data_dir=data_dir)
        print(f"Saved step-based segmentation accuracy results to: {accuracy_output_path}")
        print(f"Saved step-based segmentation cost results to: {cost_output_path}")


if __name__ == "__main__":
    main()
