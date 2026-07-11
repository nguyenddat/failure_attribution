from __future__ import annotations

from pathlib import Path

import pandas as pd

from single_fault.experiments.shared import MethodConfig, run_method_configs_for_dataset
from single_fault.methods.step_based_multi_step import (
    DEFAULT_MODEL_NAME,
    ContextMode,
    build_method_name,
    step_based_multi_step_single_file,
)
from single_fault.utils.datasets import DATASET_DIRS
from single_fault.utils.experiment_paths import STEP_BASED_CONTEXT_MODE_COMPARISON_OUTPUT_DIR
from single_fault.utils.results import build_cost_columns, build_result_columns
from single_fault.utils.schema import Metadata


EXPERIMENT_NAME = "step_based_context_mode_comparison"
DATASET_KEY = "ww_hand_crafted"
FIXED_NUM_STEPS = 5
COMPARISON_CONTEXT_MODES = [
    ContextMode.PREVIOUS_ONLY,
    ContextMode.NEXT_ONLY,
]


def build_runner(context_mode: ContextMode):
    def run_single_file(data: dict, metadata: Metadata):
        return step_based_multi_step_single_file(
            data=data,
            num_steps=FIXED_NUM_STEPS,
            context_mode=context_mode,
            metadata=metadata,
        )

    return run_single_file


def build_method_configs() -> list[MethodConfig]:
    return [
        MethodConfig(
            method_name=build_method_name(FIXED_NUM_STEPS, context_mode),
            metadata_method="step_by_step",
            run_single_file=build_runner(context_mode),
            include_predictions=True,
            progress_fields={
                "window": FIXED_NUM_STEPS,
                "mode": context_mode.value,
            },
        )
        for context_mode in COMPARISON_CONTEXT_MODES
    ]


def build_summary(accuracy_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for config in build_method_configs():
        agent_col, step_col = build_result_columns(config.method_name)
        latency_col, input_tokens_col, output_tokens_col = build_cost_columns(config.method_name)
        records.append(
            {
                "method": config.method_name,
                "agent_accuracy": float(pd.to_numeric(accuracy_df[agent_col], errors="coerce").mean()),
                "step_accuracy": float(pd.to_numeric(accuracy_df[step_col], errors="coerce").mean()),
                "avg_latency": float(pd.to_numeric(cost_df[latency_col], errors="coerce").mean()),
                "avg_input_tokens": float(pd.to_numeric(cost_df[input_tokens_col], errors="coerce").mean()),
                "avg_output_tokens": float(pd.to_numeric(cost_df[output_tokens_col], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(records).sort_values("method").reset_index(drop=True)


def run_experiment(model_name: str = DEFAULT_MODEL_NAME) -> tuple[Path, Path, Path]:
    data_dir = DATASET_DIRS[DATASET_KEY]
    accuracy_output_path, cost_output_path = run_method_configs_for_dataset(
        dataset_key=DATASET_KEY,
        data_dir=data_dir,
        model_name=model_name,
        output_dir=STEP_BASED_CONTEXT_MODE_COMPARISON_OUTPUT_DIR,
        accuracy_file_name=f"{DATASET_KEY}_step_based_context_mode.csv",
        cost_file_name=f"{DATASET_KEY}_step_based_context_mode_cost.csv",
        experiment_name=EXPERIMENT_NAME,
        method_configs=build_method_configs(),
    )

    accuracy_df = pd.read_csv(accuracy_output_path)
    cost_df = pd.read_csv(cost_output_path)
    summary_df = build_summary(accuracy_df, cost_df)
    summary_path = STEP_BASED_CONTEXT_MODE_COMPARISON_OUTPUT_DIR / f"{DATASET_KEY}_step_based_context_mode_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return accuracy_output_path, cost_output_path, summary_path


def main() -> None:
    accuracy_output_path, cost_output_path, summary_path = run_experiment()
    print(f"Saved context-mode accuracy results to: {accuracy_output_path}")
    print(f"Saved context-mode cost results to: {cost_output_path}")
    print(f"Saved context-mode summary to: {summary_path}")


if __name__ == "__main__":
    main()
