from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

from baseline.who_and_when.methods import (  # noqa: E402
    all_at_once_single_file,
    step_by_step_single_file,
)
from experiments.single_fault.experiments.shared import MethodConfig, run_method_configs_for_dataset  # noqa: E402
from experiments.single_fault.utils.experiment_paths import BASELINE_OUTPUT_DIR  # noqa: E402
from experiments.single_fault.utils.schema import AccuracyMetrics, CostMetrics, Metadata  # noqa: E402


DEFAULT_MODEL_NAME = "gpt-4o-mini"
DATA_DIR = REPO_ROOT / "data" / "error_localization" / "single_fault"

DATASET_DIRS = {
    "ww_algorithm_generated": DATA_DIR / "who_and_when__algorithm-generated",
    "ww_hand_crafted": DATA_DIR / "who_and_when__hand-crafted",
}


def _to_legacy_metrics(accuracy: dict, cost: dict):
    accuracy_metrics = AccuracyMetrics(
        gt_agent=accuracy["gt_agent"],
        gt_step=accuracy["gt_step"],
        pred_agent=accuracy["pred_agent"],
        pred_step=accuracy["pred_step"],
        agent_accuracy=accuracy["agent_accuracy"],
        step_accuracy=accuracy["step_accuracy"],
    )
    cost_metrics = CostMetrics(
        num_input_steps=0,
        latency=cost["latency"],
        input_tokens=cost["input_tokens"],
        output_tokens=cost["output_tokens"],
        input_cost=0.0,
        output_cost=0.0,
        total_cost=0.0,
    )
    return accuracy_metrics, cost_metrics


def _run_all_at_once(data: dict, metadata: Metadata):
    accuracy, cost = all_at_once_single_file(data, model_name=metadata.model_name)
    return _to_legacy_metrics(accuracy, cost)


def _run_step_by_step(data: dict, metadata: Metadata):
    accuracy, cost = step_by_step_single_file(data, model_name=metadata.model_name)
    return _to_legacy_metrics(accuracy, cost)


def build_method_configs() -> list[MethodConfig]:
    return [
        MethodConfig(
            method_name="all_at_once",
            metadata_method="all_at_once",
            run_single_file=_run_all_at_once,
        ),
        MethodConfig(
            method_name="step_by_step",
            metadata_method="step_by_step",
            run_single_file=_run_step_by_step,
        ),
    ]


def main() -> None:
    for dataset_key, data_dir in DATASET_DIRS.items():
        accuracy_path, cost_path = run_method_configs_for_dataset(
            dataset_key=dataset_key,
            data_dir=data_dir,
            model_name=DEFAULT_MODEL_NAME,
            output_dir=BASELINE_OUTPUT_DIR,
            accuracy_file_name=f"{dataset_key}.csv",
            cost_file_name=f"{dataset_key}_cost.csv",
            experiment_name="baseline_who_and_when",
            method_configs=build_method_configs(),
        )
        print(f"Saved who_and_when baseline accuracy results to: {accuracy_path}")
        print(f"Saved who_and_when baseline cost results to: {cost_path}")


if __name__ == "__main__":
    main()
