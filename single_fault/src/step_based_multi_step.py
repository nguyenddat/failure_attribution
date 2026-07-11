from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from single_fault.utils.accuracy import agent_names_match
    from single_fault.utils.file import format_agent_behaviors, load_json
    from single_fault.utils.schema import AccuracyMetrics, CostMetrics, Metadata, StepByStepInput
    from single_fault.utils.datasets import DATASET_DIRS, OUTPUT_DIR
else:
    from ..utils.accuracy import agent_names_match
    from ..utils.file import format_agent_behaviors, load_json
    from ..utils.schema import AccuracyMetrics, CostMetrics, Metadata, StepByStepInput
    from ..utils.datasets import DATASET_DIRS, OUTPUT_DIR


WINDOW_SIZES = range(5, 9)
DEFAULT_MODEL_NAME = "gpt-4o-mini"


class ContextMode(str, Enum):
    SURROUNDING = "surrounding"
    PREVIOUS_ONLY = "previous_only"
    NEXT_ONLY = "next_only"


CONTEXT_MODES = [
    ContextMode.SURROUNDING,
    ContextMode.PREVIOUS_ONLY,
    ContextMode.NEXT_ONLY,
]


def build_method_name(num_steps: int, context_mode: ContextMode) -> str:
    if context_mode is ContextMode.SURROUNDING:
        return f"step_based_multi_step_w{num_steps}"
    if context_mode is ContextMode.PREVIOUS_ONLY:
        return f"step_based_multi_step_prev_w{num_steps}"
    if context_mode is ContextMode.NEXT_ONLY:
        return f"step_based_multi_step_next_w{num_steps}"
    raise ValueError(f"Unsupported context mode: {context_mode}")


def get_surrounding_steps(trajectory: list[dict], current_step: int, num_steps: int) -> list[dict]:
    before_target = num_steps // 2
    after_target = num_steps - before_target

    before_start = max(0, current_step - before_target)
    before_steps = trajectory[before_start:current_step]
    after_steps = trajectory[current_step + 1 : current_step + 1 + after_target]

    missing_before = before_target - len(before_steps)
    if missing_before > 0:
        extra_after_end = current_step + 1 + after_target + missing_before
        after_steps = trajectory[current_step + 1 : extra_after_end]

    missing_after = after_target - len(after_steps)
    if missing_after > 0:
        extra_before_start = max(0, before_start - missing_after)
        before_steps = trajectory[extra_before_start:current_step]

    return before_steps + after_steps


def get_previous_steps(trajectory: list[dict], current_step: int, num_steps: int) -> list[dict]:
    start_idx = max(0, current_step - num_steps)
    return trajectory[start_idx:current_step]


def get_next_steps(trajectory: list[dict], current_step: int, num_steps: int) -> list[dict]:
    end_idx = current_step + 1 + num_steps
    return trajectory[current_step + 1 : end_idx]


def get_context_steps(
    trajectory: list[dict],
    current_step: int,
    num_steps: int,
    context_mode: ContextMode,
) -> list[dict]:
    if context_mode is ContextMode.SURROUNDING:
        return get_surrounding_steps(trajectory, current_step, num_steps)
    if context_mode is ContextMode.PREVIOUS_ONLY:
        return get_previous_steps(trajectory, current_step, num_steps)
    if context_mode is ContextMode.NEXT_ONLY:
        return get_next_steps(trajectory, current_step, num_steps)
    raise ValueError(f"Unsupported context mode: {context_mode}")


def step_based_multi_step_single_file(
    data: dict,
    num_steps: int,
    context_mode: ContextMode,
    metadata: Metadata,
    accuracy_metrics: AccuracyMetrics | None = None,
    cost_metrics: CostMetrics | None = None,
) -> Tuple[AccuracyMetrics, CostMetrics]:
    if __package__ is None or __package__ == "":
        from single_fault.utils.get_chat_completion import get_chat_completion
    else:
        from ..utils.get_chat_completion import get_chat_completion

    trajectory = data.get("trajectory", [])
    total_steps = len(trajectory)

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

    for current_step in range(total_steps):
        current_step_entry = trajectory[current_step]
        surrounding_steps = get_context_steps(trajectory, current_step, num_steps, context_mode)
        agent_name = current_step_entry.get("agent_name", "Unknown Agent")

        method_input = StepByStepInput(
            problem=data["question"],
            current_step_content=format_agent_behaviors([current_step_entry]),
            chat_content=format_agent_behaviors(surrounding_steps),
        )

        result, metrics = get_chat_completion(metadata, method_input)

        cost_metrics.latency += metrics["latency"]
        cost_metrics.input_tokens += metrics["input_tokens"]
        cost_metrics.output_tokens += metrics["output_tokens"]
        cost_metrics.num_input_steps += 1 + len(surrounding_steps)

        if result["error_found"]:
            accuracy_metrics.pred_agent = agent_name
            accuracy_metrics.pred_step = current_step
            accuracy_metrics.step_accuracy = float(accuracy_metrics.gt_step == current_step)
            accuracy_metrics.agent_accuracy = float(agent_names_match(accuracy_metrics.gt_agent, agent_name))
            return accuracy_metrics, cost_metrics

    return accuracy_metrics, cost_metrics


def run_dataset(dataset_key: str, data_dir: Path, model_name: str = DEFAULT_MODEL_NAME) -> tuple[Path, Path]:
    if __package__ is None or __package__ == "":
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
        from ..utils.results import (
            has_complete_method_cost,
            has_complete_method_result,
            load_or_init_dataset_results,
            sort_results,
            update_method_cost,
            update_method_result,
            upsert_base_row,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    accuracy_csv_path = OUTPUT_DIR / f"{dataset_key}_step_based_multi_step.csv"
    cost_csv_path = OUTPUT_DIR / f"{dataset_key}_step_based_multi_step_cost.csv"

    accuracy_df = load_or_init_dataset_results(accuracy_csv_path)
    cost_df = load_or_init_dataset_results(cost_csv_path)

    file_paths = sorted(data_dir.glob("*.json"), key=lambda path: int(path.stem))
    total_runs = len(file_paths) * len(WINDOW_SIZES) * len(CONTEXT_MODES)
    progress = tqdm(total=total_runs, desc=f"{dataset_key}:step_based_multi_step", unit="run")

    try:
        for num_steps in WINDOW_SIZES:
            for context_mode in CONTEXT_MODES:
                method_name = build_method_name(num_steps, context_mode)
                metadata = Metadata(model_name=model_name, method="step_by_step")

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

                    progress.set_postfix(window=num_steps, mode=context_mode.value, file=file_path.name)

                    if has_complete_method_result(accuracy_df, file_path.name, method_name) and has_complete_method_cost(
                        cost_df, file_path.name, method_name
                    ):
                        progress.update(1)
                        continue

                    accuracy_metrics, cost_metrics = step_based_multi_step_single_file(
                        data=data,
                        num_steps=num_steps,
                        context_mode=context_mode,
                        metadata=metadata,
                    )
                    accuracy_df = update_method_result(
                        df=accuracy_df,
                        file_name=file_path.name,
                        method_name=method_name,
                        agent_accuracy=accuracy_metrics.agent_accuracy,
                        step_accuracy=accuracy_metrics.step_accuracy,
                    )
                    cost_df = update_method_cost(
                        df=cost_df,
                        file_name=file_path.name,
                        method_name=method_name,
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


if __name__ == "__main__":
    for dataset_key, data_dir in DATASET_DIRS.items():
        accuracy_output_path, cost_output_path = run_dataset(
            dataset_key=dataset_key,
            data_dir=data_dir,
        )
        print(f"Saved segmentation accuracy results to: {accuracy_output_path}")
        print(f"Saved segmentation cost results to: {cost_output_path}")
