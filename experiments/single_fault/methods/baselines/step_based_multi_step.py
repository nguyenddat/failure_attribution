from __future__ import annotations

from enum import Enum
from typing import Tuple

from experiments.single_fault.utils.accuracy import agent_names_match
from experiments.single_fault.utils.file import format_agent_behaviors
from experiments.single_fault.utils.schema import (
    AccuracyMetrics,
    CostMetrics,
    Metadata,
    StepByStepInput,
)


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


def get_surrounding_steps(
    trajectory: list[dict], current_step: int, num_steps: int
) -> list[dict]:
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


def get_previous_steps(
    trajectory: list[dict], current_step: int, num_steps: int
) -> list[dict]:
    start_idx = max(0, current_step - num_steps)
    return trajectory[start_idx:current_step]


def get_next_steps(
    trajectory: list[dict], current_step: int, num_steps: int
) -> list[dict]:
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
    from experiments.get_chat_completion import get_chat_completion

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
        surrounding_steps = get_context_steps(
            trajectory, current_step, num_steps, context_mode
        )
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
            accuracy_metrics.step_accuracy = float(
                accuracy_metrics.gt_step == current_step
            )
            accuracy_metrics.agent_accuracy = float(
                agent_names_match(accuracy_metrics.gt_agent, agent_name)
            )
            return accuracy_metrics, cost_metrics

    return accuracy_metrics, cost_metrics
