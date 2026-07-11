from __future__ import annotations

from typing import Tuple

from single_fault.utils.accuracy import agent_names_match
from single_fault.utils.file import format_agent_behaviors
from single_fault.utils.schema import AccuracyMetrics, CostMetrics, Metadata, StepByStepInput


def step_by_step_single_file(
    data: dict,
    current_step: int,
    total_steps: int,
    metadata: Metadata,
    accuracy_metrics: AccuracyMetrics | None = None,
    cost_metrics: CostMetrics | None = None,
) -> Tuple[AccuracyMetrics, CostMetrics]:
    from single_fault.utils.get_chat_completion import get_chat_completion

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
        chat_content=format_agent_behaviors(trajectory[: current_step + 1]),
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
