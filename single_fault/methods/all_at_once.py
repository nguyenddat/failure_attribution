from __future__ import annotations

from typing import Tuple

from single_fault.utils.accuracy import agent_names_match
from single_fault.utils.file import format_agent_behaviors
from single_fault.utils.schema import AccuracyMetrics, AllAtOnceInput, CostMetrics, Metadata


def all_at_once_single_file(
    data: dict,
    metadata: Metadata,
    accuracy_metrics: AccuracyMetrics | None = None,
    cost_metrics: CostMetrics | None = None,
) -> Tuple[AccuracyMetrics, CostMetrics]:
    from single_fault.utils.get_chat_completion import get_chat_completion

    trajectory = data.get("trajectory", [])
    step_to_agent_name = {}
    for item in trajectory:
        step_to_agent_name[int(item["step"])] = item["agent_name"]

    if not cost_metrics:
        cost_metrics = CostMetrics(
            num_input_steps=len(trajectory),
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

    method_input = AllAtOnceInput(
        problem=data["question"],
        chat_content=format_agent_behaviors(trajectory),
    )

    result, metrics = get_chat_completion(metadata, method_input)
    pred_step = int(result["step_number"])

    accuracy_metrics.pred_step = pred_step
    accuracy_metrics.pred_agent = step_to_agent_name[pred_step]
    accuracy_metrics.step_accuracy = float(accuracy_metrics.gt_step == accuracy_metrics.pred_step)
    accuracy_metrics.agent_accuracy = float(
        agent_names_match(accuracy_metrics.gt_agent, accuracy_metrics.pred_agent)
    )

    cost_metrics.latency += metrics["latency"]
    cost_metrics.input_tokens += metrics["input_tokens"]
    cost_metrics.output_tokens += metrics["output_tokens"]
    return accuracy_metrics, cost_metrics
