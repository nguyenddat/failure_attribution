from __future__ import annotations

from typing import Tuple

from single_fault.utils.accuracy import agent_names_match
from single_fault.utils.file import format_agent_behaviors
from single_fault.utils.schema import AccuracyMetrics, CostMetrics, Metadata, StepByStepInput


GPT_4O_MINI_MAX_TOKENS = 128_000
TOKEN_LEVELS = [0.25, 0.40, 0.50]
DEFAULT_MODEL_NAME = "gpt-4o-mini"


def build_method_name(level_ratio: float) -> str:
    level_percent = int(round(level_ratio * 100))
    return f"token_based_multi_step_{level_percent}pct"


def token_budget_from_ratio(level_ratio: float) -> int:
    return int(GPT_4O_MINI_MAX_TOKENS * level_ratio)


def count_tokens(text: str, model_name: str) -> int:
    from single_fault.utils.models import get_model

    model = get_model(model_name)
    return model.get_num_tokens(text)


def get_token_limited_surrounding_steps(
    trajectory: list[dict],
    current_step: int,
    max_chat_tokens: int,
    model_name: str,
) -> list[dict]:
    selected_before: list[dict] = []
    selected_after: list[dict] = []

    left = current_step - 1
    right = current_step + 1

    while left >= 0 or right < len(trajectory):
        added = False

        if left >= 0:
            candidate_before = [trajectory[left]] + selected_before
            candidate_steps = candidate_before + selected_after
            candidate_tokens = count_tokens(format_agent_behaviors(candidate_steps), model_name)
            if candidate_tokens <= max_chat_tokens:
                selected_before = candidate_before
                left -= 1
                added = True
            else:
                left = -1

        if right < len(trajectory):
            candidate_after = selected_after + [trajectory[right]]
            candidate_steps = selected_before + candidate_after
            candidate_tokens = count_tokens(format_agent_behaviors(candidate_steps), model_name)
            if candidate_tokens <= max_chat_tokens:
                selected_after = candidate_after
                right += 1
                added = True
            else:
                right = len(trajectory)

        if not added:
            break

    return selected_before + selected_after


def token_based_multi_step_single_file(
    data: dict,
    max_chat_tokens: int,
    metadata: Metadata,
    accuracy_metrics: AccuracyMetrics | None = None,
    cost_metrics: CostMetrics | None = None,
) -> Tuple[AccuracyMetrics, CostMetrics]:
    from single_fault.utils.get_chat_completion import get_chat_completion

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
        surrounding_steps = get_token_limited_surrounding_steps(
            trajectory=trajectory,
            current_step=current_step,
            max_chat_tokens=max_chat_tokens,
            model_name=metadata.model_name,
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
            accuracy_metrics.step_accuracy = float(accuracy_metrics.gt_step == current_step)
            accuracy_metrics.agent_accuracy = float(agent_names_match(accuracy_metrics.gt_agent, agent_name))
            return accuracy_metrics, cost_metrics

    return accuracy_metrics, cost_metrics
