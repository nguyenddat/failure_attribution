from __future__ import annotations

from baseline.who_and_when.system_prompt import (
    AllAtOnceInput,
    StepByStepInput,
    all_at_once_parser,
    all_at_once_prompt,
    step_by_step_parser,
    step_by_step_prompt,
)
from experiments.single_fault.utils.accuracy import agent_names_match
from baseline.llm import invoke_structured


def format_trajectory(trajectory: list[dict]) -> str:
    return "\n".join(
        f"- step {item['step']} - {item['agent_name']}: {item['content']}"
        for item in trajectory
    )


def _score(gt_agent: str, gt_step: int, pred_agent: str, pred_step: int) -> dict:
    return {
        "gt_agent": gt_agent,
        "gt_step": gt_step,
        "pred_agent": pred_agent,
        "pred_step": pred_step,
        "agent_accuracy": float(agent_names_match(gt_agent, pred_agent)),
        "step_accuracy": float(gt_step == pred_step),
    }


def all_at_once_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    trajectory = data["trajectory"]
    step_to_agent_name = {int(item["step"]): item["agent_name"] for item in trajectory}

    method_input = AllAtOnceInput(
        problem=data["question"],
        chat_content=format_trajectory(trajectory),
    )
    result, cost = invoke_structured(
        model_name=model_name,
        prompt_template=all_at_once_prompt,
        parser=all_at_once_parser,
        prompt_params=method_input,
    )

    pred_step = int(result["step_number"])
    pred_agent = step_to_agent_name.get(pred_step, "Not Found")
    accuracy = _score(
        gt_agent=data["mistake_agent"],
        gt_step=int(data["mistake_step"]),
        pred_agent=pred_agent,
        pred_step=pred_step,
    )
    return accuracy, cost


def step_by_step_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    trajectory = data["trajectory"]
    all_chat_content = format_trajectory(trajectory)
    total_cost = {"latency": 0.0, "input_tokens": 0, "output_tokens": 0}

    for current_step, item in enumerate(trajectory):
        method_input = StepByStepInput(
            problem=data["question"],
            current_step_content=format_trajectory([item]),
            chat_content=format_trajectory(trajectory[: current_step + 1]),
        )
        result, cost = invoke_structured(
            model_name=model_name,
            prompt_template=step_by_step_prompt,
            parser=step_by_step_parser,
            prompt_params=method_input,
        )
        total_cost = {
            "latency": total_cost["latency"] + cost["latency"],
            "input_tokens": total_cost["input_tokens"] + cost["input_tokens"],
            "output_tokens": total_cost["output_tokens"] + cost["output_tokens"],
        }

        if result["error_found"]:
            accuracy = _score(
                gt_agent=data["mistake_agent"],
                gt_step=int(data["mistake_step"]),
                pred_agent=item["agent_name"],
                pred_step=current_step,
            )
            return accuracy, total_cost

    accuracy = _score(
        gt_agent=data["mistake_agent"],
        gt_step=int(data["mistake_step"]),
        pred_agent="Not Found",
        pred_step=-1,
    )
    return accuracy, total_cost
