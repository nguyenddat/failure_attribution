from __future__ import annotations

import re

from llm import invoke_structured
from system_prompt import (
    BothInput,
    NextOnlyInput,
    PrevOnlyInput,
    both_prompt,
    next_only_prompt,
    prev_only_prompt,
    response_parser,
)
from windows import ContextMode, before_after_slices


def _normalize_agent_name(name: str | None) -> str:
    if not name:
        return ""
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return re.sub(r"\s*\(.*?\)\s*$", "", normalized)


def _agent_names_match(gt_agent: str | None, pred_agent: str | None) -> bool:
    gt_normalized = _normalize_agent_name(gt_agent)
    pred_normalized = _normalize_agent_name(pred_agent)
    if not gt_normalized or not pred_normalized:
        return False
    return gt_normalized == pred_normalized


def format_steps(steps: list[dict]) -> str:
    if not steps:
        return "(none)"
    return "\n".join(
        f"- step {item['step']} - {item['agent_name']}: {item['content']}"
        for item in steps
    )


def _empty_cost() -> dict:
    return {"latency": 0.0, "input_tokens": 0, "output_tokens": 0}


def _accumulate_cost(total: dict, delta: dict) -> dict:
    return {
        "latency": total["latency"] + delta["latency"],
        "input_tokens": total["input_tokens"] + delta["input_tokens"],
        "output_tokens": total["output_tokens"] + delta["output_tokens"],
    }


def _score(gt_agent: str, gt_step: int, pred_agent: str, pred_step: int) -> dict:
    return {
        "gt_agent": gt_agent,
        "gt_step": gt_step,
        "pred_agent": pred_agent,
        "pred_step": pred_step,
        "agent_accuracy": float(_agent_names_match(gt_agent, pred_agent)),
        "step_accuracy": float(gt_step == pred_step),
    }


def _ask_error_found(
    mode: ContextMode,
    problem: str,
    current_step_content: str,
    before: list[dict],
    after: list[dict],
    model_name: str,
) -> tuple[bool, dict]:
    if mode is ContextMode.BOTH:
        method_input = BothInput(
            problem=problem,
            current_step_content=current_step_content,
            before_content=format_steps(before),
            after_content=format_steps(after),
        )
        prompt_template = both_prompt
    elif mode is ContextMode.PREV_ONLY:
        method_input = PrevOnlyInput(
            problem=problem,
            current_step_content=current_step_content,
            before_content=format_steps(before),
        )
        prompt_template = prev_only_prompt
    elif mode is ContextMode.NEXT_ONLY:
        method_input = NextOnlyInput(
            problem=problem,
            current_step_content=current_step_content,
            after_content=format_steps(after),
        )
        prompt_template = next_only_prompt
    else:
        raise ValueError(f"Unsupported context mode: {mode}")

    result, cost = invoke_structured(
        model_name=model_name,
        prompt_template=prompt_template,
        parser=response_parser,
        prompt_params=method_input,
    )
    return bool(result["error_found"]), cost


def fixed_window_single_file(
    data: dict, window_size: int, mode: ContextMode, model_name: str
) -> tuple[dict, dict]:
    trajectory = data["trajectory"]
    total_cost = _empty_cost()

    for current_step, item in enumerate(trajectory):
        before, after = before_after_slices(trajectory, current_step, window_size, mode)
        error_found, cost = _ask_error_found(
            mode=mode,
            problem=data["question"],
            current_step_content=format_steps([item]),
            before=before,
            after=after,
            model_name=model_name,
        )
        total_cost = _accumulate_cost(total_cost, cost)

        if error_found:
            return (
                _score(
                    gt_agent=data["mistake_agent"],
                    gt_step=int(data["mistake_step"]),
                    pred_agent=item["agent_name"],
                    pred_step=current_step,
                ),
                total_cost,
            )

    return (
        _score(
            gt_agent=data["mistake_agent"],
            gt_step=int(data["mistake_step"]),
            pred_agent="Not Found",
            pred_step=-1,
        ),
        total_cost,
    )
