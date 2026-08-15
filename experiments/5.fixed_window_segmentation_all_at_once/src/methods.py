from __future__ import annotations

import re

from llm import invoke_structured
from segments import make_segments
from system_prompt import SegmentInput, response_parser, segment_prompt


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


def _ask_segment_error(problem: str, segment: list[dict], model_name: str) -> tuple[dict, dict]:
    method_input = SegmentInput(problem=problem, segment_content=format_steps(segment))
    result, cost = invoke_structured(
        model_name=model_name,
        prompt_template=segment_prompt,
        parser=response_parser,
        prompt_params=method_input,
    )
    return result, cost


def fixed_window_all_at_once_single_file(
    data: dict, window_size: int, model_name: str
) -> tuple[dict, dict]:
    trajectory = data["trajectory"]
    total_cost = _empty_cost()

    for segment in make_segments(trajectory, window_size):
        result, cost = _ask_segment_error(
            problem=data["question"], segment=segment, model_name=model_name
        )
        total_cost = _accumulate_cost(total_cost, cost)

        segment_step_ids = {item["step"] for item in segment}
        if result["error_found"] and result["step_id"] in segment_step_ids:
            pred_step = result["step_id"]
            pred_agent = next(
                item["agent_name"] for item in segment if item["step"] == pred_step
            )
            return (
                _score(
                    gt_agent=data["mistake_agent"],
                    gt_step=int(data["mistake_step"]),
                    pred_agent=pred_agent,
                    pred_step=pred_step,
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
