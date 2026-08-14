from __future__ import annotations

from baseline.telbench.metrics import (
    compute_metrics,
    first_error_span,
)
from baseline.telbench.system_prompt import (
    AllAtOnceInput,
    StepByStepInput,
    all_at_once_parser,
    all_at_once_prompt,
    step_by_step_parser,
    step_by_step_prompt,
)
from baseline.llm import invoke_structured, is_context_length_exceeded


def format_spans(spans: list[dict]) -> str:
    return "\n".join(f"- {span['id']}: {span['raw']}" for span in spans)


def _empty_cost() -> dict:
    return {"latency": 0.0, "input_tokens": 0, "output_tokens": 0}


def _accumulate_cost(total: dict, delta: dict) -> dict:
    return {
        "latency": total["latency"] + delta["latency"],
        "input_tokens": total["input_tokens"] + delta["input_tokens"],
        "output_tokens": total["output_tokens"] + delta["output_tokens"],
    }


def _build_accuracy(data: dict, pred_span: str | None, exceeded: bool) -> dict:
    span_ids_in_order = [span["id"] for span in data["spans"]]
    gold_spans = data["gold"]["error_span_ids"]
    gt_first_error = first_error_span(span_ids_in_order, gold_spans)
    metrics = compute_metrics(pred_span, gold_spans, gt_first_error)
    return {
        "pred_span": pred_span,
        "metrics": metrics,
        "exceeded_max_token_limit": exceeded,
    }


def all_at_once_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    method_input = AllAtOnceInput(
        question=data["question"],
        spans_content=format_spans(data["spans"]),
    )

    try:
        result, cost = invoke_structured(
            model_name=model_name,
            prompt_template=all_at_once_prompt,
            parser=all_at_once_parser,
            prompt_params=method_input,
        )
    except Exception as error:
        if not is_context_length_exceeded(error):
            raise
        return _build_accuracy(data, pred_span=None, exceeded=True), _empty_cost()

    return _build_accuracy(data, pred_span=result["span_id"], exceeded=False), cost


def step_by_step_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    spans = data["spans"]
    all_spans_content = format_spans(spans)
    total_cost = _empty_cost()

    for span in spans:
        method_input = StepByStepInput(
            question=data["question"],
            current_span_content=f"- {span['id']}: {span['raw']}",
            spans_content=all_spans_content,
        )

        try:
            result, cost = invoke_structured(
                model_name=model_name,
                prompt_template=step_by_step_prompt,
                parser=step_by_step_parser,
                prompt_params=method_input,
            )
        except Exception as error:
            if not is_context_length_exceeded(error):
                raise
            return _build_accuracy(data, pred_span=None, exceeded=True), total_cost

        total_cost = _accumulate_cost(total_cost, cost)

        if result["error_found"]:
            return _build_accuracy(data, pred_span=span["id"], exceeded=False), total_cost

    return _build_accuracy(data, pred_span=None, exceeded=False), total_cost
