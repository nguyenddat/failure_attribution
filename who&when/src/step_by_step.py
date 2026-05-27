import os
import sys
import json
from pathlib import Path
from typing import Any, Dict

try:
    from .utils.models import get_model
    from .utils.file import load_json
    from .utils.get_chat_completion import get_chat_completion
except ImportError:
    # Allow direct script execution: python src/step_by_step.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.utils.models import get_model
    from src.utils.file import load_json
    from src.utils.get_chat_completion import get_chat_completion

def step_by_step_single_file(
    model_name: str,
    data: Dict[str, Any],
    current_step: int,
    total_steps: int,
    metrics_acc: Dict[str, Any] | None = None,
    max_retries_per_step: int = 3,
):  
    if metrics_acc is None:
        metrics_acc = {
            "num_calls": 0,
            "latency_s_total": 0.0,
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "total_tokens_total": 0,
            "cost_usd_estimate_total": 0.0,
        }

    print(f"Evaluating step {current_step} out of {total_steps}...")
    if current_step >= total_steps:
        return {
            "mistake_agent": None,
            "mistake_step": None,
            "reason": "No mistake found in any step.",
            "metrics": metrics_acc,
        }
        
    # load data + model
    model = get_model(model_name)
    method_params = {
        "method": "step_by_step",
        "with_gt": True,
        "model_name": model_name,
    }
    
    # prepare prompt params
    problem = data["question"]
    problem_gt = data["ground_truth"]
    chat_history = data["history"]
    if total_steps != len(chat_history):
        total_steps = len(chat_history)
    if current_step < 0 or current_step >= total_steps:
        raise ValueError(f"current_step should be between 0 and {total_steps - 1}")

    current_rows = []
    for i in range(current_step + 1):
        entry = chat_history[i]
        agent = entry.get("name", "Unknown Agent")
        content = entry.get("content", "")
        current_rows.append(f"Step {i} - {agent}: {content}")
    current_conversation_history = "\n".join(current_rows)
    agent_name = chat_history[current_step].get("name", "Unknown Agent")
    
    prompt_params = {
        "problem": problem,
        "ground_truth": problem_gt,
        "current_conversation_history": current_conversation_history,
        "idx": current_step,
        "agent_name": agent_name
    }
    
    # predict (retry only this step if call/parsing fails)
    last_err = None
    response = None
    for attempt in range(1, max_retries_per_step + 1):
        try:
            response = get_chat_completion(model, method_params, prompt_params)
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(
                f"[WARN] Step {current_step} failed (attempt {attempt}/{max_retries_per_step}): {e}"
            )

    if response is None:
        return {
            "mistake_agent": None,
            "mistake_step": None,
            "reason": f"Step {current_step} failed after {max_retries_per_step} retries: {last_err}",
            "metrics": metrics_acc,
        }

    m = response.get("metrics", {})
    tu = m.get("token_usage", {})
    metrics_acc["num_calls"] += 1
    metrics_acc["latency_s_total"] += m.get("latency_s", 0.0) or 0.0
    metrics_acc["input_tokens_total"] += tu.get("input_tokens", 0) or 0
    metrics_acc["output_tokens_total"] += tu.get("output_tokens", 0) or 0
    metrics_acc["total_tokens_total"] += tu.get("total_tokens", 0) or 0
    metrics_acc["cost_usd_estimate_total"] += m.get("cost_usd_estimate", 0.0) or 0.0

    error_found = response["error_found"]
    reason = response["reason"]
    if error_found:
        return {
            "mistake_agent": agent_name,
            "mistake_step": current_step,
            "reason": reason,
            "metrics": metrics_acc,
        }
    else:
        return step_by_step_single_file(
            model_name=model_name,
            data=data,
            current_step=current_step + 1,
            total_steps=total_steps,
            metrics_acc=metrics_acc,
            max_retries_per_step=max_retries_per_step,
        )

if __name__ == "__main__":
    # Example usage
    model_name = "gpt-4o-mini"
    
    file_path = os.path.join(os.getcwd(), "Who&When\\Algorithm-Generated\\1.json")
    data = load_json(file_path)
    total_steps = len(data["history"])

    # Predict 
    result = step_by_step_single_file(model_name, data, current_step=0, total_steps=total_steps)
    
    # Eval
    gt_mistake_agent = data.get("mistake_agent", None)
    gt_mistake_step = int(data.get("mistake_step", None)) if data.get("mistake_step", None) is not None else None
    print(f"Predicted mistake agent: {result['mistake_agent']}, step: {result['mistake_step']}")
    print(f"Ground truth mistake agent: {gt_mistake_agent}, step: {gt_mistake_step}")
    print(f"Reason: {result['reason']}")
        
    print(f"\t--> Agent level: {'Correct' if (result['mistake_agent'] == gt_mistake_agent) else 'Incorrect'}")
    print(f"\t--> Step level: {'Correct' if (result['mistake_step'] == gt_mistake_step) else 'Incorrect'}")

    # Print metrics
    metrics = result["metrics"]
    print("\nMetrics:")
    print(f"\tNumber of calls: {metrics['num_calls']}")
    print(f"\tTotal latency (s): {metrics['latency_s_total']:.2f}")
    print(f"\tTotal input tokens: {metrics['input_tokens_total']}")
    print(f"\tTotal output tokens: {metrics['output_tokens_total']}")
    print(f"\tTotal tokens: {metrics['total_tokens_total']}")
    print(f"\tTotal cost estimate (USD): ${metrics['cost_usd_estimate_total']:.4f}")
