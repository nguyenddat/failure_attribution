import os
import sys
import json
from pathlib import Path
from typing import Any, Dict
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.file import load_json
from src.utils.get_chat_completion import get_chat_completion
from src.utils.schema import Metadata, AccuracyMetrics, CostMetrics, \
    StepByStepInput

def step_by_step_single_file(
    data: dict,
    current_step: int,
    total_steps: int,
    metadata: Metadata,
    accuracy_metrics: AccuracyMetrics | None = None,
    cost_metrics: CostMetrics | None = None,
    progress_bar: tqdm | None = None,
):  
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
            gt_agent=data.get("mistake_agent", None),
            gt_step=data.get("mistake_step", None),
            pred_agent="Not Found",
            pred_step=-1,
            agent_accuracy=0,
            step_accuracy=0
        )
    
    if progress_bar is not None:
        progress_bar.set_description(f"step_by_step step {current_step + 1}/{total_steps}")

    if current_step >= total_steps:
        accuracy_metrics.pred_agent = "Not Found"
        accuracy_metrics.pred_step = -1
        accuracy_metrics.step_accuracy = 0
        accuracy_metrics.agent_accuracy = 0
        return accuracy_metrics, cost_metrics
    
    conversation_history = []
    for i in range(current_step + 1):
        entry = data["history"][i]
        agent = entry.get("name", "Unknown Agent")
        content = entry.get("content", "")
        conversation_history.append(f"{agent}: {content}")
    current_conversation_history = "\n".join(conversation_history)
    agent_name = data["history"][current_step].get("name", "Unknown Agent")
            
    method_input = StepByStepInput(
        problem=data["question"],
        ground_truth=data["ground_truth"],
        current_conversation_history=current_conversation_history,
        idx=current_step,
        agent_name=agent_name
    )
    
    # Predict
    result, metrics = get_chat_completion(metadata, method_input)
    
    cost_metrics.latency += metrics["latency"]
    cost_metrics.input_tokens += metrics["input_tokens"]
    cost_metrics.output_tokens += metrics["output_tokens"]
    cost_metrics.num_input_steps += len(conversation_history)
    if result["error_found"]:
        accuracy_metrics.pred_agent = agent_name
        accuracy_metrics.pred_step = current_step
        accuracy_metrics.step_accuracy = accuracy_metrics.gt_step == current_step
        accuracy_metrics.agent_accuracy = accuracy_metrics.gt_agent == agent_name
        return accuracy_metrics, cost_metrics
    
    return step_by_step_single_file(
        data=data,
        current_step=current_step + 1,
        total_steps=total_steps,
        metadata=metadata,
        accuracy_metrics=accuracy_metrics,
        cost_metrics=cost_metrics,
        progress_bar=progress_bar
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    file_path = project_root / "Who&When" / "Algorithm-Generated" / "1.json"
    data = load_json(file_path)
    total_steps = len(data["history"])

    # Predict
    metadata = Metadata(model_name="gpt-4o-mini", method="step_by_step", with_gt=True)
    accuracy_metrics, cost_metrics = step_by_step_single_file(data, current_step=0, total_steps=total_steps, metadata=metadata)
    print(f"Predicted mistake agent: {accuracy_metrics.pred_agent}, step: {accuracy_metrics.pred_step}")
    print((
        f"Cost:\n",
        f"\tLatency: {cost_metrics.latency}s\n",
        f"\tInput tokens: {cost_metrics.input_tokens}\n",
        f"\tOutput tokens: {cost_metrics.output_tokens}\n"
        f"\nNumber of input steps: {cost_metrics.num_input_steps}"
    )
    )
