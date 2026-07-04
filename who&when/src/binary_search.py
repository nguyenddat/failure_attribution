import os
import sys
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.file import load_json
from src.utils.get_chat_completion import get_chat_completion
from src.utils.schema import Metadata, AccuracyMetrics, CostMetrics, BinarySearchInput


def binary_search_single_file(
    data: dict,
    metadata: Metadata,
    start_step: int,
    end_step: int,
    accuracy_metrics: AccuracyMetrics | None = None,
    cost_metrics: CostMetrics | None = None,
) -> Tuple[AccuracyMetrics, CostMetrics]: 
    if not cost_metrics:
        cost_metrics = CostMetrics(num_input_steps=len(data["history"]), latency=0.0, input_tokens=0, output_tokens=0)
    
    if not accuracy_metrics:
        accuracy_metrics = AccuracyMetrics(
            gt_agent=data.get("mistake_agent", None),
            gt_step=data.get("mistake_step", None),
            pred_agent="Not Found",
            pred_step=-1,
            agent_accuracy=0,
            step_accuracy=0
        )
    
    if start_step > end_step:
        accuracy_metrics.pred_agent = "Not Found"
        accuracy_metrics.pred_step = -1
        accuracy_metrics.step_accuracy = 0
        accuracy_metrics.agent_accuracy = 0
        return accuracy_metrics, cost_metrics
    
    if start_step == end_step:
        accuracy_metrics.pred_agent = data["history"][start_step].get("name", "Unknown Agent")
        accuracy_metrics.pred_step = start_step
        accuracy_metrics.step_accuracy = accuracy_metrics.gt_step == start_step
        accuracy_metrics.agent_accuracy = accuracy_metrics.gt_agent == accuracy_metrics.pred_agent
        return accuracy_metrics, cost_metrics
    
    conversation_history = []
    for i in range(start_step, end_step):
        entry = data["history"][i]
        agent = entry.get("name", "Unknown Agent")
        content = entry.get("content", "")
        conversation_history.append(f"{agent} (step {i}): {content}")
    chat_content = "\n".join(conversation_history)

    method_input = BinarySearchInput(
        problem=data["question"],
        ground_truth=data["ground_truth"],
        chat_segment_history=chat_content,
        start_step=start_step,
        end_step=end_step
    )
    
    # Predict
    result, metrics = get_chat_completion(metadata, method_input)
    
    cost_metrics.latency += metrics["latency"]
    cost_metrics.input_tokens += metrics["input_tokens"]
    cost_metrics.output_tokens += metrics["output_tokens"]
    cost_metrics.num_input_steps += len(conversation_history)
    if result["direction"] == "upper":
        return binary_search_single_file(
            data=data,
            metadata=metadata,
            start_step=(start_step + end_step)//2,
            end_step=end_step,
            accuracy_metrics=accuracy_metrics,
            cost_metrics=cost_metrics
        )
    else:
        return binary_search_single_file(
            data=data,
            metadata=metadata,
            start_step=start_step,
            end_step=(start_step + end_step)//2,
            accuracy_metrics=accuracy_metrics,
            cost_metrics=cost_metrics
        )
            

