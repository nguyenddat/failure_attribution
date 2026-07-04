import os
import sys
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.file import load_json
from src.utils.get_chat_completion import get_chat_completion
from src.utils.schema import Metadata, AccuracyMetrics, CostMetrics, AllAtOnceInput


def all_at_once_single_file(
    data: dict,
    metadata: Metadata,
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

    conversation_history = []
    for i in range(len(data["history"])):
        entry = data["history"][i]
        agent = entry.get("name", "Unknown Agent")
        content = entry.get("content", "")
        conversation_history.append(f"{agent}: {content}")
    chat_content = "\n".join(conversation_history)

    method_input = AllAtOnceInput(
        problem=data["question"],
        ground_truth=data["ground_truth"],
        chat_content=chat_content
    )
    
    # Predict
    result, metrics = get_chat_completion(metadata, method_input)
    
    cost_metrics.latency += metrics["latency"]
    cost_metrics.input_tokens += metrics["input_tokens"]
    cost_metrics.output_tokens += metrics["output_tokens"]
    
    accuracy_metrics.pred_agent = result["agent_name"]
    accuracy_metrics.pred_step = result["step_number"]
    accuracy_metrics.step_accuracy = accuracy_metrics.gt_step == accuracy_metrics.pred_step
    accuracy_metrics.agent_accuracy = accuracy_metrics.gt_agent == accuracy_metrics.pred_agent
    return accuracy_metrics, cost_metrics

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "who&when\\Who&When\\Algorithm-Generated\\1.json")
    data = load_json(file_path)
    total_steps = len(data["history"])

    # Predict
    metadata = Metadata(model_name="gpt-4o-mini", method="all_at_once", with_gt=True)
    accuracy_metrics, cost_metrics = all_at_once_single_file(data, metadata=metadata)
    print(f"Predicted mistake agent: {accuracy_metrics.pred_agent}, step: {accuracy_metrics.pred_step}")
    print((
        f"Cost:\n",
        f"\tLatency: {cost_metrics.latency}s\n",
        f"\tInput tokens: {cost_metrics.input_tokens}\n",
        f"\tOutput tokens: {cost_metrics.output_tokens}\n"
        f"\nNumber of input steps: {cost_metrics.num_input_steps}"
    )
    )