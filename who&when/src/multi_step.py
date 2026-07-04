import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.file import load_json
from src.utils.get_chat_completion import get_chat_completion
from src.utils.schema import (
    Metadata,
    AccuracyMetrics,
    CostMetrics,
    MultiStepCoarseInput,
    MultiStepRefineInput,
)


@dataclass
class ChunkScanResult:
    start_step: int
    end_step: int
    summary: str
    suspect_agents: list[str]
    earliest_suspect_step: int | None
    latest_suspect_step: int | None
    confidence: float


def _build_step_block(data: dict, start_step: int, end_step: int) -> str:
    lines: list[str] = []
    for i in range(start_step, end_step):
        entry = data["history"][i]
        agent = entry.get("name", "Unknown Agent")
        content = entry.get("content", "")
        lines.append(f"[step={i}] [agent={agent}] {content}")
    return "\n".join(lines)


def _accumulate_cost(cost_metrics: CostMetrics, metrics: dict, num_steps: int) -> None:
    cost_metrics.latency += metrics["latency"]
    cost_metrics.input_tokens += metrics["input_tokens"]
    cost_metrics.output_tokens += metrics["output_tokens"]
    cost_metrics.num_input_steps += num_steps


def coarse_scan_conversation(
    data: dict,
    chunk_size: int,
    metadata: Metadata,
    cost_metrics: CostMetrics,
    progress_bar: tqdm | None = None,
) -> list[ChunkScanResult]:
    results: list[ChunkScanResult] = []
    total_steps = len(data["history"])
    current_step = 0
    previous_summaries: list[str] = []

    while current_step < total_steps:
        end_step = min(current_step + chunk_size, total_steps)
        if progress_bar is not None:
            progress_bar.set_description(f"Multistep coarse: step {current_step}-{end_step} / {total_steps}")

        method_input = MultiStepCoarseInput(
            problem=data["question"],
            ground_truth=data["ground_truth"],
            previous_summaries="\n".join(previous_summaries),
            current_chunk=_build_step_block(data, current_step, end_step),
        )

        coarse_metadata = Metadata(
            model_name=metadata.model_name,
            method="multistep_coarse",
            with_gt=metadata.with_gt,
        )
        result, metrics = get_chat_completion(coarse_metadata, method_input)
        _accumulate_cost(cost_metrics, metrics, end_step - current_step)

        chunk_result = ChunkScanResult(
            start_step=current_step,
            end_step=end_step,
            summary=result["summary"],
            suspect_agents=result["suspect_agents"],
            earliest_suspect_step=result["earliest_suspect_step"],
            latest_suspect_step=result["latest_suspect_step"],
            confidence=float(result["confidence"]),
        )
        results.append(chunk_result)

        previous_summaries.append(
            f"Chunk {current_step}-{end_step - 1}: {chunk_result.summary}"
        )
        current_step = end_step

    return results


def pick_suspicious_chunk(chunk_results: list[ChunkScanResult]) -> ChunkScanResult | None:
    suspicious_chunks = [
        result
        for result in chunk_results
        if result.suspect_agents
        and result.earliest_suspect_step is not None
        and result.latest_suspect_step is not None
    ]
    if not suspicious_chunks:
        return None

    return max(
        suspicious_chunks,
        key=lambda result: (result.confidence, -result.earliest_suspect_step, -(result.end_step - result.start_step)),
    )


def refine_suspicious_chunk(
    data: dict,
    suspicious_chunk: ChunkScanResult,
    metadata: Metadata,
    cost_metrics: CostMetrics,
    total_steps: int,
) -> tuple[str, int, str]:
    refine_start = max(0, suspicious_chunk.earliest_suspect_step - 1)
    refine_end = min(total_steps, suspicious_chunk.latest_suspect_step + 2)

    coarse_context_lines = []
    coarse_context_lines.append(
        f"Suspicious chunk: steps {suspicious_chunk.start_step}-{suspicious_chunk.end_step - 1}"
    )
    coarse_context_lines.append(f"Summary: {suspicious_chunk.summary}")
    coarse_context_lines.append(
        f"Suspect agents: {', '.join(suspicious_chunk.suspect_agents) if suspicious_chunk.suspect_agents else 'None'}"
    )
    coarse_context_lines.append(
        f"Suspect range: {suspicious_chunk.earliest_suspect_step} - {suspicious_chunk.latest_suspect_step}"
    )
    coarse_context_lines.append(f"Confidence: {suspicious_chunk.confidence:.2f}")

    method_input = MultiStepRefineInput(
        problem=data["question"],
        ground_truth=data["ground_truth"],
        coarse_scan_context="\n".join(coarse_context_lines),
        refinement_window=_build_step_block(data, refine_start, refine_end),
    )
    refine_metadata = Metadata(
        model_name=metadata.model_name,
        method="multistep_refine",
        with_gt=metadata.with_gt,
    )
    result, metrics = get_chat_completion(refine_metadata, method_input)
    _accumulate_cost(cost_metrics, metrics, refine_end - refine_start)

    pred_step = result["step_number"]
    pred_agent = result["agent_name"]
    reason = result["reason"]

    if pred_step is None or pred_agent is None:
        fallback_step = suspicious_chunk.earliest_suspect_step
        fallback_agent = data["history"][fallback_step].get("name", "Unknown Agent")
        return fallback_agent, fallback_step, reason

    if not (refine_start <= pred_step < refine_end):
        fallback_step = suspicious_chunk.earliest_suspect_step
        fallback_agent = data["history"][fallback_step].get("name", "Unknown Agent")
        return fallback_agent, fallback_step, reason

    valid_agents = {
        data["history"][i].get("name", "Unknown Agent")
        for i in range(refine_start, refine_end)
    }
    if pred_agent not in valid_agents:
        fallback_step = suspicious_chunk.earliest_suspect_step
        fallback_agent = data["history"][fallback_step].get("name", "Unknown Agent")
        return fallback_agent, fallback_step, reason

    return pred_agent, int(pred_step), reason


def multistep_single_file(
    data: dict,
    num_steps: int,
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
        progress_bar.set_description(f"Multistep: step {current_step} - {current_step+num_steps} / {total_steps}")

    if current_step >= total_steps:
        accuracy_metrics.pred_agent = "Not Found"
        accuracy_metrics.pred_step = -1
        accuracy_metrics.step_accuracy = 0
        accuracy_metrics.agent_accuracy = 0
        return accuracy_metrics, cost_metrics

    chunk_results = coarse_scan_conversation(
        data=data,
        chunk_size=num_steps,
        metadata=metadata,
        cost_metrics=cost_metrics,
        progress_bar=progress_bar,
    )
    suspicious_chunk = pick_suspicious_chunk(chunk_results)

    if suspicious_chunk is None:
        accuracy_metrics.pred_agent = "Not Found"
        accuracy_metrics.pred_step = -1
        accuracy_metrics.step_accuracy = 0
        accuracy_metrics.agent_accuracy = 0
        return accuracy_metrics, cost_metrics

    pred_agent, pred_step, _ = refine_suspicious_chunk(
        data=data,
        suspicious_chunk=suspicious_chunk,
        metadata=metadata,
        cost_metrics=cost_metrics,
        total_steps=total_steps,
    )

    accuracy_metrics.pred_agent = pred_agent
    accuracy_metrics.pred_step = pred_step
    accuracy_metrics.step_accuracy = accuracy_metrics.gt_step == pred_step
    accuracy_metrics.agent_accuracy = accuracy_metrics.gt_agent == pred_agent
    return accuracy_metrics, cost_metrics


if __name__ == "__main__":
    # directory
    project_root = Path(__file__).resolve().parents[1]
    
    data_dir = project_root / "Who&When" / "Algorithm-Generated"
    
    accuracy_dir = project_root / ".." / "outputs" / "accuracy"
    accuracy_file =  accuracy_dir / "multi_step_accuracy.csv"
    accuracy_cols = ["file name", "gt_agent", "gt_step", "pred_agent", "pred_step", "agent_accuracy", "step_accuracy"]
    accuracy_df = pd.DataFrame(columns=accuracy_cols)
    if accuracy_file.exists():
        accuracy_df = pd.read_csv(accuracy_file)
    
    cost_dir = project_root / ".." / "outputs" / "cost"
    cost_file = cost_dir / "multi_step_cost.csv"
    cost_cols = ["file name", "latency", "input_tokens", "output_tokens", "num_input_steps"]
    cost_df = pd.DataFrame(columns=cost_cols)
    if cost_file.exists():
        cost_df = pd.read_csv(cost_file)
    
    # run benchmarks
    metadata = Metadata(model_name="gpt-4o-mini", method="multistep", with_gt=True)
    for file_name in os.listdir(data_dir):
        file = data_dir / file_name
        if not file.suffix == ".json":
            continue
        
        # check if already run before
        if accuracy_df[accuracy_df["file name"] == file.name].shape[0] > 0:
            if cost_df[cost_df["file name"] == file.name].shape[0] > 0:
                continue
        
        data = load_json(file)
        total_steps = len(data["history"])
        accuracy_metrics, cost_metrics = multistep_single_file(
            data=data,
            num_steps=3,
            current_step=0,
            total_steps=total_steps,
            metadata=metadata
        )
        
        accuracy_df.loc[len(accuracy_df)] = {
            "file name": file_name,
            "gt_agent": accuracy_metrics.gt_agent,
            "gt_step": accuracy_metrics.gt_step,
            "pred_agent": accuracy_metrics.pred_agent,
            "pred_step": accuracy_metrics.pred_step,
            "agent_accuracy": accuracy_metrics.agent_accuracy,
            "step_accuracy": accuracy_metrics.step_accuracy,
        }

        cost_df.loc[len(cost_df)] = {
            "file name": file_name,
            "latency": cost_metrics.latency,
            "input_tokens": cost_metrics.input_tokens,
            "output_tokens": cost_metrics.output_tokens,
            "num_input_steps": cost_metrics.num_input_steps,
        }

        accuracy_df.to_csv(accuracy_file, index=False)
        cost_df.to_csv(cost_file, index=False)

