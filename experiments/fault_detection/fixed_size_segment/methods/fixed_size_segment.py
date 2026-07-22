import math
from typing import List, Tuple

import tiktoken
from pydantic import BaseModel

from data.fault_detection.mast import Sample, Metadata, render_taxonomy
from ...metrics import calculate_performance_metrics, PerformanceMetrics, CostMetrics
from ..get_chat_completion import get_chat_completion
from ...baseline.methods.all_at_once import ExperimentMetadata

SEGMENT_LEVELS = [0.25, 0.50, 0.70, 1.0]

_ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")


class FixedSizeSegmentIn(BaseModel):
    trajectory: str
    failure_mode: str
    is_final_segment: bool


def build_segments(raw_trajectory: str, segment_ratio: float) -> List[str]:
    tokens = _ENCODING.encode(raw_trajectory)
    total_tokens = len(tokens)
    segment_size = max(1, math.ceil(total_tokens * segment_ratio))
    return [
        _ENCODING.decode(tokens[start : start + segment_size])
        for start in range(0, total_tokens, segment_size)
    ]


def fixed_size_segment(
    sample: Sample,
    data_metadata: Metadata,
    experiment_metadata: ExperimentMetadata,
    segment_ratio: float,
) -> Tuple[PerformanceMetrics, CostMetrics]:
    segments = build_segments(sample.raw_trajectory, segment_ratio)
    failure_mode_text = render_taxonomy(data_metadata)

    pred_faults: set[str] = set()
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0

    for index, segment_text in enumerate(segments):
        prompt_params = FixedSizeSegmentIn(
            trajectory=segment_text,
            failure_mode=failure_mode_text,
            is_final_segment=(index == len(segments) - 1),
        )

        result, cost_metrics = get_chat_completion(
            metadata=experiment_metadata,
            method="fixed_size_segment",
            prompt_params=prompt_params,
        )

        pred_faults.update(result["faults"])
        total_input_tokens += cost_metrics["input_tokens"]
        total_output_tokens += cost_metrics["output_tokens"]
        total_latency += cost_metrics["latency"]

    performance_metrics = calculate_performance_metrics(
        gt_faults=sample.faults,
        pred_faults=sorted(pred_faults),
    )
    cost_metrics_total = CostMetrics(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        latency=total_latency,
    )

    return performance_metrics, cost_metrics_total
