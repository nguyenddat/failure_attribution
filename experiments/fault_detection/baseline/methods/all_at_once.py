from typing import Optional, Tuple

from pydantic import BaseModel

from data.fault_detection.mast import Sample, Metadata, render_taxonomy
from ...metrics import calculate_performance_metrics, PerformanceMetrics, CostMetrics
from ..get_chat_completion import get_chat_completion

class AllAtOnceIn(BaseModel):
    trajectory: str
    failure_mode: str

class ExperimentMetadata(BaseModel):
    model_name: str
    dataset_name: str
    segmentation: Optional[str] = None

def all_at_once(
    sample: Sample,
    data_metadata: Metadata,
    experiment_metadata: ExperimentMetadata
) -> Tuple[PerformanceMetrics, CostMetrics]:
    raw_trajectory = sample.raw_trajectory
    gt_faults = sample.faults

    prompt_params = AllAtOnceIn(
        trajectory=raw_trajectory,
        failure_mode=render_taxonomy(data_metadata),
    )

    result, cost_metrics = get_chat_completion(
        metadata=experiment_metadata,
        method="all_at_once",
        prompt_params=prompt_params,
    )

    pred_faults = result["faults"]

    performance_metrics = calculate_performance_metrics(
        gt_faults=gt_faults,
        pred_faults=pred_faults,
    )
    cost_metrics = CostMetrics(**cost_metrics)

    return performance_metrics, cost_metrics
    
