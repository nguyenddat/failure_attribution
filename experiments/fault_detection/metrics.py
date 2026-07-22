from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel

MAST_LABEL_SPACE_SIZE = 14

def _group_of(code: str) -> str:
    return code.split(".", 1)[0]


class PerformanceMetrics(BaseModel):
    gt_faults: list[str]
    pred_faults: list[str]

    exact_match_ratio: Literal[0, 1]
    hamming_loss: float

    precision: float
    recall: float
    f1: float

    group_precision: float              # group tolerance
    group_recall: float                 # group tolerance
    group_f1: float                     # group tolerance


def calculate_performance_metrics(
    gt_faults: list[str],
    pred_faults: list[str],
    total_label_space_size: int = MAST_LABEL_SPACE_SIZE,
) -> PerformanceMetrics:
    gt_set = set(gt_faults)
    pred_set = set(pred_faults)

    # 1. Exact Match Ratio (Subset Accuracy)
    exact_match = 1 if gt_set == pred_set else 0

    # 2. Precision, Recall, F1
    true_positives = len(gt_set & pred_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    # Precision khong xac dinh khi khong predict gi ca -> quy uoc: 1.0 neu gt cung rong, nguoc lai 0.0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else (1.0 if len(gt_set) == 0 else 0.0)
    )

    # Recall khong xac dinh khi gt rong (khong co gi de "nho lai") -> luon 1.0, bat ke pred co du thua hay khong
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 1.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    symmetric_difference = false_positives + false_negatives
    hamming_loss = (
        symmetric_difference / total_label_space_size
        if total_label_space_size > 0
        else 0.0
    )

    # 3. Group-tolerant: khop theo nhom loi, gioi han boi so luong moi nhom (khong dem trung)
    pred_groups = Counter(_group_of(code) for code in pred_faults)
    gt_groups = Counter(_group_of(code) for code in gt_faults)
    group_true_positives = sum((pred_groups & gt_groups).values())
    group_false_positives = len(pred_faults) - group_true_positives
    group_false_negatives = len(gt_faults) - group_true_positives

    group_precision = (
        group_true_positives / (group_true_positives + group_false_positives)
        if (group_true_positives + group_false_positives) > 0
        else (1.0 if len(gt_faults) == 0 else 0.0)
    )
    group_recall = (
        group_true_positives / (group_true_positives + group_false_negatives)
        if (group_true_positives + group_false_negatives) > 0
        else 1.0
    )
    if group_precision + group_recall == 0:
        group_f1 = 0.0
    else:
        group_f1 = (
            2 * (group_precision * group_recall) / (group_precision + group_recall)
        )

    return PerformanceMetrics(
        gt_faults=gt_faults,
        pred_faults=pred_faults,
        hamming_loss=round(hamming_loss, 4),
        exact_match_ratio=exact_match,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        group_precision=round(group_precision, 4),
        group_recall=round(group_recall, 4),
        group_f1=round(group_f1, 4),
    )
