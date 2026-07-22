from __future__ import annotations

import re


def normalize_agent_name(name: str | None) -> str:
    if not name:
        return ""

    normalized = name.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*\(.*?\)\s*$", "", normalized)
    return normalized


def agent_names_match(gt_agent: str | None, pred_agent: str | None) -> bool:
    gt_normalized = normalize_agent_name(gt_agent)
    pred_normalized = normalize_agent_name(pred_agent)

    if not gt_normalized or not pred_normalized:
        return False

    if gt_normalized == pred_normalized:
        return True

    # Fallback for label variants such as "Orchestrator" vs "Orchestrator (thought)".
    return gt_normalized in pred_normalized or pred_normalized in gt_normalized
