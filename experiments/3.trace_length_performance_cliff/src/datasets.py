from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIRS: dict[str, Path] = {
    "who_and_when__hand-crafted": REPO_ROOT
    / "data/error_localization/single_fault/who_and_when__hand-crafted",
    "trace_elephant": REPO_ROOT
    / "data/error_localization/single_fault/trace_elephant",
    "telbench": REPO_ROOT / "data/error_localization/multi_fault/telbench",
}


def normalize_who_and_when(data: dict) -> Optional[dict]:
    if data.get("mistake_step") is None or data.get("mistake_agent") is None:
        return None
    return {
        "question": data["question"],
        "trajectory": data["trajectory"],
        "mistake_agent": data["mistake_agent"],
        "mistake_step": int(data["mistake_step"]),
    }


def _extract_trace_elephant_content(step: dict) -> str:
    choices = step.get("output", {}).get("choices") or [{}]
    message = choices[0].get("message", {})
    content = message.get("content") or ""
    if content:
        return content

    parts = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        parts.append(f"[tool_call: {fn.get('name')}({fn.get('arguments')})]")
    return " ".join(parts)


def normalize_trace_elephant(data: dict) -> Optional[dict]:
    if data.get("mistake_step") is None or data.get("mistake_agent") is None:
        return None

    trajectory = [
        {
            "step": idx,
            "agent_name": step["agent_name"],
            "content": _extract_trace_elephant_content(step),
        }
        for idx, step in enumerate(data["trajectory"])
    ]
    return {
        "question": data["task_instruction"],
        "trajectory": trajectory,
        "mistake_agent": data["mistake_agent"],
        "mistake_step": int(data["mistake_step"]),
    }


def normalize_telbench(data: dict) -> Optional[dict]:
    spans = data["spans"]
    error_span_ids = data["gold"]["error_span_ids"]
    if not spans or not error_span_ids:
        return None

    span_ids = [span["id"] for span in spans]
    target_id = error_span_ids[0]
    if target_id not in span_ids:
        return None

    trajectory = [
        {"step": idx, "agent_name": "agent", "content": span["raw"]}
        for idx, span in enumerate(spans)
    ]
    return {
        "question": data["question"],
        "trajectory": trajectory,
        "mistake_agent": "agent",
        "mistake_step": span_ids.index(target_id),
    }


NORMALIZERS: dict[str, Callable[[dict], Optional[dict]]] = {
    "who_and_when__hand-crafted": normalize_who_and_when,
    "trace_elephant": normalize_trace_elephant,
    "telbench": normalize_telbench,
}
