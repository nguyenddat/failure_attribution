from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "error_localization" / "single_fault"

DATASET_DIRS: dict[str, Path] = {
    "who_and_when__hand-crafted": DATA_DIR / "who_and_when__hand-crafted",
    "trace_elephant": DATA_DIR / "trace_elephant",
}


def normalize_who_and_when(data: dict) -> Optional[dict]:
    if data.get("mistake_step") is None or data.get("mistake_agent") is None:
        return None
    return {
        "question": data["question"],
        "trajectory": [
            {
                "step": int(item["step"]),
                "agent_name": item["agent_name"],
                "content": item["content"],
            }
            for item in data["trajectory"]
        ],
        "mistake_agent": data["mistake_agent"],
        "mistake_step": int(data["mistake_step"]),
    }


def _trace_elephant_content(step: dict) -> str:
    choices = step.get("output", {}).get("choices") or [{}]
    message = choices[0].get("message", {})
    content = message.get("content") or ""
    if content:
        return content

    # swe-agent traces put tool calls in step["tool_logs"] instead of
    # message["tool_calls"]; fall back to it when the message has none.
    calls = message.get("tool_calls") or step.get("tool_logs") or []
    parts = [
        f"[tool_call: {call.get('function', {}).get('name')}"
        f"({call.get('function', {}).get('arguments')})]"
        for call in calls
    ]
    return " ".join(parts)


def normalize_trace_elephant(data: dict) -> Optional[dict]:
    if data.get("mistake_step") is None or data.get("mistake_agent") is None:
        return None
    return {
        "question": data["task_instruction"],
        "trajectory": [
            {
                "step": idx,
                "agent_name": step["agent_name"],
                "content": _trace_elephant_content(step),
            }
            for idx, step in enumerate(data["trajectory"])
        ],
        "mistake_agent": data["mistake_agent"],
        # raw mistake_step is on the 1-based step_id scale; trajectory
        # steps above are re-indexed to 0-based idx, so shift to match.
        "mistake_step": int(data["mistake_step"]) - 1,
    }


NORMALIZERS: dict[str, Callable[[dict], Optional[dict]]] = {
    "who_and_when__hand-crafted": normalize_who_and_when,
    "trace_elephant": normalize_trace_elephant,
}
