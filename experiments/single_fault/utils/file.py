import json
from pathlib import Path
from typing import Any, Dict, Iterable


def load_json(file_url: str) -> Dict[str, Any]:
    file_path = Path(file_url)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_agent_behaviors(agent_behaviors: Iterable[Any]) -> str:
    formatted_steps = []

    for index, behavior in enumerate(agent_behaviors):
        if isinstance(behavior, dict):
            step = behavior.get("step", index)
            agent_name = behavior.get("agent_name", "Unknown Agent")
            content = behavior.get("content", "")
        else:
            step = getattr(behavior, "step", index)
            agent_name = getattr(behavior, "agent_name", "Unknown Agent")
            content = getattr(behavior, "content", "")

        formatted_steps.append(f"- step {step} - {agent_name}: {content}")

    return "\n".join(formatted_steps)
