from __future__ import annotations

import math
from typing import List

from pydantic import BaseModel

try:
    import tiktoken
except Exception:
    tiktoken = None


class AgentBehavior(BaseModel):
    step: int
    agent_name: str
    content: str


def build_token_counter():
    if tiktoken is None:
        return approximate_token_count

    try:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        return lambda text: len(tokenizer.encode(text or ""))
    except Exception:
        return approximate_token_count


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


count_tokens = build_token_counter()


def token_based_segment(
    trajectory: List[AgentBehavior],
    max_token_per_window: int,
    overlap_ratio: float = 0.5,
) -> List[List[AgentBehavior]]:
    segments: List[List[AgentBehavior]] = []
    start_idx = 0

    while start_idx < len(trajectory):
        current_segment: List[AgentBehavior] = []
        current_token_count = 0
        end_idx = start_idx

        while end_idx < len(trajectory):
            step = trajectory[end_idx]
            step_token_count = count_tokens(step.content)
            if current_segment and current_token_count + step_token_count > max_token_per_window:
                break

            current_segment.append(step)
            current_token_count += step_token_count
            end_idx += 1

        if not current_segment:
            current_segment.append(trajectory[start_idx])
            end_idx = start_idx + 1

        segments.append(list(current_segment))

        if end_idx >= len(trajectory):
            break

        overlap_steps = int(len(current_segment) * overlap_ratio)
        overlap_steps = min(overlap_steps, max(len(current_segment) - 1, 0))
        stride = max(1, len(current_segment) - overlap_steps)
        start_idx += stride

    return segments
