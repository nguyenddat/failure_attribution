from typing import List, Any

from pydantic import BaseModel

class AgentBehavior(BaseModel):
    step: int
    agent_name: str
    content: str

def step_based_segment(
    trajectory: List[AgentBehavior],
    slide_window: int,
    overlapping: int
) -> List[List[AgentBehavior]]:
    
    segments: List[List[AgentBehavior]] = []
    stride = slide_window - overlapping
    
    for start_idx in range(0, len(trajectory), stride):
        end_idx = start_idx + slide_window
        segment = trajectory[start_idx:end_idx]

        if segment:
            segments.append(segment)

        if end_idx >= len(trajectory):
            break

    return segments