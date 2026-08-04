"""Schema for the who&when dataset (Kevin355/Who_and_When).

Shared by the loaders under ``data/error_localization/single_fault`` and by any
experiment reading the generated JSON files.
"""

from typing import List

from pydantic import BaseModel


class AgentBehavior(BaseModel):
    step: int
    agent_name: str
    content: str


class Data(BaseModel):
    # problem fields
    question: str

    # trajectory fields
    trajectory: List[AgentBehavior]

    # labels
    mistake_step: int
    mistake_agent: str
