from typing import List

from pydantic import BaseModel

# SCHEMAS

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

# UTIL FUNCTIONS
def dataset_name_to_filename(name: str) -> str:
    return name.replace("/", "__").replace("&", "_and_")
