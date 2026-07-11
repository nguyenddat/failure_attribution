from typing import List, Optional

from pydantic import BaseModel

class AgentBehavior(BaseModel):
    user_request: Optional[str] = None
    
    step: int
    agent_name: str
    content: str

class FaultyAgent(BaseModel):
    step: Optional[int] = None
    agent_name: Optional[str] = None
    error_type: Optional[str] = None

class Data(BaseModel):
    # problem fields
    question: Optional[str] = None

    # trajectory fields
    trajectory: List[AgentBehavior]

    # labels
    faulty_agents: List[FaultyAgent]


# UTIL FUNCTIONS
def dataset_name_to_filename(name: str) -> str:
    return name.replace("/", "__").replace("&", "_and_")
