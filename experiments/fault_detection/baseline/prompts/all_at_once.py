from typing import List

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Response(BaseModel):
    faults: List[str] = Field(
        default_factory=list,
        alias="Faults",
        description=(
            "List of failure mode codes (e.g. '1.1', '2.3') present in the trajectory. "
            "Return an empty list if the trajectory contains no failure."
        ),
    )


parser = PydanticOutputParser(pydantic_object=Response)

prompt = """
You are an AI assistant tasked with analyzing a multi-agent conversation trajectory for a real-world problem-solving task.

You will be provided with:
1. The taxonomy of known failure modes, each with a code, name, and description.
2. The complete trajectory of the multi-agent system, organized as a sequence of steps.

Your task is to identify every failure mode from the taxonomy that occurred anywhere in the trajectory.

Important rules:
- A trajectory may contain zero, one, or multiple failure modes. Many trajectories succeed completely and contain none — do not force a prediction if you find no clear evidence of failure.
- Only return a failure mode code if there is clear evidence of it in the trajectory. Do not guess.
- A failure mode label applies to the trajectory as a whole; you do not need to point to a specific step or agent.
- Base your prediction only on the given taxonomy and trajectory, do not use outside knowledge of the task domain.
- Return each applicable code at most once.

The failure mode taxonomy is:
{failure_mode}

The full multi-agent trajectory is:
{trajectory}

Please answer strictly in the following JSON format:
"""
