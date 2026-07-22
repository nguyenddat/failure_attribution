from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    subtask: str = Field(description="A concise description of the semantic subtask.")
    from_step: int = Field(
        description="The first step index belonging to this subtask."
    )
    to_step: int = Field(
        description="The last step index belonging to this subtask, inclusive."
    )


class Response(BaseModel):
    keep: bool = Field(
        description="Whether the current decomposition is already aligned and needs no improvement."
    )
    subtasks: Optional[List[SubTask]] = Field(
        default=None,
        description="The revised subtasks if the current decomposition needs improvement; otherwise null.",
    )


parser = PydanticOutputParser(pydantic_object=Response)

prompt = """
You are an AI assistant performing trajectory-aligned reflection for CHIEF task decomposition.

You will be provided with:
1. The original problem the agents attempted to solve.
2. The complete normalized multi-agent trajectory.
3. A current subtask decomposition draft.

Your task is to judge whether the current decomposition is already aligned with the trajectory.

Reflection rules:
- Set `keep` to true only if the current decomposition is already good enough and does not need improvement.
- If `keep` is true, set `subtasks` to null.
- If `keep` is false, return a revised `subtasks` list.
- Each item in `subtasks` must contain `subtask`, `from_step`, and `to_step`.
- Each subtask must represent one continuous range of steps.
- The subtasks must be ordered by `from_step`.
- The decomposition must follow the actual execution flow in the trajectory.
- Revise only when necessary.

Original problem:
{problem}

Full normalized multi-agent trajectory:
{chat_content}

Current subtask decomposition:
{subtasks}

Please answer strictly in the following JSON format:
"""
