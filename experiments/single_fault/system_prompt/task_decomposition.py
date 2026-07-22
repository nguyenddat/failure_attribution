from typing import List

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class SubTask(BaseModel):
    subtask: str = Field(description="A concise description of the semantic subtask.")
    from_step: int = Field(
        description="The first step index belonging to this subtask."
    )
    to_step: int = Field(
        description="The last step index belonging to this subtask, inclusive."
    )


class Response(BaseModel):
    subtasks: List[SubTask]


parser = PydanticOutputParser(pydantic_object=Response)

# Cải tiến prompt: Thêm {format_instructions} và tối ưu hóa các quy tắc ràng buộc
prompt = """
You are an AI assistant performing CHIEF task decomposition for a failed multi-agent trajectory.

You will be provided with:
1. The original problem the agents attempted to solve.
2. Retrieved task-solving exemplars from a knowledge base.
3. The complete multi-agent trajectory, represented as normalized steps.
4. The total number of normalized steps.

Your task is to divide the trajectory into a sequence of semantic subtasks.

Decomposition rules:
- Each item in `subtasks` must contain `subtask`, `from_step`, and `to_step`.
- Each subtask must represent one continuous range of steps.
- The subtasks must be ordered by `from_step`.
- The first subtask must start at step 0.
- Consecutive subtasks must be contiguous: the next `from_step` must equal the previous `to_step` + 1.
- Every trajectory step must belong to exactly one subtask. No steps should be omitted or overlapped.
- Create boundaries only when the semantic objective or type of work changes. Do not create a separate subtask for every individual message.
- Base the decomposition on the actual trajectory. Use the exemplars only as guidance for the level of abstraction.
- Describe what the agents attempted to accomplish, not whether the attempt succeeded or failed.

Original problem:
{problem}

Retrieved exemplars:
{exemplars}

Full normalized multi-agent trajectory:
{chat_content}

Please answer strictly in the following JSON format:
"""
