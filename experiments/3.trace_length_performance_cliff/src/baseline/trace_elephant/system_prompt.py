from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class AllAtOnceInput(BaseModel):
    problem: str
    chat_content: str


class StepByStepInput(BaseModel):
    problem: str
    current_step_content: str
    chat_content: str


class AllAtOnceResponse(BaseModel):
    step_number: int = Field(
        ...,
        alias="Step Number",
        description="The step number where the first important mistake occurred.",
    )


class StepByStepResponse(BaseModel):
    error_found: bool = Field(
        ...,
        alias="Error Found",
        description="Whether the current step contains an important mistake.",
    )


all_at_once_parser = PydanticOutputParser(pydantic_object=AllAtOnceResponse)
step_by_step_parser = PydanticOutputParser(pydantic_object=StepByStepResponse)

all_at_once_prompt = """
You are an AI assistant tasked with analyzing a multi-agent conversation history for a real-world problem-solving task.

You will be provided with:
1. The original problem that the agents are trying to solve.
2. The complete conversation history of the agents, organized as a sequence of steps.

Your task is to identify the first step in which any agent made an important mistake that could directly lead to an incorrect final solution.

Important rules:
- Return only the first step where an important mistake occurred.
- Do not mark minor wording issues or harmless inaccuracies as mistakes.
- If multiple mistakes appear later, ignore them and return only the earliest important mistake.
- If the conversation does not contain an obvious mistake, choose the step that is most likely responsible for the incorrect final solution.
- Base your prediction only on the given problem and conversation.

The problem is:
{problem}

The full multi-agent conversation is:
{chat_content}

Please answer strictly in the following JSON format:
"""

step_by_step_prompt = """
You are an AI assistant tasked with evaluating a specific step in a multi-agent conversation for a real-world problem-solving task.

You will be provided with:
1. The original problem that the agents are trying to solve.
2. The content of the current step to evaluate.
3. The surrounding conversation context from the full multi-agent conversation.

Your task is to determine whether the current step contains an important mistake that could directly lead to an incorrect final solution.

The problem is:
{problem}

The content of the current step is:
{current_step_content}

The surrounding conversation context is:
{chat_content}

Important rules:
- Evaluate only the current step, not other steps.
- Use the surrounding conversation context only to understand whether the current step is correct or incorrect.
- Return true only if the current step contains an important mistake that could meaningfully affect the final solution.
- Do not mark minor wording issues, incomplete but harmless reasoning, or stylistic problems as mistakes.
- If the current step is reasonable based on the available context, return false.
- If the current step repeats, relies on, or amplifies an earlier wrong assumption in a way that affects the final solution, return true.
- Base your judgment only on the given problem, the current step, and the provided conversation context.

Please answer strictly in the following JSON format:
"""
