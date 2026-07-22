from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Response(BaseModel):
    error_found: bool = Field(
        ..., description="Whether the current step contains an important mistake."
    )


parser = PydanticOutputParser(pydantic_object=Response)

prompt = """
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
