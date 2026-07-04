from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

prompt = """
You are an AI assistant performing a local refinement pass over a narrow region of a multi-agent conversation to identify the agent and step most responsible for the error.

Problem: {problem}
Ground truth answer: {ground_truth}

Coarse scan findings:
{coarse_scan_context}

Refinement window:
{refinement_window}

Your task:
1. Use the refinement window plus the coarse findings to identify the most likely first important mistake in this local region.
2. Return the exact step number and agent name only if you can justify them from the refinement window.
3. If the local region still does not contain a clear important mistake, return null for step_number and agent_name, and explain why.

Important rules:
- Prefer the first important mistake, not a later downstream consequence.
- Only use a step number that appears in the refinement window.
- Only use an agent name that appears in the refinement window.
- Do not guess if the window is inconclusive.

Respond strictly in the JSON format:
"""


class Response(BaseModel):
    step_number: int | None = Field(
        default=None,
        description="The exact step number of the first important mistake in the refinement window, or null if inconclusive."
    )
    agent_name: str | None = Field(
        default=None,
        description="The exact agent name responsible for the chosen step, or null if inconclusive."
    )
    reason: str = Field(..., description="Explanation for the final refinement judgment.")


parser = PydanticOutputParser(pydantic_object=Response)
