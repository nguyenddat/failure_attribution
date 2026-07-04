from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

prompt = """
You are an AI assistant performing a coarse scan over one chunk of a multi-agent conversation that is solving a real-world problem.

Problem: {problem}
Ground truth answer: {ground_truth}

Previous coarse summaries:
{previous_summaries}

Current chunk:
{current_chunk}

Your task:
1. Read the current chunk in context of the problem and the previous coarse summaries.
2. Decide whether this chunk contains a plausible mistake that could materially contribute to a wrong final answer.
3. If suspicious, identify the suspect agents and the narrowest plausible step range inside this chunk.
4. If not suspicious, still provide a concise summary for downstream reasoning.

Important rules:
- This is a coarse scan, so do not force an exact final answer.
- Only use step numbers that appear in the current chunk.
- If there is no meaningful suspicion, return an empty suspect_agents list and null suspect step bounds.
- Confidence must be a number between 0 and 1.

Respond strictly in the JSON format:
"""


class Response(BaseModel):
    summary: str = Field(..., description="A concise summary of the current chunk.")
    suspect_agents: list[str] = Field(
        default_factory=list,
        description="List of suspect agent names in this chunk. Empty if there is no meaningful suspicion."
    )
    earliest_suspect_step: int | None = Field(
        default=None,
        description="Earliest suspicious step in the current chunk, or null if there is no meaningful suspicion."
    )
    latest_suspect_step: int | None = Field(
        default=None,
        description="Latest suspicious step in the current chunk, or null if there is no meaningful suspicion."
    )
    confidence: float = Field(
        ...,
        description="Confidence from 0 to 1 that this chunk contains the mistake or the earliest clearly relevant mistake."
    )


parser = PydanticOutputParser(pydantic_object=Response)
