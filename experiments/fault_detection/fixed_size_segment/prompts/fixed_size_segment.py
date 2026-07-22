from typing import List

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Response(BaseModel):
    faults: List[str] = Field(
        default_factory=list,
        alias="Faults",
        description=(
            "List of failure mode codes (e.g. '1.1', '2.3') present in THIS segment. "
            "Return an empty list if the segment contains no clear evidence of failure."
        ),
    )


parser = PydanticOutputParser(pydantic_object=Response)

prompt = """
You are an AI assistant tasked with analyzing ONE SEGMENT of a multi-agent conversation trajectory for a real-world problem-solving task. The full trajectory has been split into fixed-size segments; you are only shown one of them.

You will be provided with:
1. The taxonomy of known failure modes, each with a code, name, and description.
2. Whether this segment is the FINAL segment of the trajectory (true/false).
3. The content of this segment, organized as a sequence of log lines.

Your task is to identify every failure mode from the taxonomy for which there is clear, self-contained evidence WITHIN THIS SEGMENT.

Important rules:
- This segment may start or end mid-action; do not assume missing context implies a failure.
- Some failure modes can only be judged once the trajectory has actually ended (e.g. premature termination, unawareness of termination conditions, missing or incorrect verification). If `is_final_segment` is false, do NOT flag these end-of-trajectory failure modes unless the segment itself already contains explicit, unambiguous evidence of them. If `is_final_segment` is true, judge them normally using this segment's content.
- A segment may contain zero, one, or multiple failure modes. Do not force a prediction if you find no clear evidence.
- Only return a failure mode code if there is clear evidence of it in this segment. Do not guess.
- Base your prediction only on the given taxonomy and segment content, do not use outside knowledge of the task domain.
- Return each applicable code at most once.

The failure mode taxonomy is:
{failure_mode}

is_final_segment: {is_final_segment}

The segment content is:
{trajectory}

Please answer strictly in the following JSON format:
"""
