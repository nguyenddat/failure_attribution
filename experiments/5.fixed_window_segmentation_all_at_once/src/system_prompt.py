from __future__ import annotations

from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class SegmentInput(BaseModel):
    problem: str
    segment_content: str


class Response(BaseModel):
    error_found: bool = Field(
        ..., description="Whether any step in the given segment contains an important mistake."
    )
    step_id: Optional[int] = Field(
        default=None,
        description=(
            "The step id of the step where the mistake occurred, if error_found "
            "is true. Must be one of the step ids listed in the segment. Null if "
            "error_found is false."
        ),
    )


response_parser = PydanticOutputParser(pydantic_object=Response)


segment_prompt = """
You are an AI assistant evaluating a segment of consecutive steps from a
multi-agent conversation for a real-world problem-solving task. You are given
ALL steps in this segment at once, each labeled with its step id.

Examine every step in the segment and determine whether ANY of them contains
an important mistake that could directly lead to an incorrect final solution.
Minor wording issues or harmless inaccuracies are not mistakes.

If you find such a mistake, return error_found=true and step_id set to the
id of the step where the mistake first occurred. If no step in the segment
contains such a mistake, return error_found=false and step_id=null.

The problem is:
{problem}

The steps in this segment are:
{segment_content}

Please answer strictly in the following JSON format:
"""
