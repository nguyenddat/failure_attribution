from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

prompt = """
You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.
Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.
The problem to address is as follows: {problem}
The Answer for the problem is: {ground_truth}
Review the following conversation segment from step {start_step} to step {end_step}:

{chat_segment_history}

Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({{upper}}) or the lower half ({{lower}}) of this segment.
Please provide your prediction strictly in the JSON format:
"""

class Response(BaseModel):
    direction: Literal["upper", "lower"] = Field(..., description="Either upper half or lower half.")

parser = PydanticOutputParser(pydantic_object=Response)
