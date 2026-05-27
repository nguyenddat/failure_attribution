from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

prompt = """
You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem.
The problem being addressed is: {problem}.
The Answer for the problem is: {ground_truth}

Here is the conversation history up to the current step:
{current_conversation_history}

The most recent step ({idx}) was by '{agent_name}'.

Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution.
Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process.

Respond strictly in the JSON format:
"""

class Response(BaseModel):
    error_found: bool = Field(..., description="Indicates whether an error was found in the most recent step.")
    reason: str = Field(..., description="A clear explanation for the judgment made regarding the most recent step.")

parser = PydanticOutputParser(pydantic_object=Response)
