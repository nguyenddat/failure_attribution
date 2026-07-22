from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Response(BaseModel):
    step_number: int = Field(
        ...,
        alias="Step Number",
        description="The step number where the first important mistake occurred.",
    )


parser = PydanticOutputParser(pydantic_object=Response)

prompt = """
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
