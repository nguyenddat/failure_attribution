from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

prompt = """
You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.
The problem is: {problem}
The Answer for the problem is: {ground_truth}

Identify which agent made an error, at which step, and explain the reason for the error.
Here's the conversation:

{chat_content}

Based on this conversation, please predict the following:
1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.
2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows:
{{
    "agent a": "xx",
    "agent b": "xxxx",
    "agent c": "xxxxx",
    "agent a": "xxxxxxx"
}},
each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.
3. The reason for your prediction.

Please answer strictly in the JSON format:
"""

class Response(BaseModel):
    agent_name: str = Field(..., alias="Agent Name", description="The agent directly responsible for the wrong solution.")
    step_number: int = Field(..., alias="Step Number", description="The step where the first mistake occurred.")
    reason_for_mistake: str = Field(..., alias="Reason for Mistake", description="The reason for the prediction.")

    class Config:
        populate_by_name = True
        extra = "ignore"


parser = PydanticOutputParser(pydantic_object=Response)
