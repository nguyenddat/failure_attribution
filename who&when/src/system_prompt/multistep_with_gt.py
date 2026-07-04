from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

prompt = """
You are an AI assistant tasked with evaluating a batch of recent steps in an ongoing multi-agent conversation that is solving a real-world problem.

Problem: {problem}
Ground truth answer: {ground_truth}

Here is the summary of previous evaluations so far:
{previous_evaluation_history}

Here are the current {num_steps} step(s) to evaluate together:
{current_steps}

Your task:
1. Evaluate the current batch of steps using both the ground truth and the previous evaluation history.
2. If you can clearly identify a step in the current batch that contains an important error which could hinder the reasoning process or lead to an incorrect final answer, return:
   - the step_number of the first clearly erroneous step, and
   - a concise explanation of why it is incorrect.
3. If you cannot confidently identify such an error yet, do not guess. Instead, return a concise summary of what happened in the current batch so it can be used as context for later evaluations.

Important rules:
- Do not be overly critical. Only flag errors that clearly matter.
- Only identify an error if it occurs in the current batch of steps, not in the previous summarized history.
- Use the previous evaluation history only as context, not as a target for identifying errors.
- If multiple current steps appear problematic, return the first clearly important erroneous step.
- If no clear error can be identified yet, return only a summary.
- The returned step_number must correspond exactly to the step number shown in the current batch.

Respond strictly in the following JSON format:
"""

class Response(BaseModel):
    step_number: int | None = Field(
        default=None,
        description="The number of the first clearly erroneous step in the current batch. Null if no clear error is identified."
    )
    reason: str | None = Field(
        default=None,
        description="A concise explanation of why the identified step is incorrect. Null if no clear error is identified."
    )
    summary: str | None = Field(
        default=None,
        description="A concise summary of the current batch when no clear error can be identified yet. Null if an error is identified."
    )

parser = PydanticOutputParser(pydantic_object=Response)