from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class AllAtOnceInput(BaseModel):
    question: str
    spans_content: str


class StepByStepInput(BaseModel):
    question: str
    current_span_content: str
    spans_content: str


class AllAtOnceResponse(BaseModel):
    span_id: str = Field(
        ...,
        alias="Span ID",
        description="The id of the span where the first important mistake occurred.",
    )


class StepByStepResponse(BaseModel):
    error_found: bool = Field(
        ...,
        alias="Error Found",
        description="Whether the current span contains an important mistake.",
    )


all_at_once_parser = PydanticOutputParser(pydantic_object=AllAtOnceResponse)
step_by_step_parser = PydanticOutputParser(pydantic_object=StepByStepResponse)

all_at_once_prompt = """
You are an AI assistant analyzing a deep-research agent trajectory that has
been segmented into an ordered sequence of semantic spans. Each span covers
one continuous local goal (planning, retrieval, verification, comparison,
finalization).

You will be provided with:
1. The original research question.
2. The full ordered sequence of spans.

Your task is to identify the id of the FIRST span in which the agent made
an important mistake (an unsupported, contradicted, or prematurely
committed claim) that could directly affect the final answer.

Important rules:
- Return only the id of the first span containing an important mistake.
- Do not mark normal exploration, failed searches, tentative hypotheses,
  already-corrected errors, or harmless tool noise as mistakes.
- If multiple mistakes appear later, ignore them and return only the
  earliest one.
- Base your prediction only on the given question and spans.

The research question is:
{question}

The ordered spans are:
{spans_content}

Please answer strictly in the following JSON format:
"""

step_by_step_prompt = """
You are an AI assistant evaluating one span of a deep-research agent
trajectory that has been segmented into an ordered sequence of semantic
spans. Each span covers one continuous local goal (planning, retrieval,
verification, comparison, finalization).

You will be provided with:
1. The original research question.
2. The content of the current span to evaluate.
3. The full ordered sequence of spans as surrounding context.

Your task is to determine whether the current span contains an important
mistake (an unsupported, contradicted, or prematurely committed claim) that
could directly affect the final answer.

The research question is:
{question}

The content of the current span is:
{current_span_content}

The full ordered spans are:
{spans_content}

Important rules:
- Evaluate only the current span, not other spans.
- Use the surrounding spans only to judge whether the current span is
  correct.
- Return true only if the current span introduces, reuses, amplifies, or
  finalizes a claim that is unsupported or contradicted.
- Do not mark normal exploration, failed searches, tentative hypotheses,
  already-corrected errors, or harmless tool noise as mistakes.

Please answer strictly in the following JSON format:
"""
