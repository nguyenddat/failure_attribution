from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class BothInput(BaseModel):
    problem: str
    current_step_content: str
    before_content: str
    after_content: str


class PrevOnlyInput(BaseModel):
    problem: str
    current_step_content: str
    before_content: str


class NextOnlyInput(BaseModel):
    problem: str
    current_step_content: str
    after_content: str


class Response(BaseModel):
    error_found: bool = Field(
        ..., description="Whether the current step contains an important mistake."
    )


response_parser = PydanticOutputParser(pydantic_object=Response)


both_prompt = """
You are an AI assistant evaluating one step of a multi-agent conversation for a
real-world problem-solving task. You are given a fixed-size window of context
around the current step: the steps immediately BEFORE it and the steps
immediately AFTER it.

Your judgment must combine both directions:
1. Compare the current step against what the preceding steps establish — does
   it follow logically from what came before, or does it already deviate here?
2. Compare the current step against what the following steps show — are its
   consequences, as reflected afterwards, consistent with a correct step, or
   do later steps reveal that this step's action/claim was wrong?

Return true only if the current step itself — not an earlier step, not a later
step — is where an important mistake occurred that could directly lead to an
incorrect final solution. Minor wording issues or harmless inaccuracies are not
mistakes.

The problem is:
{problem}

The steps immediately before the current step are:
{before_content}

The current step to evaluate is:
{current_step_content}

The steps immediately after the current step are:
{after_content}

Please answer strictly in the following JSON format:
"""


prev_only_prompt = """
You are an AI assistant evaluating one step of a multi-agent conversation for a
real-world problem-solving task. You are given only the steps immediately
BEFORE the current step — no future steps are available to you.

Every step before the current one has already been checked and judged to be
free of important mistakes. Your task is causal: given that everything before
the current step is correct, does the current step introduce a NEW mistake
that a correct continuation would not contain?

Return true only if the current step deviates from what a correct continuation
of the given prior context would look like. If the current step is a
reasonable continuation given the (correct) prior context, return false — even
if you suspect a problem might appear later, since you cannot see later steps
and must judge only what is given here.

The problem is:
{problem}

The steps immediately before the current step (already confirmed correct) are:
{before_content}

The current step to evaluate is:
{current_step_content}

Please answer strictly in the following JSON format:
"""


next_only_prompt = """
You are an AI assistant evaluating one step of a multi-agent conversation for a
real-world problem-solving task. You are given only the steps immediately
AFTER the current step — no history before it is available to you.

Your task is root-cause reasoning, working backward from consequences: look at
the steps that follow for signs of a problem — confusion, contradictions,
corrections, failed verifications, or a wrong result propagating forward — and
judge whether the current step is plausibly the ROOT CAUSE that set that
problem in motion.

Return true only if the current step is the point that plausibly introduced
the failure visible afterwards — not merely any step that happens to precede a
bad outcome. If the following steps show no evidence of a problem traceable to
the current step, return false.

The problem is:
{problem}

The current step to evaluate is:
{current_step_content}

The steps immediately after the current step are:
{after_content}

Please answer strictly in the following JSON format:
"""
