from __future__ import annotations

from unittest.mock import patch

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage

from experiments.single_fault.utils.llm import (
    invoke_structured,
    is_context_length_exceeded,
)


class _EchoResponse(BaseModel):
    value: str = Field(..., alias="Value")


class _FakeModel:
    def __init__(self, reply_value: str):
        self.reply_value = reply_value
        self.received_prompt = None

    def invoke(self, prompt_value):
        self.received_prompt = prompt_value
        return AIMessage(
            content=f'{{"Value": "{self.reply_value}"}}',
            usage_metadata={
                "input_tokens": 42,
                "output_tokens": 7,
                "total_tokens": 49,
            },
        )


def test_invoke_structured_returns_parsed_result_and_cost():
    parser = PydanticOutputParser(pydantic_object=_EchoResponse)
    fake_model = _FakeModel(reply_value="ok")

    with patch(
        "experiments.single_fault.utils.llm.get_model", return_value=fake_model
    ):
        result, cost = invoke_structured(
            model_name="gpt-4o-mini",
            prompt_template="Say {greeting}.",
            parser=parser,
            prompt_params=_GreetingParams(greeting="hi"),
        )

    assert result == {"value": "ok"}
    assert cost == {"latency": cost["latency"], "input_tokens": 42, "output_tokens": 7}
    assert cost["latency"] >= 0.0


class _GreetingParams(BaseModel):
    greeting: str


def test_is_context_length_exceeded_matches_known_phrases():
    assert is_context_length_exceeded(Exception("Error: context_length_exceeded"))
    assert is_context_length_exceeded(
        Exception("This model's maximum context length is 128000 tokens")
    )
    assert not is_context_length_exceeded(Exception("rate limit exceeded"))
