from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers.fix import OutputFixingParser

from experiments.fault_detection.baseline.prompts.all_at_once import (
    prompt as prompt_all_at_once,
    parser as parser_all_at_once,
)
from experiments.models import get_model

if TYPE_CHECKING:
    from experiments.fault_detection.baseline.methods.all_at_once import (
        AllAtOnceIn,
        ExperimentMetadata,
    )


def get_prompt(method: str):
    if method == "all_at_once":
        prompt = prompt_all_at_once
        parser = parser_all_at_once

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Please choose a valid combination."
        )

    system_messages = ChatPromptTemplate.from_messages(
        [
            ("system", prompt + "\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return system_messages, parser


def get_chat_completion(
    metadata: ExperimentMetadata,
    method: str,
    prompt_params: AllAtOnceIn,
):
    model = get_model(metadata.model_name)
    system_messages, parser = get_prompt(method=method)

    prompt_value = system_messages.invoke(prompt_params.model_dump())

    t0 = time.perf_counter()
    ai_msg = model.invoke(prompt_value)
    latency = time.perf_counter() - t0

    cost_metrics = {}
    cost_metrics["input_tokens"] = ai_msg.usage_metadata["input_tokens"]
    cost_metrics["output_tokens"] = ai_msg.usage_metadata["output_tokens"]
    cost_metrics["latency"] = latency

    parser = OutputFixingParser.from_llm(parser=parser, llm=model)
    result = parser.invoke(ai_msg).model_dump()
    return result, cost_metrics
