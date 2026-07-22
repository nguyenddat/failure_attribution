from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers.fix import OutputFixingParser

from experiments.fault_detection.fixed_size_segment.prompts.fixed_size_segment import (
    prompt as prompt_fixed_size_segment,
    parser as parser_fixed_size_segment,
)
from experiments.models import get_model

if TYPE_CHECKING:
    from experiments.fault_detection.fixed_size_segment.methods.fixed_size_segment import (
        FixedSizeSegmentIn,
    )
    from experiments.fault_detection.baseline.methods.all_at_once import (
        ExperimentMetadata,
    )


def get_prompt(method: str):
    if method == "fixed_size_segment":
        prompt = prompt_fixed_size_segment
        parser = parser_fixed_size_segment

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
    prompt_params: FixedSizeSegmentIn,
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
