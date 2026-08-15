from __future__ import annotations

import time

from langchain_classic.output_parsers.fix import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from experiments.chat_models import get_model


def invoke_structured(
    model_name: str,
    prompt_template: str,
    parser,
    prompt_params: BaseModel,
) -> tuple[dict, dict]:
    model = get_model(model_name)
    system_messages = ChatPromptTemplate.from_messages(
        [("system", prompt_template + "\n{format_instructions}")]
    ).partial(format_instructions=parser.get_format_instructions())

    prompt_value = system_messages.invoke(prompt_params.model_dump())

    t0 = time.perf_counter()
    ai_msg = model.invoke(prompt_value)
    latency = time.perf_counter() - t0

    cost = {
        "latency": latency,
        "input_tokens": ai_msg.usage_metadata["input_tokens"],
        "output_tokens": ai_msg.usage_metadata["output_tokens"],
    }

    try:
        result = parser.invoke(ai_msg).model_dump()
    except Exception:  # noqa: BLE001 - fall back to output-fixing on any parse failure
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
        result = fixing_parser.invoke(ai_msg).model_dump()
    return result, cost
