import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers.fix import OutputFixingParser

from single_fault.system_prompt.step_by_step import (
    prompt as prompt_step_by_step_with_gt, 
    parser as parser_step_by_step_with_gt
)
from single_fault.system_prompt.all_at_one import (
    prompt as prompt_all_at_one_with_gt,
    parser as parser_all_at_one_with_gt,
)
from single_fault.utils.models import get_model
from single_fault.utils.schema import (
    AllAtOnceInput,
    StepByStepInput,
    Metadata,
)

def clean_llm_json_text(text: str) -> str:
    return (
        text.strip()
        .strip()
        .replace("\\$", "$")
    )

def get_prompt(method: str):
    if method == "step_by_step":
        prompt = prompt_step_by_step_with_gt
        parser = parser_step_by_step_with_gt
        
    elif method == "all_at_once":
        prompt = prompt_all_at_one_with_gt
        parser = parser_all_at_one_with_gt
    
    else:
        raise ValueError(f"Unsupported method '{method}'. Please choose a valid combination.")
    
    system_messages = ChatPromptTemplate.from_messages(
        [
            ("system", prompt + "\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return system_messages, parser


def get_chat_completion(
    metadata: Metadata,
    prompt_params: AllAtOnceInput | StepByStepInput,
):
    model = get_model(metadata.model_name)
    method = metadata.method
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
