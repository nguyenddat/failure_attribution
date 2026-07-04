import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers.fix import OutputFixingParser

from src.system_prompt.step_by_step_with_gt import (
    prompt as prompt_step_by_step_with_gt, 
    parser as parser_step_by_step_with_gt
)
from src.system_prompt.all_at_one_with_gt import (
    prompt as prompt_all_at_one_with_gt,
    parser as parser_all_at_one_with_gt,
)
from src.system_prompt.binary_search_with_gt import (
    prompt as prompt_binary_search_with_gt,
    parser as parser_binary_search_with_gt,
)
from src.system_prompt.multistep_with_gt import (
    prompt as prompt_multistep_with_gt,
    parser as parser_multistep_with_gt,
)
from src.system_prompt.multistep_coarse_with_gt import (
    prompt as prompt_multistep_coarse_with_gt,
    parser as parser_multistep_coarse_with_gt,
)
from src.system_prompt.multistep_refine_with_gt import (
    prompt as prompt_multistep_refine_with_gt,
    parser as parser_multistep_refine_with_gt,
)
from src.utils.models import get_model
from src.utils.schema import (
    AllAtOnceInput,
    BinarySearchInput,
    MultiStepCoarseInput,
    MultiStepInput,
    MultiStepRefineInput,
    StepByStepInput,
    Metadata,
)

def clean_llm_json_text(text: str) -> str:
    return (
        text.strip()
        .strip()
        .replace("\\$", "$")
    )

def get_prompt(method: str, with_gt: bool = False):
    if method == "step_by_step" and with_gt:
        prompt = prompt_step_by_step_with_gt
        parser = parser_step_by_step_with_gt
        
    elif method == "all_at_once" and with_gt:
        prompt = prompt_all_at_one_with_gt
        parser = parser_all_at_one_with_gt
    
    elif method == "binary_search" and with_gt:
        prompt = prompt_binary_search_with_gt
        parser = parser_binary_search_with_gt

    elif method == "multistep" and with_gt:
        prompt = prompt_multistep_with_gt
        parser = parser_multistep_with_gt

    elif method == "multistep_coarse" and with_gt:
        prompt = prompt_multistep_coarse_with_gt
        parser = parser_multistep_coarse_with_gt

    elif method == "multistep_refine" and with_gt:
        prompt = prompt_multistep_refine_with_gt
        parser = parser_multistep_refine_with_gt
    
    else:
        raise ValueError(f"Unsupported method '{method}' with_gt={with_gt}. Please choose a valid combination.")
    
    system_messages = ChatPromptTemplate.from_messages(
        [
            ("system", prompt + "\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return system_messages, parser


def get_chat_completion(
    metadata: Metadata,
    prompt_params: AllAtOnceInput | BinarySearchInput | MultiStepCoarseInput | MultiStepInput | MultiStepRefineInput | StepByStepInput,
):
    model = get_model(metadata.model_name)    
    method = metadata.method
    with_gt = metadata.with_gt
    system_messages, parser = get_prompt(method=method, with_gt=with_gt)
    
    # Latency metric
    prompt_value = system_messages.invoke(prompt_params.model_dump())
        
    t0 = time.perf_counter()
    ai_msg = ai_msg = model.invoke(prompt_value)
    latency = time.perf_counter() - t0

    # Token metric
    cost_metrics = {}
    cost_metrics["input_tokens"] = ai_msg.usage_metadata["input_tokens"]
    cost_metrics["output_tokens"] = ai_msg.usage_metadata["output_tokens"]
    cost_metrics["latency"] = latency

    parser = OutputFixingParser.from_llm(parser=parser, llm=model)
    result = parser.invoke(ai_msg).model_dump()
    return result, cost_metrics
