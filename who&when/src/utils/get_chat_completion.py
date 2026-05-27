from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time
import re

from ..system_prompt.step_by_step_with_gt import (
    prompt as prompt_step_by_step_with_gt, 
    parser as parser_step_by_step_with_gt
) 
from .models import model_pricing_per_1m

def get_prompt(
    method: str,
    with_gt: bool = False,
):
    if method == "step_by_step" and with_gt:
        prompt = prompt_step_by_step_with_gt
        parser = parser_step_by_step_with_gt
    
    else:
        raise ValueError(f"Unsupported method '{method}' with_gt={with_gt}. Please choose a valid combination.")
    
    system_messages = ChatPromptTemplate.from_messages(
        [
            ("system", prompt + "\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return system_messages, parser

def get_chat_completion(
    model: ChatOpenAI,
    method_params: dict,
    prompt_params: dict,
):
    # Get prompt + parser
    method = method_params.get("method")
    with_gt = method_params.get("with_gt", False)
    system_messages, parser = get_prompt(method=method, with_gt=with_gt)
    
    # Invoke raw model first so we can capture latency + usage metadata
    prompt_value = system_messages.invoke(prompt_params)
    t0 = time.perf_counter()
    ai_msg = model.invoke(prompt_value)
    latency_s = time.perf_counter() - t0

    raw_text = (ai_msg.content or "").strip()
    parsed = None
    parse_mode = "pydantic_json"
    try:
        parsed = parser.invoke(ai_msg).dict()
    except Exception:
        # Fallback: robust text parse for "1. Yes/No" + "Reason:"
        parse_mode = "text_fallback"
        txt = raw_text.lower()
        first_line = raw_text.splitlines()[0].strip().lower() if raw_text else ""
        error_found = (
            first_line.startswith("1. yes")
            or first_line.startswith("yes")
            or bool(re.search(r"\b1\.\s*yes\b", txt))
            or (("yes" in txt) and ("no" not in txt[:40]))
        )
        reason = raw_text
        m = re.search(r"reason\s*:\s*(.*)", raw_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            reason = m.group(1).strip()
        parsed = {
            "error_found": bool(error_found),
            "reason": reason if reason else "Parser fallback: empty model response.",
        }
    usage = ai_msg.response_metadata.get("token_usage", {}) if ai_msg.response_metadata else {}
    input_tokens = usage.get("prompt_tokens", 0) or 0
    output_tokens = usage.get("completion_tokens", 0) or 0
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    alias = method_params.get("model_name")
    pricing = model_pricing_per_1m.get(alias, {})
    input_price = pricing.get("input")
    output_price = pricing.get("output")
    cost_usd_estimate = None
    if input_price is not None and output_price is not None:
        cost_usd_estimate = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

    parsed["metrics"] = {
        "latency_s": latency_s,
        "parse_mode": parse_mode,
        "raw_response": raw_text,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "cost_usd_estimate": cost_usd_estimate,
    }
    return parsed
