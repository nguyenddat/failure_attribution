from __future__ import annotations

import tiktoken

MODEL_TOKEN_THRESHOLDS: dict[str, int] = {
    "gpt-4o-mini": 128_000 - 2_000,
    "deepseek-v4-flash": 1_050_000 - 2_000,
}

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def exceeds_threshold(model_key: str, token_count: int) -> bool:
    return token_count > MODEL_TOKEN_THRESHOLDS[model_key]
