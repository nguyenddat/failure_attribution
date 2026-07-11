import os

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

model_names = ["deepseek_v4_flash", "gpt-4o-mini"]
models = {
    "deepseek_v4_flash": "deepseek/deepseek-v4-flash",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "deepseek-v4-flash:free": "deepseek/deepseek-v4-flash:free"
}
# USD per 1M tokens (update when provider pricing changes)
model_pricing_per_1m = {
    "deepseek-v4-flash:free": {
        "input": 0.0,
        "output": 0.0,
    },
    "deepseek_v4_flash": {
        "input": 0.10,
        "output": 0.20,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    }
}
model_cache = {}

def get_model(model_name: str) -> ChatOpenAI:
    if model_name not in model_cache:
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")
        
        model = ChatOpenAI(
            model=models[model_name],
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        model_cache[model_name] = model
    return model_cache[model_name]
