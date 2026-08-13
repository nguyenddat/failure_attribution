import os

from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

model_names = ["embedding_3_small"]
models = {"embedding_3_small": "openai/text-embedding-3-small"}

model_cache = {}
def get_model(model_name: str) -> OpenAIEmbeddings:
    if model_name not in model_cache:
        if model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(models.keys())}"
            )

        model_cache[model_name] = OpenAIEmbeddings(
            model=models[model_name],
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return model_cache[model_name]
