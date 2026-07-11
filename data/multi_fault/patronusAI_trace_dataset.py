import os
import json
from pathlib import Path
from typing import Any, List, Optional


import pandas as pd
from datasets import load_dataset

from utils import AgentBehavior, FaultyAgent, Data
from utils import dataset_name_to_filename

base_dir = Path(__file__).resolve().parent

dataset_name = "patronusai/trace-dataset"
dataset_path = base_dir / "json" / dataset_name_to_filename(dataset_name).replace(".json", "")
dataset_path.mkdir(parents=True, exist_ok=True)


ds = load_dataset("PatronusAI/trace-dataset")
df = ds["train"].to_pandas()

def conversation_to_trajectory(conversation: Any):
    segments = []
    current_user_request = None
    current_content_parts = []

    for item in conversation:
        role = item.get("role")
        content = item.get("content") or ""

        if role == "user":
            if current_user_request is not None:
                segments.append({
                    "user_request": current_user_request,
                    "agent_name": "Single Agent",
                    "content": "\n\n".join(current_content_parts),
                })

            current_user_request = content
            current_content_parts = []

        else:
            if current_user_request is None:
                continue

            message_text = f"[{role}]\n{content}"

            tool_calls = item.get("tool_calls") or []
            if tool_calls:
                tool_calls_content = [
                    json.dumps(call, ensure_ascii=False, indent=2)
                    for call in tool_calls
                ]

                message_text += "\nTool calls:\n" + "\n".join(tool_calls_content)

            current_content_parts.append(message_text)

    # Save final segment
    if current_user_request is not None:
        segments.append({
            "user_request": current_user_request,
            "agent_name": "Single Agent",
            "content": "\n\n".join(current_content_parts),
        })

    for i, segment in enumerate(segments):
        segment["step"] = i

    return [AgentBehavior(**segment) for segment in segments]

def label_to_gt(label):
    steps = label.split(".")
    faulty_agents = [FaultyAgent(step=step) for step in steps]
    return faulty_agents

def load_data_path() -> Path:
    for i, row in df.iterrows():
        file_path = dataset_path / f"{i}.json"
        if os.path.exists(file_path):
            continue
        
        trajectory = conversation_to_trajectory(json.loads(row["conversation"]))
        faulty_agents = label_to_gt(row["label"])
        data = Data(trajectory=trajectory, faulty_agents=faulty_agents)
        
        with open(file_path, "w") as file:
          json.dump(data.model_dump(), file)

    return dataset_path

load_data_path()