import json
from pathlib import Path
from typing import Any, Dict

def load_json(file_url: str) -> Dict[str, Any]:
    file_path = Path(file_url)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)