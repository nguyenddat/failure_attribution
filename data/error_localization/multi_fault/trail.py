"""Download the TRAIL dataset (PatronusAI/TRAIL) into a single local folder.

TRAIL ships two Hugging Face configs, ``gaia`` and ``swe_bench``. Both are
written to the same output directory; the originating config is kept on each
record via the ``source`` field.

Each trace is stored raw; the schema lives in ``schemas/trail.py``.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download

from schemas.trail import Trace

base_dir = Path(__file__).resolve().parent

dataset_name = "trail"
dataset_repo = "PatronusAI/TRAIL"
dataset_path = base_dir / dataset_name

# Config name -> parquet file inside the HF repo.
CONFIG_FILES = {
    "gaia": "data/gaia-00000-of-00001-33a2e72d362d688a.parquet",
    "swe_bench": "data/swe_bench-00000-of-00001-91aa04220f7198b4.parquet",
}

_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def parse_json(raw: str) -> Dict[str, Any]:
    """Parse a JSON column, repairing the trailing commas some rows contain."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", raw))


def flatten_spans(spans: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Depth-first walk of the nested span tree (``child_spans``)."""
    for span in spans:
        yield span
        yield from flatten_spans(span.get("child_spans") or [])


def load_dataframe(config: str) -> pd.DataFrame:
    path = hf_hub_download(dataset_repo, CONFIG_FILES[config], repo_type="dataset")
    return pd.read_parquet(path)


def load_data_path() -> Path:
    dataset_path.mkdir(parents=True, exist_ok=True)

    index = 0
    skipped: List[Tuple[str, int, str]] = []

    for config in CONFIG_FILES:
        df = load_dataframe(config)

        for row_index, row in df.iterrows():
            file_path = dataset_path / f"{index}.json"
            index += 1

            if file_path.exists():
                continue

            try:
                trace = parse_json(row["trace"])
                labels = parse_json(row["labels"])
            except json.JSONDecodeError as error:
                skipped.append((config, int(row_index), str(error)))
                continue

            data = Trace(
                trace_id=trace["trace_id"],
                source=config,
                trace=trace,
                labels=labels,
            )

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data.model_dump(), file, ensure_ascii=False)

    if skipped:
        print(f"skipped {len(skipped)} unparsable rows:")
        for config, row_index, error in skipped:
            print(f"  {config}[{row_index}]: {error}")

    return dataset_path


if __name__ == "__main__":
    path = load_data_path()
    print(f"{len(list(path.glob('*.json')))} traces in {path}")
