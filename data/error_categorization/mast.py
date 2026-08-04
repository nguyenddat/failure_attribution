"""Loader for the MAST dataset (mcemri/MAST-Data).

The schema lives in ``schemas/mast.py``; it is re-exported here so existing
imports (``from data.error_categorization.mast import Sample, ...``) keep working.

Traces are unstructured logs, so each sample is written with the whole log in
``raw_trajectory`` and no step splitting. Segmentation is a separate, optional
step (``build_agent_behaviors.py``).
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

from schemas.mast import (  # noqa: F401  (re-exported for experiments)
    AgentBehavior,
    FailureGroup,
    FailureMode,
    MAST_METADATA,
    Metadata,
    Sample,
    render_taxonomy,
)

base_dir = Path(__file__).resolve().parent
json_dir = base_dir / "mast"

dataset_link = "hf://datasets/mcemri/MAST-Data/MAD_full_dataset.json"


def row_to_sample(row: pd.Series) -> Sample:
    annotation: dict = row["mast_annotation"]
    faults = [code for code, value in annotation.items() if value]
    return Sample(
        mas_name=row["mas_name"],
        raw_trajectory=row["trace"]["trajectory"],
        faults=faults,
    )


def write_metadata() -> None:
    metadata_path = json_dir / "metadata.json"
    metadata_path.write_text(MAST_METADATA.model_dump_json(indent=2), encoding="utf-8")


def write_samples(df: pd.DataFrame) -> None:
    for i, (_, row) in enumerate(df.iterrows()):
        file_path = json_dir / f"{i}.json"
        if file_path.exists():
            continue
        sample = row_to_sample(row)
        file_path.write_text(sample.model_dump_json(indent=2), encoding="utf-8")


def load_full_dataframe() -> pd.DataFrame:
    ds = load_dataset("json", data_files=dataset_link)
    return pd.DataFrame(ds["train"])


def load_data_path() -> Path:
    json_dir.mkdir(parents=True, exist_ok=True)
    write_metadata()
    write_samples(load_full_dataframe())
    return json_dir


if __name__ == "__main__":
    load_data_path()
