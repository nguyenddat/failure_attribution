"""Loader for the MAST dataset (mcemri/MAST-Data).

The schema lives in ``schemas/mast.py``; it is re-exported here so existing
imports (``from data.error_categorization.mast import Sample, ...``) keep
working.

Keeps every raw column from the source dataset (``mas_name``, ``llm_name``,
``benchmark_name``, ``trace_id``, ``trace.key``/``trace.index``/
``trace.trajectory``, ``mast_annotation``) — nothing is dropped. ``faults``
is a derived convenience field (codes where ``mast_annotation`` is truthy),
kept alongside the raw dict for backward compatibility.

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
    trace: dict = row["trace"]
    annotation: dict = row["mast_annotation"]
    faults = [code for code, value in annotation.items() if value]
    return Sample(
        mas_name=row["mas_name"],
        llm_name=row["llm_name"],
        benchmark_name=row["benchmark_name"],
        trace_id=row["trace_id"],
        trace_key=trace["key"],
        trace_index=trace["index"],
        raw_trajectory=trace["trajectory"],
        mast_annotation=annotation,
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
