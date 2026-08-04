"""Download the TRAIL dataset (PatronusAI/TRAIL) into a single local folder.

TRAIL ships two Hugging Face configs, ``gaia`` and ``swe_bench``. Both are
written to the same output directory, numbered consecutively.

Only the fields needed for error localization are written out; see
``schemas/trail.py`` for the resulting shape and ``build_span`` below for what
gets dropped.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download

from schemas.trail import LogRecord, Span, Trace

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


# Project/tenant bookkeeping, identical on every span of a trace.
NOISE_ATTRIBUTE_PREFIXES = ("pat.",)
NOISE_ATTRIBUTES = ("input.mime_type", "output.mime_type")

# On LLM spans these two restate llm.input_messages.*/llm.output_messages.*
# verbatim as one JSON blob and account for ~40% of the raw dataset size.
REDUNDANT_IO_ATTRIBUTES = ("input.value", "output.value")


def select_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    drop = set(NOISE_ATTRIBUTES)
    if "llm.input_messages.0.message.content" in attributes:
        drop.update(REDUNDANT_IO_ATTRIBUTES)

    return {
        key: value
        for key, value in attributes.items()
        if key not in drop and not key.startswith(NOISE_ATTRIBUTE_PREFIXES)
    }


def build_log(log: Dict[str, Any]) -> LogRecord:
    # The rest of a log record is transport metadata (trace/span ids already on
    # the span, resource/scope descriptors, empty evaluations/annotations).
    return LogRecord(
        timestamp=log["timestamp"],
        severity_text=log.get("severity_text", ""),
        body=log.get("body"),
    )


def build_span(span: Dict[str, Any]) -> Span:
    """Keep identity, ordering, content and structure; drop OTel plumbing.

    Dropped: trace_id/trace_state/trace_flags, service_name, resource_attributes,
    scope_name/scope_version, links, duration and status (``span_kind`` is
    ``Internal`` and ``status_code`` ``Unset`` on every span in the dataset).
    """
    return Span(
        span_id=span["span_id"],
        parent_span_id=span.get("parent_span_id"),
        name=span["span_name"],
        timestamp=span["timestamp"],
        attributes=select_attributes(span.get("span_attributes") or {}),
        logs=[build_log(log) for log in span.get("logs") or []],
        events=span.get("events") or [],
        child_spans=[build_span(child) for child in span.get("child_spans") or []],
    )


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
                spans=[build_span(span) for span in trace["spans"]],
                errors=labels["errors"],
                scores=labels["scores"][0],
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
