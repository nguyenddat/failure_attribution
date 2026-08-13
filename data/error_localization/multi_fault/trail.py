import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download

from schemas.trail import Error, LogRecord, Score, Span, Trace

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


def build_log(log: Dict[str, Any]) -> LogRecord:
    return LogRecord(
        timestamp=log["timestamp"],
        trace_id=log["trace_id"],
        span_id=log["span_id"],
        trace_flags=log["trace_flags"],
        severity_text=log.get("severity_text", ""),
        severity_number=log["severity_number"],
        service_name=log.get("service_name", ""),
        body=log.get("body"),
        resource_schema_url=log.get("resource_schema_url") or "",
        resource_attributes=log.get("resource_attributes") or {},
        scope_schema_url=log.get("scope_schema_url") or "",
        scope_name=log.get("scope_name") or "",
        scope_version=log.get("scope_version") or "",
        scope_attributes=log.get("scope_attributes") or {},
        log_attributes=log.get("log_attributes") or {},
        evaluations=log.get("evaluations") or [],
        annotations=log.get("annotations") or [],
    )


def build_span(span: Dict[str, Any]) -> Span:
    return Span(
        timestamp=span["timestamp"],
        trace_id=span["trace_id"],
        span_id=span["span_id"],
        parent_span_id=span.get("parent_span_id"),
        trace_state=span.get("trace_state") or "",
        span_name=span["span_name"],
        span_kind=span.get("span_kind") or "",
        service_name=span.get("service_name") or "",
        resource_attributes=span.get("resource_attributes") or {},
        scope_name=span.get("scope_name") or "",
        scope_version=span.get("scope_version") or "",
        span_attributes=span.get("span_attributes") or {},
        duration=span.get("duration") or "",
        status_code=span.get("status_code") or "",
        status_message=span.get("status_message") or "",
        events=span.get("events") or [],
        links=span.get("links") or [],
        logs=[build_log(log) for log in span.get("logs") or []],
        child_spans=[build_span(child) for child in span.get("child_spans") or []],
    )


QUESTION_PREFIX = "New task:"


def flatten_spans_model(spans: List[Span]) -> Iterator[Span]:
    for span in spans:
        yield span
        yield from flatten_spans_model(span.child_spans)


def extract_question(spans: List[Span]) -> str:
    """Recover the problem statement from the trace.

    TRAIL has no problem statement column: the task text only appears as the
    first user message handed to the agent, prefixed with "New task:". Both
    configs carry it (the GAIA question for ``gaia``, the issue statement for
    ``swe_bench``).
    """
    for span in flatten_spans_model(spans):
        for key in sorted(span.span_attributes):
            value = span.span_attributes[key]
            if not key.endswith(".message.content") or not isinstance(value, str):
                continue
            if value.startswith(QUESTION_PREFIX):
                return value[len(QUESTION_PREFIX):].strip()
    return ""


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

            spans = [build_span(span) for span in trace["spans"]]
            data = Trace(
                trace_id=trace["trace_id"],
                question=extract_question(spans),
                task_source=config,
                spans=spans,
                errors=[Error(**e) for e in labels["errors"]],
                scores=[Score(**s) for s in labels["scores"]],
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
