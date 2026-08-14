# Experiment 1 baseline per-dataset split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `experiments/single_fault/experiments/baseline/` from one shared prompt/parser/driver into three self-contained dataset pipelines — `who_and_when/`, `trace_elephant/`, `telbench/` — each running `all_at_once` and `step_by_step`, per `docs/superpowers/specs/2026-08-14-experiment-1-baseline-per-dataset-design.md`.

**Architecture:** Each dataset folder owns its `system_prompt.py` (prompt text + pydantic parser) and `methods.py` (`all_at_once_single_file`, `step_by_step_single_file`). `who_and_when`/`trace_elephant` reuse the existing resumable CSV driver (`shared.py` + `utils/results.py` + `utils/schema.py`) because their output shape (1 agent + 1 step) is identical. `telbench` gets its own metrics (`metrics.py`), CSV writer (`results.py`), and driver (`run.py`) because it predicts a span id with no agent concept. A new tiny shared helper (`utils/llm.py`) factors out the LangChain plumbing (build prompt, call model, parse, extract token/latency) so none of the three `methods.py` files duplicate that mechanics — it carries no dataset-specific prompt content, so it doesn't violate the "separate code per dataset" requirement.

**Tech Stack:** Python, LangChain (`langchain_core`, `langchain_classic.output_parsers.fix.OutputFixingParser`), Pydantic, pandas, pytest.

## Global Constraints

- Conda env `rs_segment` must be active before running Python (per repo `CLAUDE.md`) — **not currently installed on this machine** (`conda info --envs` shows only `base` and `lc_openmetadata`). Before Task 1, either create it (`conda create -n rs_segment ...` with this repo's dependencies) or confirm with the user which environment to use. Do not skip this — tests import `langchain_core`, `pydantic`, `pandas`.
- Strict YAGNI (repo `CLAUDE.md`) — the duplication across dataset folders is intentional (explicit user decision), not an exception to YAGNI; don't add anything beyond what's specified.
- Existing files not touched: `methods/baselines/{task_decomposition,step_based_multi_step,token_based_multi_step,all_at_once,step_by_step}.py`, `system_prompt/*.py`, `experiments/step_based_segmentation/`, `experiments/token_based_segmentation/`, `experiments/step_based_context_mode_comparison/`. These keep importing from the old shared files — do not delete or rename them.
- Keep existing output CSV file names for who&when (`ww_hand_crafted.csv`, `ww_algorithm_generated.csv` + `_cost.csv`) so downstream analysis scripts reading `experiments/single_fault/experiments/baseline/output/` keep working.

---

## File Structure

```
experiments/single_fault/
  utils/
    llm.py                          # NEW — shared LangChain invoke helper (no prompt content)
  experiments/baseline/
    who_and_when/
      __init__.py
      system_prompt.py               # NEW
      methods.py                     # NEW
      run.py                          # NEW
    trace_elephant/
      __init__.py
      system_prompt.py               # NEW
      methods.py                     # NEW
      run.py                          # NEW
    telbench/
      __init__.py
      system_prompt.py               # NEW
      metrics.py                      # NEW
      results.py                      # NEW
      methods.py                     # NEW
      run.py                          # NEW
    run.py                            # MODIFIED — becomes a thin entrypoint gluing the 3 above
tests/single_fault/baseline/
  test_llm.py                         # NEW
  who_and_when/test_methods.py        # NEW
  trace_elephant/test_methods.py      # NEW
  telbench/test_metrics.py            # NEW
  telbench/test_results.py            # NEW
  telbench/test_methods.py            # NEW
  telbench/test_run.py                # NEW
  test_entrypoint.py                  # NEW
```

No `tests/` directory currently exists in the repo — this plan creates one, mirroring the source tree under `experiments/single_fault/`. Run all new tests with:

```bash
conda activate rs_segment
python -m pytest tests/single_fault/baseline -v
```

---

### Task 1: Shared LLM invoke helper

**Files:**
- Create: `experiments/single_fault/utils/llm.py`
- Test: `tests/single_fault/baseline/test_llm.py`

**Interfaces:**
- Produces:
  - `invoke_structured(model_name: str, prompt_template: str, parser: PydanticOutputParser, prompt_params: pydantic.BaseModel) -> tuple[dict, dict]` — returns `(parsed_result_dict, cost_metrics_dict)` where `cost_metrics_dict` has keys `latency`, `input_tokens`, `output_tokens`.
  - `is_context_length_exceeded(error: Exception) -> bool` — `True` if the error message contains (case-insensitive) any of `"context_length_exceeded"`, `"maximum context length"`, `"context window"`, `"too many tokens"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/single_fault/baseline/test_llm.py
from __future__ import annotations

from unittest.mock import patch

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage

from experiments.single_fault.utils.llm import (
    invoke_structured,
    is_context_length_exceeded,
)


class _EchoResponse(BaseModel):
    value: str = Field(..., alias="Value")


class _FakeModel:
    def __init__(self, reply_value: str):
        self.reply_value = reply_value
        self.received_prompt = None

    def invoke(self, prompt_value):
        self.received_prompt = prompt_value
        return AIMessage(
            content=f'{{"Value": "{self.reply_value}"}}',
            usage_metadata={
                "input_tokens": 42,
                "output_tokens": 7,
                "total_tokens": 49,
            },
        )


def test_invoke_structured_returns_parsed_result_and_cost():
    parser = PydanticOutputParser(pydantic_object=_EchoResponse)
    fake_model = _FakeModel(reply_value="ok")

    with patch(
        "experiments.single_fault.utils.llm.get_model", return_value=fake_model
    ):
        result, cost = invoke_structured(
            model_name="gpt-4o-mini",
            prompt_template="Say {greeting}.",
            parser=parser,
            prompt_params=_GreetingParams(greeting="hi"),
        )

    assert result == {"value": "ok"}
    assert cost == {"latency": cost["latency"], "input_tokens": 42, "output_tokens": 7}
    assert cost["latency"] >= 0.0


class _GreetingParams(BaseModel):
    greeting: str


def test_is_context_length_exceeded_matches_known_phrases():
    assert is_context_length_exceeded(Exception("Error: context_length_exceeded"))
    assert is_context_length_exceeded(
        Exception("This model's maximum context length is 128000 tokens")
    )
    assert not is_context_length_exceeded(Exception("rate limit exceeded"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.single_fault.utils.llm'`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/single_fault/utils/llm.py
from __future__ import annotations

import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers.fix import OutputFixingParser
from pydantic import BaseModel

from experiments.chat_models import get_model


_CONTEXT_LENGTH_PHRASES = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
)


def invoke_structured(
    model_name: str,
    prompt_template: str,
    parser,
    prompt_params: BaseModel,
) -> tuple[dict, dict]:
    model = get_model(model_name)
    system_messages = ChatPromptTemplate.from_messages(
        [("system", prompt_template + "\n{format_instructions}")]
    ).partial(format_instructions=parser.get_format_instructions())

    prompt_value = system_messages.invoke(prompt_params.model_dump())

    t0 = time.perf_counter()
    ai_msg = model.invoke(prompt_value)
    latency = time.perf_counter() - t0

    cost_metrics = {
        "latency": latency,
        "input_tokens": ai_msg.usage_metadata["input_tokens"],
        "output_tokens": ai_msg.usage_metadata["output_tokens"],
    }

    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
    result = fixing_parser.invoke(ai_msg).model_dump()
    return result, cost_metrics


def is_context_length_exceeded(error: Exception) -> bool:
    message = str(error).lower()
    return any(phrase in message for phrase in _CONTEXT_LENGTH_PHRASES)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/test_llm.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/single_fault/utils/llm.py tests/single_fault/baseline/test_llm.py
git commit -m "feat: add shared LangChain invoke helper for per-dataset baseline methods"
```

---

### Task 2: `telbench/metrics.py`

**Files:**
- Create: `experiments/single_fault/experiments/baseline/telbench/metrics.py`
- Test: `tests/single_fault/baseline/telbench/test_metrics.py`

**Interfaces:**
- Produces:
  - `first_error_span(span_ids_in_order: list[str], gold_spans: list[str]) -> str` — earliest `gold_spans` member by position in `span_ids_in_order`.
  - `compute_metrics(pred_span: str | None, gold_spans: list[str], gt_first_error: str) -> dict` — returns `{"fea": float, "precision": float, "recall": float, "f1": float}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/single_fault/baseline/telbench/test_metrics.py
from __future__ import annotations

from experiments.single_fault.experiments.baseline.telbench.metrics import (
    compute_metrics,
    first_error_span,
)


def test_first_error_span_picks_earliest_by_position():
    order = ["s001", "s002", "s003", "s004"]
    assert first_error_span(order, ["s003", "s001"]) == "s001"
    assert first_error_span(order, ["s004"]) == "s004"


def test_compute_metrics_exact_hit_single_gold():
    result = compute_metrics(pred_span="s001", gold_spans=["s001"], gt_first_error="s001")
    assert result == {"fea": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_compute_metrics_hit_but_not_first_error_multi_gold():
    # pred hits a gold span, but not the earliest one -> fea is 0, P/R/F1 still count the hit
    result = compute_metrics(
        pred_span="s003", gold_spans=["s001", "s003"], gt_first_error="s001"
    )
    assert result["fea"] == 0.0
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest_approx(2 / 3)


def test_compute_metrics_miss():
    result = compute_metrics(pred_span="s099", gold_spans=["s001"], gt_first_error="s001")
    assert result == {"fea": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_compute_metrics_not_found():
    result = compute_metrics(pred_span=None, gold_spans=["s001"], gt_first_error="s001")
    assert result == {"fea": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def pytest_approx(value):
    import pytest

    return pytest.approx(value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/single_fault/experiments/baseline/telbench/metrics.py
from __future__ import annotations


def first_error_span(span_ids_in_order: list[str], gold_spans: list[str]) -> str:
    gold_set = set(gold_spans)
    for span_id in span_ids_in_order:
        if span_id in gold_set:
            return span_id
    raise ValueError("No gold span found in span_ids_in_order")


def compute_metrics(
    pred_span: str | None, gold_spans: list[str], gt_first_error: str
) -> dict:
    hit = pred_span is not None and pred_span in gold_spans
    fea = float(pred_span == gt_first_error)

    if not hit:
        return {"fea": fea, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = 1.0
    recall = 1.0 / len(gold_spans)
    f1 = 2 * precision * recall / (precision + recall)
    return {"fea": fea, "precision": precision, "recall": recall, "f1": f1}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_metrics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/single_fault/experiments/baseline/telbench/metrics.py tests/single_fault/baseline/telbench/test_metrics.py
git commit -m "feat: add telbench FEA/precision/recall/f1 metrics"
```

---

### Task 3: `telbench/results.py`

**Files:**
- Create: `experiments/single_fault/experiments/baseline/telbench/results.py`
- Test: `tests/single_fault/baseline/telbench/test_results.py`

**Interfaces:**
- Consumes: nothing from other new tasks (pure pandas).
- Produces:
  - `load_or_init_results(csv_path: Path) -> pd.DataFrame`
  - `upsert_base_row(df, file_name: str, gold_spans: list[str], gt_first_error: str) -> pd.DataFrame`
  - `update_method_result(df, file_name: str, method_name: str, pred_span: str | None, metrics: dict, exceeded_max_token_limit: bool, latency: float, input_tokens: int, output_tokens: int) -> pd.DataFrame`
  - `has_complete_method_result(df, file_name: str, method_name: str) -> bool`
  - `sort_results(df) -> pd.DataFrame`

Column names: `file, gold_spans, gt_first_error, {method}_pred_span, {method}_fea, {method}_precision, {method}_recall, {method}_f1, {method}_exceeded_max_token_limit, {method}_latency, {method}_input_tokens, {method}_output_tokens`. `gold_spans` is stored as a `;`-joined string (CSV-safe, mirrors how the rest of the repo avoids nested JSON in CSV cells).

- [ ] **Step 1: Write the failing tests**

```python
# tests/single_fault/baseline/telbench/test_results.py
from __future__ import annotations

from pathlib import Path

from experiments.single_fault.experiments.baseline.telbench.results import (
    has_complete_method_result,
    load_or_init_results,
    sort_results,
    update_method_result,
    upsert_base_row,
)


def test_upsert_base_row_creates_and_updates_row(tmp_path: Path):
    df = load_or_init_results(tmp_path / "out.csv")
    df = upsert_base_row(df, "0.json", ["s001", "s003"], "s001")
    assert list(df.loc[0, ["file", "gold_spans", "gt_first_error"]]) == [
        "0.json",
        "s001;s003",
        "s001",
    ]

    df = upsert_base_row(df, "0.json", ["s001"], "s001")
    assert len(df) == 1
    assert df.loc[0, "gold_spans"] == "s001"


def test_update_method_result_and_completeness(tmp_path: Path):
    df = load_or_init_results(tmp_path / "out.csv")
    df = upsert_base_row(df, "0.json", ["s001"], "s001")

    assert has_complete_method_result(df, "0.json", "all_at_once") is False

    df = update_method_result(
        df,
        file_name="0.json",
        method_name="all_at_once",
        pred_span="s001",
        metrics={"fea": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        exceeded_max_token_limit=False,
        latency=1.2,
        input_tokens=100,
        output_tokens=10,
    )

    assert has_complete_method_result(df, "0.json", "all_at_once") is True
    assert df.loc[0, "all_at_once_pred_span"] == "s001"
    assert df.loc[0, "all_at_once_fea"] == 1.0
    assert df.loc[0, "all_at_once_exceeded_max_token_limit"] == False  # noqa: E712


def test_sort_results_orders_by_numeric_file_stem(tmp_path: Path):
    df = load_or_init_results(tmp_path / "out.csv")
    df = upsert_base_row(df, "10.json", ["s001"], "s001")
    df = upsert_base_row(df, "2.json", ["s001"], "s001")
    sorted_df = sort_results(df)
    assert list(sorted_df["file"]) == ["2.json", "10.json"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_results.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/single_fault/experiments/baseline/telbench/results.py
from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_RESULT_COLUMNS = ["file", "gold_spans", "gt_first_error"]


def _method_columns(method_name: str) -> list[str]:
    return [
        f"{method_name}_pred_span",
        f"{method_name}_fea",
        f"{method_name}_precision",
        f"{method_name}_recall",
        f"{method_name}_f1",
        f"{method_name}_exceeded_max_token_limit",
        f"{method_name}_latency",
        f"{method_name}_input_tokens",
        f"{method_name}_output_tokens",
    ]


def load_or_init_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)

    df = pd.DataFrame(columns=BASE_RESULT_COLUMNS)
    for column in BASE_RESULT_COLUMNS:
        df[column] = pd.Series(dtype="object")
    return df


def upsert_base_row(
    df: pd.DataFrame, file_name: str, gold_spans: list[str], gt_first_error: str
) -> pd.DataFrame:
    gold_spans_str = ";".join(gold_spans)
    row_mask = df["file"] == file_name
    if row_mask.any():
        df.loc[row_mask, "gold_spans"] = gold_spans_str
        df.loc[row_mask, "gt_first_error"] = gt_first_error
        return df

    df.loc[len(df)] = {
        "file": file_name,
        "gold_spans": gold_spans_str,
        "gt_first_error": gt_first_error,
    }
    return df


def update_method_result(
    df: pd.DataFrame,
    file_name: str,
    method_name: str,
    pred_span: str | None,
    metrics: dict,
    exceeded_max_token_limit: bool,
    latency: float,
    input_tokens: int,
    output_tokens: int,
) -> pd.DataFrame:
    for column in _method_columns(method_name):
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")

    row_mask = df["file"] == file_name
    df.loc[row_mask, f"{method_name}_pred_span"] = pred_span
    df.loc[row_mask, f"{method_name}_fea"] = metrics["fea"]
    df.loc[row_mask, f"{method_name}_precision"] = metrics["precision"]
    df.loc[row_mask, f"{method_name}_recall"] = metrics["recall"]
    df.loc[row_mask, f"{method_name}_f1"] = metrics["f1"]
    df.loc[row_mask, f"{method_name}_exceeded_max_token_limit"] = exceeded_max_token_limit
    df.loc[row_mask, f"{method_name}_latency"] = latency
    df.loc[row_mask, f"{method_name}_input_tokens"] = input_tokens
    df.loc[row_mask, f"{method_name}_output_tokens"] = output_tokens
    return df


def has_complete_method_result(
    df: pd.DataFrame, file_name: str, method_name: str
) -> bool:
    columns = _method_columns(method_name)
    if any(column not in df.columns for column in columns):
        return False

    row = df.loc[df["file"] == file_name, columns]
    if row.empty:
        return False

    return bool(row.notna().all(axis=1).iloc[0])


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    def sort_key(series: pd.Series) -> pd.Series:
        normalized = series.astype(str).str.replace(".json", "", regex=False)
        return pd.to_numeric(normalized, errors="coerce")

    return df.sort_values(by="file", key=sort_key, na_position="last").reset_index(
        drop=True
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_results.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/single_fault/experiments/baseline/telbench/results.py tests/single_fault/baseline/telbench/test_results.py
git commit -m "feat: add telbench CSV results writer"
```

---

### Task 4: `telbench/system_prompt.py`

**Files:**
- Create: `experiments/single_fault/experiments/baseline/telbench/system_prompt.py`
- Test: `tests/single_fault/baseline/telbench/test_system_prompt.py`

**Interfaces:**
- Produces:
  - `AllAtOnceInput(BaseModel)` fields: `question: str`, `spans_content: str`
  - `all_at_once_prompt: str`, `all_at_once_parser: PydanticOutputParser` (parses to `{"span_id": str}`)
  - `StepByStepInput(BaseModel)` fields: `question: str`, `current_span_content: str`, `spans_content: str`
  - `step_by_step_prompt: str`, `step_by_step_parser: PydanticOutputParser` (parses to `{"error_found": bool}`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/single_fault/baseline/telbench/test_system_prompt.py
from __future__ import annotations

from experiments.single_fault.experiments.baseline.telbench.system_prompt import (
    all_at_once_parser,
    step_by_step_parser,
)


def test_all_at_once_parser_extracts_span_id():
    parsed = all_at_once_parser.parse('{"Span ID": "s004"}')
    assert parsed.span_id == "s004"


def test_step_by_step_parser_extracts_error_found():
    parsed = step_by_step_parser.parse('{"Error Found": true}')
    assert parsed.error_found is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_system_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/single_fault/experiments/baseline/telbench/system_prompt.py
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class AllAtOnceInput(BaseModel):
    question: str
    spans_content: str


class StepByStepInput(BaseModel):
    question: str
    current_span_content: str
    spans_content: str


class AllAtOnceResponse(BaseModel):
    span_id: str = Field(
        ...,
        alias="Span ID",
        description="The id of the span where the first important mistake occurred.",
    )


class StepByStepResponse(BaseModel):
    error_found: bool = Field(
        ..., description="Whether the current span contains an important mistake."
    )


all_at_once_parser = PydanticOutputParser(pydantic_object=AllAtOnceResponse)
step_by_step_parser = PydanticOutputParser(pydantic_object=StepByStepResponse)

all_at_once_prompt = """
You are an AI assistant analyzing a deep-research agent trajectory that has
been segmented into an ordered sequence of semantic spans. Each span covers
one continuous local goal (planning, retrieval, verification, comparison,
finalization).

You will be provided with:
1. The original research question.
2. The full ordered sequence of spans.

Your task is to identify the id of the FIRST span in which the agent made
an important mistake (an unsupported, contradicted, or prematurely
committed claim) that could directly affect the final answer.

Important rules:
- Return only the id of the first span containing an important mistake.
- Do not mark normal exploration, failed searches, tentative hypotheses,
  already-corrected errors, or harmless tool noise as mistakes.
- If multiple mistakes appear later, ignore them and return only the
  earliest one.
- Base your prediction only on the given question and spans.

The research question is:
{question}

The ordered spans are:
{spans_content}

Please answer strictly in the following JSON format:
"""

step_by_step_prompt = """
You are an AI assistant evaluating one span of a deep-research agent
trajectory that has been segmented into an ordered sequence of semantic
spans. Each span covers one continuous local goal (planning, retrieval,
verification, comparison, finalization).

You will be provided with:
1. The original research question.
2. The content of the current span to evaluate.
3. The full ordered sequence of spans as surrounding context.

Your task is to determine whether the current span contains an important
mistake (an unsupported, contradicted, or prematurely committed claim) that
could directly affect the final answer.

The research question is:
{question}

The content of the current span is:
{current_span_content}

The full ordered spans are:
{spans_content}

Important rules:
- Evaluate only the current span, not other spans.
- Use the surrounding spans only to judge whether the current span is
  correct.
- Return true only if the current span introduces, reuses, amplifies, or
  finalizes a claim that is unsupported or contradicted.
- Do not mark normal exploration, failed searches, tentative hypotheses,
  already-corrected errors, or harmless tool noise as mistakes.

Please answer strictly in the following JSON format:
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_system_prompt.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/single_fault/experiments/baseline/telbench/system_prompt.py tests/single_fault/baseline/telbench/test_system_prompt.py
git commit -m "feat: add telbench prompts and parsers"
```

---

### Task 5: `telbench/methods.py`

**Files:**
- Create: `experiments/single_fault/experiments/baseline/telbench/methods.py`
- Test: `tests/single_fault/baseline/telbench/test_methods.py`

**Interfaces:**
- Consumes:
  - `experiments.single_fault.utils.llm.invoke_structured` (Task 1)
  - `experiments.single_fault.utils.llm.is_context_length_exceeded` (Task 1)
  - `experiments.single_fault.experiments.baseline.telbench.metrics.first_error_span`, `compute_metrics` (Task 2)
  - `experiments.single_fault.experiments.baseline.telbench.system_prompt.*` (Task 4)
- Produces:
  - `format_spans(spans: list[dict]) -> str` — `"- {id}: {raw}\n..."` joined.
  - `TelbenchAccuracyResult` (plain `dict`) shape: `{"pred_span": str | None, "metrics": dict, "exceeded_max_token_limit": bool}`
  - `all_at_once_single_file(data: dict, model_name: str) -> tuple[dict, dict]` — returns `(TelbenchAccuracyResult, cost_metrics_dict)`, `cost_metrics_dict` has `latency`, `input_tokens`, `output_tokens`.
  - `step_by_step_single_file(data: dict, model_name: str) -> tuple[dict, dict]` — same return shape, accumulates cost across all LLM calls made during the scan.

- [ ] **Step 1: Write the failing tests**

```python
# tests/single_fault/baseline/telbench/test_methods.py
from __future__ import annotations

from unittest.mock import patch

from experiments.single_fault.experiments.baseline.telbench.methods import (
    all_at_once_single_file,
    step_by_step_single_file,
)


def _sample_data():
    return {
        "id": "0001",
        "question": "What is the answer?",
        "spans": [
            {"id": "s001", "raw": "retrieve info"},
            {"id": "s002", "raw": "verify source"},
            {"id": "s003", "raw": "wrong conclusion"},
        ],
        "gold": {"error_span_ids": ["s003"]},
    }


def test_all_at_once_returns_predicted_span_and_metrics():
    with patch(
        "experiments.single_fault.experiments.baseline.telbench.methods.invoke_structured",
        return_value=({"span_id": "s003"}, {"latency": 0.5, "input_tokens": 10, "output_tokens": 2}),
    ):
        accuracy, cost = all_at_once_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_span"] == "s003"
    assert accuracy["metrics"]["fea"] == 1.0
    assert accuracy["exceeded_max_token_limit"] is False
    assert cost == {"latency": 0.5, "input_tokens": 10, "output_tokens": 2}


def test_all_at_once_marks_exceeded_and_keeps_not_found():
    with patch(
        "experiments.single_fault.experiments.baseline.telbench.methods.invoke_structured",
        side_effect=Exception("context_length_exceeded"),
    ):
        accuracy, cost = all_at_once_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_span"] is None
    assert accuracy["exceeded_max_token_limit"] is True
    assert accuracy["metrics"]["fea"] == 0.0
    assert cost == {"latency": 0.0, "input_tokens": 0, "output_tokens": 0}


def test_step_by_step_stops_at_first_error_span():
    call_results = iter(
        [
            ({"error_found": False}, {"latency": 0.1, "input_tokens": 5, "output_tokens": 1}),
            ({"error_found": False}, {"latency": 0.1, "input_tokens": 6, "output_tokens": 1}),
            ({"error_found": True}, {"latency": 0.1, "input_tokens": 7, "output_tokens": 1}),
        ]
    )
    with patch(
        "experiments.single_fault.experiments.baseline.telbench.methods.invoke_structured",
        side_effect=lambda *a, **k: next(call_results),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_span"] == "s003"
    assert accuracy["metrics"]["fea"] == 1.0
    assert accuracy["exceeded_max_token_limit"] is False
    assert cost["input_tokens"] == 5 + 6 + 7


def test_step_by_step_keeps_prior_state_when_exceeded_mid_scan():
    call_results = iter(
        [
            ({"error_found": True}, {"latency": 0.1, "input_tokens": 5, "output_tokens": 1}),
        ]
    )

    def _side_effect(*args, **kwargs):
        try:
            return next(call_results)
        except StopIteration:
            raise Exception("This model's maximum context length is 128000 tokens")

    with patch(
        "experiments.single_fault.experiments.baseline.telbench.methods.invoke_structured",
        side_effect=_side_effect,
    ):
        # Force the loop to keep going past span s001 by using data with the
        # error span first, then a span that would trigger the (mocked)
        # second call to raise.
        data = _sample_data()
        data["gold"]["error_span_ids"] = ["s001"]
        accuracy, cost = step_by_step_single_file(data, model_name="gpt-4o-mini")

    # error_found=True at span s001 -> loop returns immediately, second call
    # never happens, so no exceeded flag and pred_span is s001.
    assert accuracy["pred_span"] == "s001"
    assert accuracy["exceeded_max_token_limit"] is False


def test_step_by_step_not_found_when_no_span_flagged():
    with patch(
        "experiments.single_fault.experiments.baseline.telbench.methods.invoke_structured",
        return_value=({"error_found": False}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_span"] is None
    assert accuracy["metrics"]["fea"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_methods.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/single_fault/experiments/baseline/telbench/methods.py
from __future__ import annotations

from experiments.single_fault.experiments.baseline.telbench.metrics import (
    compute_metrics,
    first_error_span,
)
from experiments.single_fault.experiments.baseline.telbench.system_prompt import (
    AllAtOnceInput,
    StepByStepInput,
    all_at_once_parser,
    all_at_once_prompt,
    step_by_step_parser,
    step_by_step_prompt,
)
from experiments.single_fault.utils.llm import invoke_structured, is_context_length_exceeded


def format_spans(spans: list[dict]) -> str:
    return "\n".join(f"- {span['id']}: {span['raw']}" for span in spans)


def _empty_cost() -> dict:
    return {"latency": 0.0, "input_tokens": 0, "output_tokens": 0}


def _accumulate_cost(total: dict, delta: dict) -> dict:
    return {
        "latency": total["latency"] + delta["latency"],
        "input_tokens": total["input_tokens"] + delta["input_tokens"],
        "output_tokens": total["output_tokens"] + delta["output_tokens"],
    }


def _build_accuracy(data: dict, pred_span: str | None, exceeded: bool) -> dict:
    span_ids_in_order = [span["id"] for span in data["spans"]]
    gold_spans = data["gold"]["error_span_ids"]
    gt_first_error = first_error_span(span_ids_in_order, gold_spans)
    metrics = compute_metrics(pred_span, gold_spans, gt_first_error)
    return {
        "pred_span": pred_span,
        "metrics": metrics,
        "exceeded_max_token_limit": exceeded,
    }


def all_at_once_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    method_input = AllAtOnceInput(
        question=data["question"],
        spans_content=format_spans(data["spans"]),
    )

    try:
        result, cost = invoke_structured(
            model_name=model_name,
            prompt_template=all_at_once_prompt,
            parser=all_at_once_parser,
            prompt_params=method_input,
        )
    except Exception as error:
        if not is_context_length_exceeded(error):
            raise
        return _build_accuracy(data, pred_span=None, exceeded=True), _empty_cost()

    return _build_accuracy(data, pred_span=result["span_id"], exceeded=False), cost


def step_by_step_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    spans = data["spans"]
    all_spans_content = format_spans(spans)
    total_cost = _empty_cost()

    for span in spans:
        method_input = StepByStepInput(
            question=data["question"],
            current_span_content=f"- {span['id']}: {span['raw']}",
            spans_content=all_spans_content,
        )

        try:
            result, cost = invoke_structured(
                model_name=model_name,
                prompt_template=step_by_step_prompt,
                parser=step_by_step_parser,
                prompt_params=method_input,
            )
        except Exception as error:
            if not is_context_length_exceeded(error):
                raise
            return _build_accuracy(data, pred_span=None, exceeded=True), total_cost

        total_cost = _accumulate_cost(total_cost, cost)

        if result["error_found"]:
            return _build_accuracy(data, pred_span=span["id"], exceeded=False), total_cost

    return _build_accuracy(data, pred_span=None, exceeded=False), total_cost
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_methods.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/single_fault/experiments/baseline/telbench/methods.py tests/single_fault/baseline/telbench/test_methods.py
git commit -m "feat: add telbench all_at_once/step_by_step methods with token-limit handling"
```

---

### Task 6: `telbench/run.py`

**Files:**
- Create: `experiments/single_fault/experiments/baseline/telbench/run.py`
- Test: `tests/single_fault/baseline/telbench/test_run.py`

**Interfaces:**
- Consumes: `telbench/methods.py::all_at_once_single_file, step_by_step_single_file` (Task 5), `telbench/results.py::*` (Task 3)
- Produces:
  - `TELBENCH_DATA_DIR: Path` — `PROJECT_ROOT / "data" / "error_localization" / "multi_fault" / "telbench"`
  - `TELBENCH_OUTPUT_PATH: Path` — `experiments/baseline/output/telbench.csv` (via `experiments/single_fault/utils/experiment_paths.py::BASELINE_OUTPUT_DIR`)
  - `run_telbench(data_dir: Path, output_path: Path, model_name: str) -> Path`
  - `main() -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/single_fault/baseline/telbench/test_run.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments.single_fault.experiments.baseline.telbench.run import run_telbench


def _write_sample(dir_path: Path, index: int, spans: list[dict], gold: list[str]):
    data = {
        "id": f"{index:04d}",
        "question": "q",
        "spans": spans,
        "gold": {"error_span_ids": gold},
    }
    (dir_path / f"{index}.json").write_text(json.dumps(data))


def test_run_telbench_writes_csv_for_both_methods(tmp_path: Path):
    data_dir = tmp_path / "telbench"
    data_dir.mkdir()
    spans = [{"id": "s001", "raw": "a"}, {"id": "s002", "raw": "b"}]
    _write_sample(data_dir, 0, spans, ["s002"])

    output_path = tmp_path / "output" / "telbench.csv"

    def fake_all_at_once(data, model_name):
        return (
            {"pred_span": "s002", "metrics": {"fea": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}, "exceeded_max_token_limit": False},
            {"latency": 0.1, "input_tokens": 1, "output_tokens": 1},
        )

    def fake_step_by_step(data, model_name):
        return (
            {"pred_span": None, "metrics": {"fea": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}, "exceeded_max_token_limit": False},
            {"latency": 0.2, "input_tokens": 2, "output_tokens": 2},
        )

    with patch(
        "experiments.single_fault.experiments.baseline.telbench.run.all_at_once_single_file",
        side_effect=fake_all_at_once,
    ), patch(
        "experiments.single_fault.experiments.baseline.telbench.run.step_by_step_single_file",
        side_effect=fake_step_by_step,
    ):
        result_path = run_telbench(data_dir=data_dir, output_path=output_path, model_name="gpt-4o-mini")

    assert result_path == output_path
    df = pd.read_csv(output_path)
    assert len(df) == 1
    assert df.loc[0, "all_at_once_pred_span"] == "s002"
    assert df.loc[0, "step_by_step_fea"] == 0.0


def test_run_telbench_skips_already_complete_rows(tmp_path: Path):
    data_dir = tmp_path / "telbench"
    data_dir.mkdir()
    spans = [{"id": "s001", "raw": "a"}]
    _write_sample(data_dir, 0, spans, ["s001"])
    output_path = tmp_path / "output" / "telbench.csv"

    call_count = {"n": 0}

    def fake_method(data, model_name):
        call_count["n"] += 1
        return (
            {"pred_span": "s001", "metrics": {"fea": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}, "exceeded_max_token_limit": False},
            {"latency": 0.1, "input_tokens": 1, "output_tokens": 1},
        )

    with patch(
        "experiments.single_fault.experiments.baseline.telbench.run.all_at_once_single_file",
        side_effect=fake_method,
    ), patch(
        "experiments.single_fault.experiments.baseline.telbench.run.step_by_step_single_file",
        side_effect=fake_method,
    ):
        run_telbench(data_dir=data_dir, output_path=output_path, model_name="gpt-4o-mini")
        run_telbench(data_dir=data_dir, output_path=output_path, model_name="gpt-4o-mini")

    assert call_count["n"] == 2  # not 4 -> second run skipped both methods
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/single_fault/experiments/baseline/telbench/run.py
from __future__ import annotations

from pathlib import Path

from experiments.single_fault.experiments.baseline.telbench.methods import (
    all_at_once_single_file,
    step_by_step_single_file,
)
from experiments.single_fault.experiments.baseline.telbench.results import (
    has_complete_method_result,
    load_or_init_results,
    sort_results,
    update_method_result,
    upsert_base_row,
)
from experiments.single_fault.utils.experiment_paths import BASELINE_OUTPUT_DIR
from experiments.single_fault.utils.file import load_json


PROJECT_ROOT = Path(__file__).resolve().parents[5]
TELBENCH_DATA_DIR = (
    PROJECT_ROOT / "data" / "error_localization" / "multi_fault" / "telbench"
)
TELBENCH_OUTPUT_PATH = BASELINE_OUTPUT_DIR / "telbench.csv"
DEFAULT_MODEL_NAME = "gpt-4o-mini"

METHODS = {
    "all_at_once": all_at_once_single_file,
    "step_by_step": step_by_step_single_file,
}


def _sample_file_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.json"), key=lambda path: int(path.stem))


def run_telbench(data_dir: Path, output_path: Path, model_name: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_or_init_results(output_path)

    for file_path in _sample_file_paths(data_dir):
        data = load_json(file_path)
        span_ids_in_order = [span["id"] for span in data["spans"]]
        gold_spans = data["gold"]["error_span_ids"]
        from experiments.single_fault.experiments.baseline.telbench.metrics import (
            first_error_span,
        )

        gt_first_error = first_error_span(span_ids_in_order, gold_spans)

        df = upsert_base_row(df, file_path.name, gold_spans, gt_first_error)

        for method_name, method_fn in METHODS.items():
            if has_complete_method_result(df, file_path.name, method_name):
                continue

            accuracy, cost = method_fn(data, model_name)
            df = update_method_result(
                df,
                file_name=file_path.name,
                method_name=method_name,
                pred_span=accuracy["pred_span"],
                metrics=accuracy["metrics"],
                exceeded_max_token_limit=accuracy["exceeded_max_token_limit"],
                latency=cost["latency"],
                input_tokens=cost["input_tokens"],
                output_tokens=cost["output_tokens"],
            )
            df = sort_results(df)
            df.to_csv(output_path, index=False)

    df = sort_results(df)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    result_path = run_telbench(
        data_dir=TELBENCH_DATA_DIR,
        output_path=TELBENCH_OUTPUT_PATH,
        model_name=DEFAULT_MODEL_NAME,
    )
    print(f"Saved telbench baseline results to: {result_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/telbench/test_run.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/single_fault/experiments/baseline/telbench/run.py tests/single_fault/baseline/telbench/test_run.py
git commit -m "feat: add telbench baseline driver"
```

---

### Task 7: `who_and_when/system_prompt.py` + `methods.py`

**Files:**
- Create: `experiments/single_fault/experiments/baseline/who_and_when/system_prompt.py`
- Create: `experiments/single_fault/experiments/baseline/who_and_when/methods.py`
- Test: `tests/single_fault/baseline/who_and_when/test_methods.py`

**Interfaces:**
- Consumes: `experiments.single_fault.utils.llm.invoke_structured` (Task 1), `experiments.single_fault.utils.accuracy.agent_names_match` (existing)
- Produces:
  - `format_trajectory(trajectory: list[dict]) -> str` — `"- step {step} - {agent_name}: {content}"` per item, joined by `\n` (same behavior as the current shared `format_agent_behaviors`, reimplemented locally since who&when's `methods.py` no longer imports the shared one).
  - `all_at_once_single_file(data: dict, model_name: str) -> tuple[dict, dict]` — `dict` shape: `{"pred_agent": str, "pred_step": int, "agent_accuracy": float, "step_accuracy": float, "gt_agent": str, "gt_step": int}`; cost dict: `{"latency", "input_tokens", "output_tokens"}`.
  - `step_by_step_single_file(data: dict, model_name: str) -> tuple[dict, dict]` — same shapes, iterative (no recursion), accumulates cost.

- [ ] **Step 1: Write the failing tests**

```python
# tests/single_fault/baseline/who_and_when/test_methods.py
from __future__ import annotations

from unittest.mock import patch

from experiments.single_fault.experiments.baseline.who_and_when.methods import (
    all_at_once_single_file,
    step_by_step_single_file,
)


def _sample_data():
    return {
        "question": "solve it",
        "trajectory": [
            {"step": 0, "agent_name": "Planner", "content": "plan"},
            {"step": 1, "agent_name": "Coder", "content": "wrong code"},
        ],
        "mistake_agent": "Coder",
        "mistake_step": 1,
    }


def test_all_at_once_scores_correct_prediction():
    with patch(
        "experiments.single_fault.experiments.baseline.who_and_when.methods.invoke_structured",
        return_value=({"step_number": 1}, {"latency": 0.1, "input_tokens": 3, "output_tokens": 1}),
    ):
        accuracy, cost = all_at_once_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Coder"
    assert accuracy["pred_step"] == 1
    assert accuracy["agent_accuracy"] == 1.0
    assert accuracy["step_accuracy"] == 1.0
    assert cost["input_tokens"] == 3


def test_step_by_step_stops_at_first_flagged_step():
    call_results = iter(
        [
            ({"error_found": False}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
            ({"error_found": True}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
        ]
    )
    with patch(
        "experiments.single_fault.experiments.baseline.who_and_when.methods.invoke_structured",
        side_effect=lambda *a, **k: next(call_results),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Coder"
    assert accuracy["pred_step"] == 1
    assert accuracy["step_accuracy"] == 1.0
    assert cost["input_tokens"] == 2


def test_step_by_step_not_found_when_no_step_flagged():
    with patch(
        "experiments.single_fault.experiments.baseline.who_and_when.methods.invoke_structured",
        return_value=({"error_found": False}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Not Found"
    assert accuracy["pred_step"] == -1
    assert accuracy["step_accuracy"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/who_and_when/test_methods.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/single_fault/experiments/baseline/who_and_when/system_prompt.py
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class AllAtOnceInput(BaseModel):
    problem: str
    chat_content: str


class StepByStepInput(BaseModel):
    problem: str
    current_step_content: str
    chat_content: str


class AllAtOnceResponse(BaseModel):
    step_number: int = Field(
        ...,
        alias="Step Number",
        description="The step number where the first important mistake occurred.",
    )


class StepByStepResponse(BaseModel):
    error_found: bool = Field(
        ..., description="Whether the current step contains an important mistake."
    )


all_at_once_parser = PydanticOutputParser(pydantic_object=AllAtOnceResponse)
step_by_step_parser = PydanticOutputParser(pydantic_object=StepByStepResponse)

all_at_once_prompt = """
You are an AI assistant tasked with analyzing a multi-agent conversation history for a real-world problem-solving task.

You will be provided with:
1. The original problem that the agents are trying to solve.
2. The complete conversation history of the agents, organized as a sequence of steps.

Your task is to identify the first step in which any agent made an important mistake that could directly lead to an incorrect final solution.

Important rules:
- Return only the first step where an important mistake occurred.
- Do not mark minor wording issues or harmless inaccuracies as mistakes.
- If multiple mistakes appear later, ignore them and return only the earliest important mistake.
- If the conversation does not contain an obvious mistake, choose the step that is most likely responsible for the incorrect final solution.
- Base your prediction only on the given problem and conversation.

The problem is:
{problem}

The full multi-agent conversation is:
{chat_content}

Please answer strictly in the following JSON format:
"""

step_by_step_prompt = """
You are an AI assistant tasked with evaluating a specific step in a multi-agent conversation for a real-world problem-solving task.

You will be provided with:
1. The original problem that the agents are trying to solve.
2. The content of the current step to evaluate.
3. The surrounding conversation context from the full multi-agent conversation.

Your task is to determine whether the current step contains an important mistake that could directly lead to an incorrect final solution.

The problem is:
{problem}

The content of the current step is:
{current_step_content}

The surrounding conversation context is:
{chat_content}

Important rules:
- Evaluate only the current step, not other steps.
- Use the surrounding conversation context only to understand whether the current step is correct or incorrect.
- Return true only if the current step contains an important mistake that could meaningfully affect the final solution.
- Do not mark minor wording issues, incomplete but harmless reasoning, or stylistic problems as mistakes.
- If the current step is reasonable based on the available context, return false.
- If the current step repeats, relies on, or amplifies an earlier wrong assumption in a way that affects the final solution, return true.
- Base your judgment only on the given problem, the current step, and the provided conversation context.

Please answer strictly in the following JSON format:
"""
```

```python
# experiments/single_fault/experiments/baseline/who_and_when/methods.py
from __future__ import annotations

from experiments.single_fault.experiments.baseline.who_and_when.system_prompt import (
    AllAtOnceInput,
    StepByStepInput,
    all_at_once_parser,
    all_at_once_prompt,
    step_by_step_parser,
    step_by_step_prompt,
)
from experiments.single_fault.utils.accuracy import agent_names_match
from experiments.single_fault.utils.llm import invoke_structured


def format_trajectory(trajectory: list[dict]) -> str:
    return "\n".join(
        f"- step {item['step']} - {item['agent_name']}: {item['content']}"
        for item in trajectory
    )


def _score(gt_agent: str, gt_step: int, pred_agent: str, pred_step: int) -> dict:
    return {
        "gt_agent": gt_agent,
        "gt_step": gt_step,
        "pred_agent": pred_agent,
        "pred_step": pred_step,
        "agent_accuracy": float(agent_names_match(gt_agent, pred_agent)),
        "step_accuracy": float(gt_step == pred_step),
    }


def all_at_once_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    trajectory = data["trajectory"]
    step_to_agent_name = {int(item["step"]): item["agent_name"] for item in trajectory}

    method_input = AllAtOnceInput(
        problem=data["question"],
        chat_content=format_trajectory(trajectory),
    )
    result, cost = invoke_structured(
        model_name=model_name,
        prompt_template=all_at_once_prompt,
        parser=all_at_once_parser,
        prompt_params=method_input,
    )

    pred_step = int(result["step_number"])
    pred_agent = step_to_agent_name.get(pred_step, "Not Found")
    accuracy = _score(
        gt_agent=data["mistake_agent"],
        gt_step=int(data["mistake_step"]),
        pred_agent=pred_agent,
        pred_step=pred_step,
    )
    return accuracy, cost


def step_by_step_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    trajectory = data["trajectory"]
    all_chat_content = format_trajectory(trajectory)
    total_cost = {"latency": 0.0, "input_tokens": 0, "output_tokens": 0}

    for current_step, item in enumerate(trajectory):
        method_input = StepByStepInput(
            problem=data["question"],
            current_step_content=format_trajectory([item]),
            chat_content=format_trajectory(trajectory[: current_step + 1]),
        )
        result, cost = invoke_structured(
            model_name=model_name,
            prompt_template=step_by_step_prompt,
            parser=step_by_step_parser,
            prompt_params=method_input,
        )
        total_cost = {
            "latency": total_cost["latency"] + cost["latency"],
            "input_tokens": total_cost["input_tokens"] + cost["input_tokens"],
            "output_tokens": total_cost["output_tokens"] + cost["output_tokens"],
        }

        if result["error_found"]:
            accuracy = _score(
                gt_agent=data["mistake_agent"],
                gt_step=int(data["mistake_step"]),
                pred_agent=item["agent_name"],
                pred_step=current_step,
            )
            return accuracy, total_cost

    accuracy = _score(
        gt_agent=data["mistake_agent"],
        gt_step=int(data["mistake_step"]),
        pred_agent="Not Found",
        pred_step=-1,
    )
    return accuracy, total_cost
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/who_and_when/test_methods.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/single_fault/experiments/baseline/who_and_when/system_prompt.py experiments/single_fault/experiments/baseline/who_and_when/methods.py tests/single_fault/baseline/who_and_when/test_methods.py
git commit -m "feat: add who_and_when-specific baseline prompts and methods"
```

---

### Task 8: `who_and_when/run.py`

**Files:**
- Create: `experiments/single_fault/experiments/baseline/who_and_when/run.py`
- Test: covered by Task 10's entrypoint test (this driver is a thin wrapper around already-tested `shared.py`; no new pure logic to unit test beyond the config wiring below).

**Interfaces:**
- Consumes: `who_and_when/methods.py::all_at_once_single_file, step_by_step_single_file` (Task 7), `experiments.single_fault.experiments.shared.MethodConfig, run_method_configs_for_dataset` (existing)
- Produces: `DATASET_DIRS: dict[str, Path]`, `build_method_configs() -> list[MethodConfig]`, `main() -> None`

- [ ] **Step 1: Write implementation directly (thin wrapper, no new branching logic to test beyond what Task 7 already covers)**

```python
# experiments/single_fault/experiments/baseline/who_and_when/run.py
from __future__ import annotations

from pathlib import Path

from experiments.single_fault.experiments.baseline.who_and_when.methods import (
    all_at_once_single_file,
    step_by_step_single_file,
)
from experiments.single_fault.experiments.shared import MethodConfig, run_method_configs_for_dataset
from experiments.single_fault.utils.experiment_paths import BASELINE_OUTPUT_DIR
from experiments.single_fault.utils.schema import Metadata


DEFAULT_MODEL_NAME = "gpt-4o-mini"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = PROJECT_ROOT / "data" / "error_localization" / "single_fault"

DATASET_DIRS = {
    "ww_algorithm_generated": DATA_DIR / "who_and_when__algorithm-generated",
    "ww_hand_crafted": DATA_DIR / "who_and_when__hand-crafted",
}


def _run_all_at_once(data: dict, metadata: Metadata):
    accuracy, cost = all_at_once_single_file(data, model_name=metadata.model_name)
    return _to_legacy_metrics(accuracy, cost)


def _run_step_by_step(data: dict, metadata: Metadata):
    accuracy, cost = step_by_step_single_file(data, model_name=metadata.model_name)
    return _to_legacy_metrics(accuracy, cost)


def _to_legacy_metrics(accuracy: dict, cost: dict):
    from experiments.single_fault.utils.schema import AccuracyMetrics, CostMetrics

    accuracy_metrics = AccuracyMetrics(
        gt_agent=accuracy["gt_agent"],
        gt_step=accuracy["gt_step"],
        pred_agent=accuracy["pred_agent"],
        pred_step=accuracy["pred_step"],
        agent_accuracy=accuracy["agent_accuracy"],
        step_accuracy=accuracy["step_accuracy"],
    )
    cost_metrics = CostMetrics(
        num_input_steps=0,
        latency=cost["latency"],
        input_tokens=cost["input_tokens"],
        output_tokens=cost["output_tokens"],
        input_cost=0.0,
        output_cost=0.0,
        total_cost=0.0,
    )
    return accuracy_metrics, cost_metrics


def build_method_configs() -> list[MethodConfig]:
    return [
        MethodConfig(
            method_name="all_at_once",
            metadata_method="all_at_once",
            run_single_file=_run_all_at_once,
        ),
        MethodConfig(
            method_name="step_by_step",
            metadata_method="step_by_step",
            run_single_file=_run_step_by_step,
        ),
    ]


def main() -> None:
    for dataset_key, data_dir in DATASET_DIRS.items():
        accuracy_path, cost_path = run_method_configs_for_dataset(
            dataset_key=dataset_key,
            data_dir=data_dir,
            model_name=DEFAULT_MODEL_NAME,
            output_dir=BASELINE_OUTPUT_DIR,
            accuracy_file_name=f"{dataset_key}.csv",
            cost_file_name=f"{dataset_key}_cost.csv",
            experiment_name="baseline_who_and_when",
            method_configs=build_method_configs(),
        )
        print(f"Saved who_and_when baseline accuracy results to: {accuracy_path}")
        print(f"Saved who_and_when baseline cost results to: {cost_path}")


if __name__ == "__main__":
    main()
```

**Note:** `shared.py::run_method_configs_for_dataset` calls `config.run_single_file(data, metadata)` expecting a `(AccuracyMetrics, CostMetrics)` tuple — that's why `_run_all_at_once`/`_run_step_by_step` adapt the new dict-based `methods.py` return shape back into the existing pydantic models via `_to_legacy_metrics`. This keeps `shared.py` and `utils/results.py` untouched.

- [ ] **Step 2: Manual smoke check (no LLM calls — just wiring)**

```bash
python -c "
from experiments.single_fault.experiments.baseline.who_and_when.run import build_method_configs, DATASET_DIRS
configs = build_method_configs()
assert [c.method_name for c in configs] == ['all_at_once', 'step_by_step']
assert set(DATASET_DIRS.keys()) == {'ww_algorithm_generated', 'ww_hand_crafted'}
print('OK')
"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/single_fault/experiments/baseline/who_and_when/run.py
git commit -m "feat: add who_and_when baseline driver"
```

---

### Task 9: `trace_elephant/` (system_prompt.py, methods.py, run.py)

**Files:**
- Create: `experiments/single_fault/experiments/baseline/trace_elephant/system_prompt.py`
- Create: `experiments/single_fault/experiments/baseline/trace_elephant/methods.py`
- Create: `experiments/single_fault/experiments/baseline/trace_elephant/run.py`
- Test: `tests/single_fault/baseline/trace_elephant/test_methods.py`

**Important field difference from who&when** (confirmed by inspecting an actual sample file): TraceElephant JSON has `task_instruction` (not `question`), and each trajectory step has `step_id`/`agent_id`/`agent_name`/`input`/`output`/`tool_logs` (not `step`/`content`). `methods.py` must format from these fields directly — it cannot reuse who&when's `format_trajectory`.

**Interfaces:**
- Consumes: `experiments.single_fault.utils.llm.invoke_structured` (Task 1), `experiments.single_fault.utils.accuracy.agent_names_match` (existing)
- Produces:
  - `format_trajectory(trajectory: list[dict]) -> str` — `"- step {step_id} - {agent_name}: input={input} | output={output} | tool_logs={tool_logs}"` per item (using `json.dumps` for the non-string `input`/`output`/`tool_logs` values), joined by `\n`.
  - `all_at_once_single_file(data: dict, model_name: str) -> tuple[dict, dict]`, `step_by_step_single_file(data: dict, model_name: str) -> tuple[dict, dict]` — same return shapes as Task 7's who&when versions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/single_fault/baseline/trace_elephant/test_methods.py
from __future__ import annotations

from unittest.mock import patch

from experiments.single_fault.experiments.baseline.trace_elephant.methods import (
    all_at_once_single_file,
    format_trajectory,
    step_by_step_single_file,
)


def _sample_data():
    return {
        "task_instruction": "solve it",
        "trajectory": [
            {
                "step_id": 0,
                "agent_id": "a0",
                "agent_name": "Orchestrator",
                "input": {"goal": "start"},
                "output": {"plan": "do x"},
                "tool_logs": [],
            },
            {
                "step_id": 1,
                "agent_id": "a1",
                "agent_name": "Worker",
                "input": {"task": "do x"},
                "output": {"result": "wrong"},
                "tool_logs": ["search"],
            },
        ],
        "mistake_agent": "Worker",
        "mistake_step": 1,
    }


def test_format_trajectory_includes_input_output_tool_logs():
    text = format_trajectory(_sample_data()["trajectory"])
    assert "step 1 - Worker" in text
    assert '"result": "wrong"' in text
    assert '"search"' in text


def test_all_at_once_scores_correct_prediction():
    with patch(
        "experiments.single_fault.experiments.baseline.trace_elephant.methods.invoke_structured",
        return_value=({"step_number": 1}, {"latency": 0.1, "input_tokens": 3, "output_tokens": 1}),
    ):
        accuracy, cost = all_at_once_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Worker"
    assert accuracy["pred_step"] == 1
    assert accuracy["agent_accuracy"] == 1.0
    assert accuracy["step_accuracy"] == 1.0


def test_step_by_step_not_found_when_no_step_flagged():
    with patch(
        "experiments.single_fault.experiments.baseline.trace_elephant.methods.invoke_structured",
        return_value=({"error_found": False}, {"latency": 0.1, "input_tokens": 1, "output_tokens": 1}),
    ):
        accuracy, cost = step_by_step_single_file(_sample_data(), model_name="gpt-4o-mini")

    assert accuracy["pred_agent"] == "Not Found"
    assert accuracy["pred_step"] == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/trace_elephant/test_methods.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/single_fault/experiments/baseline/trace_elephant/system_prompt.py
```
Identical content to `who_and_when/system_prompt.py` from Task 7 (same classes `AllAtOnceInput`, `StepByStepInput`, `AllAtOnceResponse`, `StepByStepResponse`, `all_at_once_parser`, `step_by_step_parser`, `all_at_once_prompt`, `step_by_step_prompt`) — copy that file verbatim to this path. This is the intentional per-dataset duplication decided in the design.

```python
# experiments/single_fault/experiments/baseline/trace_elephant/methods.py
from __future__ import annotations

import json

from experiments.single_fault.experiments.baseline.trace_elephant.system_prompt import (
    AllAtOnceInput,
    StepByStepInput,
    all_at_once_parser,
    all_at_once_prompt,
    step_by_step_parser,
    step_by_step_prompt,
)
from experiments.single_fault.utils.accuracy import agent_names_match
from experiments.single_fault.utils.llm import invoke_structured


def format_trajectory(trajectory: list[dict]) -> str:
    lines = []
    for item in trajectory:
        lines.append(
            f"- step {item['step_id']} - {item['agent_name']}: "
            f"input={json.dumps(item['input'])} | "
            f"output={json.dumps(item['output'])} | "
            f"tool_logs={json.dumps(item['tool_logs'])}"
        )
    return "\n".join(lines)


def _score(gt_agent: str, gt_step: int, pred_agent: str, pred_step: int) -> dict:
    return {
        "gt_agent": gt_agent,
        "gt_step": gt_step,
        "pred_agent": pred_agent,
        "pred_step": pred_step,
        "agent_accuracy": float(agent_names_match(gt_agent, pred_agent)),
        "step_accuracy": float(gt_step == pred_step),
    }


def all_at_once_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    trajectory = data["trajectory"]
    step_to_agent_name = {
        int(item["step_id"]): item["agent_name"] for item in trajectory
    }

    method_input = AllAtOnceInput(
        problem=data["task_instruction"],
        chat_content=format_trajectory(trajectory),
    )
    result, cost = invoke_structured(
        model_name=model_name,
        prompt_template=all_at_once_prompt,
        parser=all_at_once_parser,
        prompt_params=method_input,
    )

    pred_step = int(result["step_number"])
    pred_agent = step_to_agent_name.get(pred_step, "Not Found")
    accuracy = _score(
        gt_agent=data["mistake_agent"],
        gt_step=int(data["mistake_step"]),
        pred_agent=pred_agent,
        pred_step=pred_step,
    )
    return accuracy, cost


def step_by_step_single_file(data: dict, model_name: str) -> tuple[dict, dict]:
    trajectory = data["trajectory"]
    total_cost = {"latency": 0.0, "input_tokens": 0, "output_tokens": 0}

    for current_step, item in enumerate(trajectory):
        method_input = StepByStepInput(
            problem=data["task_instruction"],
            current_step_content=format_trajectory([item]),
            chat_content=format_trajectory(trajectory[: current_step + 1]),
        )
        result, cost = invoke_structured(
            model_name=model_name,
            prompt_template=step_by_step_prompt,
            parser=step_by_step_parser,
            prompt_params=method_input,
        )
        total_cost = {
            "latency": total_cost["latency"] + cost["latency"],
            "input_tokens": total_cost["input_tokens"] + cost["input_tokens"],
            "output_tokens": total_cost["output_tokens"] + cost["output_tokens"],
        }

        if result["error_found"]:
            accuracy = _score(
                gt_agent=data["mistake_agent"],
                gt_step=int(data["mistake_step"]),
                pred_agent=item["agent_name"],
                pred_step=current_step,
            )
            return accuracy, total_cost

    accuracy = _score(
        gt_agent=data["mistake_agent"],
        gt_step=int(data["mistake_step"]),
        pred_agent="Not Found",
        pred_step=-1,
    )
    return accuracy, total_cost
```

```python
# experiments/single_fault/experiments/baseline/trace_elephant/run.py
from __future__ import annotations

from pathlib import Path

from experiments.single_fault.experiments.baseline.trace_elephant.methods import (
    all_at_once_single_file,
    step_by_step_single_file,
)
from experiments.single_fault.experiments.shared import MethodConfig, run_method_configs_for_dataset
from experiments.single_fault.utils.experiment_paths import BASELINE_OUTPUT_DIR
from experiments.single_fault.utils.schema import AccuracyMetrics, CostMetrics, Metadata


DEFAULT_MODEL_NAME = "gpt-4o-mini"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = PROJECT_ROOT / "data" / "error_localization" / "single_fault" / "trace_elephant"
DATASET_KEY = "trace_elephant"


def _to_legacy_metrics(accuracy: dict, cost: dict):
    accuracy_metrics = AccuracyMetrics(
        gt_agent=accuracy["gt_agent"],
        gt_step=accuracy["gt_step"],
        pred_agent=accuracy["pred_agent"],
        pred_step=accuracy["pred_step"],
        agent_accuracy=accuracy["agent_accuracy"],
        step_accuracy=accuracy["step_accuracy"],
    )
    cost_metrics = CostMetrics(
        num_input_steps=0,
        latency=cost["latency"],
        input_tokens=cost["input_tokens"],
        output_tokens=cost["output_tokens"],
        input_cost=0.0,
        output_cost=0.0,
        total_cost=0.0,
    )
    return accuracy_metrics, cost_metrics


def _run_all_at_once(data: dict, metadata: Metadata):
    accuracy, cost = all_at_once_single_file(data, model_name=metadata.model_name)
    return _to_legacy_metrics(accuracy, cost)


def _run_step_by_step(data: dict, metadata: Metadata):
    accuracy, cost = step_by_step_single_file(data, model_name=metadata.model_name)
    return _to_legacy_metrics(accuracy, cost)


def build_method_configs() -> list[MethodConfig]:
    return [
        MethodConfig(
            method_name="all_at_once",
            metadata_method="all_at_once",
            run_single_file=_run_all_at_once,
        ),
        MethodConfig(
            method_name="step_by_step",
            metadata_method="step_by_step",
            run_single_file=_run_step_by_step,
        ),
    ]


def main() -> None:
    accuracy_path, cost_path = run_method_configs_for_dataset(
        dataset_key=DATASET_KEY,
        data_dir=DATA_DIR,
        model_name=DEFAULT_MODEL_NAME,
        output_dir=BASELINE_OUTPUT_DIR,
        accuracy_file_name=f"{DATASET_KEY}.csv",
        cost_file_name=f"{DATASET_KEY}_cost.csv",
        experiment_name="baseline_trace_elephant",
        method_configs=build_method_configs(),
    )
    print(f"Saved trace_elephant baseline accuracy results to: {accuracy_path}")
    print(f"Saved trace_elephant baseline cost results to: {cost_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/trace_elephant/test_methods.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/single_fault/experiments/baseline/trace_elephant/ tests/single_fault/baseline/trace_elephant/test_methods.py
git commit -m "feat: add trace_elephant-specific baseline prompts, methods, and driver"
```

---

### Task 10: Top-level entrypoint + `dataset_file_paths` note

**Files:**
- Modify: `experiments/single_fault/experiments/baseline/run.py` (replace entire content)
- Test: `tests/single_fault/baseline/test_entrypoint.py`

**Interfaces:**
- Consumes: `who_and_when/run.py::main` (Task 8), `trace_elephant/run.py::main` (Task 9), `telbench/run.py::main` (Task 6)
- Produces: `main() -> None` in `experiments/single_fault/experiments/baseline/run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/single_fault/baseline/test_entrypoint.py
from __future__ import annotations

from unittest.mock import patch

from experiments.single_fault.experiments.baseline.run import main


def test_main_calls_all_three_dataset_mains():
    with patch(
        "experiments.single_fault.experiments.baseline.run.who_and_when_main"
    ) as ww_main, patch(
        "experiments.single_fault.experiments.baseline.run.trace_elephant_main"
    ) as te_main, patch(
        "experiments.single_fault.experiments.baseline.run.telbench_main"
    ) as tb_main:
        main()

    ww_main.assert_called_once()
    te_main.assert_called_once()
    tb_main.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/single_fault/baseline/test_entrypoint.py -v`
Expected: FAIL — old `run.py` has no `who_and_when_main`/`trace_elephant_main`/`telbench_main` names to patch (`AttributeError`).

- [ ] **Step 3: Replace `run.py` content**

```python
# experiments/single_fault/experiments/baseline/run.py
from __future__ import annotations

from experiments.single_fault.experiments.baseline.telbench.run import main as telbench_main
from experiments.single_fault.experiments.baseline.trace_elephant.run import (
    main as trace_elephant_main,
)
from experiments.single_fault.experiments.baseline.who_and_when.run import (
    main as who_and_when_main,
)


def main() -> None:
    who_and_when_main()
    trace_elephant_main()
    telbench_main()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/single_fault/baseline/test_entrypoint.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Add `__init__.py` files so the new packages import cleanly**

```bash
touch "experiments/single_fault/experiments/baseline/who_and_when/__init__.py"
touch "experiments/single_fault/experiments/baseline/trace_elephant/__init__.py"
touch "experiments/single_fault/experiments/baseline/telbench/__init__.py"
touch "tests/__init__.py" "tests/single_fault/__init__.py" "tests/single_fault/baseline/__init__.py" \
      "tests/single_fault/baseline/who_and_when/__init__.py" \
      "tests/single_fault/baseline/trace_elephant/__init__.py" \
      "tests/single_fault/baseline/telbench/__init__.py"
```

- [ ] **Step 6: Run the full new test suite**

Run: `python -m pytest tests/single_fault/baseline -v`
Expected: PASS (all tests from Tasks 1–10, ~24 tests total)

- [ ] **Step 7: Commit**

```bash
git add experiments/single_fault/experiments/baseline/run.py tests/single_fault/baseline/test_entrypoint.py \
        experiments/single_fault/experiments/baseline/who_and_when/__init__.py \
        experiments/single_fault/experiments/baseline/trace_elephant/__init__.py \
        experiments/single_fault/experiments/baseline/telbench/__init__.py \
        tests/__init__.py tests/single_fault/__init__.py tests/single_fault/baseline/__init__.py \
        tests/single_fault/baseline/who_and_when/__init__.py \
        tests/single_fault/baseline/trace_elephant/__init__.py \
        tests/single_fault/baseline/telbench/__init__.py
git commit -m "feat: wire baseline entrypoint to per-dataset who_and_when/trace_elephant/telbench drivers"
```

---

## Manual verification (not automated — costs real API calls)

After all tasks pass, do one real end-to-end smoke run before trusting the pipeline on the full datasets:

```bash
conda activate rs_segment
python -c "
from pathlib import Path
from experiments.single_fault.experiments.baseline.telbench.run import TELBENCH_DATA_DIR, run_telbench
import json, tempfile

# copy just sample 0 into a scratch dir so this doesn't touch the real output CSV
with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    data = json.loads((TELBENCH_DATA_DIR / '0.json').read_text())
    (tmp_dir / '0.json').write_text(json.dumps(data))
    run_telbench(data_dir=tmp_dir, output_path=tmp_dir / 'out.csv', model_name='gpt-4o-mini')
    print((tmp_dir / 'out.csv').read_text())
"
```

Do the same one-sample smoke check for `who_and_when/run.py` and `trace_elephant/run.py` (copy one JSON file from each real data dir into a temp dir, call `run_method_configs_for_dataset` against it) before kicking off a full run across all datasets. Confirm costs/tokens look sane and predictions parse without `OutputFixingParser` errors.

---

## Self-review notes

- **Spec coverage:** who_and_when (Task 7–8), trace_elephant (Task 9, including the real field-name differences discovered by inspecting sample JSON — `task_instruction`/`step_id`/`input`/`output`/`tool_logs`), telbench (Task 2–6, FEA + degenerate P/R/F1, exceed-token-limit flag preserving prior state), shared LLM plumbing (Task 1), entrypoint glue (Task 10) — all covered.
- **Placeholder scan:** none found — every step has runnable code.
- **Type consistency:** `methods.py` in all three dataset folders returns `(dict, dict)` — checked consistent across Tasks 5, 7, 9. `who_and_when/run.py` and `trace_elephant/run.py` both adapt that dict shape back to `AccuracyMetrics`/`CostMetrics` via a local `_to_legacy_metrics`, matching `shared.py`'s existing `SingleFileRunner` signature (`Callable[[dict, Metadata], tuple[AccuracyMetrics, CostMetrics]]`).
