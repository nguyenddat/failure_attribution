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
