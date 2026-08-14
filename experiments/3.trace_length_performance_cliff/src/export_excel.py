from __future__ import annotations

from pathlib import Path

import pandas as pd

ACCURACY_COLUMNS = [
    "model",
    "dataset",
    "method",
    "file",
    "gt_agent",
    "gt_step",
    "pred_agent",
    "pred_step",
    "agent_accuracy",
    "step_accuracy",
    "num_steps",
    "avg_token_count",
    "status",
]

COST_COLUMNS = [
    "model",
    "dataset",
    "method",
    "file",
    "latency",
    "input_tokens",
    "output_tokens",
    "num_steps",
    "avg_token_count",
    "status",
]


def load_or_init(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_excel(path)
    return pd.DataFrame(columns=columns)


def _row_mask(df: pd.DataFrame, model: str, dataset: str, method: str, file: str) -> pd.Series:
    return (
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["method"] == method)
        & (df["file"] == file)
    )


def is_row_done(df: pd.DataFrame, model: str, dataset: str, method: str, file: str) -> bool:
    if df.empty:
        return False
    match = df[_row_mask(df, model, dataset, method, file)]
    if match.empty:
        return False
    return bool(match["status"].notna().iloc[0])


def upsert_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    if not df.empty:
        mask = _row_mask(df, row["model"], row["dataset"], row["method"], row["file"])
        if mask.any():
            for key, value in row.items():
                df.loc[mask, key] = value
            return df
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)
