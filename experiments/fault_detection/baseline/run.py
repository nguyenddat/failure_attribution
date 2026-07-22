from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.fault_detection.mast import Sample, MAST_METADATA, json_dir
from experiments.fault_detection.baseline.methods.all_at_once import (
    all_at_once,
    ExperimentMetadata,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_CSV = OUTPUT_DIR / "all_at_once.csv"

MAX_WORKERS = 5
REQUESTS_PER_MINUTE = 60

METRIC_COLUMNS = [
    "exact_match_ratio",
    "hamming_loss",
    "precision",
    "recall",
    "f1",
    "group_precision",
    "group_recall",
    "group_f1",
    "input_tokens",
    "output_tokens",
    "latency",
]
BASE_COLUMNS = ["file_name", "model", "dataset", "segmentation", "gt_faults", "pred_faults"]
ALL_COLUMNS = BASE_COLUMNS + METRIC_COLUMNS + ["error"]

MAX_TOKEN_ERROR_KEYWORDS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "context length",
    "too many tokens",
    "prompt is too long",
)


def is_max_token_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in MAX_TOKEN_ERROR_KEYWORDS)


class RateLimiter:
    """Throttle call starts to at most `requests_per_minute`, shared across threads."""

    def __init__(self, requests_per_minute: int):
        self.min_interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.last_call = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = self.last_call + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self.last_call = now


def dataset_file_paths() -> list[Path]:
    return sorted(
        (path for path in json_dir.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def load_or_init_results() -> pd.DataFrame:
    if OUTPUT_CSV.exists():
        return pd.read_csv(OUTPUT_CSV)
    return pd.DataFrame(columns=ALL_COLUMNS)


def is_row_complete(df: pd.DataFrame, file_name: str) -> bool:
    matched = df[df["file_name"] == file_name]
    if matched.empty:
        return False
    row = matched.iloc[0]
    # "exceed max token" la loi vinh vien (khong retry). Cac loi khac (rate limit,
    # network...) van duoc ghi vao CSV de nhin thay, nhung khong tinh la complete
    # nen lan chay sau se tu retry va ghi de dong nay.
    if row.get("error") == "exceed max token":
        return True
    return bool(row[METRIC_COLUMNS].notna().all())


def upsert_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df = df[df["file_name"] != row["file_name"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(
        "file_name", key=lambda col: col.map(lambda name: int(Path(name).stem))
    ).reset_index(drop=True)
    return df


def process_file(
    file_path: Path,
    experiment_metadata: ExperimentMetadata,
    limiter: RateLimiter,
):
    limiter.acquire()
    sample = Sample.model_validate_json(file_path.read_text(encoding="utf-8"))
    performance_metrics, cost_metrics = all_at_once(
        sample=sample,
        data_metadata=MAST_METADATA,
        experiment_metadata=experiment_metadata,
    )
    return performance_metrics, cost_metrics


def run(
    model_name: str = "gpt-4o-mini",
    dataset_name: str = "mast",
    max_workers: int = MAX_WORKERS,
    requests_per_minute: int = REQUESTS_PER_MINUTE,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_init_results()
    experiment_metadata = ExperimentMetadata(
        model_name=model_name, dataset_name=dataset_name, segmentation=None
    )

    file_paths = dataset_file_paths()
    pending = [fp for fp in file_paths if not is_row_complete(df, fp.name)]
    skipped = len(file_paths) - len(pending)

    limiter = RateLimiter(requests_per_minute)
    write_lock = threading.Lock()

    progress = tqdm(total=len(file_paths), desc="fault_detection:all_at_once", unit="file")
    progress.update(skipped)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_file, file_path, experiment_metadata, limiter
                ): file_path
                for file_path in pending
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    performance_metrics, cost_metrics = future.result()
                except Exception as exc:
                    progress.write(f"Failed {file_path.name}: {exc}")

                    error_message = (
                        "exceed max token" if is_max_token_error(exc) else str(exc)
                    )
                    sample = Sample.model_validate_json(
                        file_path.read_text(encoding="utf-8")
                    )
                    row = {
                        "file_name": file_path.name,
                        "model": model_name,
                        "dataset": dataset_name,
                        "segmentation": experiment_metadata.segmentation,
                        "gt_faults": "|".join(sample.faults),
                        "pred_faults": "",
                        "error": error_message,
                    }
                    with write_lock:
                        df = upsert_row(df, row)
                        df.to_csv(OUTPUT_CSV, index=False)

                    progress.set_postfix({"file": file_path.name})
                    progress.update(1)
                    continue

                row = {
                    "file_name": file_path.name,
                    "model": model_name,
                    "dataset": dataset_name,
                    "segmentation": experiment_metadata.segmentation,
                    "gt_faults": "|".join(performance_metrics.gt_faults),
                    "pred_faults": "|".join(performance_metrics.pred_faults),
                    "error": "",
                    **{
                        col: getattr(performance_metrics, col)
                        for col in METRIC_COLUMNS
                        if hasattr(performance_metrics, col)
                    },
                    **{
                        col: getattr(cost_metrics, col)
                        for col in METRIC_COLUMNS
                        if hasattr(cost_metrics, col)
                    },
                }

                with write_lock:
                    df = upsert_row(df, row)
                    df.to_csv(OUTPUT_CSV, index=False)

                progress.set_postfix({"file": file_path.name})
                progress.update(1)
    finally:
        progress.close()

    return OUTPUT_CSV


if __name__ == "__main__":
    output_path = run()
    print(f"Saved fault detection results to: {output_path}")
