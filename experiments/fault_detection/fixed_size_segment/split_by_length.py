"""Split fixed_size_segment (token-ratio) output CSVs into short/long subsets,
using the step-count threshold manifest from data/error_categorization/split_by_length.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.error_categorization.mast import json_dir

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
LENGTH_SPLIT_PATH = json_dir / "length_split.json"


def load_ids() -> tuple[set[int], set[int]]:
    manifest = json.loads(LENGTH_SPLIT_PATH.read_text(encoding="utf-8"))
    return set(manifest["short_ids"]), set(manifest["long_ids"])


def split_csv(csv_path: Path, short_ids: set[int], long_ids: set[int]) -> None:
    df = pd.read_csv(csv_path)
    file_ids = df["file_name"].map(lambda name: int(Path(name).stem))

    short_dir = OUTPUT_DIR / "short"
    long_dir = OUTPUT_DIR / "long"
    short_dir.mkdir(parents=True, exist_ok=True)
    long_dir.mkdir(parents=True, exist_ok=True)

    df[file_ids.isin(short_ids)].to_csv(short_dir / csv_path.name, index=False)
    df[file_ids.isin(long_ids)].to_csv(long_dir / csv_path.name, index=False)


def main() -> None:
    short_ids, long_ids = load_ids()
    csv_paths = sorted(OUTPUT_DIR.glob("fixed_size_segment_*pct.csv"))

    for csv_path in csv_paths:
        split_csv(csv_path, short_ids, long_ids)
        print(f"split {csv_path.name}: short/{csv_path.name}, long/{csv_path.name}")


if __name__ == "__main__":
    main()
