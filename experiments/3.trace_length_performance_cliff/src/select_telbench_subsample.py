"""One-off: pick 250/1000 telbench files, stratified by trace length (num_steps).

Preserves the full-dataset length distribution (proportional per decile,
minimum floor per bin so the long-trace tail stays visible), and prefers
files that already have a completed run in accuracy.xlsx so existing LLM
calls are not wasted. Writes the chosen filenames to telbench_subsample.json;
run.py filters telbench to that manifest when it is present.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from datasets import DATASET_DIRS, NORMALIZERS  # noqa: E402

TARGET_TOTAL = 250
NUM_BINS = 10
MIN_PER_BIN = 15
SEED = 0

SRC_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SRC_DIR / "telbench_subsample.json"
ACCURACY_PATH = SRC_DIR.parent / "results" / "tables" / "accuracy.xlsx"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_telbench_lengths() -> pd.DataFrame:
    files = sorted(DATASET_DIRS["telbench"].glob("*.json"), key=lambda p: int(p.stem))
    rows = []
    for f in files:
        normalized = NORMALIZERS["telbench"](load_json(f))
        if normalized is None:
            continue
        rows.append({"file": f.name, "num_steps": len(normalized["trajectory"])})
    return pd.DataFrame(rows)


def already_done_files() -> set[str]:
    if not ACCURACY_PATH.exists():
        return set()
    acc = pd.read_excel(ACCURACY_PATH)
    mask = (acc["dataset"] == "telbench") & acc["status"].notna()
    return set(acc.loc[mask, "file"])


def allocate_quota(bin_sizes: pd.Series, target_total: int, min_per_bin: int) -> pd.Series:
    """Proportional allocation floored at min(min_per_bin, bin size), water-filled to hit
    target_total exactly (each unit goes to the bin furthest below its ideal share)."""
    alloc = bin_sizes.clip(upper=min_per_bin).astype(int)
    remaining = target_total - int(alloc.sum())
    if remaining < 0:
        raise ValueError("min_per_bin * num_bins exceeds target_total")

    ideal = bin_sizes / bin_sizes.sum() * target_total
    for _ in range(remaining):
        deficit = (ideal - alloc).where(alloc < bin_sizes, other=float("-inf"))
        alloc[deficit.idxmax()] += 1

    return alloc


def main() -> None:
    df = valid_telbench_lengths()
    df["bin"] = pd.qcut(df["num_steps"], NUM_BINS, duplicates="drop")

    done = already_done_files()
    rng = random.Random(SEED)

    bins = df.groupby("bin", observed=True)
    bin_sizes = bins.size()
    quota = allocate_quota(bin_sizes, TARGET_TOTAL, MIN_PER_BIN)

    selected: list[str] = []
    for bin_label, group in bins:
        pool = group["file"].tolist()
        done_in_bin = [f for f in pool if f in done]
        rest = [f for f in pool if f not in done]
        rng.shuffle(done_in_bin)
        rng.shuffle(rest)
        chosen = (done_in_bin + rest)[: quota[bin_label]]
        selected.extend(chosen)

    selected.sort(key=lambda name: int(Path(name).stem))
    MANIFEST_PATH.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    summary = df.assign(selected=df["file"].isin(selected))
    print(f"valid telbench files: {len(df)}")
    print(f"selected: {len(selected)} (target {TARGET_TOTAL})")
    print(f"reused already-done files: {len(set(selected) & done)}")
    print(summary.groupby("bin", observed=True)["selected"].agg(["sum", "count"]))


if __name__ == "__main__":
    main()
