from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "data" / "single_fault" / "json"

DATASET_DIRS = {
    "ww_algorithm_generated": DATA_DIR / "who_and_when__algorithm-generated",
    "ww_hand_crafted": DATA_DIR / "who_and_when__hand-crafted",
}
