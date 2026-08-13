"""Correlation between ``mas_name`` (framework) and ``mast_annotation`` (14 MAST codes).

Reads every sample under ``data/error_categorization/mast/`` (written by
``data/error_categorization/mast.py`` — run ``make load_mast`` first) and, for
each of the 14 fault codes, tests whether its presence/absence is associated
with the framework via a chi-square test of independence on the
``mas_name x code`` contingency table. Effect size is Cramer's V.

Also reports how confounded ``mas_name`` is with ``llm_name`` and
``benchmark_name`` in this dataset (each framework is not run on every
model/benchmark combination), since a significant mas_name/code association
may really be a benchmark or model effect.

AppWorld/HyperAgent/OpenManus are dropped entirely before any analysis (see
``mast_known_issues.py``) — their mast_annotation is a known index-aligned
duplication bug, not real per-framework signal.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/analyze_mas_annotation_correlation.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

from schemas.mast import MAST_METADATA
from mast_known_issues import DUPLICATE_ANNOTATION_BUG

json_dir = REPO_ROOT / "data/error_categorization/mast"

CODES = [mode.code for group in MAST_METADATA.groups for mode in group.failure_modes]
CODE_TO_GROUP = {
    mode.code: group.name for group in MAST_METADATA.groups for mode in group.failure_modes
}


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name == "metadata.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["mas_name"] in DUPLICATE_ANNOTATION_BUG:
            continue
        row = {
            "mas_name": data["mas_name"],
            "llm_name": data["llm_name"],
            "benchmark_name": data["benchmark_name"],
        }
        row.update({code: data["mast_annotation"][code] for code in CODES})
        rows.append(row)
    return pd.DataFrame(rows)


def confound_report(df: pd.DataFrame) -> None:
    print("=== Confound check: mas_name x benchmark_name x llm_name ===")
    combo = df.groupby(["mas_name", "benchmark_name", "llm_name"]).size()
    combo = combo[combo > 0]
    per_mas = combo.reset_index().groupby("mas_name").size()
    print("Distinct (benchmark, llm) pairs per framework:")
    print(per_mas.to_string())
    print(
        "-> most frameworks map to exactly 1 (benchmark, llm) pair: "
        "mas_name is almost fully confounded with benchmark_name/llm_name here.\n"
    )


def code_association(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    results = []
    for code in CODES:
        table = pd.crosstab(df["mas_name"], df[code])
        chi2, p, dof, _ = chi2_contingency(table)
        k = min(table.shape)
        cramers_v = (chi2 / (n * (k - 1))) ** 0.5 if k > 1 else float("nan")
        results.append(
            {
                "code": code,
                "group": CODE_TO_GROUP[code],
                "rate_overall": df[code].mean(),
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
                "cramers_v": cramers_v,
            }
        )
    out = pd.DataFrame(results).sort_values("cramers_v", ascending=False)
    return out


def rate_by_framework(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("mas_name")[CODES].mean().round(3)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_mast` first.")
    df = load_dataframe()
    print(f"Loaded {len(df)} samples, {df['mas_name'].nunique()} frameworks.\n")

    confound_report(df)

    print("=== Rate of each code by framework (mean of 0/1) ===")
    print(rate_by_framework(df).to_string())
    print()

    print("=== Chi-square association: mas_name x code (sorted by Cramer's V) ===")
    assoc = code_association(df)
    print(assoc.to_string(index=False))

    out_path = RESULTS_DIR / "mas_name_vs_mast_annotation.csv"
    assoc.to_csv(out_path, index=False)
    print(f"\nSaved association table -> {out_path}")


if __name__ == "__main__":
    main()
