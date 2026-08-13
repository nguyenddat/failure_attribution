"""Failure-mode distribution by MAS framework (MAST dataset).

Stacked bar chart, one bar per ``mas_name``, segments = 14 MAST failure-mode
codes (colored by category: System Design / Inter-Agent / Task Verification),
mirroring Figure 4 of the MAST paper. Y-axis is rate per 100 traces (not raw
count) since frameworks have very different n (AG2 n=597 vs AppWorld n=30) —
raw counts would just track n, not real prevalence.

AppWorld/HyperAgent/OpenManus are dropped entirely before any analysis (see
mast_known_issues.py): their mast_annotation is a known index-aligned
duplication bug, not a real per-framework signal.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_mast_failure_distribution.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

from schemas.mast import MAST_METADATA
from mast_known_issues import DUPLICATE_ANNOTATION_BUG

json_dir = REPO_ROOT / "data/error_categorization/mast"
fig_path = RESULTS_DIR / "mast_failure_distribution_by_framework.png"

GROUP_CMAPS = {
    "System Design Issues": "Purples",
    "Inter-Agent Misalignment": "Reds",
    "Task Verification": "Greens",
}


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name == "metadata.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["mas_name"] in DUPLICATE_ANNOTATION_BUG:
            continue
        row = {"mas_name": data["mas_name"]}
        row.update(data["mast_annotation"])
        rows.append(row)
    return pd.DataFrame(rows)


def code_colors() -> dict:
    colors = {}
    for group in MAST_METADATA.groups:
        cmap = plt.get_cmap(GROUP_CMAPS[group.name])
        n = len(group.failure_modes)
        for i, mode in enumerate(group.failure_modes):
            # skip the very light end of the colormap so bars stay visible
            colors[mode.code] = cmap(0.35 + 0.55 * i / max(n - 1, 1))
    return colors


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_mast` first.")
    df = load_dataframe()

    codes = [m.code for g in MAST_METADATA.groups for m in g.failure_modes]
    code_names = {m.code: m.name for g in MAST_METADATA.groups for m in g.failure_modes}
    rate = df.groupby("mas_name")[codes].mean() * 100  # per 100 traces
    n_per_mas = df.groupby("mas_name").size()
    rate = rate.reindex(rate.sum(axis=1).sort_values(ascending=False).index)

    colors = code_colors()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bottom = pd.Series(0.0, index=rate.index)
    for code in codes:
        ax.bar(rate.index, rate[code], bottom=bottom, color=colors[code], label=f"{code} {code_names[code]}", width=0.6)
        bottom += rate[code]

    xticklabels = [f"{name}\n(n={n_per_mas[name]})" for name in rate.index]
    ax.set_xticks(range(len(rate.index)))
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Expected # failure-mode tags per 100 traces")
    ax.set_title(
        "MAST: failure-mode distribution by framework\n"
        "AppWorld/HyperAgent/OpenManus excluded (index-aligned annotation duplication bug)"
    )

    # group codes into 3 legend blocks, matching the paper's category colors
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1, title="Failure mode")

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(rate.round(1).to_string())
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
