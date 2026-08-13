"""Confusion-matrix-style heatmap: mas_name x benchmark_name (MAST dataset).

Visualizes how confounded ``mas_name`` (framework) is with ``benchmark_name``
in ``data/error_categorization/mast/`` — most frameworks map to exactly one
benchmark, so any mas_name x mast_annotation association may really be a
benchmark effect. See ``analyze_mas_annotation_correlation.py``.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_mas_name_vs_benchmark.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

json_dir = REPO_ROOT / "data/error_categorization/mast"
fig_path = RESULTS_DIR / "mast_mas_name_x_benchmark_name.png"


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name == "metadata.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"mas_name": data["mas_name"], "benchmark_name": data["benchmark_name"]})
    return pd.DataFrame(rows)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_mast` first.")
    df = load_dataframe()

    matrix = pd.crosstab(df["mas_name"], df["benchmark_name"])

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, matrix.shape[1] * 1.1), max(4, matrix.shape[0] * 0.7)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "# traces"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(f"MAST: mas_name x benchmark_name (n={len(df)} traces)")
    plt.xlabel("benchmark_name")
    plt.ylabel("mas_name")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(matrix.to_string())
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
