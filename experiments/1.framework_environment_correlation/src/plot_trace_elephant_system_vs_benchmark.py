"""Confusion-matrix-style heatmap: system_name x benchmark_name (TraceElephant).

Same purpose as MAST's plot_mas_name_vs_benchmark.py: visualize how
confounded system_name is with benchmark_name. Here it's near-total except
for GAIA, which both captain-agent and magentic-one run — the only
benchmark-matched comparison point between 2 systems in this dataset.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_trace_elephant_system_vs_benchmark.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

json_dir = REPO_ROOT / "data/error_localization/single_fault/trace_elephant"
fig_path = RESULTS_DIR / "trace_elephant_system_x_benchmark.png"


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"system_name": data["system_name"], "benchmark_name": data["benchmark_name"]})
    return pd.DataFrame(rows)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_trace_elephant` first.")
    df = load_dataframe()

    matrix = pd.crosstab(df["system_name"], df["benchmark_name"])

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, matrix.shape[1] * 1.4), max(4, matrix.shape[0] * 0.9)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "# traces"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(f"TraceElephant: system_name x benchmark_name (n={len(df)} traces)")
    plt.xlabel("benchmark_name")
    plt.ylabel("system_name")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(matrix.to_string())
    print(f"\nSaved -> {fig_path}")


if __name__ == "__main__":
    main()
