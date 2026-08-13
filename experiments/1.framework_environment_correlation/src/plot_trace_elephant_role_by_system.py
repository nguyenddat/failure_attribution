"""Per-system pie charts: failure-role composition (TraceElephant).

One pie per system_name (3 pies); each slice = Orchestrator / Worker /
Verification (trace_elephant_agent_roles.py). Mirrors MAST's
plot_mast_pie_by_group.py.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_trace_elephant_role_by_system.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

from trace_elephant_agent_roles import ROLE_ORDER, classify

json_dir = REPO_ROOT / "data/error_localization/single_fault/trace_elephant"
fig_path = RESULTS_DIR / "trace_elephant_pie_role_by_system.png"

ROLE_COLORS = {
    "Orchestrator": plt.get_cmap("Purples")(0.6),
    "Worker": plt.get_cmap("Reds")(0.6),
    "Verification": plt.get_cmap("Greens")(0.6),
}


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"system_name": data["system_name"], "role": classify(data["mistake_agent"])})
    return pd.DataFrame(rows)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_trace_elephant` first.")
    df = load_dataframe()

    n_per_sys = df.groupby("system_name").size()
    systems = n_per_sys.sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, len(systems), figsize=(len(systems) * 4.2, 5))

    for ax, sysn in zip(axes, systems):
        counts = df[df["system_name"] == sysn]["role"].value_counts().reindex(ROLE_ORDER).dropna()
        ax.pie(
            counts.values,
            colors=[ROLE_COLORS[r] for r in counts.index],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title(f"{sysn}\n(n={n_per_sys[sysn]} traces)", fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLORS[r]) for r in ROLE_ORDER]
    fig.legend(handles, ROLE_ORDER, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("TraceElephant: failure-role composition by system", fontsize=12)
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
