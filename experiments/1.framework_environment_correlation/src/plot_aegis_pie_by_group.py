"""Per-framework pie charts: injected-error-category composition (AEGIS).

One pie per framework; each slice = one of the 3 MAST categories (System
Design Issues / Inter-Agent Misalignment / Task Verification), reusing
schemas/mast.py's MAST_METADATA for grouping + colors so this reads
consistently with MAST's plot_mast_pie_by_group.py.

magentic_one/smoagents are shown separately from the other 4 frameworks:
they always inject exactly 1 agent with 1 injection_strategy (see
analyze_aegis_correlation.py) and are 100% confounded with their own
benchmark — not directly comparable to the pooled 4.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_aegis_pie_by_group.py
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

json_dir = REPO_ROOT / "data/error_localization/multi_fault/aegis"
fig_path = RESULTS_DIR / "aegis_pie_group_by_framework.png"

SINGLE_INJECTION_FRAMEWORKS = {"magentic_one", "smoagents"}

GROUP_COLORS = {
    "System Design Issues": plt.get_cmap("Purples")(0.6),
    "Inter-Agent Misalignment": plt.get_cmap("Reds")(0.6),
    "Task Verification": plt.get_cmap("Greens")(0.6),
}


def code_to_group() -> dict:
    return {m.code: g.name for g in MAST_METADATA.groups for m in g.failure_modes}


def load_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        framework = data["metadata"]["framework"]
        for agent in data["ground_truth"]["injected_agents"]:
            # error_type is "FM-1.1" etc.; MAST taxonomy codes are "1.1" etc.
            code = agent["error_type"].removeprefix("FM-")
            rows.append({"framework": framework, "code": code})
    return pd.DataFrame(rows)


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_aegis` first.")
    df = load_dataframe()
    code2group = code_to_group()
    df["group"] = df["code"].map(code2group)
    group_names = [g.name for g in MAST_METADATA.groups]

    n_per_fw = df.groupby("framework").size()
    pooled = [f for f in n_per_fw.index if f not in SINGLE_INJECTION_FRAMEWORKS]
    single = [f for f in n_per_fw.index if f in SINGLE_INJECTION_FRAMEWORKS]
    frameworks = sorted(pooled, key=lambda f: -n_per_fw[f]) + sorted(single, key=lambda f: -n_per_fw[f])

    fig, axes = plt.subplots(1, len(frameworks), figsize=(len(frameworks) * 3.4, 5))

    for i, (ax, fw) in enumerate(zip(axes, frameworks)):
        counts = df[df["framework"] == fw]["group"].value_counts().reindex(group_names).dropna()
        ax.pie(
            counts.values,
            colors=[GROUP_COLORS[g] for g in counts.index],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        tag = " (single-injection)" if fw in SINGLE_INJECTION_FRAMEWORKS else ""
        ax.set_title(f"{fw}{tag}\n(n={n_per_fw[fw]} traces)", fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in group_names]
    fig.legend(handles, group_names, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "AEGIS: injected error-category composition by framework\n"
        "magentic_one/smoagents: always-1-injection regime, shown separately (not pooled)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 0.88])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
