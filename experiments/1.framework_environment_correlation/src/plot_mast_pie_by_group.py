"""Per-framework pie charts: failure-CATEGORY composition of each mas_name (MAST).

One pie per ``mas_name`` (7 pies); each slice = one of the 3 MAST categories
(System Design Issues / Inter-Agent Misalignment / Task Verification), sized
by how many tags of that category the framework's traces carry. Coarser
version of ``plot_mast_pie_by_error_type.py`` (which breaks each pie into the
14 individual codes instead of the 3 groups).

AppWorld/HyperAgent/OpenManus are dropped entirely before any analysis (see
mast_known_issues.py): their mast_annotation is a known index-aligned
duplication bug, not a real per-framework signal. That leaves 4 frameworks
(AG2, ChatDev, Magentic, MetaGPT).

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_mast_pie_by_group.py
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
fig_path = RESULTS_DIR / "mast_pie_group_by_framework.png"

GROUP_COLORS = {
    "System Design Issues": plt.get_cmap("Purples")(0.6),
    "Inter-Agent Misalignment": plt.get_cmap("Reds")(0.6),
    "Task Verification": plt.get_cmap("Greens")(0.6),
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


def code_to_group() -> dict:
    return {m.code: g.name for g in MAST_METADATA.groups for m in g.failure_modes}


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_mast` first.")
    df = load_dataframe()

    code2group = code_to_group()
    codes = list(code2group)
    group_names = [g.name for g in MAST_METADATA.groups]
    n_per_mas = df.groupby("mas_name").size()
    frameworks = n_per_mas.sort_values(ascending=False).index.tolist()

    ncols, nrows = len(frameworks), 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 4.6))
    axes = axes if hasattr(axes, "__len__") else [axes]

    for ax, fw in zip(axes, frameworks):
        tags = df[df["mas_name"] == fw][codes].sum()
        by_group = tags.groupby(code2group).sum().reindex(group_names).fillna(0)
        by_group = by_group[by_group > 0]
        total_tags = int(by_group.sum())
        label = f"{fw}\n(n={n_per_mas[fw]} traces, {total_tags} tags)"
        ax.pie(
            by_group.values,
            colors=[GROUP_COLORS[g] for g in by_group.index],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax.set_title(label, fontsize=10)

    for ax in axes[len(frameworks):]:
        ax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in group_names]
    fig.legend(handles, group_names, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(
        "MAST: failure-category composition by framework\n"
        "AppWorld/HyperAgent/OpenManus excluded (index-aligned annotation duplication bug)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
