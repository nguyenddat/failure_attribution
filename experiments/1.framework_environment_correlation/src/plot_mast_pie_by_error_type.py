"""Per-framework pie charts: failure-mode composition of each mas_name (MAST).

One pie per ``mas_name`` (7 pies); each slice = one of the 14 MAST codes,
sized by how many traces of that framework carry the code (multi-label, so
slices sum to total tag count, not n traces). Code -> color mapping matches
``plot_mast_failure_distribution.py`` (purple=System Design, red=Inter-Agent,
green=Task Verification) so the two figures read consistently together.

AppWorld/HyperAgent/OpenManus are dropped entirely before any analysis (see
mast_known_issues.py): their mast_annotation is a known index-aligned
duplication bug, not a real per-framework signal. That leaves 4 frameworks
(AG2, ChatDev, Magentic, MetaGPT).

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/plot_mast_pie_by_error_type.py
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
fig_path = RESULTS_DIR / "mast_pie_error_type_by_framework.png"

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
            colors[mode.code] = cmap(0.35 + 0.55 * i / max(n - 1, 1))
    return colors


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_mast` first.")
    df = load_dataframe()

    codes = [m.code for g in MAST_METADATA.groups for m in g.failure_modes]
    code_names = {m.code: m.name for g in MAST_METADATA.groups for m in g.failure_modes}
    colors = code_colors()
    n_per_mas = df.groupby("mas_name").size()
    frameworks = n_per_mas.sort_values(ascending=False).index.tolist()

    ncols, nrows = len(frameworks), 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 4.6))
    axes = axes if hasattr(axes, "__len__") else [axes]

    for ax, fw in zip(axes, frameworks):
        counts = df[df["mas_name"] == fw][codes].sum()
        counts = counts[counts > 0]
        total_tags = int(counts.sum())
        label = f"{fw}\n(n={n_per_mas[fw]} traces, {total_tags} tags)"
        ax.pie(
            counts.values,
            colors=[colors[c] for c in counts.index],
            autopct=lambda p: f"{p:.0f}%" if p >= 5 else "",
            startangle=90,
            textprops={"fontsize": 7},
        )
        ax.set_title(label, fontsize=10)

    for ax in axes[len(frameworks):]:
        ax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in codes]
    fw_labels = [f"{c} {code_names[c]}" for c in codes]
    fig.legend(handles, fw_labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "MAST: failure-mode composition by framework\n"
        "AppWorld/HyperAgent/OpenManus excluded (index-aligned annotation duplication bug)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.22, 1, 0.9])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
