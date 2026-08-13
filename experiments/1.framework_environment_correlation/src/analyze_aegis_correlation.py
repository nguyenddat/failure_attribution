"""Correlation between ``framework`` and injected ``error_type`` (AEGIS).

Mirrors MAST's analyze_mas_annotation_correlation.py, but here the "fault
tags" are literal — each trace has 1-4 explicitly injected agents, each
carrying one MAST failure-mode code (``FM-1.1`` .. ``FM-3.3``, reused from
``schemas/mast.py``). No annotation-reliability bug like MAST's was found
here; instead this reports 2 real data-quality caveats:

1. ``metadata.num_agents``/``num_injected_agents`` are unreliable — verify
   against the actual conversation_history/injected_agents lists, don't
   trust the metadata counts directly.
2. ``magentic_one``/``smoagents`` are structurally different from the
   other 4 frameworks: each is 100% confounded with its own
   framework-specific benchmark label and *always* injects exactly 1
   agent using exactly 1 injection_strategy — a deliberate pipeline
   limitation (not a bug), so treat them as a separate regime rather than
   averaging them in with agentverse/dylan/llm_debate/macnet.

Part of experiment 1.framework_environment_correlation. Run directly:
python experiments/1.framework_environment_correlation/src/analyze_aegis_correlation.py
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

json_dir = REPO_ROOT / "data/error_localization/multi_fault/aegis"

SINGLE_INJECTION_FRAMEWORKS = {"magentic_one", "smoagents"}


def load_frame_level(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    injected = data["ground_truth"]["injected_agents"]
    return {
        "framework": data["metadata"]["framework"],
        "benchmark": data["metadata"]["benchmark"],
        "model": data["metadata"]["model"],
        "num_agents_meta": data["metadata"]["num_agents"],
        "num_injected_meta": data["metadata"]["num_injected_agents"],
        "n_conv_agents_actual": len(set(s["agent_name"] for s in data["input"]["conversation_history"])),
        "n_injected_actual": len(injected),
        "injection_strategies": tuple(sorted(a["injection_strategy"] for a in injected)),
        "error_types": tuple(a["error_type"] for a in injected),
    }


def load_dataframe() -> pd.DataFrame:
    return pd.DataFrame(load_frame_level(p) for p in sorted(json_dir.glob("*.json")))


def metadata_reliability_report(df: pd.DataFrame) -> None:
    print("=== Metadata reliability ===")
    agree_agents = (df["num_agents_meta"] == df["n_conv_agents_actual"]).mean()
    agree_injected = (df["num_injected_meta"] == df["n_injected_actual"]).mean()
    print(f"num_agents (metadata) matches actual unique agent count: {agree_agents:.1%}")
    print(f"num_injected_agents (metadata) matches len(injected_agents): {agree_injected:.1%}")
    print("-> do not trust metadata counts; use the actual lists.\n")


def single_injection_report(df: pd.DataFrame) -> None:
    print("=== magentic_one / smoagents: deliberate single-injection regime ===")
    for fw in sorted(SINGLE_INJECTION_FRAMEWORKS):
        sub = df[df["framework"] == fw]
        strategies = sub["injection_strategies"].value_counts()
        print(
            f"{fw}: n={len(sub)}, n_injected always 1? {(sub['n_injected_actual']==1).all()}, "
            f"injection_strategies used: {strategies.to_dict()}"
        )
    print(
        "-> both frameworks always inject exactly 1 agent using exactly 1 "
        "injection_strategy — a pipeline design limit, not a bug. Excluded "
        "from the pooled chi-square below; benchmark/model are also 100% "
        "confounded with these 2 frameworks (see plot_aegis_framework_vs_benchmark.py).\n"
    )


def error_type_association(df: pd.DataFrame) -> pd.DataFrame:
    print("=== framework x error_type, agentverse/dylan/llm_debate/macnet only ===")
    pooled = df[~df["framework"].isin(SINGLE_INJECTION_FRAMEWORKS)]
    long = (
        pooled[["framework", "error_types"]]
        .explode("error_types")
        .rename(columns={"error_types": "error_type"})
        .reset_index(drop=True)
    )
    table = pd.crosstab(long["framework"], long["error_type"])

    results = []
    for code in table.columns:
        sub_table = pd.crosstab(long["framework"], long["error_type"] == code)
        chi2, p, dof, _ = chi2_contingency(sub_table)
        n = sub_table.values.sum()
        k = min(sub_table.shape)
        v = (chi2 / (n * (k - 1))) ** 0.5 if k > 1 else float("nan")
        results.append({"error_type": code, "n_tags": int(table[code].sum()), "p_value": p, "cramers_v": v})
    out = pd.DataFrame(results).sort_values("cramers_v", ascending=False)
    print(out.to_string(index=False))
    print()
    return out


def main() -> None:
    if not json_dir.exists():
        raise SystemExit(f"{json_dir} not found — run `make load_aegis` first.")
    df = load_dataframe()
    print(f"Loaded {len(df)} traces, {df['framework'].nunique()} frameworks.\n")

    metadata_reliability_report(df)
    single_injection_report(df)
    assoc = error_type_association(df)

    out_path = RESULTS_DIR / "aegis_framework_vs_error_type.csv"
    assoc.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
