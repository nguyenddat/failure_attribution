"""Draw the trace length distribution of every dataset onto one pair of axes.

Writes ``figures/trace_token_length.png`` and ``figures/trace_step_length.png``
next to this module, and prints per-dataset statistics.
"""

from pathlib import Path

from data.error_categorization.mast_length import DATASET as MAST
from data.error_localization.multi_fault.aegis_length import DATASET as AEGIS
from data.error_localization.multi_fault.trail_length import DATASET as TRAIL
from data.error_localization.single_fault.agent_error_bench_length import DATASET as AEB
from data.error_localization.single_fault.ww_algorithm_generated_length import DATASET as WW_ALGO
from data.error_localization.single_fault.ww_hand_crafted_length import DATASET as WW_HAND
from data.trace_length import run

figures_dir = Path(__file__).resolve().parent / "figures"

# Plot order, which is also the colour order.
DATASETS = [WW_HAND, WW_ALGO, AEB, MAST, AEGIS, TRAIL]

if __name__ == "__main__":
    run(DATASETS, figures_dir)
