from __future__ import annotations

from unittest.mock import patch

from baseline.run import main


def test_main_calls_all_three_dataset_mains():
    with patch(
        "baseline.run.who_and_when_main"
    ) as ww_main, patch(
        "baseline.run.trace_elephant_main"
    ) as te_main, patch(
        "baseline.run.telbench_main"
    ) as tb_main:
        main()

    ww_main.assert_called_once()
    te_main.assert_called_once()
    tb_main.assert_called_once()
