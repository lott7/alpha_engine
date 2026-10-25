from __future__ import annotations

import matplotlib
import pandas as pd

from alpha_engine.backtest import BacktestResult
from alpha_engine.plotting import plot_backtest_result

matplotlib.use("Agg")


def test_plot_backtest_result_returns_axes() -> None:
    cumulative_returns = pd.DataFrame(
        {
            "strategy": [1.0, 1.02, 1.05],
            "IWV": [1.0, 1.01, 1.03],
            "SPY": [1.0, 1.015, 1.04],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )
    result = BacktestResult(cumulative_returns=cumulative_returns)

    ax = plot_backtest_result(result)

    assert ax.get_title() == "Strategy vs. Benchmark Returns"
    assert len(ax.lines) == 3
