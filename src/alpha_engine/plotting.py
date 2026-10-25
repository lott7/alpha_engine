from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from alpha_engine.backtest import BacktestResult


def plot_backtest_result(
    result: BacktestResult,
    *,
    title: str = "Strategy vs. Benchmark Returns",
    ax: Axes | None = None,
) -> Axes:
    """Plot strategy and benchmark cumulative returns on a single axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 7))

    for column in result.cumulative_returns.columns:
        ax.plot(result.cumulative_returns.index, result.cumulative_returns[column], label=column)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.show()
    # return ax
