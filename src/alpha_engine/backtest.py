from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from alpha_engine.data import ETF_SYMBOLS, BacktestData
from alpha_engine.data import load_backtest_data as _load_backtest_data
from alpha_engine.data import load_backtest_returns_data as _load_backtest_returns_data
from alpha_engine.eigen import EigenPortfolio


@dataclass(frozen=True, slots=True)
class BacktestResult:
    cumulative_returns: pd.DataFrame


def load_backtest_returns_data(
    symbols: Sequence[str],
    start_date: str | dt.date,
    end_date: str | dt.date,
    *,
    engine: Engine | None = None,
    benchmark_symbols: Sequence[str] = ETF_SYMBOLS,
) -> pd.DataFrame:
    return _load_backtest_returns_data(
        symbols,
        start_date,
        end_date,
        engine=engine,
        benchmark_symbols=benchmark_symbols,
    )


def load_backtest_data(
    symbols: Sequence[str],
    start_date: str | dt.date,
    end_date: str | dt.date,
    *,
    engine: Engine | None = None,
    benchmark_symbols: Sequence[str] = ETF_SYMBOLS,
) -> BacktestData:
    return _load_backtest_data(
        symbols,
        start_date,
        end_date,
        engine=engine,
        benchmark_symbols=benchmark_symbols,
    )


def run_backtest(
    portfolio: EigenPortfolio,
    backtest: BacktestData | pd.DataFrame,
    *,
    benchmark_symbols: Sequence[str] | None = None,
) -> BacktestResult:
    """Run a simple cumulative-return backtest against ETF benchmarks."""
    if isinstance(backtest, BacktestData):
        test_returns = backtest.returns
        active_benchmarks = tuple(benchmark_symbols or backtest.benchmark_symbols)
    else:
        test_returns = backtest
        active_benchmarks = tuple(benchmark_symbols or ETF_SYMBOLS)

    required_columns = [*active_benchmarks, *portfolio.symbols]
    selected_returns = test_returns.reindex(columns=required_columns)
    benchmark_count = len(active_benchmarks)

    strategy_returns = np.dot(
        selected_returns.iloc[:, benchmark_count:].to_numpy(),
        np.array(portfolio.weights.to_numpy()),
    )
    cumulative_returns = pd.DataFrame(index=selected_returns.index)
    cumulative_returns["strategy"] = np.cumprod(1 + strategy_returns).flatten()

    for symbol in active_benchmarks:
        cumulative_returns[symbol] = (1 + selected_returns[symbol]).cumprod()

    cumulative_returns.index.name = selected_returns.index.name
    return BacktestResult(cumulative_returns=cumulative_returns)
