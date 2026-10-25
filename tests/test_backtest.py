from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from alpha_engine.backtest import (
    BacktestData,
    BacktestResult,
    load_backtest_data,
    load_backtest_returns_panel,
    run_backtest,
)
from alpha_engine.data import Data
from alpha_engine.eigen import EigenPortfolio


def test_run_backtest_returns_strategy_and_benchmark_cumulative_returns() -> None:
    portfolio = EigenPortfolio(
        symbols=("AAA", "BBB"),
        weights=pd.Series([0.6, 0.4], index=["AAA", "BBB"], name="port_wt"),
        method="covariance_eigh_method",
        rank=-1,
    )
    returns = pd.DataFrame(
        {
            "IWV": [0.01, 0.02],
            "SPY": [0.015, 0.01],
            "QQQ": [0.02, 0.005],
            "AAA": [0.03, 0.01],
            "BBB": [0.00, 0.02],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )

    result = run_backtest(portfolio, returns)

    assert isinstance(result, BacktestResult)
    assert list(result.cumulative_returns.columns) == ["strategy", "IWV", "SPY", "QQQ"]
    assert result.cumulative_returns.loc[pd.Timestamp("2026-01-02"), "strategy"] == 1.018
    assert result.cumulative_returns.loc[pd.Timestamp("2026-01-03"), "IWV"] == 1.0302


def test_load_backtest_returns_panel_queries_selected_symbols_and_benchmarks() -> None:
    raw_df = pd.DataFrame(
        [
            {"date": "2026-01-01", "symbol": "IWV", "adjusted_close": 100.0},
            {"date": "2026-01-02", "symbol": "IWV", "adjusted_close": 101.0},
            {"date": "2026-01-01", "symbol": "AAA", "adjusted_close": 10.0},
            {"date": "2026-01-02", "symbol": "AAA", "adjusted_close": 10.5},
            {"date": "2026-01-01", "symbol": "BBB", "adjusted_close": 20.0},
            {"date": "2026-01-02", "symbol": "BBB", "adjusted_close": 21.0},
        ]
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    connection = MagicMock()
    transaction = MagicMock()
    transaction.__enter__.return_value = connection
    transaction.__exit__.return_value = None

    engine = MagicMock()
    engine.connect.return_value = transaction

    with patch("alpha_engine.data.pd.read_sql", return_value=raw_df) as read_sql:
        result = load_backtest_returns_panel(
            ["AAA", "BBB"], "2026-01-01", "2026-01-03", engine=engine
        )

    read_sql.assert_called_once()
    assert list(result.columns) == ["IWV", "AAA", "BBB"]
    assert result.index[0] == pd.Timestamp("2026-01-02")


def test_data_object_reuses_cached_backtest_returns() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [10.0, 10.5, 10.6],
            "BBB": [20.0, 21.0, 21.5],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )
    data = Data(
        data=panel,
        start_date="2026-01-01",
        end_date="2026-01-04",
        etf_symbols=("IWV", "SPY", "QQQ"),
        target_symbol_count=2,
    )
    cached_returns = pd.DataFrame(
        {
            "IWV": [0.01, 0.02],
            "SPY": [0.015, 0.01],
            "QQQ": [0.02, 0.005],
            "AAA": [0.03, 0.01],
            "BBB": [0.00, 0.02],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    portfolio = EigenPortfolio(
        symbols=("AAA", "BBB"),
        weights=pd.Series([0.6, 0.4], index=["AAA", "BBB"], name="port_wt"),
        method="covariance_eigh_method",
        rank=-1,
    )
    backtest_data = BacktestData(
        returns=cached_returns,
        benchmark_symbols=("IWV", "SPY", "QQQ"),
    )

    with patch(
        "alpha_engine.data.load_backtest_data",
        return_value=backtest_data,
    ) as load_backtest:
        cached_backtest = data.load_backtest("2026-01-01", "2026-01-04")
        first_result = run_backtest(portfolio, cached_backtest)
        second_result = run_backtest(portfolio, cached_backtest)

    load_backtest.assert_called_once()
    assert isinstance(first_result, BacktestResult)
    assert first_result.cumulative_returns.equals(second_result.cumulative_returns)


def test_load_backtest_uses_attached_engine_when_no_override() -> None:
    engine = MagicMock()
    data = Data(
        data=pd.DataFrame(
            {"AAA": [10.0, 10.5]}, index=pd.to_datetime(["2026-01-01", "2026-01-02"])
        ),
        start_date="2026-01-01",
        end_date="2026-01-03",
        etf_symbols=("IWV", "SPY", "QQQ"),
        target_symbol_count=1,
        engine=engine,
    )
    backtest_data = BacktestData(
        returns=pd.DataFrame({"IWV": [0.01], "AAA": [0.02]}, index=pd.to_datetime(["2026-01-02"])),
        benchmark_symbols=("IWV",),
    )

    with patch("alpha_engine.data.load_backtest_data", return_value=backtest_data) as load_backtest:
        result = data.load_backtest("2026-01-01", "2026-01-03", benchmark_symbols=("IWV",))

    load_backtest.assert_called_once_with(
        ("AAA",),
        "2026-01-01",
        "2026-01-03",
        engine=engine,
        benchmark_symbols=("IWV",),
    )
    assert result is backtest_data


def test_load_backtest_data_wraps_returns_and_benchmarks() -> None:
    returns = pd.DataFrame(
        {
            "IWV": [0.01, 0.02],
            "AAA": [0.03, 0.01],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )

    with patch(
        "alpha_engine.data.load_backtest_returns_panel",
        return_value=returns,
    ) as load_returns:
        result = load_backtest_data(
            ["AAA"],
            "2026-01-01",
            "2026-01-04",
            benchmark_symbols=("IWV",),
        )

    load_returns.assert_called_once()
    assert isinstance(result, BacktestData)
    assert result.returns.equals(returns)
    assert result.benchmark_symbols == ("IWV",)


def test_data_initializes_engine_by_default() -> None:
    engine = MagicMock()
    connection = MagicMock()
    transaction = MagicMock()
    transaction.__enter__.return_value = connection
    transaction.__exit__.return_value = None
    engine.connect.return_value = transaction
    raw_df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "symbol": "AAA",
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "AAA",
                "close": 11.0,
                "adjusted_close": 11.0,
                "volume": 1_000,
            },
        ]
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    with (
        patch("alpha_engine.data.get_engine", return_value=engine),
        patch(
            "alpha_engine.data.pd.read_sql",
            return_value=raw_df,
        ),
    ):
        result = Data.load_strategy("2026-01-01", "2026-01-03", target_symbol_count=1)

    assert result.engine is engine


def test_build_strategy_excludes_high_volatility_and_q90_volume_symbols() -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    rows: list[dict[str, object]] = []

    for date in dates:
        rows.append(
            {
                "date": date,
                "symbol": "KEEP",
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 120,
            }
        )

    for idx, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "symbol": "SPIKE",
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 1_000 if idx == len(dates) - 1 else 100,
            }
        )

    for idx, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "symbol": "VOL",
                "close": 4.0 if idx % 2 else 1.0,
                "adjusted_close": 4.0 if idx % 2 else 1.0,
                "volume": 1_000,
            }
        )

    raw_df = pd.DataFrame(rows)

    result = Data.build_strategy(raw_df, target_symbol_count=1, etf_symbols=())

    assert result.symbols == ("KEEP",)
    assert list(result.data.columns) == ["KEEP"]
