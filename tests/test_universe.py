from __future__ import annotations

import pandas as pd
import pytest

from alpha_engine.data import Data
from alpha_engine.universe import build_strategy_adjusted_close_panel, build_strategy_universe


def test_build_strategy_adjusted_close_panel_excludes_etfs_and_filters_universe() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "symbol": "IWV",
                "close": 300.0,
                "adjusted_close": 300.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "SPY",
                "close": 600.0,
                "adjusted_close": 600.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "QQQ",
                "close": 500.0,
                "adjusted_close": 500.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "IWV",
                "close": 301.0,
                "adjusted_close": 301.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "SPY",
                "close": 601.0,
                "adjusted_close": 601.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "QQQ",
                "close": 501.0,
                "adjusted_close": 501.0,
                "volume": 1_000,
            },
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
                "volume": 1_100,
            },
            {
                "date": "2026-01-01",
                "symbol": "BBB",
                "close": 20.0,
                "adjusted_close": 20.0,
                "volume": 2_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "BBB",
                "close": 21.0,
                "adjusted_close": 21.0,
                "volume": 2_200,
            },
            {
                "date": "2026-01-01",
                "symbol": "CCC",
                "close": 30.0,
                "adjusted_close": 30.0,
                "volume": 3_000,
            },
        ]
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    result = build_strategy_adjusted_close_panel(raw_df, target_symbol_count=1)

    assert "IWV" not in result.columns
    assert "SPY" not in result.columns
    assert "QQQ" not in result.columns
    assert "CCC" not in result.columns
    assert list(result.columns) == ["BBB"]
    assert result.index.name == "date"
    assert result.columns.name == "symbol"
    assert result.loc[pd.Timestamp("2026-01-02"), "BBB"] == 21.0


def test_build_strategy_adjusted_close_panel_keeps_quantile_ties_above_threshold() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "symbol": "IWV",
                "close": 300.0,
                "adjusted_close": 300.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "IWV",
                "close": 301.0,
                "adjusted_close": 301.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "SPY",
                "close": 600.0,
                "adjusted_close": 600.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "SPY",
                "close": 601.0,
                "adjusted_close": 601.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "QQQ",
                "close": 500.0,
                "adjusted_close": 500.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "QQQ",
                "close": 501.0,
                "adjusted_close": 501.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "AAA",
                "close": 100.0,
                "adjusted_close": 100.0,
                "volume": 10,
            },
            {
                "date": "2026-01-02",
                "symbol": "AAA",
                "close": 100.0,
                "adjusted_close": 100.0,
                "volume": 10,
            },
            {
                "date": "2026-01-01",
                "symbol": "BBB",
                "close": 90.0,
                "adjusted_close": 90.0,
                "volume": 10,
            },
            {
                "date": "2026-01-02",
                "symbol": "BBB",
                "close": 90.0,
                "adjusted_close": 90.0,
                "volume": 10,
            },
            {
                "date": "2026-01-01",
                "symbol": "CCC",
                "close": 90.0,
                "adjusted_close": 90.0,
                "volume": 10,
            },
            {
                "date": "2026-01-02",
                "symbol": "CCC",
                "close": 90.0,
                "adjusted_close": 90.0,
                "volume": 10,
            },
            {
                "date": "2026-01-01",
                "symbol": "DDD",
                "close": 70.0,
                "adjusted_close": 70.0,
                "volume": 10,
            },
            {
                "date": "2026-01-02",
                "symbol": "DDD",
                "close": 70.0,
                "adjusted_close": 70.0,
                "volume": 10,
            },
        ]
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    with pytest.raises(AssertionError, match="Expected 2 symbols, but found 3"):
        build_strategy_adjusted_close_panel(raw_df, target_symbol_count=2)


def test_build_strategy_adjusted_close_panel_returns_filled_selected_strategy_columns() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "symbol": "IWV",
                "close": 300.0,
                "adjusted_close": 300.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "IWV",
                "close": 301.0,
                "adjusted_close": 301.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-03",
                "symbol": "IWV",
                "close": 302.0,
                "adjusted_close": 302.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "SPY",
                "close": 600.0,
                "adjusted_close": 600.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "SPY",
                "close": 601.0,
                "adjusted_close": 601.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-03",
                "symbol": "SPY",
                "close": 602.0,
                "adjusted_close": 602.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "QQQ",
                "close": 500.0,
                "adjusted_close": 500.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "QQQ",
                "close": 501.0,
                "adjusted_close": 501.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-03",
                "symbol": "QQQ",
                "close": 502.0,
                "adjusted_close": 502.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "AAA",
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 2_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "AAA",
                "close": 11.0,
                "adjusted_close": None,
                "volume": 2_000,
            },
            {
                "date": "2026-01-03",
                "symbol": "AAA",
                "close": 12.0,
                "adjusted_close": 12.0,
                "volume": 2_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "BBB",
                "close": 9.0,
                "adjusted_close": 9.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "BBB",
                "close": 9.5,
                "adjusted_close": 9.5,
                "volume": 1_000,
            },
            {
                "date": "2026-01-03",
                "symbol": "BBB",
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 1_000,
            },
        ]
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    result = build_strategy_adjusted_close_panel(raw_df, target_symbol_count=1)

    assert not result.loc[:, ["AAA"]].isna().any().any()
    assert result.loc[pd.Timestamp("2026-01-02"), "AAA"] == 12.0


def test_build_strategy_universe_returns_dataclass_with_metadata() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "symbol": "IWV",
                "close": 300.0,
                "adjusted_close": 300.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "IWV",
                "close": 301.0,
                "adjusted_close": 301.0,
                "volume": 1_000,
            },
            {
                "date": "2026-01-01",
                "symbol": "AAA",
                "close": 10.0,
                "adjusted_close": 10.0,
                "volume": 2_000,
            },
            {
                "date": "2026-01-02",
                "symbol": "AAA",
                "close": 11.0,
                "adjusted_close": 11.0,
                "volume": 2_100,
            },
        ]
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    result = build_strategy_universe(
        raw_df,
        start_date="2026-01-01",
        end_date="2026-01-03",
        target_symbol_count=1,
    )

    assert isinstance(result, Data)
    assert result.start_date == "2026-01-01"
    assert result.end_date == "2026-01-03"
    assert result.target_symbol_count == 1
    assert result.symbols == ("AAA",)
    assert list(result.data.columns) == ["AAA"]
