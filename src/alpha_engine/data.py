from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from util.db.connection import get_engine  # pyright: ignore[reportMissingTypeStubs]

ETF_SYMBOLS: tuple[str, ...] = ("IWV", "SPY", "QQQ")


@dataclass(frozen=True, slots=True)
class BacktestData:
    returns: pd.DataFrame
    benchmark_symbols: tuple[str, ...]


@dataclass(slots=True)
class Data:
    data: pd.DataFrame
    start_date: str | dt.date | None
    end_date: str | dt.date | None
    etf_symbols: tuple[str, ...]
    target_symbol_count: int
    engine: Engine = field(default_factory=get_engine, repr=False)
    _returns_cache: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _backtest_cache: dict[tuple[str | dt.date, str | dt.date, tuple[str, ...]], BacktestData] = (
        field(
            default_factory=dict,
            init=False,
            repr=False,
        )
    )

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(symbol) for symbol in self.data.columns)

    @classmethod
    def load_strategy(
        cls,
        start_date: str | dt.date,
        end_date: str | dt.date,
        engine: Engine | None = None,
        etf_symbols: Sequence[str] = ETF_SYMBOLS,
        target_symbol_count: int = 2000,
    ) -> Data:
        active_engine = engine or get_engine()

        query = text(
            """
            SELECT symbol, exchange, date, [close], adjusted_close, volume
            FROM dbo.us_equities
            WHERE date >= :start_date
                AND date <  :end_date;
            """
        )

        with active_engine.connect() as conn:
            raw_df = pd.read_sql(
                query,
                conn,
                parse_dates=["date"],
                params={"start_date": start_date, "end_date": end_date},
            )

        return cls.build_strategy(
            raw_df,
            target_symbol_count=target_symbol_count,
            etf_symbols=etf_symbols,
            start_date=start_date,
            end_date=end_date,
            engine=active_engine,
        )

    @classmethod
    def build_strategy(
        cls,
        raw_df: pd.DataFrame,
        target_symbol_count: int,
        etf_symbols: Sequence[str] = ETF_SYMBOLS,
        start_date: str | dt.date | None = None,
        end_date: str | dt.date | None = None,
        engine: Engine | None = None,
    ) -> Data:
        active_engine = engine or get_engine()
        df = raw_df.copy()
        etf_symbols_list = list(etf_symbols)

        wide_df = _pivot_adjusted_close(df)
        strategy_mask = ~wide_df.columns.isin(etf_symbols_list)
        strategy_wide = wide_df.loc[:, strategy_mask]

        ## Data Cleaning with details discovered during initial exploration phase
        # The training set must contain a full year of data to estimate the model
        # Exclude symbols with an adjusted close value=999999.9999
        strategy_wide = strategy_wide.replace(999999.9999, np.nan)
        filled_strategy_wide = strategy_wide.bfill(limit=1).ffill(limit_area="inside")
        counts = filled_strategy_wide.count()
        complete_symbols = counts.index[counts >= counts.mode().iloc[0]]

        # Since market capitalization is unavailable, dollar volume is used as an approximation
        strategy_df = df.loc[~df["symbol"].isin(etf_symbols_list)].copy()
        strategy_df["dollar_volume"] = strategy_df["close"] * strategy_df["volume"]
        strategy_df["return"] = strategy_df.groupby("symbol")["adjusted_close"].pct_change()

        avg_day_volume = strategy_df.groupby("symbol")["dollar_volume"].mean()
        symbol_std = strategy_df.groupby("symbol")["return"].std()

        # Exclude symbols with return volatility at or above 1.
        std_threshold_list = symbol_std.loc[symbol_std >= 1].index.tolist()

        q90_volume = strategy_df.groupby("symbol")["dollar_volume"].quantile(q=0.90)
        # Exclude symbols whose 90th percentile dollar volume is below their average.
        q90_threshold_list = q90_volume[q90_volume < avg_day_volume].index.tolist()

        excluded_symbols = set(std_threshold_list).union(q90_threshold_list)
        eligible_symbols = complete_symbols[~complete_symbols.isin(excluded_symbols)]
        avg_day_volume = avg_day_volume.loc[eligible_symbols]

        # Reducing the training set to a targeted number
        selected_count = min(target_symbol_count, len(avg_day_volume))
        q = 1 - selected_count / len(avg_day_volume)
        threshold = avg_day_volume.quantile(q)
        strategy_symbols = avg_day_volume.loc[avg_day_volume >= threshold].index.tolist()

        assert target_symbol_count == len(strategy_symbols), (
            f"Expected {target_symbol_count} symbols, but found {len(strategy_symbols)}"
        )

        data = filled_strategy_wide.reindex(columns=strategy_symbols)
        return cls(
            data=data,
            start_date=start_date,
            end_date=end_date,
            etf_symbols=tuple(etf_symbols),
            target_symbol_count=target_symbol_count,
            engine=active_engine,
        )

    def returns(self, *, refresh: bool = False) -> pd.DataFrame:
        if refresh or self._returns_cache is None:
            cleaned_df = self.data.replace(0, np.nan).ffill(limit_area="inside")
            returns = cleaned_df.pct_change().iloc[1:, :]
            returns.index.name = self.data.index.name
            returns.columns.name = self.data.columns.name
            self._returns_cache = returns
        return self._returns_cache

    def load_backtest(
        self,
        start_date: str | dt.date,
        end_date: str | dt.date,
        benchmark_symbols: Sequence[str] = (),
        refresh: bool = False,
    ) -> BacktestData:
        benchmarks = tuple(benchmark_symbols or ETF_SYMBOLS)
        cache_key = (start_date, end_date, benchmarks)
        if refresh or cache_key not in self._backtest_cache:
            self._backtest_cache[cache_key] = load_backtest_data(
                self.symbols,
                start_date,
                end_date,
                engine=self.engine,
                benchmark_symbols=benchmarks,
            )
        return self._backtest_cache[cache_key]

    # def load_backtest_returns(
    #     self,
    #     start_date: str | dt.date,
    #     end_date: str | dt.date,
    #     benchmark_symbols: Sequence[str] = (),
    #     refresh: bool = False,
    # ) -> pd.DataFrame:
    #     return self.load_backtest(
    #         start_date,
    #         end_date,
    #         benchmark_symbols=benchmark_symbols,
    #         refresh=refresh,
    #     ).returns


# def load_strategy_adjusted_close_data(
#     start_date: str | dt.date,
#     end_date: str | dt.date,
#     etf_symbols: Sequence[str] = ETF_SYMBOLS,
#     target_symbol_count: int = 2000,
# ) -> pd.DataFrame:
#     """Load us_equities rows and build the final adjusted-close data."""
#     return Data.load_strategy(
#         start_date,
#         end_date,
#         etf_symbols=etf_symbols,
#         target_symbol_count=target_symbol_count,
#     ).data


# def load_strategy_universe(
#     start_date: str | dt.date,
#     end_date: str | dt.date,
#     engine: Engine | None = None,
#     etf_symbols: Sequence[str] = ETF_SYMBOLS,
#     target_symbol_count: int = 2000,
# ) -> Data:
#     """Load us_equities rows and build a strategy universe."""
#     return Data.load_strategy(
#         start_date,
#         end_date,
#         engine=engine,
#         etf_symbols=etf_symbols,
#         target_symbol_count=target_symbol_count,
#     )


# def build_strategy_adjusted_close_data(
#     raw_df: pd.DataFrame,
#     target_symbol_count: int,
#     etf_symbols: Sequence[str] = ETF_SYMBOLS,
# ) -> pd.DataFrame:
#     """Return a date-by-symbol adjusted-close data excluding configured ETFs."""
#     return Data.build_strategy(
#         raw_df,
#         target_symbol_count=target_symbol_count,
#         etf_symbols=etf_symbols,
#     ).data


# def build_strategy_universe(
#     raw_df: pd.DataFrame,
#     target_symbol_count: int,
#     etf_symbols: Sequence[str] = ETF_SYMBOLS,
#     *,
#     start_date: str | dt.date | None = None,
#     end_date: str | dt.date | None = None,
#     engine: Engine | None = None,
# ) -> Data:
#     """Return a filtered strategy universe with metadata and adjusted-close data."""
#     return Data.build_strategy(
#         raw_df,
#         target_symbol_count=target_symbol_count,
#         etf_symbols=etf_symbols,
#         start_date=start_date,
#         end_date=end_date,
#         engine=engine,
#     )


def load_backtest_returns_data(
    symbols: Sequence[str],
    start_date: str | dt.date,
    end_date: str | dt.date,
    engine: Engine | None = None,
    benchmark_symbols: Sequence[str] = ETF_SYMBOLS,
) -> pd.DataFrame:
    """Load benchmark and portfolio prices, then return a daily returns data."""
    active_engine = engine or get_engine()
    query = text(
        """
        SELECT symbol, date, adjusted_close
        FROM dbo.us_equities
        WHERE date >= :start_date
            AND date <  :end_date
        """
    )
    params = cast(
        dict[str, Any],
        {
            "start_date": start_date,
            "end_date": end_date,
        },
    )

    with active_engine.connect() as conn:
        raw_df = pd.read_sql(
            query,
            conn,
            parse_dates=["date"],
            params=cast(Any, params),
        )

    ordered_columns = list(benchmark_symbols) + list(symbols)
    raw_df = raw_df.loc[raw_df["symbol"].isin(ordered_columns)]
    wide_df = raw_df.pivot(index="date", columns="symbol", values="adjusted_close")

    returns = wide_df.reindex(columns=ordered_columns).ffill().pct_change().iloc[1:, :].fillna(0)
    returns.index.name = "date"
    returns.columns.name = "symbol"

    assert not returns.isna().any(axis=None), "Missing value(s) in the testing set!"

    return returns


def load_backtest_data(
    symbols: Sequence[str],
    start_date: str | dt.date,
    end_date: str | dt.date,
    *,
    engine: Engine | None = None,
    benchmark_symbols: Sequence[str] = ETF_SYMBOLS,
) -> BacktestData:
    returns = load_backtest_returns_data(
        symbols,
        start_date,
        end_date,
        engine=engine,
        benchmark_symbols=benchmark_symbols,
    )
    return BacktestData(
        returns=returns,
        benchmark_symbols=tuple(benchmark_symbols),
    )


def _pivot_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=pd.Index([], name="date"))

    wide_df = df.pivot(index="date", columns="symbol", values="adjusted_close")
    wide_df.columns.name = "symbol"
    wide_df.index.name = "date"
    return wide_df.sort_index()
