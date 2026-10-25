from __future__ import annotations

import math
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from alpha_engine.data import Data
from alpha_engine.eigen import (
    EigenPortfolio,
    build_eigen_portfolio,
    compute_returns,
    covariance_eigh_method,
    pca_method,
)


def test_compute_returns_accepts_universe_dataclass() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [100.0, 0.0, 102.0, 104.0],
            "BBB": [50.0, 51.0, 52.0, 54.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
    )
    panel.index.name = "date"
    panel.columns.name = "symbol"
    universe = Data(
        data=panel,
        start_date="2026-01-01",
        end_date="2026-01-05",
        etf_symbols=("IWV", "SPY", "QQQ"),
        target_symbol_count=2,
    )

    result = compute_returns(universe)
    actual_return = cast(float, result.loc[pd.Timestamp("2026-01-03"), "AAA"])

    assert list(result.columns) == ["AAA", "BBB"]
    assert result.index[0] == pd.Timestamp("2026-01-02")
    assert round(actual_return, 10) == 0.02


def test_covariance_and_pca_methods_return_symbol_indexed_series() -> None:
    returns = _sample_returns()

    eigh_weights = covariance_eigh_method(returns, rank=0)
    pca_weights = pca_method(returns, rank=0)

    assert list(eigh_weights.index) == ["AAA", "BBB", "CCC"]
    assert list(pca_weights.index) == ["AAA", "BBB", "CCC"]


def test_build_eigen_portfolio_selects_requested_number_of_symbols() -> None:
    returns = _sample_returns()

    result = build_eigen_portfolio(returns, rank=0, top_n=2)

    assert isinstance(result, EigenPortfolio)
    assert len(result.symbols) == 2
    assert math.isclose(float(result.weights.sum()), 1.0)
    assert result.method == "covariance_eigh_method"


def test_build_eigen_portfolio_optimizes_overweight_covariance_positions() -> None:
    returns = pd.DataFrame(
        np.tile(np.linspace(0.01, 0.1, 10), (4, 1)),
        columns=[f"S{i}" for i in range(10)],
    )
    target_weights = pd.Series(
        [0.19, 0.13, 0.11, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.06],
        index=returns.columns,
        dtype=float,
    )
    optimized_weights = pd.Series(
        [0.10] * 10,
        index=returns.columns,
        dtype=float,
    )

    with (
        patch("alpha_engine.eigen.cov_eigh_method", return_value=target_weights),
        patch(
            "alpha_engine.eigen.optimize_portfolio_weights", return_value=optimized_weights
        ) as optimize,
    ):
        result = build_eigen_portfolio(returns, top_n=10)

    optimize.assert_called_once()
    assert result.weights.name == "port_wt"
    assert math.isclose(float(result.weights.max()), 0.10)
    assert math.isclose(float(result.weights.sum()), 1.0)


def test_data_object_reuses_cached_returns_across_portfolio_runs() -> None:
    panel = pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 103.0, 105.0],
            "BBB": [50.0, 49.0, 51.0, 52.0],
            "CCC": [80.0, 81.0, 83.0, 84.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
    )
    data = Data(
        data=panel,
        start_date="2026-01-01",
        end_date="2026-01-05",
        etf_symbols=("IWV", "SPY", "QQQ"),
        target_symbol_count=3,
    )

    first_returns = data.returns()
    second_returns = data.returns()
    first_portfolio = build_eigen_portfolio(data, rank=0, top_n=2)
    second_portfolio = build_eigen_portfolio(data, method=pca_method, rank=0, top_n=2)

    assert first_returns is second_returns
    assert len(first_portfolio.symbols) == 2
    assert len(second_portfolio.symbols) == 2


def test_build_eigen_portfolio_accepts_data_object() -> None:
    returns = _sample_returns()
    prices = (1 + returns).cumprod()
    data = Data(
        data=prices,
        start_date="2026-01-01",
        end_date="2026-01-06",
        etf_symbols=("IWV", "SPY", "QQQ"),
        target_symbol_count=3,
    )

    result = build_eigen_portfolio(data, rank=0, top_n=2)

    assert len(result.symbols) == 2


def test_covariance_eigh_method_selects_eigenvector_column() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.02, 0.01, 0.03, 0.02],
            "BBB": [0.01, 0.015, 0.011, 0.016],
        }
    )
    covariance = returns.cov().to_numpy()
    expected_vector = np.linalg.eigh(covariance)[1][:, -1]
    expected = pd.Series(
        expected_vector / expected_vector.sum(), index=returns.columns, dtype=float
    )

    result = covariance_eigh_method(returns, rank=0)

    pd.testing.assert_series_equal(result, expected)


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAA": [0.02, 0.01, 0.015, 0.025],
            "BBB": [0.01, 0.015, 0.02, 0.018],
            "CCC": [0.03, 0.028, 0.031, 0.029],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]),
    )
