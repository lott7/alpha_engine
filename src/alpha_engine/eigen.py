from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  # pyright: ignore[reportMissingTypeStubs]
from sklearn.preprocessing import StandardScaler

from alpha_engine.data import Data


class EigenMethod(Protocol):
    def __call__(self, returns: pd.DataFrame, rank: int) -> pd.Series: ...


@dataclass(frozen=True, slots=True)
class EigenPortfolio:
    symbols: tuple[str, ...]
    weights: pd.Series
    method: str
    rank: int


MAX_POSITION_WEIGHT = 0.10


def cov_eigh_method(returns: pd.DataFrame, rank: int = 0) -> pd.Series:
    """Return correlation-eigenvector loadings for the requested rank."""
    _validate_returns(returns)
    standardized = (returns - returns.mean()) / returns.std()
    covariance = standardized.cov()
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance.to_numpy()
    )  # Eigenvalues (L), Eigenvectors (Q)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    eigenportfolio = eigenvectors[:, rank] / np.sum(eigenvectors[:, rank])
    return pd.Series(eigenportfolio, index=returns.columns, dtype=float)


def pca_method(returns: pd.DataFrame, rank: int = 0) -> pd.Series:
    """Return PCA component loadings for the requested rank."""
    _validate_returns(returns)
    X = StandardScaler().fit_transform(returns)

    pca = PCA()
    pca.fit(X)

    cumulative = pca.explained_variance_ratio_.cumsum()

    plt.plot(range(1, len(cumulative) + 1), cumulative)
    plt.axhline(0.70, color="red", linestyle="--", label="70%")
    plt.axhline(0.90, color="orange", linestyle="--", label="90%")
    plt.xlabel("Number of PCs")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
    plt.show()

    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.show()

    for i, x in enumerate(cumulative, 1):
        if x >= 0.75:
            print(f"{i} PCs explain {x:.2%} of variance")
            break

    for n_components, variance in enumerate(pca.explained_variance_ratio_, 1):
        if variance < 0.01:
            print(f"PC {n_components} explains only {variance:.2%} of variance")
            break

    loadings = pca.components_[:n_components]

    # Importance of each stock across the selected PCs
    stock_score = np.sqrt(
        np.sum(loadings**2, axis=0)  # !Drop negative weights!
    )

    return pd.Series(stock_score, index=returns.columns, dtype=float)


def compute_returns(data: Data | pd.DataFrame) -> pd.DataFrame:
    """Return daily returns from either a strategy Data object or a prices panel."""
    if isinstance(data, Data):
        return data.returns()

    _validate_returns_input(data)
    cleaned_df = data.replace(0, np.nan).ffill(limit_area="inside")
    returns = cleaned_df.pct_change().iloc[1:, :]
    returns.index.name = data.index.name
    returns.columns.name = data.columns.name
    return returns


def build_portfolio(
    returns: Data | pd.DataFrame,
    method: EigenMethod = cov_eigh_method,
    rank: int = 0,
    top_n: int = 20,
) -> EigenPortfolio:
    """Build a normalized eigen-portfolio from a returns panel."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0")

    if isinstance(returns, Data):
        returns_df = returns.returns()
    else:
        returns_df = returns

    raw_weights = method(returns_df, rank=rank).sort_values(ascending=False)

    selected_weights = raw_weights.iloc[:top_n].copy()
    weight_sum = float(selected_weights.sum())
    if np.isclose(weight_sum, 0.0):
        raise ValueError("selected eigen-portfolio weights sum to zero")

    normalized_weights = optimize_portfolio_weights(
        selected_weights,
        max_weight=MAX_POSITION_WEIGHT,
    )

    normalized_weights.name = "port_wt"
    return EigenPortfolio(
        symbols=tuple(str(symbol) for symbol in normalized_weights.index),
        weights=normalized_weights,
        method=_method_name(method),
        rank=rank,
    )


def _method_name(method: EigenMethod) -> str:
    return getattr(method, "__name__", method.__class__.__name__)


def build_eigen_portfolio(
    returns: Data | pd.DataFrame,
    method: EigenMethod | None = None,
    rank: int = 0,
    top_n: int = 20,
) -> EigenPortfolio:
    """Backward-compatible alias for build_portfolio."""
    active_method = covariance_eigh_method if method is None else method
    return build_portfolio(returns, method=active_method, rank=rank, top_n=top_n)


def covariance_eigh_method(returns: pd.DataFrame, rank: int = 0) -> pd.Series:
    """Backward-compatible alias for cov_eigh_method."""
    return cov_eigh_method(returns, rank=rank)


def optimize_portfolio_weights(
    target_weights: pd.Series,
    max_weight: float = MAX_POSITION_WEIGHT,
) -> pd.Series:
    """Project weights onto long-only capped weights while staying close to the target."""
    if target_weights.empty:
        raise ValueError("target_weights must not be empty")
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be between 0 and 1")
    if len(target_weights) * max_weight < 1:
        raise ValueError("max_weight is infeasible for the number of selected positions")

    n = len(target_weights)
    weights = cp.Variable(n)

    target_values = target_weights.to_numpy(dtype=float)
    equal_weight = 1.0 / n

    # Fine tuning for equal weighting
    lambda_reg = 5.0

    objective = cp.Minimize(
        cp.sum_squares(weights - target_values)
        + lambda_reg * cp.sum_squares(weights - equal_weight)
    )

    constraints = [
        cp.sum(weights) == 1,
        weights <= max_weight,
        weights >= 0,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError(f"weight optimization failed with status: {problem.status}")

    optimized_weights = np.asarray(weights.value, dtype=float).reshape(-1)
    return pd.Series(optimized_weights, index=target_weights.index, dtype=float)


def _validate_returns(returns: pd.DataFrame) -> None:
    if returns.empty:
        raise ValueError("returns must not be empty")
    if returns.shape[1] == 0:
        raise ValueError("returns must contain at least one symbol column")
    if returns.isna().any().any():
        raise ValueError("returns must not contain NaN values")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns must contain only finite values")


def _validate_returns_input(data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError("data must not be empty")
    if data.shape[1] == 0:
        raise ValueError("data must contain at least one symbol column")


def _should_optimize_covariance_weights(
    method: EigenMethod,
    normalized_weights: pd.Series,
    *,
    max_weight: float = MAX_POSITION_WEIGHT,
) -> bool:
    method_name = _method_name(method)
    if method_name not in {"cov_eigh_method", "covariance_eigh_method"}:
        return False
    if len(normalized_weights) * max_weight < 1:
        return False
    return bool((normalized_weights > max_weight).any())
