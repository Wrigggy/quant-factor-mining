"""Simple mean-variance optimizer for reproducible research demos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import cap_position_size, enforce_long_only
from .risk_model import make_psd


def mean_variance_optimize(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 5.0,
    max_weight: float = 0.2,
) -> pd.Series:
    """Solve long-only mean-variance with SLSQP and explicit constraints."""
    assets = expected_returns.index.intersection(covariance.index)
    if len(assets) == 0:
        raise ValueError("No common assets between expected returns and covariance")

    mu = expected_returns.loc[assets].values
    cov = make_psd(covariance.loc[assets, assets]).values
    n = len(assets)

    x0 = np.repeat(1.0 / n, n)

    def objective(w: np.ndarray) -> float:
        ret = float(mu @ w)
        risk = float(w @ cov @ w)
        return -(ret - risk_aversion * risk)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(n)]

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Optimizer failed: {result.message}")

    weights = pd.Series(result.x, index=assets)
    weights = enforce_long_only(weights)
    weights = cap_position_size(weights, max_weight=max_weight)
    return weights
