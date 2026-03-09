"""Data quality checks used by scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .contracts import ensure_market_data


@dataclass(frozen=True)
class QualityCheckResult:
    passed: bool
    message: str


def check_min_history(data: pd.DataFrame, min_days: int) -> QualityCheckResult:
    """Ensure the dataset spans enough trading days for research windows."""
    market = ensure_market_data(data)
    n_days = market.index.get_level_values("date").nunique()
    if n_days < min_days:
        return QualityCheckResult(False, f"Insufficient history: {n_days} < {min_days}")
    return QualityCheckResult(True, f"History OK: {n_days} trading days")


def check_min_universe_size(data: pd.DataFrame, min_tickers: int) -> QualityCheckResult:
    """Ensure cross-sectional signal tests have enough names."""
    market = ensure_market_data(data)
    n_tickers = market.index.get_level_values("ticker").nunique()
    if n_tickers < min_tickers:
        return QualityCheckResult(False, f"Insufficient universe size: {n_tickers} < {min_tickers}")
    return QualityCheckResult(True, f"Universe OK: {n_tickers} tickers")
