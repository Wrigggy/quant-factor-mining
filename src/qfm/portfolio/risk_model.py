"""Risk model utilities for covariance estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Sample covariance over optional rolling window."""
    data = returns.tail(window) if window is not None else returns
    return data.cov()


def ewma_covariance(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """EWMA covariance matrix for the latest date (asset × asset)."""
    panel = returns.ewm(span=span, min_periods=max(5, span // 3)).cov()
    if panel.empty:
        raise ValueError("Cannot compute EWMA covariance on empty returns")

    latest_date = panel.index.get_level_values(0).max()
    latest = panel.xs(latest_date, level=0)
    latest = latest.reindex(index=returns.columns, columns=returns.columns)
    return latest


def make_psd(cov: pd.DataFrame, floor: float = 1e-8) -> pd.DataFrame:
    """Project covariance matrix to PSD via eigenvalue clipping."""
    vals, vecs = np.linalg.eigh(cov.values)
    vals = np.maximum(vals, floor)
    repaired = vecs @ np.diag(vals) @ vecs.T
    return pd.DataFrame(repaired, index=cov.index, columns=cov.columns)
