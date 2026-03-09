"""Benchmark construction utilities."""

from __future__ import annotations

import pandas as pd


def equal_weight_benchmark(asset_returns: pd.DataFrame) -> pd.Series:
    """Equal-weight daily benchmark from available asset returns."""
    if asset_returns.empty:
        raise ValueError("asset_returns is empty")
    return asset_returns.mean(axis=1).rename("benchmark_return")
