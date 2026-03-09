"""Tabular reporting helpers."""

from __future__ import annotations

import pandas as pd


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """Format metrics dict into a one-column DataFrame."""
    return pd.DataFrame({"value": metrics}).rename_axis("metric")


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics into compact summary statistics."""
    cols = [c for c in ["total_return", "annual_return", "sharpe", "max_drawdown"] if c in fold_metrics.columns]
    if not cols:
        return pd.DataFrame()
    return fold_metrics[cols].agg(["mean", "std", "min", "max"])
