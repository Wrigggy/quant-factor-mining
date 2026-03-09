"""Preprocessing helpers that preserve index contracts and avoid silent leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .contracts import ensure_market_data


@dataclass(frozen=True)
class PreprocessReport:
    initial_rows: int
    final_rows: int
    initial_nulls: int
    final_nulls: int
    dropped_rows: int


def preprocess_market_data(data: pd.DataFrame, max_ffill_days: int = 3) -> Tuple[pd.DataFrame, PreprocessReport]:
    """Clean market data while preserving canonical MultiIndex shape.

    Steps:
    1. Validate and normalize contract.
    2. Remove impossible OHLC rows (high < low).
    3. Forward-fill by ticker with a conservative cap.
    4. Keep deterministic sorted index.
    """
    normalized = ensure_market_data(data)

    initial_rows = len(normalized)
    initial_nulls = int(normalized.isna().sum().sum())

    cleaned = normalized.copy()

    invalid_mask = cleaned["high"] < cleaned["low"]
    if invalid_mask.any():
        cleaned = cleaned.loc[~invalid_mask]

    cleaned = cleaned.groupby(level="ticker", group_keys=False).ffill(limit=max_ffill_days)
    cleaned = cleaned.sort_index()

    final_rows = len(cleaned)
    final_nulls = int(cleaned.isna().sum().sum())

    report = PreprocessReport(
        initial_rows=initial_rows,
        final_rows=final_rows,
        initial_nulls=initial_nulls,
        final_nulls=final_nulls,
        dropped_rows=initial_rows - final_rows,
    )

    return cleaned, report


def close_prices_wide(data: pd.DataFrame) -> pd.DataFrame:
    """Pivot close prices into date × ticker frame."""
    market = ensure_market_data(data)
    close = market["close"].unstack("ticker").sort_index()
    close.columns = close.columns.astype(str)
    return close


def daily_returns_from_close(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute close-to-close daily returns with deterministic shape."""
    returns = close_wide.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns


def summarize_market_data(data: pd.DataFrame) -> Dict[str, object]:
    """Compact data summary used in run artifacts."""
    market = ensure_market_data(data)
    dates = market.index.get_level_values("date")
    tickers = market.index.get_level_values("ticker")
    return {
        "rows": int(len(market)),
        "n_tickers": int(tickers.nunique()),
        "start_date": str(dates.min().date()),
        "end_date": str(dates.max().date()),
        "null_ratio": float(market.isna().mean().mean()),
    }
