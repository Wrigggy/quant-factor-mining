"""Benchmark construction and resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency path
    yf = None


def equal_weight_benchmark(asset_returns: pd.DataFrame) -> pd.Series:
    """Equal-weight daily benchmark from available asset returns."""
    if asset_returns.empty:
        raise ValueError("asset_returns is empty")
    return asset_returns.mean(axis=1).rename("benchmark_return")


@dataclass(frozen=True)
class BenchmarkResolution:
    """Resolved benchmark series plus provenance metadata."""

    returns: pd.Series
    source: str
    requested_mode: str
    requested_ticker: str | None
    fallback_reason: str | None
    coverage_ratio: float
    non_null_days: int
    total_days: int


def _normalize_return_series(series: pd.Series) -> pd.Series:
    out = pd.Series(series).copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.name = "benchmark_return"
    return out


def _extract_ticker_returns(market_data: pd.DataFrame, ticker: str) -> pd.Series | None:
    if market_data.empty:
        return None
    if "close" not in market_data.columns:
        return None
    close = market_data["close"].unstack("ticker").sort_index()
    if ticker not in close.columns:
        return None
    return close[ticker].pct_change(fill_method=None).rename("benchmark_return")


def _fetch_ticker_returns_yfinance(ticker: str, start: str, end: str) -> tuple[pd.Series | None, str | None]:
    """Fetch benchmark returns from yfinance and return (series, error)."""
    if yf is None:
        return None, "yfinance_not_available"

    try:
        history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    except Exception as exc:  # pragma: no cover - network/runtime variability
        return None, f"fetch_exception:{exc.__class__.__name__}"

    if history.empty:
        return None, "empty_history"
    if "Close" not in history.columns:
        return None, "close_column_missing"

    returns = history["Close"].pct_change(fill_method=None).rename("benchmark_return")
    returns.index = pd.to_datetime(returns.index).tz_localize(None)
    return returns, None


def resolve_benchmark_returns(
    asset_returns: pd.DataFrame,
    market_data: pd.DataFrame,
    mode: str = "equal_weight",
    ticker: str = "SPY",
    fallback: str = "equal_weight",
    fetch_if_missing: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BenchmarkResolution:
    """Resolve benchmark return series for attribution with deterministic fallback."""
    if asset_returns.empty:
        raise ValueError("asset_returns is empty")
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise ValueError("asset_returns index must be DatetimeIndex")

    mode = str(mode).lower()
    fallback = str(fallback).lower()
    ticker = str(ticker).upper()

    if mode == "equal_weight":
        resolved = equal_weight_benchmark(asset_returns)
        total_days = int(len(resolved))
        return BenchmarkResolution(
            returns=resolved,
            source="equal_weight",
            requested_mode=mode,
            requested_ticker=None,
            fallback_reason=None,
            coverage_ratio=1.0 if total_days > 0 else 0.0,
            non_null_days=total_days,
            total_days=total_days,
        )

    if mode != "spy":
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    spy_returns = _extract_ticker_returns(market_data, ticker=ticker)
    fallback_reason: str | None = None

    if spy_returns is None and fetch_if_missing:
        start = start_date or str(asset_returns.index.min().date())
        end = end_date or str((asset_returns.index.max() + pd.Timedelta(days=1)).date())
        fetched, err = _fetch_ticker_returns_yfinance(ticker=ticker, start=start, end=end)
        spy_returns = fetched
        if spy_returns is None:
            fallback_reason = f"spy_fetch_failed:{err}"
    elif spy_returns is None:
        fallback_reason = "spy_not_in_market_data_and_fetch_disabled"

    if spy_returns is None:
        if fallback != "equal_weight":
            raise ValueError(f"Unsupported benchmark fallback: {fallback}")
        fallback_returns = equal_weight_benchmark(asset_returns)
        total_days = int(len(fallback_returns))
        return BenchmarkResolution(
            returns=fallback_returns,
            source="equal_weight_fallback",
            requested_mode=mode,
            requested_ticker=ticker,
            fallback_reason=fallback_reason or "spy_unavailable",
            coverage_ratio=1.0 if total_days > 0 else 0.0,
            non_null_days=total_days,
            total_days=total_days,
        )

    aligned = _normalize_return_series(spy_returns).reindex(asset_returns.index)
    non_null_days = int(aligned.notna().sum())
    total_days = int(len(aligned))
    coverage_ratio = float(non_null_days / total_days) if total_days > 0 else 0.0

    if non_null_days == 0:
        if fallback != "equal_weight":
            raise ValueError(f"Unsupported benchmark fallback: {fallback}")
        fallback_returns = equal_weight_benchmark(asset_returns)
        total_days = int(len(fallback_returns))
        return BenchmarkResolution(
            returns=fallback_returns,
            source="equal_weight_fallback",
            requested_mode=mode,
            requested_ticker=ticker,
            fallback_reason="spy_no_date_overlap",
            coverage_ratio=1.0 if total_days > 0 else 0.0,
            non_null_days=total_days,
            total_days=total_days,
        )

    return BenchmarkResolution(
        returns=aligned.rename("benchmark_return"),
        source="spy",
        requested_mode=mode,
        requested_ticker=ticker,
        fallback_reason=fallback_reason,
        coverage_ratio=coverage_ratio,
        non_null_days=non_null_days,
        total_days=total_days,
    )
