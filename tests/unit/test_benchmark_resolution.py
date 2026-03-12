from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.backtest.benchmarks import equal_weight_benchmark, resolve_benchmark_returns
from qfm.data.contracts import ensure_market_data


def _append_spy_rows(market_data: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None):
    close_wide = market_data["close"].unstack("ticker").sort_index()
    dates = close_wide.index
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates <= end]

    anchor = close_wide.iloc[:, 0].reindex(dates).astype(float)
    spy_close = anchor * 0.97
    spy = pd.DataFrame(
        {
            "open": spy_close * 0.999,
            "high": spy_close * 1.002,
            "low": spy_close * 0.998,
            "close": spy_close,
            "volume": pd.Series(5_000_000.0, index=dates),
        }
    )
    spy["date"] = dates
    spy["ticker"] = "SPY"
    spy = spy.set_index(["date", "ticker"]).sort_index()

    merged = pd.concat([market_data, spy], axis=0).sort_index()
    return ensure_market_data(merged)


def test_resolve_spy_benchmark_from_local_market_data(market_data):
    data_with_spy = _append_spy_rows(market_data)
    asset_returns = (
        data_with_spy["close"]
        .unstack("ticker")
        .sort_index()
        .pct_change(fill_method=None)
        .fillna(0.0)
    )

    resolved = resolve_benchmark_returns(
        asset_returns=asset_returns,
        market_data=data_with_spy,
        mode="spy",
        ticker="SPY",
        fetch_if_missing=False,
    )

    assert resolved.source == "spy"
    assert resolved.requested_ticker == "SPY"
    assert resolved.fallback_reason is None
    assert 0.95 <= resolved.coverage_ratio <= 1.0
    assert resolved.non_null_days < resolved.total_days
    assert resolved.returns.name == "benchmark_return"


def test_resolve_spy_benchmark_falls_back_to_equal_weight_when_missing(market_data):
    asset_returns = (
        market_data["close"]
        .unstack("ticker")
        .sort_index()
        .pct_change(fill_method=None)
        .fillna(0.0)
    )

    resolved = resolve_benchmark_returns(
        asset_returns=asset_returns,
        market_data=market_data,
        mode="spy",
        ticker="SPY",
        fetch_if_missing=False,
    )

    assert resolved.source == "equal_weight_fallback"
    assert resolved.fallback_reason == "spy_not_in_market_data_and_fetch_disabled"
    assert resolved.coverage_ratio == 1.0
    assert resolved.non_null_days == resolved.total_days == len(asset_returns)

    expected = equal_weight_benchmark(asset_returns)
    assert np.allclose(resolved.returns.to_numpy(), expected.to_numpy(), equal_nan=False)


def test_resolve_spy_benchmark_with_partial_date_overlap(market_data):
    dates = market_data.index.get_level_values("date").unique().sort_values()
    start = dates[int(len(dates) * 0.25)]
    end = dates[int(len(dates) * 0.75)]
    partial_spy_data = _append_spy_rows(market_data, start=start, end=end)
    asset_returns = (
        partial_spy_data["close"]
        .unstack("ticker")
        .sort_index()
        .pct_change(fill_method=None)
        .fillna(0.0)
    )

    resolved = resolve_benchmark_returns(
        asset_returns=asset_returns,
        market_data=partial_spy_data,
        mode="spy",
        ticker="SPY",
        fetch_if_missing=False,
    )

    assert resolved.source == "spy"
    assert 0.0 < resolved.coverage_ratio < 1.0
    assert 0 < resolved.non_null_days < resolved.total_days
