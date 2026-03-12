"""Leakage-safe backtest engine.

Signal at date t is used to rebalance at the close of t and becomes active at t+1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .costs import LiquidityCostModel, linear_transaction_cost, liquidity_transaction_cost


def _apply_weight_cap(weights: pd.Series, max_single_weight: float) -> pd.Series:
    """Apply per-asset max weight and renormalize while preserving long-only weights."""
    w = weights.clip(lower=0.0).astype(float)
    if w.sum() <= 0:
        return w
    w = w / w.sum()

    cap = float(max_single_weight)
    if cap >= 1.0:
        return w
    if cap <= 0.0:
        raise ValueError("max_single_weight must be > 0")
    if len(w) * cap < 1.0 - 1e-12:
        raise ValueError("max_single_weight is too small for selected asset count")

    w = w.clip(upper=cap)
    for _ in range(32):
        deficit = 1.0 - float(w.sum())
        if deficit <= 1e-12:
            break
        room = (cap - w).clip(lower=0.0)
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            break
        w = (w + room / room_sum * deficit).clip(upper=cap)

    if w.sum() <= 0:
        return weights
    return w / w.sum()


def _target_weights_from_scores(
    score_row: pd.Series,
    top_n: int,
    universe: pd.Index,
    weighting_mode: str = "equal_weight",
    score_temperature: float = 1.0,
    max_single_weight: float = 1.0,
) -> pd.Series:
    """Convert cross-sectional score into long-only target portfolio weights."""
    weights = pd.Series(0.0, index=universe)

    clean = score_row.dropna()
    if clean.empty:
        return weights

    n = min(top_n, len(clean))
    selected = clean.nlargest(n)
    if len(selected) == 0:
        return weights

    mode = str(weighting_mode).lower()
    if mode == "equal_weight":
        target = pd.Series(1.0 / len(selected), index=selected.index)
    elif mode == "score_tilted":
        temperature = max(float(score_temperature), 1e-6)
        centered = (selected.astype(float) - float(selected.max())) / temperature
        raw = np.exp(centered.to_numpy(dtype=float))
        raw_sum = float(raw.sum())
        if raw_sum <= 0.0:
            target = pd.Series(1.0 / len(selected), index=selected.index)
        else:
            target = pd.Series(raw / raw_sum, index=selected.index)
        target = _apply_weight_cap(target, max_single_weight=max_single_weight)
    else:
        raise ValueError(f"Unsupported weighting_mode: {weighting_mode}")

    weights.loc[target.index] = target.values
    return weights


def run_backtest_from_scores(
    scores: pd.DataFrame,
    asset_returns: pd.DataFrame,
    top_n: int = 20,
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 1_000_000,
    execution_prices: Optional[pd.DataFrame] = None,
    execution_volumes: Optional[pd.DataFrame] = None,
    liquidity_cost_model: Optional[LiquidityCostModel] = None,
    weighting_mode: str = "equal_weight",
    score_temperature: float = 1.0,
    max_single_weight: float = 1.0,
) -> pd.DataFrame:
    """Run long-only strategy from score matrix without same-day look-ahead."""
    if scores.empty:
        raise ValueError("scores is empty")
    if asset_returns.empty:
        raise ValueError("asset_returns is empty")

    common_dates = scores.index.intersection(asset_returns.index)
    if len(common_dates) == 0:
        raise ValueError("No common dates between scores and asset_returns")

    common_assets = scores.columns.intersection(asset_returns.columns)
    if len(common_assets) == 0:
        raise ValueError("No common assets between scores and asset_returns")

    scores = scores.loc[common_dates, common_assets].sort_index()
    returns = asset_returns.loc[common_dates, common_assets].sort_index().fillna(0.0)
    prices = (
        execution_prices.reindex(index=common_dates, columns=common_assets).sort_index()
        if execution_prices is not None
        else None
    )
    volumes = (
        execution_volumes.reindex(index=common_dates, columns=common_assets).sort_index()
        if execution_volumes is not None
        else None
    )

    portfolio_value = float(initial_capital)
    active_weights = pd.Series(0.0, index=common_assets)

    rows = []

    for i, date in enumerate(common_dates):
        value_before = portfolio_value

        gross_return = float((active_weights * returns.loc[date]).sum())
        portfolio_value *= (1.0 + gross_return)

        turnover = 0.0
        cost_rate = 0.0

        if i % rebalance_frequency == 0:
            target = _target_weights_from_scores(
                scores.loc[date],
                top_n=top_n,
                universe=common_assets,
                weighting_mode=weighting_mode,
                score_temperature=score_temperature,
                max_single_weight=max_single_weight,
            )
            trade = target - active_weights
            turnover = float(trade.abs().sum())

            if liquidity_cost_model is not None and prices is not None and volumes is not None:
                cost_rate = liquidity_transaction_cost(
                    trade_weights=trade,
                    portfolio_value=portfolio_value,
                    prices=prices.loc[date],
                    volumes=volumes.loc[date],
                    model=liquidity_cost_model,
                )
            else:
                cost_rate = linear_transaction_cost(turnover, transaction_cost_bps)
            portfolio_value *= (1.0 - cost_rate)

            # Rebalance at close of date; these weights are used from next day onward.
            active_weights = target

        net_return = portfolio_value / value_before - 1.0

        rows.append(
            {
                "date": date,
                "portfolio_value": portfolio_value,
                "portfolio_return": net_return,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost_rate": cost_rate,
                "is_rebalance": i % rebalance_frequency == 0,
            }
        )

    result = pd.DataFrame(rows).set_index("date")
    return result
