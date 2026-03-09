"""Backtest performance metrics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _annualize_return_from_series(returns: pd.Series) -> float:
    """Annualize compounded return from daily return series."""
    if returns.empty:
        return 0.0
    total = float((1.0 + returns).prod() - 1.0)
    years = max(len(returns) / TRADING_DAYS, 1 / TRADING_DAYS)
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def compute_benchmark_attribution(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """Compute benchmark-relative performance attribution metrics."""
    aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join="inner")
    aligned_strategy = aligned_strategy.dropna()
    aligned_benchmark = aligned_benchmark.reindex(aligned_strategy.index).dropna()
    aligned_strategy = aligned_strategy.reindex(aligned_benchmark.index)

    if len(aligned_strategy) < 2:
        return {
            "benchmark_total_return": 0.0,
            "benchmark_annual_return": 0.0,
            "excess_total_return": 0.0,
            "alpha_annual": 0.0,
            "beta": 0.0,
            "tracking_error_annual": 0.0,
            "information_ratio": 0.0,
        }

    rf_daily = risk_free_rate / TRADING_DAYS
    strategy_excess = aligned_strategy - rf_daily
    benchmark_excess = aligned_benchmark - rf_daily

    benchmark_var = float(benchmark_excess.var())
    if benchmark_var <= 1e-12:
        beta = 0.0
    else:
        covariance = float(np.cov(strategy_excess, benchmark_excess, ddof=1)[0, 1])
        beta = covariance / benchmark_var

    alpha_daily = float(strategy_excess.mean() - beta * benchmark_excess.mean())
    alpha_annual = alpha_daily * TRADING_DAYS

    active = aligned_strategy - aligned_benchmark
    tracking_error_annual = float(active.std() * np.sqrt(TRADING_DAYS))
    information_ratio = float(
        active.mean() / (active.std() + 1e-12) * np.sqrt(TRADING_DAYS)
    )

    benchmark_total = float((1.0 + aligned_benchmark).prod() - 1.0)
    strategy_total = float((1.0 + aligned_strategy).prod() - 1.0)

    return {
        "benchmark_total_return": benchmark_total,
        "benchmark_annual_return": _annualize_return_from_series(aligned_benchmark),
        "excess_total_return": strategy_total - benchmark_total,
        "alpha_annual": alpha_annual,
        "beta": float(beta),
        "tracking_error_annual": tracking_error_annual,
        "information_ratio": information_ratio,
    }


def compute_performance_metrics(
    backtest: pd.DataFrame,
    risk_free_rate: float = 0.02,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute core performance metrics from backtest results."""
    if backtest.empty:
        raise ValueError("backtest is empty")

    returns = backtest["portfolio_return"].fillna(0.0)
    value = backtest["portfolio_value"]

    total_return = float(value.iloc[-1] / value.iloc[0] - 1.0)
    years = max(len(returns) / TRADING_DAYS, 1 / TRADING_DAYS)
    annual_return = float((1 + total_return) ** (1 / years) - 1)

    vol_daily = float(returns.std())
    vol_annual = float(vol_daily * np.sqrt(TRADING_DAYS))

    excess = returns - risk_free_rate / TRADING_DAYS
    sharpe = float(excess.mean() / (excess.std() + 1e-12) * np.sqrt(TRADING_DAYS))

    rolling_max = value.cummax()
    drawdown = (value - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    turnover_total = float(backtest["turnover"].sum())
    avg_turnover = float(backtest.loc[backtest["is_rebalance"], "turnover"].mean())
    if np.isnan(avg_turnover):
        avg_turnover = 0.0
    total_cost = float((backtest["cost_rate"] * backtest["portfolio_value"]).sum())

    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility_annual": vol_annual,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": float((returns > 0).mean()),
        "total_turnover": turnover_total,
        "avg_turnover": avg_turnover,
        "total_cost_dollars": total_cost,
        "n_days": int(len(returns)),
    }

    if benchmark_returns is not None:
        metrics.update(
            compute_benchmark_attribution(
                strategy_returns=returns,
                benchmark_returns=benchmark_returns,
                risk_free_rate=risk_free_rate,
            )
        )

    return metrics
