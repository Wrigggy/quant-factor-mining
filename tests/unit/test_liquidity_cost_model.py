from __future__ import annotations

import pandas as pd

from qfm.backtest.costs import LiquidityCostModel, liquidity_transaction_cost
from qfm.backtest.engine import run_backtest_from_scores


def test_liquidity_transaction_cost_penalizes_low_volume():
    trade = pd.Series({"A": 0.5, "B": -0.5})
    price = pd.Series({"A": 100.0, "B": 100.0})
    high_volume = pd.Series({"A": 5_000_000.0, "B": 5_000_000.0})
    low_volume = pd.Series({"A": 10_000.0, "B": 10_000.0})
    model = LiquidityCostModel()

    high_liquidity_cost = liquidity_transaction_cost(
        trade_weights=trade,
        portfolio_value=1_000_000.0,
        prices=price,
        volumes=high_volume,
        model=model,
    )
    low_liquidity_cost = liquidity_transaction_cost(
        trade_weights=trade,
        portfolio_value=1_000_000.0,
        prices=price,
        volumes=low_volume,
        model=model,
    )

    assert low_liquidity_cost > high_liquidity_cost > 0.0


def test_backtest_uses_liquidity_model_when_prices_and_volumes_provided():
    dates = pd.bdate_range("2024-01-01", periods=6)
    scores = pd.DataFrame({"A": 1.0, "B": 0.0}, index=dates)
    returns = pd.DataFrame({"A": 0.0, "B": 0.0}, index=dates)
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    low_volume = pd.DataFrame({"A": 8_000.0, "B": 8_000.0}, index=dates)

    linear_bt = run_backtest_from_scores(
        scores=scores,
        asset_returns=returns,
        top_n=1,
        rebalance_frequency=1,
        transaction_cost_bps=10.0,
        initial_capital=100_000.0,
    )
    liquidity_bt = run_backtest_from_scores(
        scores=scores,
        asset_returns=returns,
        top_n=1,
        rebalance_frequency=1,
        transaction_cost_bps=10.0,
        initial_capital=100_000.0,
        execution_prices=prices,
        execution_volumes=low_volume,
        liquidity_cost_model=LiquidityCostModel(),
    )

    assert liquidity_bt["cost_rate"].iloc[0] > linear_bt["cost_rate"].iloc[0]
