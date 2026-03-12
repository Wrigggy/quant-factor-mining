from __future__ import annotations

import pandas as pd
import pytest

from qfm.backtest.engine import run_backtest_from_scores


@pytest.mark.parametrize("weighting_mode", ["equal_weight", "score_tilted"])
def test_backtest_signal_is_not_used_same_day(weighting_mode: str):
    dates = pd.bdate_range("2024-01-01", periods=8)
    returns = pd.DataFrame(index=dates, columns=["A", "B"], data=0.0)

    # Winner alternates by day. If same-day leakage exists, strategy would capture each winner.
    for i, date in enumerate(dates):
        if i % 2 == 0:
            returns.loc[date, "A"] = 0.1
        else:
            returns.loc[date, "B"] = 0.1

    # Scores intentionally point to same-day winner.
    scores = pd.DataFrame(index=dates, columns=["A", "B"], data=0.0)
    for i, date in enumerate(dates):
        if i % 2 == 0:
            scores.loc[date, "A"] = 1.0
        else:
            scores.loc[date, "B"] = 1.0

    bt = run_backtest_from_scores(
        scores=scores,
        asset_returns=returns,
        top_n=1,
        rebalance_frequency=1,
        transaction_cost_bps=0.0,
        initial_capital=100.0,
        weighting_mode=weighting_mode,
        score_temperature=0.8,
        max_single_weight=1.0,
    )

    # With correct t -> t+1 execution, this alternating setup should not harvest same-day winners.
    total_return = bt["portfolio_value"].iloc[-1] / bt["portfolio_value"].iloc[0] - 1
    assert total_return < 0.2
