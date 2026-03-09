from __future__ import annotations

import pandas as pd

from qfm.factors.mean_reversion import MeanReversionFactor
from qfm.factors.momentum import MomentumFactor
from qfm.factors.volatility import LowVolatilityFactor
from qfm.modeling.walkforward import run_walkforward_research


def test_walkforward_reserves_untouched_holdout_window(market_data):
    factors = [
        MomentumFactor(lookback=126, skip=21),
        MeanReversionFactor(lookback=21),
        LowVolatilityFactor(window=42),
    ]

    result = run_walkforward_research(
        market_data=market_data,
        factors=factors,
        train_size=252,
        test_size=63,
        top_n=3,
        rebalance_frequency=21,
        transaction_cost_bps=10.0,
        initial_capital=1_000_000,
        holdout_size=63,
        evaluate_holdout=True,
    )

    fold_metrics = result["fold_metrics"]
    split = result["split"]
    holdout_metrics = result["holdout_metrics"]

    assert holdout_metrics is not None
    holdout_start = pd.Timestamp(split["holdout_start"])
    max_fold_test_end = pd.to_datetime(fold_metrics["test_end"]).max()
    assert max_fold_test_end < holdout_start
    assert holdout_metrics["holdout_start"] == split["holdout_start"]
    assert holdout_metrics["n_days"] <= 63
