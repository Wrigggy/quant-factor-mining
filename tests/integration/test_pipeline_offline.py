from __future__ import annotations

import numpy as np

from qfm.factors.mean_reversion import MeanReversionFactor
from qfm.factors.momentum import MomentumFactor
from qfm.factors.volatility import LowVolatilityFactor
from qfm.modeling.walkforward import run_walkforward_research


def test_walkforward_pipeline_offline(market_data):
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
    )

    fold_metrics = result["fold_metrics"]
    aggregate = result["aggregate"]

    assert not fold_metrics.empty
    assert aggregate["n_folds"] >= 1
    assert np.isfinite(aggregate["mean_fold_sharpe"])
    assert np.isfinite(aggregate["mean_fold_alpha_annual"])
    assert "information_ratio" in fold_metrics.columns
    assert fold_metrics["n_days"].max() <= 63
