from __future__ import annotations

import pandas as pd

from qfm.modeling.stability import apply_holdout_gate, evaluate_holdout_gate, rank_candidates


def test_stability_first_ranking_filters_holdout_failures():
    candidates = pd.DataFrame(
        [
            {
                "momentum_lookback": 126,
                "mean_reversion_lookback": 10,
                "volatility_window": 42,
                "mean_fold_sharpe": 1.50,
                "mean_fold_total_return": 0.10,
                "fold_sharpe_std": 0.70,
                "positive_sharpe_ratio": 0.80,
                "worst_fold_max_drawdown": -0.20,
                "holdout_sharpe": -0.10,
                "holdout_excess_total_return": 0.02,
            },
            {
                "momentum_lookback": 252,
                "mean_reversion_lookback": 10,
                "volatility_window": 42,
                "mean_fold_sharpe": 0.90,
                "mean_fold_total_return": 0.08,
                "fold_sharpe_std": 0.25,
                "positive_sharpe_ratio": 0.90,
                "worst_fold_max_drawdown": -0.11,
                "holdout_sharpe": 0.25,
                "holdout_excess_total_return": 0.03,
            },
            {
                "momentum_lookback": 126,
                "mean_reversion_lookback": 21,
                "volatility_window": 63,
                "mean_fold_sharpe": 0.70,
                "mean_fold_total_return": 0.07,
                "fold_sharpe_std": 0.30,
                "positive_sharpe_ratio": 0.75,
                "worst_fold_max_drawdown": -0.13,
                "holdout_sharpe": 0.10,
                "holdout_excess_total_return": 0.01,
            },
        ]
    )

    gated = apply_holdout_gate(candidates, min_holdout_sharpe=0.0, min_holdout_excess_total_return=0.0)
    ranked = rank_candidates(gated, selection_mode="stability_first")

    assert bool(ranked.iloc[0]["gate_pass"])
    assert int(ranked.iloc[0]["momentum_lookback"]) == 252
    assert bool(ranked.iloc[-1]["gate_pass"]) is False


def test_holdout_gate_fails_when_any_threshold_is_not_met():
    passed, reason = evaluate_holdout_gate(
        holdout_sharpe=-0.05,
        holdout_excess_total_return=0.02,
        min_holdout_sharpe=0.0,
        min_holdout_excess_total_return=0.0,
    )
    assert not passed
    assert "holdout_sharpe" in reason

    passed, reason = evaluate_holdout_gate(
        holdout_sharpe=0.05,
        holdout_excess_total_return=-0.02,
        min_holdout_sharpe=0.0,
        min_holdout_excess_total_return=0.0,
    )
    assert not passed
    assert "holdout_excess" in reason
