"""Simple, deterministic parameter search for reproducible research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from qfm.backtest.costs import LiquidityCostModel
from qfm.factors.base import BaseFactor
from qfm.factors.mean_reversion import MeanReversionFactor
from qfm.factors.momentum import MomentumFactor
from qfm.factors.volatility import LowVolatilityFactor
from qfm.modeling.stability import summarize_fold_stability
from qfm.modeling.walkforward import run_walkforward_research


@dataclass(frozen=True)
class SearchSpace:
    momentum_lookback: Sequence[int]
    mean_reversion_lookback: Sequence[int]
    volatility_window: Sequence[int]


DEFAULT_SPACE = SearchSpace(
    momentum_lookback=(126, 252),
    mean_reversion_lookback=(10, 21),
    volatility_window=(42, 63),
)


def build_factor_set(
    momentum_lookback: int,
    mean_reversion_lookback: int,
    volatility_window: int,
    momentum_skip: int = 21,
) -> List[BaseFactor]:
    """Build canonical factor list for one parameter combination."""
    return [
        MomentumFactor(lookback=int(momentum_lookback), skip=int(momentum_skip)),
        MeanReversionFactor(lookback=int(mean_reversion_lookback)),
        LowVolatilityFactor(window=int(volatility_window)),
    ]


def grid_search(
    market_data: pd.DataFrame,
    search_space: SearchSpace = DEFAULT_SPACE,
    train_size: int = 504,
    test_size: int = 126,
    step_size: int | None = None,
    top_n: int = 20,
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 1_000_000,
    holdout_size: int = 0,
    evaluate_holdout: bool = False,
    liquidity_cost_model: LiquidityCostModel | None = None,
    momentum_skip: int = 21,
    weighting_mode: str = "equal_weight",
    score_temperature: float = 1.0,
    max_single_weight: float = 1.0,
) -> pd.DataFrame:
    """Run deterministic grid search and return ranked results."""
    rows: List[Dict[str, object]] = []

    for mom_lb, rev_lb, vol_win in product(
        search_space.momentum_lookback,
        search_space.mean_reversion_lookback,
        search_space.volatility_window,
    ):
        factors = build_factor_set(
            momentum_lookback=mom_lb,
            mean_reversion_lookback=rev_lb,
            volatility_window=vol_win,
            momentum_skip=momentum_skip,
        )

        result = run_walkforward_research(
            market_data=market_data,
            factors=factors,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            top_n=top_n,
            rebalance_frequency=rebalance_frequency,
            transaction_cost_bps=transaction_cost_bps,
            initial_capital=initial_capital,
            holdout_size=holdout_size,
            evaluate_holdout=evaluate_holdout,
            liquidity_cost_model=liquidity_cost_model,
            weighting_mode=weighting_mode,
            score_temperature=score_temperature,
            max_single_weight=max_single_weight,
        )

        aggregate = result["aggregate"]
        fold_stability = summarize_fold_stability(result["fold_metrics"])
        holdout_metrics = result.get("holdout_metrics")
        rows.append(
            {
                "momentum_lookback": mom_lb,
                "mean_reversion_lookback": rev_lb,
                "volatility_window": vol_win,
                "mean_fold_sharpe": aggregate["mean_fold_sharpe"],
                "mean_fold_total_return": aggregate["mean_fold_total_return"],
                "mean_fold_alpha_annual": aggregate["mean_fold_alpha_annual"],
                "mean_fold_information_ratio": aggregate["mean_fold_information_ratio"],
                "n_folds": aggregate["n_folds"],
                "fold_sharpe_std": fold_stability["fold_sharpe_std"],
                "positive_sharpe_ratio": fold_stability["positive_sharpe_ratio"],
                "worst_fold_max_drawdown": fold_stability["worst_fold_max_drawdown"],
                "holdout_total_return": (
                    float(holdout_metrics.get("total_return", np.nan))
                    if holdout_metrics is not None
                    else np.nan
                ),
                "holdout_sharpe": (
                    float(holdout_metrics.get("sharpe", np.nan))
                    if holdout_metrics is not None
                    else np.nan
                ),
                "holdout_excess_total_return": (
                    float(holdout_metrics.get("excess_total_return", np.nan))
                    if holdout_metrics is not None
                    else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["mean_fold_sharpe", "mean_fold_total_return"],
        ascending=False,
    )
    return out.reset_index(drop=True)
