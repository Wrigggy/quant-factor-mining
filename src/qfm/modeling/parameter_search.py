"""Simple, deterministic parameter search for interview-friendly reproducibility."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from qfm.factors.mean_reversion import MeanReversionFactor
from qfm.factors.momentum import MomentumFactor
from qfm.factors.volatility import LowVolatilityFactor
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


def grid_search(
    market_data: pd.DataFrame,
    search_space: SearchSpace = DEFAULT_SPACE,
    train_size: int = 504,
    test_size: int = 126,
    top_n: int = 20,
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """Run deterministic grid search and return ranked results."""
    rows: List[Dict[str, object]] = []

    for mom_lb, rev_lb, vol_win in product(
        search_space.momentum_lookback,
        search_space.mean_reversion_lookback,
        search_space.volatility_window,
    ):
        factors = [
            MomentumFactor(lookback=mom_lb),
            MeanReversionFactor(lookback=rev_lb),
            LowVolatilityFactor(window=vol_win),
        ]

        result = run_walkforward_research(
            market_data=market_data,
            factors=factors,
            train_size=train_size,
            test_size=test_size,
            top_n=top_n,
            rebalance_frequency=rebalance_frequency,
            transaction_cost_bps=transaction_cost_bps,
            initial_capital=initial_capital,
        )

        aggregate = result["aggregate"]
        rows.append(
            {
                "momentum_lookback": mom_lb,
                "mean_reversion_lookback": rev_lb,
                "volatility_window": vol_win,
                "mean_fold_sharpe": aggregate["mean_fold_sharpe"],
                "mean_fold_total_return": aggregate["mean_fold_total_return"],
                "n_folds": aggregate["n_folds"],
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["mean_fold_sharpe", "mean_fold_total_return"],
        ascending=False,
    )
    return out.reset_index(drop=True)
