"""Walk-forward research loop with strict time ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from qfm.backtest.benchmarks import equal_weight_benchmark
from qfm.backtest.engine import run_backtest_from_scores
from qfm.backtest.metrics import compute_performance_metrics
from qfm.data.contracts import ensure_market_data
from qfm.factors.base import BaseFactor
from qfm.labels.forward_returns import compute_forward_returns
from qfm.modeling.feature_builder import build_factor_matrix


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    dates: pd.DatetimeIndex,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> List[WalkForwardFold]:
    """Generate expanding walk-forward folds with fixed train/test widths."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")

    dates = pd.DatetimeIndex(sorted(dates.unique()))
    step = step_size or test_size
    folds: List[WalkForwardFold] = []

    i = 0
    fold_id = 1
    while i + train_size + test_size <= len(dates):
        train_slice = dates[i : i + train_size]
        test_slice = dates[i + train_size : i + train_size + test_size]
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=train_slice[0],
                train_end=train_slice[-1],
                test_start=test_slice[0],
                test_end=test_slice[-1],
            )
        )
        fold_id += 1
        i += step

    return folds


def _cross_sectional_ic_by_date(factor: pd.Series, fwd_returns: pd.Series) -> pd.Series:
    """Compute daily cross-sectional Spearman IC."""
    aligned = pd.DataFrame({"factor": factor, "fwd": fwd_returns}).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    ic = aligned.groupby(level="date").apply(
        lambda frame: frame["factor"].corr(frame["fwd"], method="spearman")
    )
    return ic.dropna()


def estimate_train_factor_weights(
    factor_matrix: pd.DataFrame,
    forward_returns: pd.Series,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Estimate factor combination weights from train IC means only."""
    train_mask = (
        (factor_matrix.index.get_level_values("date") >= train_start)
        & (factor_matrix.index.get_level_values("date") < train_end)
    )
    train_features = factor_matrix.loc[train_mask]

    if train_features.empty:
        raise ValueError("No training features in selected fold")

    ic_stats: Dict[str, float] = {}
    for col in train_features.columns:
        daily_ic = _cross_sectional_ic_by_date(train_features[col], forward_returns)
        ic_stats[col] = float(daily_ic.mean()) if not daily_ic.empty else 0.0

    ic_series = pd.Series(ic_stats)
    positive = ic_series.clip(lower=0.0)
    if positive.sum() <= 0:
        weights = pd.Series(1.0 / len(ic_series), index=ic_series.index)
    else:
        weights = positive / positive.sum()

    return weights, ic_stats


def run_walkforward_research(
    market_data: pd.DataFrame,
    factors: Iterable[BaseFactor],
    train_size: int,
    test_size: int,
    top_n: int,
    rebalance_frequency: int,
    transaction_cost_bps: float,
    initial_capital: float,
) -> Dict[str, object]:
    """Execute walk-forward pipeline and return fold metrics + aggregate outputs."""
    data = ensure_market_data(market_data)

    factor_matrix = build_factor_matrix(data, factors, normalize=True)
    fwd_returns_1d = compute_forward_returns(data, horizon=1)
    close_wide = data["close"].unstack("ticker").sort_index()
    asset_returns = close_wide.pct_change().fillna(0.0)

    dates = pd.DatetimeIndex(sorted(close_wide.index.unique()))
    folds = generate_folds(dates, train_size=train_size, test_size=test_size)
    if not folds:
        raise ValueError("No folds generated; adjust train/test sizes")

    fold_rows = []
    equity_frames = []

    for fold in folds:
        weights, train_ic = estimate_train_factor_weights(
            factor_matrix=factor_matrix,
            forward_returns=fwd_returns_1d,
            train_start=fold.train_start,
            train_end=fold.train_end,
        )

        # Build composite score using train-estimated static weights.
        composite = factor_matrix.mul(weights, axis=1).sum(axis=1)

        test_mask = (
            (composite.index.get_level_values("date") >= fold.test_start)
            & (composite.index.get_level_values("date") <= fold.test_end)
        )
        test_scores = composite.loc[test_mask]
        test_dates = asset_returns.index[
            (asset_returns.index >= fold.test_start) & (asset_returns.index <= fold.test_end)
        ]
        test_score_wide = test_scores.unstack("ticker").reindex(test_dates).sort_index()
        test_asset_returns = asset_returns.reindex(test_dates).sort_index()

        bt = run_backtest_from_scores(
            scores=test_score_wide,
            asset_returns=test_asset_returns,
            top_n=top_n,
            rebalance_frequency=rebalance_frequency,
            transaction_cost_bps=transaction_cost_bps,
            initial_capital=initial_capital,
        )

        benchmark = equal_weight_benchmark(test_asset_returns).reindex(bt.index)
        metrics = compute_performance_metrics(bt, benchmark_returns=benchmark)
        metrics["fold_id"] = fold.fold_id
        metrics["train_start"] = str(fold.train_start.date())
        metrics["train_end"] = str(fold.train_end.date())
        metrics["test_start"] = str(fold.test_start.date())
        metrics["test_end"] = str(fold.test_end.date())
        metrics["train_ic_mean"] = float(np.mean(list(train_ic.values())))

        fold_rows.append(metrics)
        bt = bt.copy()
        bt["fold_id"] = fold.fold_id
        equity_frames.append(bt)

    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold_id")
    equity_curve = pd.concat(equity_frames).sort_index()

    aggregate = {
        "mean_fold_sharpe": float(fold_metrics["sharpe"].mean()),
        "mean_fold_total_return": float(fold_metrics["total_return"].mean()),
        "mean_fold_alpha_annual": float(fold_metrics["alpha_annual"].mean()),
        "mean_fold_information_ratio": float(fold_metrics["information_ratio"].mean()),
        "n_folds": int(len(fold_metrics)),
    }

    return {
        "fold_metrics": fold_metrics,
        "equity_curve": equity_curve,
        "aggregate": aggregate,
    }
