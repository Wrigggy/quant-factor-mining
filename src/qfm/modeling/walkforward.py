"""Walk-forward research loop with strict time ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from qfm.backtest.benchmarks import equal_weight_benchmark
from qfm.backtest.costs import LiquidityCostModel
from qfm.backtest.engine import run_backtest_from_scores
from qfm.backtest.metrics import (
    bootstrap_benchmark_attribution_ci,
    compute_benchmark_attribution,
    compute_performance_metrics,
)
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


def _split_research_and_holdout(
    dates: pd.DatetimeIndex,
    holdout_size: int,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split chronological dates into research period and untouched holdout."""
    if holdout_size < 0:
        raise ValueError("holdout_size must be >= 0")

    if holdout_size == 0:
        return dates, pd.DatetimeIndex([])

    if holdout_size >= len(dates):
        raise ValueError("holdout_size is too large for available date history")

    research_dates = dates[:-holdout_size]
    holdout_dates = dates[-holdout_size:]
    return research_dates, holdout_dates


def _run_period_backtest(
    composite_scores: pd.Series,
    asset_returns: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    top_n: int,
    rebalance_frequency: int,
    transaction_cost_bps: float,
    initial_capital: float,
    liquidity_cost_model: Optional[LiquidityCostModel],
    benchmark_returns: Optional[pd.Series] = None,
    weighting_mode: str = "equal_weight",
    score_temperature: float = 1.0,
    max_single_weight: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Backtest one date range and return strategy curve + benchmark series."""
    test_mask = (
        (composite_scores.index.get_level_values("date") >= period_start)
        & (composite_scores.index.get_level_values("date") <= period_end)
    )
    period_scores = composite_scores.loc[test_mask]
    period_dates = asset_returns.index[
        (asset_returns.index >= period_start) & (asset_returns.index <= period_end)
    ]

    score_wide = period_scores.unstack("ticker").reindex(period_dates).sort_index()
    period_returns = asset_returns.reindex(period_dates).sort_index()
    period_prices = prices.reindex(period_dates).sort_index()
    period_volumes = volumes.reindex(period_dates).sort_index()

    bt = run_backtest_from_scores(
        scores=score_wide,
        asset_returns=period_returns,
        top_n=top_n,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        initial_capital=initial_capital,
        execution_prices=period_prices,
        execution_volumes=period_volumes,
        liquidity_cost_model=liquidity_cost_model,
        weighting_mode=weighting_mode,
        score_temperature=score_temperature,
        max_single_weight=max_single_weight,
    )
    if benchmark_returns is None:
        benchmark = equal_weight_benchmark(period_returns).reindex(bt.index)
    else:
        benchmark = benchmark_returns.reindex(bt.index).rename("benchmark_return")
    return bt, benchmark


def run_walkforward_research(
    market_data: pd.DataFrame,
    factors: Iterable[BaseFactor],
    train_size: int,
    test_size: int,
    top_n: int,
    rebalance_frequency: int,
    transaction_cost_bps: float,
    initial_capital: float,
    holdout_size: int = 0,
    evaluate_holdout: bool = True,
    bootstrap_samples: int = 0,
    bootstrap_ci_level: float = 0.95,
    bootstrap_seed: int = 42,
    liquidity_cost_model: Optional[LiquidityCostModel] = None,
    weighting_mode: str = "equal_weight",
    score_temperature: float = 1.0,
    max_single_weight: float = 1.0,
    benchmark_returns: Optional[pd.Series] = None,
    benchmark_source: str = "equal_weight",
    benchmark_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Execute walk-forward pipeline and return fold metrics + aggregate outputs."""
    data = ensure_market_data(market_data)

    factor_matrix = build_factor_matrix(data, factors, normalize=True)
    fwd_returns_1d = compute_forward_returns(data, horizon=1)
    close_wide = data["close"].unstack("ticker").sort_index()
    volume_wide = data["volume"].unstack("ticker").sort_index()
    asset_returns = close_wide.pct_change().fillna(0.0)
    resolved_benchmark = None
    if benchmark_returns is not None:
        resolved_benchmark = pd.Series(benchmark_returns).copy()
        resolved_benchmark.index = pd.to_datetime(resolved_benchmark.index)
        resolved_benchmark = resolved_benchmark.sort_index().reindex(asset_returns.index)
        resolved_benchmark.name = "benchmark_return"

    dates = pd.DatetimeIndex(sorted(close_wide.index.unique()))
    research_dates, holdout_dates = _split_research_and_holdout(dates, holdout_size)

    folds = generate_folds(research_dates, train_size=train_size, test_size=test_size)
    if not folds:
        raise ValueError("No folds generated; adjust train/test/holdout sizes")

    fold_rows = []
    equity_frames = []
    oos_strategy_returns = []
    oos_benchmark_returns = []

    for fold in folds:
        weights, train_ic = estimate_train_factor_weights(
            factor_matrix=factor_matrix,
            forward_returns=fwd_returns_1d,
            train_start=fold.train_start,
            train_end=fold.train_end,
        )

        composite = factor_matrix.mul(weights, axis=1).sum(axis=1)
        bt, benchmark = _run_period_backtest(
            composite_scores=composite,
            asset_returns=asset_returns,
            prices=close_wide,
            volumes=volume_wide,
            period_start=fold.test_start,
            period_end=fold.test_end,
            top_n=top_n,
            rebalance_frequency=rebalance_frequency,
            transaction_cost_bps=transaction_cost_bps,
            initial_capital=initial_capital,
            liquidity_cost_model=liquidity_cost_model,
            benchmark_returns=resolved_benchmark,
            weighting_mode=weighting_mode,
            score_temperature=score_temperature,
            max_single_weight=max_single_weight,
        )

        metrics = compute_performance_metrics(
            bt,
            benchmark_returns=benchmark,
            bootstrap_samples=bootstrap_samples,
            bootstrap_ci_level=bootstrap_ci_level,
            bootstrap_seed=bootstrap_seed + fold.fold_id,
        )
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
        oos_strategy_returns.append(bt["portfolio_return"])
        oos_benchmark_returns.append(benchmark)

    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold_id")
    equity_curve = pd.concat(equity_frames).sort_index()

    aggregate = {
        "mean_fold_sharpe": float(fold_metrics["sharpe"].mean()),
        "mean_fold_total_return": float(fold_metrics["total_return"].mean()),
        "mean_fold_alpha_annual": float(fold_metrics["alpha_annual"].mean()),
        "mean_fold_information_ratio": float(fold_metrics["information_ratio"].mean()),
        "n_folds": int(len(fold_metrics)),
        "benchmark_source": benchmark_source,
        "position_sizing_mode": weighting_mode,
        "position_sizing_score_temperature": float(score_temperature),
        "position_sizing_max_single_weight": float(max_single_weight),
    }
    if benchmark_metadata is not None:
        for key in (
            "requested_mode",
            "requested_ticker",
            "coverage_ratio",
            "non_null_days",
            "total_days",
            "fallback_reason",
        ):
            aggregate[f"benchmark_{key}"] = benchmark_metadata.get(key)

    oos_strategy = pd.concat(oos_strategy_returns).sort_index()
    oos_benchmark = pd.concat(oos_benchmark_returns).sort_index()
    oos_attr = compute_benchmark_attribution(oos_strategy, oos_benchmark)
    aggregate.update({f"oos_{k}": v for k, v in oos_attr.items()})
    if bootstrap_samples > 0:
        oos_ci = bootstrap_benchmark_attribution_ci(
            strategy_returns=oos_strategy,
            benchmark_returns=oos_benchmark,
            n_bootstrap=bootstrap_samples,
            ci_level=bootstrap_ci_level,
            random_state=bootstrap_seed,
        )
        aggregate.update({f"oos_{k}": v for k, v in oos_ci.items()})

    holdout_metrics = None
    holdout_equity_curve = None

    if evaluate_holdout and len(holdout_dates) > 0:
        holdout_start = holdout_dates[0]
        holdout_end = holdout_dates[-1]
        train_start = research_dates[0]
        train_end = holdout_start

        final_weights, train_ic = estimate_train_factor_weights(
            factor_matrix=factor_matrix,
            forward_returns=fwd_returns_1d,
            train_start=train_start,
            train_end=train_end,
        )
        final_composite = factor_matrix.mul(final_weights, axis=1).sum(axis=1)
        holdout_bt, holdout_benchmark = _run_period_backtest(
            composite_scores=final_composite,
            asset_returns=asset_returns,
            prices=close_wide,
            volumes=volume_wide,
            period_start=holdout_start,
            period_end=holdout_end,
            top_n=top_n,
            rebalance_frequency=rebalance_frequency,
            transaction_cost_bps=transaction_cost_bps,
            initial_capital=initial_capital,
            liquidity_cost_model=liquidity_cost_model,
            benchmark_returns=resolved_benchmark,
            weighting_mode=weighting_mode,
            score_temperature=score_temperature,
            max_single_weight=max_single_weight,
        )
        holdout_metrics = compute_performance_metrics(
            holdout_bt,
            benchmark_returns=holdout_benchmark,
            bootstrap_samples=bootstrap_samples,
            bootstrap_ci_level=bootstrap_ci_level,
            bootstrap_seed=bootstrap_seed + 999,
        )
        holdout_metrics["train_start"] = str(train_start.date())
        holdout_metrics["train_end"] = str(research_dates[-1].date())
        holdout_metrics["holdout_start"] = str(holdout_start.date())
        holdout_metrics["holdout_end"] = str(holdout_end.date())
        holdout_metrics["train_ic_mean"] = float(np.mean(list(train_ic.values())))
        holdout_metrics["benchmark_source"] = benchmark_source
        holdout_metrics["position_sizing_mode"] = weighting_mode

        holdout_equity_curve = holdout_bt
        aggregate["holdout_total_return"] = float(holdout_metrics["total_return"])
        aggregate["holdout_sharpe"] = float(holdout_metrics["sharpe"])
        aggregate["holdout_alpha_annual"] = float(holdout_metrics.get("alpha_annual", 0.0))
        aggregate["holdout_information_ratio"] = float(holdout_metrics.get("information_ratio", 0.0))
        aggregate["holdout_start"] = str(holdout_start.date())
        aggregate["holdout_end"] = str(holdout_end.date())

    return {
        "fold_metrics": fold_metrics,
        "equity_curve": equity_curve,
        "aggregate": aggregate,
        "holdout_metrics": holdout_metrics,
        "holdout_equity_curve": holdout_equity_curve,
        "split": {
            "research_start": str(research_dates[0].date()),
            "research_end": str(research_dates[-1].date()),
            "holdout_start": str(holdout_dates[0].date()) if len(holdout_dates) > 0 else None,
            "holdout_end": str(holdout_dates[-1].date()) if len(holdout_dates) > 0 else None,
            "holdout_size": int(holdout_size),
        },
    }
