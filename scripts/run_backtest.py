#!/usr/bin/env python3
"""Run a single full-period backtest from factor composite scores."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qfm.backtest.costs import LiquidityCostModel
from qfm.backtest.engine import run_backtest_from_scores
from qfm.backtest.metrics import compute_performance_metrics
from qfm.backtest.benchmarks import equal_weight_benchmark
from qfm.data.fetch import DEFAULT_SNAPSHOT_PATH, generate_synthetic_market_data, load_snapshot, save_snapshot
from qfm.data.preprocess import close_prices_wide, preprocess_market_data
from qfm.factors.mean_reversion import MeanReversionFactor
from qfm.factors.momentum import MomentumFactor
from qfm.factors.volatility import LowVolatilityFactor
from qfm.modeling.feature_builder import build_factor_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-period factor backtest")
    parser.add_argument("--config", default="configs/strategy/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    snapshot_path = Path(cfg.get("data", {}).get("snapshot_path", str(DEFAULT_SNAPSHOT_PATH)))
    if snapshot_path.exists():
        data = load_snapshot(snapshot_path)
    else:
        data = generate_synthetic_market_data(
            start=cfg.get("data", {}).get("start_date", "2018-01-01"),
            end=cfg.get("data", {}).get("end_date", "2024-12-31"),
            tickers=cfg.get("data", {}).get("tickers"),
            seed=int(cfg.get("data", {}).get("seed", 42)),
        )
        save_snapshot(data, snapshot_path)

    clean_data, _ = preprocess_market_data(data)

    factors = [
        MomentumFactor(
            lookback=int(cfg.get("factors", {}).get("momentum_lookback", 252)),
            skip=int(cfg.get("factors", {}).get("momentum_skip", 21)),
        ),
        MeanReversionFactor(
            lookback=int(cfg.get("factors", {}).get("mean_reversion_lookback", 21)),
        ),
        LowVolatilityFactor(
            window=int(cfg.get("factors", {}).get("volatility_window", 63)),
        ),
    ]

    matrix = build_factor_matrix(clean_data, factors, normalize=True)
    score = matrix.mean(axis=1).unstack("ticker")

    prices = close_prices_wide(clean_data)
    volumes = clean_data["volume"].unstack("ticker").reindex(prices.index).sort_index()
    asset_returns = prices.pct_change().fillna(0.0)
    research_cfg = cfg.get("research", {})
    position_cfg = research_cfg.get("position_sizing", {})
    liquidity_model = LiquidityCostModel.from_dict(research_cfg.get("liquidity_model", {}))

    bt = run_backtest_from_scores(
        scores=score,
        asset_returns=asset_returns,
        top_n=int(research_cfg.get("top_n", 20)),
        rebalance_frequency=int(research_cfg.get("rebalance_frequency", 21)),
        transaction_cost_bps=float(research_cfg.get("transaction_cost_bps", 10.0)),
        initial_capital=float(research_cfg.get("initial_capital", 1_000_000)),
        execution_prices=prices,
        execution_volumes=volumes,
        liquidity_cost_model=liquidity_model,
        weighting_mode=str(position_cfg.get("mode", "equal_weight")),
        score_temperature=float(position_cfg.get("score_temperature", 1.0)),
        max_single_weight=float(position_cfg.get("max_single_weight", 1.0)),
    )

    benchmark = equal_weight_benchmark(asset_returns).reindex(bt.index)
    metrics = compute_performance_metrics(
        bt,
        benchmark_returns=benchmark,
        bootstrap_samples=int(research_cfg.get("bootstrap_samples", 0)),
        bootstrap_ci_level=float(research_cfg.get("bootstrap_ci_level", 0.95)),
        bootstrap_seed=int(research_cfg.get("bootstrap_seed", 42)),
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_backtest"
    out_dir = Path(cfg.get("run", {}).get("output_root", "artifacts/runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    bt.to_parquet(out_dir / "equity_curve.parquet")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Backtest completed: {out_dir}")
    print(f"Total return: {metrics['total_return']:.4f}")
    print(f"Sharpe: {metrics['sharpe']:.4f}")
    print(f"Alpha (annual): {metrics['alpha_annual']:.4f}")
    print(f"Information ratio: {metrics['information_ratio']:.4f}")
    if "alpha_annual_ci_low" in metrics:
        print(
            "Alpha CI "
            f"[{metrics['alpha_annual_ci_low']:.4f}, {metrics['alpha_annual_ci_high']:.4f}]"
        )
    if "information_ratio_ci_low" in metrics:
        print(
            "IR CI "
            f"[{metrics['information_ratio_ci_low']:.4f}, {metrics['information_ratio_ci_high']:.4f}]"
        )


if __name__ == "__main__":
    main()
