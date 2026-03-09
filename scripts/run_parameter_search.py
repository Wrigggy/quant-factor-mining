#!/usr/bin/env python3
"""Run deterministic grid search over factor parameter combinations."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qfm.backtest.costs import LiquidityCostModel
from qfm.data.fetch import DEFAULT_SNAPSHOT_PATH, generate_synthetic_market_data, load_snapshot, save_snapshot
from qfm.data.preprocess import preprocess_market_data
from qfm.modeling.parameter_search import DEFAULT_SPACE, grid_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic factor parameter grid search")
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

    clean, _ = preprocess_market_data(data)
    research_cfg = cfg.get("research", {})
    factor_cfg = cfg.get("factors", {})
    liquidity_model = LiquidityCostModel.from_dict(research_cfg.get("liquidity_model", {}))

    res = grid_search(
        market_data=clean,
        search_space=DEFAULT_SPACE,
        train_size=int(research_cfg.get("train_size", 504)),
        test_size=int(research_cfg.get("test_size", 126)),
        top_n=int(research_cfg.get("top_n", 5)),
        rebalance_frequency=int(research_cfg.get("rebalance_frequency", 21)),
        transaction_cost_bps=float(research_cfg.get("transaction_cost_bps", 10.0)),
        initial_capital=float(research_cfg.get("initial_capital", 1_000_000)),
        holdout_size=int(research_cfg.get("holdout_size", 0)),
        evaluate_holdout=False,
        liquidity_cost_model=liquidity_model,
        momentum_skip=int(factor_cfg.get("momentum_skip", 21)),
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_gridsearch"
    out_dir = Path(cfg.get("run", {}).get("output_root", "artifacts/runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "grid_search_results.csv"
    res.to_csv(out_path, index=False)

    print(f"Grid search completed: {out_path}")
    print("Top 5 combinations:")
    print(res.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
