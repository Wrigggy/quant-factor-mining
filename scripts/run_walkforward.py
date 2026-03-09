#!/usr/bin/env python3
"""Run the full walk-forward research pipeline and persist artifacts."""

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
from qfm.data.fetch import (
    DEFAULT_SNAPSHOT_PATH,
    fetch_live_yfinance,
    generate_synthetic_market_data,
    load_snapshot,
    save_snapshot,
)
from qfm.data.preprocess import preprocess_market_data, summarize_market_data
from qfm.modeling.parameter_search import DEFAULT_SPACE, SearchSpace, build_factor_set, grid_search
from qfm.modeling.walkforward import run_walkforward_research
from qfm.reporting.tearsheet import build_markdown_report, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward factor research")
    parser.add_argument("--config", default="configs/strategy/default.yaml", help="Path to YAML config")
    parser.add_argument("--live", action="store_true", help="Use live yfinance data fetch for this run")
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _get_data(cfg: dict, force_live: bool):
    data_cfg = cfg.get("data", {})
    snapshot_path = Path(data_cfg.get("snapshot_path", str(DEFAULT_SNAPSHOT_PATH)))

    if force_live:
        data = fetch_live_yfinance(
            tickers=data_cfg.get("tickers", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]),
            start=data_cfg.get("start_date", "2018-01-01"),
            end=data_cfg.get("end_date", "2025-12-31"),
        )
        save_snapshot(data, snapshot_path)
        return data

    if snapshot_path.exists():
        return load_snapshot(snapshot_path)

    # Offline-first fallback: generate deterministic synthetic snapshot.
    data = generate_synthetic_market_data(
        start=data_cfg.get("start_date", "2018-01-01"),
        end=data_cfg.get("end_date", "2024-12-31"),
        tickers=data_cfg.get("tickers"),
        seed=int(data_cfg.get("seed", 42)),
    )
    save_snapshot(data, snapshot_path)
    return data


def _resolve_search_space(cfg: dict) -> SearchSpace:
    research_cfg = cfg.get("research", {})
    nested_cfg = research_cfg.get("nested_search", {})
    raw_space = nested_cfg.get("space", {})
    return SearchSpace(
        momentum_lookback=tuple(raw_space.get("momentum_lookback", DEFAULT_SPACE.momentum_lookback)),
        mean_reversion_lookback=tuple(
            raw_space.get("mean_reversion_lookback", DEFAULT_SPACE.mean_reversion_lookback)
        ),
        volatility_window=tuple(raw_space.get("volatility_window", DEFAULT_SPACE.volatility_window)),
    )


def _build_default_factors(cfg: dict):
    factor_cfg = cfg.get("factors", {})
    return build_factor_set(
        momentum_lookback=int(factor_cfg.get("momentum_lookback", 252)),
        mean_reversion_lookback=int(factor_cfg.get("mean_reversion_lookback", 21)),
        volatility_window=int(factor_cfg.get("volatility_window", 63)),
        momentum_skip=int(factor_cfg.get("momentum_skip", 21)),
    )


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    run_cfg = cfg.get("run", {})
    research_cfg = cfg.get("research", {})
    factor_cfg = cfg.get("factors", {})

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_cfg.get("output_root", "artifacts/runs")) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    data = _get_data(cfg, force_live=args.live)
    clean_data, preprocess_report = preprocess_market_data(
        data,
        max_ffill_days=int(cfg.get("data", {}).get("max_ffill_days", 3)),
    )

    liquidity_model = LiquidityCostModel.from_dict(research_cfg.get("liquidity_model", {}))
    holdout_size = int(research_cfg.get("holdout_size", 0))
    bootstrap_samples = int(research_cfg.get("bootstrap_samples", 0))
    bootstrap_ci_level = float(research_cfg.get("bootstrap_ci_level", 0.95))
    bootstrap_seed = int(research_cfg.get("bootstrap_seed", 42))

    nested_cfg = research_cfg.get("nested_search", {})
    nested_enabled = bool(nested_cfg.get("enabled", False))
    selection_metric = str(nested_cfg.get("selection_metric", "mean_fold_sharpe"))

    selected_params = None
    if nested_enabled:
        search_space = _resolve_search_space(cfg)
        search_results = grid_search(
            market_data=clean_data,
            search_space=search_space,
            train_size=int(research_cfg.get("train_size", 504)),
            test_size=int(research_cfg.get("test_size", 126)),
            top_n=int(research_cfg.get("top_n", 20)),
            rebalance_frequency=int(research_cfg.get("rebalance_frequency", 21)),
            transaction_cost_bps=float(research_cfg.get("transaction_cost_bps", 10.0)),
            initial_capital=float(research_cfg.get("initial_capital", 1_000_000)),
            holdout_size=holdout_size,
            evaluate_holdout=False,
            liquidity_cost_model=liquidity_model,
            momentum_skip=int(factor_cfg.get("momentum_skip", 21)),
        )
        if selection_metric not in search_results.columns:
            raise ValueError(f"Invalid selection metric: {selection_metric}")

        search_results = search_results.sort_values(
            [selection_metric, "mean_fold_total_return"],
            ascending=False,
        ).reset_index(drop=True)
        best = search_results.iloc[0]
        selected_params = {
            "momentum_lookback": int(best["momentum_lookback"]),
            "mean_reversion_lookback": int(best["mean_reversion_lookback"]),
            "volatility_window": int(best["volatility_window"]),
            "momentum_skip": int(factor_cfg.get("momentum_skip", 21)),
            "selection_metric": selection_metric,
        }
        factors = build_factor_set(
            momentum_lookback=selected_params["momentum_lookback"],
            mean_reversion_lookback=selected_params["mean_reversion_lookback"],
            volatility_window=selected_params["volatility_window"],
            momentum_skip=selected_params["momentum_skip"],
        )

        search_results.to_csv(run_dir / "nested_search_results.csv", index=False)
        with open(run_dir / "selected_params.json", "w", encoding="utf-8") as handle:
            json.dump(selected_params, handle, indent=2)
    else:
        factors = _build_default_factors(cfg)

    result = run_walkforward_research(
        market_data=clean_data,
        factors=factors,
        train_size=int(research_cfg.get("train_size", 504)),
        test_size=int(research_cfg.get("test_size", 126)),
        top_n=int(research_cfg.get("top_n", 20)),
        rebalance_frequency=int(research_cfg.get("rebalance_frequency", 21)),
        transaction_cost_bps=float(research_cfg.get("transaction_cost_bps", 10.0)),
        initial_capital=float(research_cfg.get("initial_capital", 1_000_000)),
        holdout_size=holdout_size,
        evaluate_holdout=holdout_size > 0,
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci_level=bootstrap_ci_level,
        bootstrap_seed=bootstrap_seed,
        liquidity_cost_model=liquidity_model,
    )

    fold_metrics = result["fold_metrics"]
    equity_curve = result["equity_curve"]
    aggregate = result["aggregate"]
    holdout_metrics = result.get("holdout_metrics")
    holdout_equity_curve = result.get("holdout_equity_curve")

    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    equity_curve.to_parquet(run_dir / "equity_curve.parquet")

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    if holdout_metrics is not None:
        with open(run_dir / "holdout_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(holdout_metrics, handle, indent=2)
    if holdout_equity_curve is not None:
        holdout_equity_curve.to_parquet(run_dir / "holdout_equity_curve.parquet")

    run_snapshot = {
        "config": cfg,
        "selected_params": selected_params,
        "split": result.get("split"),
        "data_summary": summarize_market_data(clean_data),
        "preprocess_report": preprocess_report.__dict__,
        "run_id": run_id,
    }
    with open(run_dir / "run_snapshot.json", "w", encoding="utf-8") as handle:
        json.dump(run_snapshot, handle, indent=2)

    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    report = build_markdown_report(
        run_id=run_id,
        aggregate=aggregate,
        fold_metrics=fold_metrics,
        holdout_metrics=holdout_metrics,
        selected_params=selected_params,
        notes="Generated by scripts/run_walkforward.py",
    )
    write_report(run_dir / "report.md", report)

    print(f"Run completed: {run_dir}")
    if selected_params is not None:
        print(f"Selected params: {selected_params}")
    print(f"Mean fold Sharpe: {aggregate['mean_fold_sharpe']:.4f}")
    print(f"Mean fold total return: {aggregate['mean_fold_total_return']:.4f}")
    print(f"Mean fold alpha (annual): {aggregate['mean_fold_alpha_annual']:.4f}")
    print(f"Mean fold information ratio: {aggregate['mean_fold_information_ratio']:.4f}")
    if "holdout_total_return" in aggregate:
        print(f"Holdout total return: {aggregate['holdout_total_return']:.4f}")
        print(f"Holdout Sharpe: {aggregate['holdout_sharpe']:.4f}")


if __name__ == "__main__":
    main()
