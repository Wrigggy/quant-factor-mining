#!/usr/bin/env python3
"""Run deterministic grid search over factor parameter combinations."""

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
from qfm.data.fetch import DEFAULT_SNAPSHOT_PATH, generate_synthetic_market_data, load_snapshot, save_snapshot
from qfm.data.preprocess import preprocess_market_data
from qfm.modeling.parameter_search import DEFAULT_SPACE, SearchSpace, grid_search
from qfm.modeling.stability import apply_holdout_gate, rank_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic factor parameter grid search")
    parser.add_argument("--config", default="configs/strategy/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Evaluate holdout metrics for each parameter candidate",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["mean_fold_sharpe", "stability_first"],
        default=None,
        help="Candidate ranking mode (default from config or mean_fold_sharpe)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top rows to print")
    parser.add_argument(
        "--min-holdout-sharpe",
        type=float,
        default=None,
        help="Override holdout gate threshold: Sharpe must be above this value",
    )
    parser.add_argument(
        "--min-holdout-excess",
        type=float,
        default=None,
        help="Override holdout gate threshold: excess total return must be above this value",
    )
    return parser.parse_args()


def _resolve_search_space(cfg: dict) -> SearchSpace:
    nested_cfg = cfg.get("research", {}).get("nested_search", {})
    raw_space = nested_cfg.get("space", {})
    return SearchSpace(
        momentum_lookback=tuple(raw_space.get("momentum_lookback", DEFAULT_SPACE.momentum_lookback)),
        mean_reversion_lookback=tuple(
            raw_space.get("mean_reversion_lookback", DEFAULT_SPACE.mean_reversion_lookback)
        ),
        volatility_window=tuple(raw_space.get("volatility_window", DEFAULT_SPACE.volatility_window)),
    )


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
    nested_cfg = research_cfg.get("nested_search", {})
    factor_cfg = cfg.get("factors", {})
    liquidity_model = LiquidityCostModel.from_dict(research_cfg.get("liquidity_model", {}))
    holdout_gate_cfg = research_cfg.get("holdout_gate", {})
    position_cfg = research_cfg.get("position_sizing", {})

    selection_mode = args.selection_mode or str(nested_cfg.get("selection_mode", "mean_fold_sharpe"))
    selection_metric = str(nested_cfg.get("selection_metric", "mean_fold_sharpe"))
    include_holdout = bool(args.include_holdout or selection_mode == "stability_first")
    min_holdout_sharpe = (
        float(args.min_holdout_sharpe)
        if args.min_holdout_sharpe is not None
        else float(holdout_gate_cfg.get("min_sharpe", 0.0))
    )
    min_holdout_excess = (
        float(args.min_holdout_excess)
        if args.min_holdout_excess is not None
        else float(holdout_gate_cfg.get("min_excess_total_return", 0.0))
    )
    weighting_mode = str(position_cfg.get("mode", "equal_weight"))
    score_temperature = float(position_cfg.get("score_temperature", 1.0))
    max_single_weight = float(position_cfg.get("max_single_weight", 1.0))

    res = grid_search(
        market_data=clean,
        search_space=_resolve_search_space(cfg),
        train_size=int(research_cfg.get("train_size", 504)),
        test_size=int(research_cfg.get("test_size", 126)),
        top_n=int(research_cfg.get("top_n", 5)),
        rebalance_frequency=int(research_cfg.get("rebalance_frequency", 21)),
        transaction_cost_bps=float(research_cfg.get("transaction_cost_bps", 10.0)),
        initial_capital=float(research_cfg.get("initial_capital", 1_000_000)),
        holdout_size=int(research_cfg.get("holdout_size", 0)),
        evaluate_holdout=include_holdout,
        liquidity_cost_model=liquidity_model,
        momentum_skip=int(factor_cfg.get("momentum_skip", 21)),
        weighting_mode=weighting_mode,
        score_temperature=score_temperature,
        max_single_weight=max_single_weight,
    )
    res = apply_holdout_gate(
        candidates=res,
        min_holdout_sharpe=min_holdout_sharpe,
        min_holdout_excess_total_return=min_holdout_excess,
    )
    res = rank_candidates(
        candidates=res,
        selection_mode=selection_mode,
        selection_metric=selection_metric,
    )

    if selection_mode == "stability_first" and not bool(res["gate_pass"].any()):
        raise ValueError(
            "No candidate passed holdout gate. "
            "Relax thresholds or expand search space before selecting a strategy."
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_gridsearch"
    out_dir = Path(cfg.get("run", {}).get("output_root", "artifacts/runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "grid_search_results.csv"
    scoreboard_path = out_dir / "stability_scoreboard.csv"
    summary_path = out_dir / "stability_summary.json"
    res.to_csv(out_path, index=False)
    res.to_csv(scoreboard_path, index=False)

    summary = {
        "selection_mode": selection_mode,
        "selection_metric": selection_metric,
        "include_holdout": include_holdout,
        "min_holdout_sharpe": min_holdout_sharpe,
        "min_holdout_excess_total_return": min_holdout_excess,
        "n_candidates": int(len(res)),
        "n_gate_pass": int(res["gate_pass"].sum()) if "gate_pass" in res.columns else 0,
        "top_candidate": res.iloc[0].to_dict() if not res.empty else None,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Grid search completed: {out_path}")
    print(f"Stability scoreboard: {scoreboard_path}")
    print(f"Stability summary: {summary_path}")
    print(f"Top {args.top_k} combinations:")
    print(res.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
