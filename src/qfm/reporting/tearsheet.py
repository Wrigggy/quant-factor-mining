"""Generate markdown report artifacts for walk-forward runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_markdown_report(
    run_id: str,
    aggregate: dict,
    fold_metrics: pd.DataFrame,
    holdout_metrics: dict | None = None,
    selected_params: dict | None = None,
    notes: str = "",
) -> str:
    """Return markdown report text for one run."""
    lines = [
        f"# Quant Factor Walk-Forward Report ({run_id})",
        "",
        "## Aggregate Metrics",
        "",
        f"- Number of folds: {aggregate.get('n_folds', 'N/A')}",
        f"- Mean fold Sharpe: {aggregate.get('mean_fold_sharpe', float('nan')):.4f}",
        f"- Mean fold total return: {aggregate.get('mean_fold_total_return', float('nan')):.4f}",
        f"- Mean fold alpha (annual): {aggregate.get('mean_fold_alpha_annual', float('nan')):.4f}",
        f"- Mean fold information ratio: {aggregate.get('mean_fold_information_ratio', float('nan')):.4f}",
        f"- OOS alpha (annual, concatenated folds): {aggregate.get('oos_alpha_annual', float('nan')):.4f}",
        f"- OOS information ratio (concatenated folds): {aggregate.get('oos_information_ratio', float('nan')):.4f}",
    ]
    if "benchmark_source" in aggregate:
        lines.append(f"- Benchmark source: {aggregate.get('benchmark_source')}")
    if "benchmark_coverage_ratio" in aggregate:
        coverage = float(aggregate.get("benchmark_coverage_ratio", 0.0))
        non_null = aggregate.get("benchmark_non_null_days", "N/A")
        total_days = aggregate.get("benchmark_total_days", "N/A")
        lines.append(f"- Benchmark coverage: {coverage:.2%} ({non_null}/{total_days} days)")
    if aggregate.get("benchmark_fallback_reason"):
        lines.append(f"- Benchmark fallback reason: {aggregate.get('benchmark_fallback_reason')}")
    if "holdout_gate_pass" in aggregate:
        gate_status = "PASS" if aggregate.get("holdout_gate_pass") else "FAIL"
        gate_reason = aggregate.get("holdout_gate_reason", "n/a")
        lines.append(f"- Holdout gate: {gate_status} ({gate_reason})")

    lines += [
        "",
        "## Fold Metrics",
        "",
    ]

    if fold_metrics.empty:
        lines.append("No fold metrics available.")
    else:
        lines.append("```text")
        lines.append(fold_metrics.to_string(index=False))
        lines.append("```")

    if selected_params:
        lines += [
            "",
            "## Selected Parameters",
            "",
            "```json",
            json.dumps(selected_params, indent=2),
            "```",
        ]

    if holdout_metrics:
        lines += [
            "",
            "## Holdout Metrics",
            "",
            "```json",
            json.dumps(holdout_metrics, indent=2),
            "```",
        ]

    if notes:
        lines += ["", "## Notes", "", notes]

    lines += [
        "",
        "## Methodology Highlights",
        "",
        "- Walk-forward split with train/test separation.",
        "- Optional untouched holdout window after parameter selection.",
        "- No same-day execution: signal at t is applied from t+1.",
        "- Transaction costs include linear and optional liquidity/slippage-aware model.",
        "- Benchmark-relative attribution includes alpha, beta, tracking error, and information ratio.",
        "- Optional bootstrap confidence intervals for alpha and information ratio.",
    ]

    lines += [
        "",
        "## Equation Map (Condensed)",
        "",
        "```text",
        "OHLCV panel -> factor signals -> cross-sectional z-score -> train IC weights",
        "-> composite score S(i,t) = sum_j w_j * z_j(i,t)",
        "-> rank S(i,t), select top-N, equal weight portfolio",
        "-> execute at close(t), active from t+1",
        "-> strategy returns + benchmark returns -> attribution (alpha/beta/TE/IR)",
        "```",
        "",
        "Full math and intuition: `docs/model_equations.md`",
    ]

    return "\n".join(lines) + "\n"


def write_report(path: Path, content: str) -> None:
    """Write markdown report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
