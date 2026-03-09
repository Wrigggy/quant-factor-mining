"""Generate markdown report artifacts for walk-forward runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_markdown_report(
    run_id: str,
    aggregate: dict,
    fold_metrics: pd.DataFrame,
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

    if notes:
        lines += ["", "## Notes", "", notes]

    lines += [
        "",
        "## Methodology Highlights",
        "",
        "- Walk-forward split with train/test separation.",
        "- No same-day execution: signal at t is applied from t+1.",
        "- Transaction costs included via linear turnover model.",
        "- Benchmark-relative attribution includes alpha, beta, tracking error, and information ratio.",
    ]

    return "\n".join(lines) + "\n"


def write_report(path: Path, content: str) -> None:
    """Write markdown report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
