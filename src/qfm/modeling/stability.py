"""Stability diagnostics and candidate selection helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def summarize_fold_stability(fold_metrics: pd.DataFrame) -> Dict[str, float]:
    """Return dispersion-style fold diagnostics for robustness checks."""
    if fold_metrics.empty:
        return {
            "fold_sharpe_std": float("nan"),
            "positive_sharpe_ratio": float("nan"),
            "worst_fold_max_drawdown": float("nan"),
        }

    sharpe = pd.to_numeric(fold_metrics.get("sharpe"), errors="coerce").dropna()
    max_drawdown = pd.to_numeric(fold_metrics.get("max_drawdown"), errors="coerce").dropna()

    return {
        "fold_sharpe_std": float(sharpe.std(ddof=0)) if not sharpe.empty else float("nan"),
        "positive_sharpe_ratio": float((sharpe > 0).mean()) if not sharpe.empty else float("nan"),
        "worst_fold_max_drawdown": float(max_drawdown.min()) if not max_drawdown.empty else float("nan"),
    }


def evaluate_holdout_gate(
    holdout_sharpe: float | None,
    holdout_excess_total_return: float | None,
    min_holdout_sharpe: float = 0.0,
    min_holdout_excess_total_return: float = 0.0,
) -> Tuple[bool, str]:
    """Check strict holdout gate and return (pass, reason)."""
    if holdout_sharpe is None or holdout_excess_total_return is None:
        return False, "missing_holdout_metrics"

    if np.isnan(holdout_sharpe) or np.isnan(holdout_excess_total_return):
        return False, "missing_holdout_metrics"

    if holdout_sharpe <= min_holdout_sharpe:
        return False, f"holdout_sharpe_not_above_{min_holdout_sharpe:.4f}"

    if holdout_excess_total_return <= min_holdout_excess_total_return:
        return False, f"holdout_excess_not_above_{min_holdout_excess_total_return:.4f}"

    return True, "pass"


def apply_holdout_gate(
    candidates: pd.DataFrame,
    min_holdout_sharpe: float = 0.0,
    min_holdout_excess_total_return: float = 0.0,
) -> pd.DataFrame:
    """Append holdout gate status columns to candidate table."""
    out = candidates.copy()
    gate_pass = []
    gate_reason = []

    if "holdout_sharpe" not in out.columns or "holdout_excess_total_return" not in out.columns:
        out["gate_pass"] = False
        out["gate_reason"] = "missing_holdout_metrics"
        return out

    for row in out.itertuples(index=False):
        passed, reason = evaluate_holdout_gate(
            holdout_sharpe=float(getattr(row, "holdout_sharpe")) if pd.notna(getattr(row, "holdout_sharpe")) else None,
            holdout_excess_total_return=(
                float(getattr(row, "holdout_excess_total_return"))
                if pd.notna(getattr(row, "holdout_excess_total_return"))
                else None
            ),
            min_holdout_sharpe=min_holdout_sharpe,
            min_holdout_excess_total_return=min_holdout_excess_total_return,
        )
        gate_pass.append(passed)
        gate_reason.append(reason)

    out["gate_pass"] = gate_pass
    out["gate_reason"] = gate_reason
    return out


def rank_candidates(
    candidates: pd.DataFrame,
    selection_mode: str = "mean_fold_sharpe",
    selection_metric: str = "mean_fold_sharpe",
) -> pd.DataFrame:
    """Return ranked candidate table with deterministic ordering."""
    out = candidates.copy()

    if out.empty:
        out["selection_rank"] = pd.Series(dtype=int)
        return out

    if selection_mode == "stability_first":
        if "gate_pass" not in out.columns:
            out = apply_holdout_gate(out)

        sort_cols = [
            "gate_pass",
            "mean_fold_sharpe",
            "positive_sharpe_ratio",
            "fold_sharpe_std",
            "worst_fold_max_drawdown",
            "holdout_sharpe",
            "holdout_excess_total_return",
        ]
        for col in sort_cols:
            if col not in out.columns:
                out[col] = np.nan

        out = out.sort_values(
            by=sort_cols,
            ascending=[False, False, False, True, False, False, False],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
        out["selection_rank"] = np.arange(1, len(out) + 1, dtype=int)
        return out

    if selection_metric not in out.columns:
        raise ValueError(f"Selection metric not found: {selection_metric}")

    tie_breaker = "mean_fold_total_return" if "mean_fold_total_return" in out.columns else selection_metric
    out = out.sort_values(
        by=[selection_metric, tie_breaker],
        ascending=[False, False],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1, dtype=int)
    return out

