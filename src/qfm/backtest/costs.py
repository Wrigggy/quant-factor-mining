"""Transaction cost helpers."""

from __future__ import annotations


def bps_to_rate(bps: float) -> float:
    """Convert basis points to decimal rate."""
    return float(bps) / 10_000.0


def linear_transaction_cost(turnover: float, cost_bps: float) -> float:
    """Simple linear cost model as fraction of portfolio value."""
    return float(turnover) * bps_to_rate(cost_bps)
