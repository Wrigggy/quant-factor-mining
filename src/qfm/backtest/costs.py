"""Transaction cost helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


def bps_to_rate(bps: float) -> float:
    """Convert basis points to decimal rate."""
    return float(bps) / 10_000.0


def linear_transaction_cost(turnover: float, cost_bps: float) -> float:
    """Simple linear cost model as fraction of portfolio value."""
    return float(turnover) * bps_to_rate(cost_bps)


@dataclass(frozen=True)
class LiquidityCostModel:
    """Config for a liquidity/slippage-aware transaction cost model."""

    commission_bps: float = 4.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 8.0
    participation_cap: float = 0.10
    cap_breach_penalty_bps: float = 20.0
    impact_exponent: float = 0.50
    min_dollar_volume: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict | None) -> Optional["LiquidityCostModel"]:
        """Create config from mapping; return None when disabled/missing."""
        if raw is None:
            return None
        if not bool(raw.get("enabled", False)):
            return None
        return cls(
            commission_bps=float(raw.get("commission_bps", 4.0)),
            half_spread_bps=float(raw.get("half_spread_bps", 2.0)),
            slippage_bps=float(raw.get("slippage_bps", 8.0)),
            participation_cap=float(raw.get("participation_cap", 0.10)),
            cap_breach_penalty_bps=float(raw.get("cap_breach_penalty_bps", 20.0)),
            impact_exponent=float(raw.get("impact_exponent", 0.50)),
            min_dollar_volume=float(raw.get("min_dollar_volume", 1.0)),
        )


def liquidity_transaction_cost(
    trade_weights: pd.Series,
    portfolio_value: float,
    prices: pd.Series,
    volumes: pd.Series,
    model: LiquidityCostModel,
) -> float:
    """Estimate rebalance cost using turnover + spread + liquidity impact."""
    abs_trade = trade_weights.abs()
    turnover = float(abs_trade.sum())
    if turnover <= 0.0 or portfolio_value <= 0.0:
        return 0.0

    traded_assets = abs_trade[abs_trade > 0.0].index
    if len(traded_assets) == 0:
        return 0.0

    aligned = pd.DataFrame(
        {
            "dw": abs_trade.reindex(traded_assets).fillna(0.0),
            "price": prices.reindex(traded_assets),
            "volume": volumes.reindex(traded_assets),
        }
    ).dropna()
    if aligned.empty:
        # Fallback to linear commission+spread when microstructure inputs are missing.
        return linear_transaction_cost(turnover, model.commission_bps + model.half_spread_bps)

    base_rate = bps_to_rate(model.commission_bps + model.half_spread_bps)
    cap = max(model.participation_cap, 1e-9)

    cost_dollars = 0.0
    for _, row in aligned.iterrows():
        trade_notional = float(row["dw"] * portfolio_value)
        dollar_volume = max(float(row["price"] * row["volume"]), model.min_dollar_volume)
        participation = trade_notional / dollar_volume

        impact = bps_to_rate(model.slippage_bps) * (participation / cap) ** model.impact_exponent
        if participation > cap:
            breach = (participation - cap) / cap
            impact += bps_to_rate(model.cap_breach_penalty_bps) * breach

        cost_dollars += trade_notional * (base_rate + impact)

    return float(cost_dollars / portfolio_value)
