"""Forward return labeling utilities."""

from __future__ import annotations

import pandas as pd

from qfm.data.contracts import ensure_market_data
from qfm.factors.base import safe_stack


def compute_forward_returns(market_data: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Compute forward close-to-close returns at given horizon.

    Label at date t equals return from close(t) to close(t+horizon).
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    data = ensure_market_data(market_data)
    close = data["close"].unstack("ticker")
    forward = close.pct_change(horizon).shift(-horizon)
    return safe_stack(forward).rename(f"fwd_{horizon}d")
