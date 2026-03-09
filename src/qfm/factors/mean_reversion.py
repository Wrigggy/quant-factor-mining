"""Mean reversion factor implementations."""

from __future__ import annotations

import pandas as pd

from .base import BaseFactor, safe_stack


class MeanReversionFactor(BaseFactor):
    """Short-term reversal: negative recent return over lookback window."""

    def __init__(self, lookback: int = 21, name: str = "mean_reversion"):
        super().__init__(name=name)
        self.lookback = lookback

    def compute_raw(self, market_data: pd.DataFrame) -> pd.Series:
        close = market_data["close"].unstack("ticker")
        reversion = -(close / close.shift(self.lookback) - 1)
        return safe_stack(reversion).rename(self.name)
