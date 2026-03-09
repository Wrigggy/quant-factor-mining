"""Portfolio constraints and helper checks."""

from __future__ import annotations

import pandas as pd


def enforce_long_only(weights: pd.Series) -> pd.Series:
    """Project weights to long-only simplex by clipping and renormalizing."""
    w = weights.clip(lower=0)
    total = w.sum()
    if total <= 0:
        return pd.Series(1.0 / len(w), index=w.index)
    return w / total


def cap_position_size(weights: pd.Series, max_weight: float) -> pd.Series:
    """Apply naive cap and renormalize; deterministic and transparent."""
    capped = weights.clip(upper=max_weight)
    total = capped.sum()
    if total <= 0:
        return pd.Series(1.0 / len(capped), index=capped.index)
    return capped / total
