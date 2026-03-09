from __future__ import annotations

from qfm.factors.base import cross_sectional_zscore
from qfm.factors.momentum import MomentumFactor


def test_normalization_keeps_two_level_index(market_data):
    factor = MomentumFactor(lookback=63, skip=5)
    raw = factor.compute_raw(market_data)
    normalized = cross_sectional_zscore(raw)
    assert normalized.index.nlevels == 2
    assert normalized.index.names == ["date", "ticker"]
