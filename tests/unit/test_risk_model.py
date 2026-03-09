from __future__ import annotations

from qfm.data.preprocess import close_prices_wide
from qfm.portfolio.risk_model import ewma_covariance


def test_ewma_covariance_shape(market_data):
    close = close_prices_wide(market_data)
    returns = close.pct_change().dropna()
    cov = ewma_covariance(returns, span=30)
    assert cov.shape[0] == cov.shape[1] == len(returns.columns)
    assert list(cov.index) == list(returns.columns)
    assert list(cov.columns) == list(returns.columns)
