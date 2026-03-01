import pytest
import pandas as pd
import numpy as np
from src.optimization.parameter_optimizer import ParameterOptimizer
from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor


def _make_demo_data(n_days=400, n_tickers=5):
    np.random.seed(42)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    prices = np.cumprod(1 + np.random.randn(len(idx)) * 0.01) * 100
    data = pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1_000, 10_000, len(idx)).astype(float),
    }, index=idx)
    close = data["close"].unstack("ticker")
    return data, close


def test_parameter_optimizer_runs():
    data, prices = _make_demo_data(n_days=400)
    factors = [
        MomentumFactor(lookback=63),
        MeanReversionFactor(lookback=21),
        VolatilityFactor(window=42),
    ]
    optimizer = ParameterOptimizer(
        factors=factors,
        data=data,
        prices=prices,
        n_trials=2,
        train_size=150,
        test_size=50,
    )
    best_params = optimizer.optimize()
    assert isinstance(best_params, dict)
    for key in ["ic_window", "decay_halflife", "min_weight", "normalization_method", "forward_period"]:
        assert key in best_params, f"Missing key: {key}"


def test_parameter_optimizer_get_study():
    data, prices = _make_demo_data(n_days=400)
    factors = [MomentumFactor(lookback=63), MeanReversionFactor(lookback=21)]
    optimizer = ParameterOptimizer(
        factors=factors, data=data, prices=prices,
        n_trials=1, train_size=150, test_size=50,
    )
    optimizer.optimize()
    import optuna
    study = optimizer.get_study()
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 1
