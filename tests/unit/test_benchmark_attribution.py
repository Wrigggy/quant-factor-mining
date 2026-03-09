from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.backtest.metrics import compute_benchmark_attribution


def test_benchmark_attribution_outputs_expected_keys():
    idx = pd.bdate_range("2024-01-01", periods=30)
    strategy = pd.Series(np.linspace(0.001, 0.002, len(idx)), index=idx)
    benchmark = pd.Series(np.linspace(0.0008, 0.0018, len(idx)), index=idx)

    metrics = compute_benchmark_attribution(strategy, benchmark)

    required = {
        "benchmark_total_return",
        "benchmark_annual_return",
        "excess_total_return",
        "alpha_annual",
        "beta",
        "tracking_error_annual",
        "information_ratio",
    }
    assert required.issubset(metrics.keys())
    assert np.isfinite(metrics["beta"])
