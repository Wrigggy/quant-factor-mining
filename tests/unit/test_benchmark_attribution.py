from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.backtest.metrics import bootstrap_benchmark_attribution_ci, compute_benchmark_attribution


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


def test_bootstrap_ci_is_deterministic_and_has_bounds():
    idx = pd.bdate_range("2024-01-01", periods=80)
    strategy = pd.Series(np.linspace(-0.002, 0.003, len(idx)), index=idx)
    benchmark = pd.Series(np.linspace(-0.0015, 0.0025, len(idx)), index=idx)

    first = bootstrap_benchmark_attribution_ci(
        strategy,
        benchmark,
        n_bootstrap=120,
        ci_level=0.90,
        random_state=7,
    )
    second = bootstrap_benchmark_attribution_ci(
        strategy,
        benchmark,
        n_bootstrap=120,
        ci_level=0.90,
        random_state=7,
    )

    assert first == second
    assert first["alpha_annual_ci_low"] <= first["alpha_annual_ci_high"]
    assert first["information_ratio_ci_low"] <= first["information_ratio_ci_high"]
