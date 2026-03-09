"""
Parameter optimizer for AdaptiveCompositeFactor using Optuna (Issue #9).

Walk-forward objective: maximize mean ICIR across folds.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import optuna
from loguru import logger

from src.factors.base import BaseFactor
from src.factors.adaptive_composite import AdaptiveCompositeFactor, AdaptiveCompositeManager

optuna.logging.set_verbosity(optuna.logging.WARNING)

_BEST_PARAMS_PATH = Path(__file__).parent.parent.parent / "outputs" / "optimization" / "best_params.json"


class ParameterOptimizer:
    """
    Hyper-parameter optimizer for AdaptiveCompositeFactor.

    Uses Optuna TPE to maximize mean ICIR across walk-forward folds.

    Parameters
    ----------
    factors : list of BaseFactor
        Instantiated base factors (momentum, reversion, vol, etc.).
    data : pd.DataFrame
        Full history OHLCV data with MultiIndex (date, ticker).
    prices : pd.DataFrame
        Close prices unstacked to wide format (date index, ticker columns).
        Pass ``data['close'].unstack('ticker')`` if you only have the combined frame.
    n_trials : int, default 100
        Number of Optuna trials.
    train_size : int, default 252
        Number of trading days in each training window.
    test_size : int, default 63
        Number of trading days in each test window (rolled forward).
    """

    def __init__(
        self,
        factors: List[BaseFactor],
        data: pd.DataFrame,
        prices: pd.DataFrame,
        n_trials: int = 100,
        train_size: int = 252,
        test_size: int = 63,
    ):
        self.factors = factors
        self.data = data
        self.prices = prices
        self.n_trials = n_trials
        self.train_size = train_size
        self.test_size = test_size

        self._study: Optional[optuna.Study] = None

        # Build a sorted list of unique dates once
        self._dates = sorted(data.index.get_level_values("date").unique())
        logger.info(
            f"ParameterOptimizer: {len(factors)} factors, {len(self._dates)} dates, "
            f"train={train_size} test={test_size} n_trials={n_trials}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> Dict:
        """Run the Optuna study and return the best params dict."""
        self._study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        self._study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=False)

        best = self._study.best_params
        logger.info(f"Best params: {best}  (ICIR={self._study.best_value:.4f})")

        _BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_BEST_PARAMS_PATH, "w") as f:
            json.dump(best, f, indent=2)
        logger.info(f"Best params saved to {_BEST_PARAMS_PATH}")

        return best

    def get_study(self) -> optuna.Study:
        """Return the full Optuna study for inspection (call after optimize())."""
        if self._study is None:
            raise RuntimeError("Call optimize() first.")
        return self._study

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _objective(self, trial: optuna.Trial) -> float:
        """Walk-forward mean ICIR objective for one Optuna trial."""
        ic_window = trial.suggest_int("ic_window", 21, 126)
        decay_halflife = trial.suggest_int("decay_halflife", 5, 63)
        min_weight = trial.suggest_float("min_weight", 0.01, 0.20)
        normalization_method = trial.suggest_categorical(
            "normalization_method", ["softmax", "relu", "rank"]
        )
        forward_period = trial.suggest_int("forward_period", 5, 42)

        fold_icirs = []
        n_dates = len(self._dates)

        start = 0
        while start + self.train_size + self.test_size <= n_dates:
            fold_end = start + self.train_size + self.test_size
            window_dates = self._dates[start:fold_end]
            test_start_idx = self.train_size

            fold_icir = self._evaluate_fold(
                window_dates=window_dates,
                test_start_idx=test_start_idx,
                ic_window=ic_window,
                decay_halflife=decay_halflife,
                min_weight=min_weight,
                normalization_method=normalization_method,
                forward_period=forward_period,
            )
            if not math.isnan(fold_icir):
                fold_icirs.append(fold_icir)

            start += self.test_size  # roll forward by one test window

        if not fold_icirs:
            return float("-inf")
        return float(np.mean(fold_icirs))

    def _evaluate_fold(
        self,
        window_dates,
        test_start_idx: int,
        ic_window: int,
        decay_halflife: int,
        min_weight: float,
        normalization_method: str,
        forward_period: int,
    ) -> float:
        """
        Run one walk-forward fold and return the ICIR on the test window.

        Spins up a fresh AdaptiveCompositeFactor + Manager, iterates all dates
        in the window (train then test), and computes ICIR from the IC values
        accumulated during the test portion.
        """
        af = AdaptiveCompositeFactor(
            factors=self.factors,
            ic_window=ic_window,
            decay_halflife=decay_halflife,
            min_weight=min_weight,
            normalization_method=normalization_method,
        )
        manager = AdaptiveCompositeManager(af, forward_period=forward_period)

        test_ic_accumulator: Dict[str, List[float]] = {f.name: [] for f in self.factors}

        for i, date in enumerate(window_dates):
            date = pd.Timestamp(date)

            # Slice data up to and including current date
            date_mask = self.data.index.get_level_values("date") <= date
            data_slice = self.data[date_mask]

            # prices is wide-format DataFrame (date index × ticker columns)
            price_slice = self.prices[self.prices.index <= date]
            if price_slice.empty:
                continue
            prices_today = price_slice.iloc[-1]

            try:
                manager.compute_factor_with_update(data_slice, prices_today, date)
            except Exception:
                continue

            # Collect IC values recorded for test window dates
            if i >= test_start_idx:
                for factor_name, ic_list in af.ic_history.items():
                    if ic_list and pd.Timestamp(ic_list[-1][0]) == date:
                        test_ic_accumulator[factor_name].append(ic_list[-1][1])

        # Compute mean ICIR across all factors for this fold
        factor_icirs = []
        for factor_name, ic_vals in test_ic_accumulator.items():
            if len(ic_vals) >= 2:
                arr = np.array(ic_vals)
                std = arr.std()
                if std > 1e-9:
                    factor_icirs.append(arr.mean() / std)

        if not factor_icirs:
            return float("nan")
        return float(np.mean(factor_icirs))
