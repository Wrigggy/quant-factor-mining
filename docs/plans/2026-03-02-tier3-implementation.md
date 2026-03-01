# Tier 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement parameter optimization (Optuna), 5 explanatory Jupyter notebooks, and a Streamlit dashboard for the quant-factor-mining project.

**Architecture:** `ParameterOptimizer` wraps Optuna's TPE sampler around walk-forward ICIR; each trial spins up a fresh `AdaptiveCompositeManager` with trial params, simulates train+test windows, and returns mean ICIR across folds. Notebooks use the two cached parquet files in `data/raw/` (no live download needed). The dashboard is a pure read-only Streamlit app that reads from parquet output files and the FactorMonitor SQLite DB.

**Tech Stack:** optuna, streamlit, plotly, pandas, numpy, matplotlib, seaborn, scipy, loguru

---

## Task 1: Add new dependencies to requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Edit requirements.txt**

Append the three new packages at the end of `requirements.txt`:

```
# Parameter optimization
optuna==3.6.1

# Dashboard
streamlit==1.35.0
plotly==5.22.0
```

**Step 2: Verify install works**

```bash
cd /Users/kevinwu/Coding/quant-factor-mining
pip install optuna==3.6.1 streamlit==1.35.0 plotly==5.22.0
```

Expected: all three install without error (or are already installed at compatible versions).

---

## Task 2: Implement ParameterOptimizer (Issue #9)

**Files:**
- Create: `src/optimization/parameter_optimizer.py`
- Modify: `src/optimization/__init__.py`

**Background — how the optimizer works:**

`AdaptiveCompositeFactor` accepts `ic_window`, `decay_halflife`, `min_weight`, `normalization_method`.
`AdaptiveCompositeManager` wraps it and accepts `forward_period`.

The objective function:
1. Slices the historical date index into overlapping windows of 252 train + 63 test days.
2. For each window: creates a fresh `AdaptiveCompositeFactor` + `AdaptiveCompositeManager` with trial params, iterates every date in the window calling `compute_factor_with_update`, then computes ICIR on the IC values accumulated during the test portion.
3. Returns the **mean ICIR across all folds** (Optuna maximizes this).

**Step 1: Write the failing test**

Create `test_parameter_optimizer.py` at the project root:

```python
import pytest
import pandas as pd
import numpy as np
from src.optimization.parameter_optimizer import ParameterOptimizer
from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor


def _make_demo_data(n_days=400, n_tickers=5):
    """Build minimal MultiIndex OHLCV DataFrame for testing."""
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
    # Use n_trials=2 and small windows to keep the test fast
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
```

**Step 2: Run test to confirm it fails**

```bash
python3 -m pytest test_parameter_optimizer.py -v
```

Expected: `ImportError: cannot import name 'ParameterOptimizer'`

**Step 3: Create `src/optimization/parameter_optimizer.py`**

```python
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
from scipy.stats import spearmanr

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
        Close prices with MultiIndex (date, ticker) — used for IC calculation.
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
        that were accumulated during the *test* portion.
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

            if isinstance(self.prices, pd.DataFrame):
                # Wide-format prices: filter rows
                price_slice = self.prices[self.prices.index <= date]
                if price_slice.empty:
                    continue
                prices_today = price_slice.iloc[-1]
                # Convert to Series with ticker as index
                prices_series = prices_today
            else:
                # Already a stacked Series
                mask = self.prices.index.get_level_values("date") <= date
                prices_series = self.prices[mask]
                prices_series = prices_series[
                    prices_series.index.get_level_values("date") == date
                ]

            try:
                manager.compute_factor_with_update(data_slice, prices_series, date)
            except Exception:
                continue

            # Collect IC values for test window dates
            if i >= test_start_idx:
                for factor_name, ic_list in af.ic_history.items():
                    if ic_list and ic_list[-1][0] == date:
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
```

**Step 4: Update `src/optimization/__init__.py`**

Replace the contents (currently just a blank line or empty) with:

```python
from .parameter_optimizer import ParameterOptimizer

__all__ = ["ParameterOptimizer"]
```

**Step 5: Run tests**

```bash
python3 -m pytest test_parameter_optimizer.py -v
```

Expected: both tests PASS. Each trial runs 1–2 folds. Total runtime < 30 seconds.

If FAIL with `ModuleNotFoundError: No module named 'optuna'`, install it first:
```bash
pip install optuna==3.6.1
```

---

## Task 3: Notebook 02 — Factor Analysis

**Files:**
- Create: `notebooks/02_factor_analysis.ipynb`

**Overview:** Loads the cached demo data, runs `AdaptiveCompositeManager` for a short date range, then produces three charts:
1. IC time-series per factor
2. IC decay curve (IC mean vs. lag distance from today)
3. IC distribution histogram per factor

**Step 1: Create the notebook**

Create `notebooks/02_factor_analysis.ipynb` with the following cells:

Cell 0 (markdown):
```
# Notebook 02: Factor Analysis

This notebook examines the Information Coefficient (IC) of each factor:

- **IC time series**: how IC evolves over time
- **IC decay curve**: how quickly IC decays as the prediction horizon grows
- **IC distribution**: the statistical properties of each factor's IC
```

Cell 1 (code) — imports:
```python
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from scipy.stats import spearmanr

from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor
from src.factors.adaptive_composite import AdaptiveCompositeFactor, AdaptiveCompositeManager

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
%matplotlib inline

print('✅ Imports successful')
```

Cell 2 (markdown):
```
## 1. Load Demo Data
```

Cell 3 (code) — load data:
```python
import glob
from pathlib import Path

raw_dir = Path('../data/raw')
parquet_files = sorted(raw_dir.glob('*.parquet'))
print(f'Found {len(parquet_files)} cached files:')
for f in parquet_files:
    print(f'  {f.name}')

# Load the 10-ticker demo file (richer data)
demo_file = [f for f in parquet_files if '10tickers' in f.name]
if demo_file:
    data = pd.read_parquet(demo_file[0])
else:
    data = pd.read_parquet(parquet_files[0])

print(f'\nLoaded: {data.shape}')
print(f'Columns: {list(data.columns)}')
print(f'Tickers: {data.index.get_level_values("ticker").unique().tolist()}')
print(f'Date range: {data.index.get_level_values("date").min()} → {data.index.get_level_values("date").max()}')
```

Cell 4 (markdown):
```
## 2. Initialize Factors and Run Adaptive Manager
```

Cell 5 (code) — run manager:
```python
factors = [
    MomentumFactor(lookback=126, name='Momentum'),
    MeanReversionFactor(lookback=21, name='MeanReversion'),
    VolatilityFactor(window=63, name='Volatility'),
]

af = AdaptiveCompositeFactor(
    factors=factors,
    ic_window=63,
    decay_halflife=21,
    min_weight=0.05,
    normalization_method='softmax',
)
manager = AdaptiveCompositeManager(af, forward_period=21)

dates = sorted(data.index.get_level_values('date').unique())

# Run through all dates (may take ~1-2 minutes for 4 years of data)
print(f'Simulating {len(dates)} dates...')
for i, date in enumerate(dates):
    date = pd.Timestamp(date)
    data_slice = data[data.index.get_level_values('date') <= date]
    prices_wide = data['close'].unstack('ticker')
    prices_today = prices_wide[prices_wide.index <= date].iloc[-1]
    try:
        manager.compute_factor_with_update(data_slice, prices_today, date)
    except Exception:
        pass
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{len(dates)} dates processed')

print('✅ Simulation complete')
ic_summary = af.get_ic_summary()
print(f'\nIC Summary:\n{ic_summary}')
```

Cell 6 (markdown):
```
## 3. IC Time Series
```

Cell 7 (code) — IC time series plot:
```python
fig, axes = plt.subplots(len(factors), 1, figsize=(14, 4 * len(factors)), sharex=True)
if len(factors) == 1:
    axes = [axes]

for ax, factor in zip(axes, factors):
    ic_list = af.ic_history[factor.name]
    if not ic_list:
        ax.set_title(f'{factor.name} — no IC data')
        continue
    ic_dates = [x[0] for x in ic_list]
    ic_values = [x[1] for x in ic_list]
    ic_series = pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates))
    ic_series.plot(ax=ax, alpha=0.6, label='Daily IC')
    ic_series.rolling(21).mean().plot(ax=ax, linewidth=2, label='21-day MA')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'{factor.name} — IC Time Series (mean={ic_series.mean():.4f}, ICIR={ic_series.mean()/ic_series.std():.3f})')
    ax.set_ylabel('IC (Spearman)')
    ax.legend()

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.suptitle('Factor IC Time Series', fontsize=14, y=1.01)
plt.show()
```

Cell 8 (markdown):
```
## 4. IC Decay Curve

The IC decay curve shows how predictive power diminishes as we look further into the future.
We compute IC at horizons 1, 5, 10, 21, 42, 63 trading days.
```

Cell 9 (code) — IC decay curve:
```python
horizons = [1, 5, 10, 21, 42, 63]
close_wide = data['close'].unstack('ticker')

decay_results = {f.name: [] for f in factors}

for h in horizons:
    # Compute cross-sectional IC at each date for this horizon
    for factor in factors:
        ic_vals = []
        for i in range(h, len(dates) - h):
            d_past = dates[i - h]
            d_future = dates[i]
            try:
                fv = factor.calculate(data[data.index.get_level_values('date') <= pd.Timestamp(d_past)])
                fv_today = fv[fv.index.get_level_values('date') == pd.Timestamp(d_past)]
                fwd = (close_wide.loc[d_future] / close_wide.loc[d_past] - 1).dropna()
                aligned = pd.DataFrame({
                    'factor': fv_today.droplevel('date'),
                    'fwd': fwd,
                }).dropna()
                if len(aligned) >= 5:
                    ic, _ = spearmanr(aligned['factor'], aligned['fwd'])
                    ic_vals.append(ic)
            except Exception:
                pass
        mean_ic = np.nanmean(ic_vals) if ic_vals else float('nan')
        decay_results[factor.name].append(mean_ic)

fig, ax = plt.subplots(figsize=(10, 5))
for factor_name, values in decay_results.items():
    ax.plot(horizons, values, marker='o', label=factor_name)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Prediction Horizon (trading days)')
ax.set_ylabel('Mean IC')
ax.set_title('IC Decay Curve — Predictive Power vs. Horizon')
ax.legend()
plt.tight_layout()
plt.show()
```

Cell 10 (markdown):
```
## 5. IC Distribution per Factor
```

Cell 11 (code) — IC distribution:
```python
fig, axes = plt.subplots(1, len(factors), figsize=(6 * len(factors), 5))
if len(factors) == 1:
    axes = [axes]

for ax, factor in zip(axes, factors):
    ic_list = af.ic_history[factor.name]
    if not ic_list:
        continue
    ic_values = np.array([x[1] for x in ic_list])
    ax.hist(ic_values, bins=40, edgecolor='white', linewidth=0.5)
    ax.axvline(ic_values.mean(), color='red', linewidth=2, label=f'Mean={ic_values.mean():.4f}')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    pct_positive = (ic_values > 0).mean()
    ax.set_title(f'{factor.name}\nICIR={ic_values.mean()/ic_values.std():.3f}, {pct_positive:.0%} positive')
    ax.set_xlabel('IC Value')
    ax.set_ylabel('Frequency')
    ax.legend()

plt.suptitle('IC Distribution per Factor', fontsize=14)
plt.tight_layout()
plt.show()
```

Cell 12 (markdown):
```
## Summary

| Factor | IC Mean | IC Std | ICIR | % Positive |
|--------|---------|--------|------|------------|
```

Cell 13 (code) — summary table:
```python
summary = af.get_ic_summary()
print(summary.to_string())
```

**Step 2: Verify notebook runs**

```bash
cd /Users/kevinwu/Coding/quant-factor-mining
jupyter nbconvert --to notebook --execute notebooks/02_factor_analysis.ipynb --output notebooks/02_factor_analysis_executed.ipynb --ExecutePreprocessor.timeout=300 2>&1 | tail -5
```

Expected: `Writing ... to notebooks/02_factor_analysis_executed.ipynb`
If the kernel errors, check the traceback and fix the offending cell.
Delete the `_executed` copy after verifying: `rm notebooks/02_factor_analysis_executed.ipynb`

---

## Task 4: Notebook 03 — Regime Analysis

**Files:**
- Create: `notebooks/03_regime_analysis.ipynb`

**Step 1: Create the notebook**

Cell 0 (markdown):
```
# Notebook 03: Regime Analysis

Market regimes shape which factors work. This notebook covers:
- **Regime state timeline**: when was the market trending, mean-reverting, or volatile?
- **Weight shifts by regime**: how does the adaptive system respond to regime changes?
- **Regime frequency breakdown**: how often does each regime occur?
```

Cell 1 (code) — imports:
```python
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor
from src.factors.adaptive_composite import (
    AdaptiveCompositeFactor, AdaptiveCompositeManager,
    RegimeDetector, RegimeAwareCompositeFactor,
)

plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline
print('✅ Imports successful')
```

Cell 2 (markdown):
```
## 1. Load Demo Data
```

Cell 3 (code) — load data:
```python
raw_dir = Path('../data/raw')
parquet_files = sorted(raw_dir.glob('*.parquet'))
demo_file = [f for f in parquet_files if '10tickers' in f.name] or [parquet_files[0]]
data = pd.read_parquet(demo_file[0])

print(f'Shape: {data.shape}')
tickers = data.index.get_level_values('ticker').unique().tolist()
print(f'Tickers: {tickers}')
```

Cell 4 (markdown):
```
## 2. Compute Market Returns and Detect Regimes
```

Cell 5 (code) — regime detection:
```python
# Build equal-weight market proxy
close_wide = data['close'].unstack('ticker')
mkt_returns = np.log(close_wide / close_wide.shift(1)).mean(axis=1).dropna()
mkt_returns.name = 'market'

detector = RegimeDetector(lookback=63)
dates = mkt_returns.index.tolist()

regimes = []
for date in dates:
    hist = mkt_returns[:date]
    regime = detector.detect_regime(hist)
    regimes.append({'date': date, 'regime': regime})

regime_df = pd.DataFrame(regimes).set_index('date')
print(f'\nRegime counts:\n{regime_df["regime"].value_counts()}')
```

Cell 6 (markdown):
```
## 3. Regime State Timeline
```

Cell 7 (code) — timeline chart:
```python
REGIME_COLORS = {
    'trending':      '#2196F3',  # blue
    'mean_reverting':'#4CAF50',  # green
    'high_vol':      '#F44336',  # red
    'neutral':       '#9E9E9E',  # grey
}

fig, ax = plt.subplots(figsize=(16, 3))
for date, row in regime_df.iterrows():
    color = REGIME_COLORS.get(row['regime'], '#9E9E9E')
    ax.axvspan(date, date + pd.Timedelta(days=1), alpha=0.8, color=color)

patches = [mpatches.Patch(color=c, label=r) for r, c in REGIME_COLORS.items()]
ax.legend(handles=patches, loc='upper right')
ax.set_xlim(regime_df.index[0], regime_df.index[-1])
ax.set_title('Market Regime Timeline')
ax.set_ylabel('Regime')
ax.set_yticks([])
plt.tight_layout()
plt.show()
```

Cell 8 (markdown):
```
## 4. Weight Shifts by Regime
```

Cell 9 (code) — run manager and record weights per regime:
```python
factors = [
    MomentumFactor(lookback=126, name='Momentum'),
    MeanReversionFactor(lookback=21, name='MeanReversion'),
    VolatilityFactor(window=63, name='Volatility'),
]
af = AdaptiveCompositeFactor(factors=factors, ic_window=63, decay_halflife=21, min_weight=0.05)
manager = AdaptiveCompositeManager(af, forward_period=21)

all_dates = sorted(data.index.get_level_values('date').unique())
weight_records = []

for date in all_dates:
    date = pd.Timestamp(date)
    data_slice = data[data.index.get_level_values('date') <= date]
    prices_wide = data['close'].unstack('ticker')
    prices_today = prices_wide[prices_wide.index <= date].iloc[-1]
    try:
        _, weights = manager.compute_factor_with_update(data_slice, prices_today, date)
        regime = regime_df.loc[date, 'regime'] if date in regime_df.index else 'neutral'
        rec = {'date': date, 'regime': regime}
        rec.update(weights)
        weight_records.append(rec)
    except Exception:
        pass

weight_df = pd.DataFrame(weight_records).set_index('date')
factor_cols = [f.name for f in factors]

# Average weight by regime
weight_by_regime = weight_df.groupby('regime')[factor_cols].mean()
print('\nAverage factor weights by regime:')
print(weight_by_regime)

weight_by_regime.plot(kind='bar', figsize=(10, 5))
plt.title('Average Factor Weights by Market Regime')
plt.ylabel('Weight')
plt.xlabel('Regime')
plt.xticks(rotation=0)
plt.legend(title='Factor')
plt.tight_layout()
plt.show()
```

Cell 10 (markdown):
```
## 5. Regime Frequency Breakdown
```

Cell 11 (code) — pie chart:
```python
freq = regime_df['regime'].value_counts()
colors = [REGIME_COLORS.get(r, '#9E9E9E') for r in freq.index]

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(freq.values, labels=freq.index, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Regime Frequency Breakdown')
plt.tight_layout()
plt.show()

print(f'\nTotal trading days: {len(regime_df)}')
print(freq.to_string())
```

**Step 2: Verify notebook runs**

```bash
jupyter nbconvert --to notebook --execute notebooks/03_regime_analysis.ipynb --output notebooks/03_regime_analysis_executed.ipynb --ExecutePreprocessor.timeout=300 2>&1 | tail -5
rm -f notebooks/03_regime_analysis_executed.ipynb
```

---

## Task 5: Notebook 04 — Portfolio Construction

**Files:**
- Create: `notebooks/04_portfolio_construction.ipynb`

**Step 1: Create the notebook**

Cell 0 (markdown):
```
# Notebook 04: Portfolio Construction

Explores the mean-variance optimization layer:
- **Efficient frontier**: risk vs. return trade-off across portfolios
- **Weight evolution heatmap**: how stock weights shift over time
- **Turnover over time**: the cost of rebalancing
```

Cell 1 (code) — imports:
```python
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor
from src.factors.adaptive_composite import AdaptiveCompositeFactor, AdaptiveCompositeManager
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.optimization.risk_models import RiskModelEstimator

plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline
print('✅ Imports successful')
```

Cell 2 (code) — load data:
```python
raw_dir = Path('../data/raw')
parquet_files = sorted(raw_dir.glob('*.parquet'))
demo_file = [f for f in parquet_files if '10tickers' in f.name] or [parquet_files[0]]
data = pd.read_parquet(demo_file[0])
tickers = data.index.get_level_values('ticker').unique().tolist()
print(f'Tickers: {tickers}')
```

Cell 3 (markdown):
```
## 1. Efficient Frontier

We sample 300 random portfolios and one mean-variance optimal portfolio to visualize the frontier.
```

Cell 4 (code) — efficient frontier:
```python
close_wide = data['close'].unstack('ticker').dropna()
returns = np.log(close_wide / close_wide.shift(1)).dropna()

# Estimate annual mean returns and covariance
mu = returns.mean() * 252
cov = returns.cov() * 252
n = len(tickers)

np.random.seed(42)
n_portfolios = 300
port_returns, port_vols = [], []

for _ in range(n_portfolios):
    w = np.random.dirichlet(np.ones(n))
    port_returns.append(float(w @ mu))
    port_vols.append(float(np.sqrt(w @ cov.values @ w)))

# Optimal portfolio (max Sharpe)
from scipy.optimize import minimize

def neg_sharpe(w):
    r = w @ mu
    v = np.sqrt(w @ cov.values @ w)
    return -r / (v + 1e-9)

result = minimize(
    neg_sharpe,
    x0=np.ones(n) / n,
    bounds=[(0, 0.3)] * n,
    constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
    method='SLSQP',
)
opt_w = result.x
opt_ret = float(opt_w @ mu)
opt_vol = float(np.sqrt(opt_w @ cov.values @ opt_w))
opt_sharpe = opt_ret / opt_vol

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(port_vols, port_returns, c=np.array(port_returns) / np.array(port_vols),
                     cmap='viridis', alpha=0.5, s=10)
ax.scatter(opt_vol, opt_ret, marker='*', s=300, color='red',
           label=f'Max Sharpe ({opt_sharpe:.2f})')
plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
ax.set_xlabel('Annualized Volatility')
ax.set_ylabel('Annualized Return')
ax.set_title('Efficient Frontier (300 Random Portfolios)')
ax.legend()
plt.tight_layout()
plt.show()

print(f'\nOptimal portfolio (Max Sharpe):')
for ticker, w in zip(tickers, opt_w):
    print(f'  {ticker}: {w:.3f} ({w*100:.1f}%)')
```

Cell 5 (markdown):
```
## 2. Weight Evolution Heatmap

Rolling monthly rebalance using equal factor weights (or use adaptive weights from notebook 02).
```

Cell 6 (code) — weight evolution:
```python
# Simple rolling momentum-based weights as illustrative example
lookback = 63
dates_list = close_wide.index.tolist()
weight_records = []

for i in range(lookback, len(dates_list), 21):  # Monthly rebalance
    d = dates_list[i]
    past_prices = close_wide.iloc[i - lookback]
    curr_prices = close_wide.iloc[i]
    ret = (curr_prices / past_prices - 1).fillna(0)

    # Rank-based weights: higher return → higher weight
    ranks = ret.rank()
    w = ranks / ranks.sum()
    rec = {'date': d}
    rec.update(w.to_dict())
    weight_records.append(rec)

weights_over_time = pd.DataFrame(weight_records).set_index('date')

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    weights_over_time[tickers].T,
    ax=ax, cmap='YlOrRd', linewidths=0.5,
    xticklabels=[str(d.date()) for d in weights_over_time.index[::6]],
    yticklabels=tickers,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_title('Weight Evolution Heatmap (Monthly Rebalance, Momentum-Based)')
ax.set_xlabel('Date')
ax.set_ylabel('Ticker')
plt.tight_layout()
plt.show()
```

Cell 7 (markdown):
```
## 3. Portfolio Turnover Over Time
```

Cell 8 (code) — turnover:
```python
w_arr = weights_over_time[tickers].values
turnover = np.abs(np.diff(w_arr, axis=0)).sum(axis=1) / 2  # One-way turnover

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(weights_over_time.index[1:], turnover, alpha=0.7, color='steelblue')
ax.axhline(turnover.mean(), color='red', linestyle='--', label=f'Mean={turnover.mean():.3f}')
ax.set_xlabel('Date')
ax.set_ylabel('One-Way Turnover')
ax.set_title('Portfolio Turnover Over Time')
ax.legend()
plt.tight_layout()
plt.show()

print(f'\nMean turnover: {turnover.mean():.3f} ({turnover.mean()*100:.1f}%)')
print(f'Max turnover:  {turnover.max():.3f} ({turnover.max()*100:.1f}%)')
```

**Step 2: Verify notebook runs**

```bash
jupyter nbconvert --to notebook --execute notebooks/04_portfolio_construction.ipynb --output notebooks/04_portfolio_construction_executed.ipynb --ExecutePreprocessor.timeout=300 2>&1 | tail -5
rm -f notebooks/04_portfolio_construction_executed.ipynb
```

---

## Task 6: Notebook 05 — Backtest Results

**Files:**
- Create: `notebooks/05_backtest_results.ipynb`

**Step 1: Create the notebook**

Cell 0 (markdown):
```
# Notebook 05: Backtest Results

Full backtest analysis using `PortfolioBacktester`:
- **Equity curve**: cumulative portfolio value over time
- **Monthly returns heatmap**: returns by year × month
- **Drawdown**: when and how deep losses occurred
- **Risk metrics**: Sharpe ratio, Sortino ratio, max drawdown
```

Cell 1 (code) — imports:
```python
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor
from src.factors.adaptive_composite import AdaptiveCompositeFactor, AdaptiveCompositeManager
from src.optimization.backtester import PortfolioBacktester

plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline
print('✅ Imports successful')
```

Cell 2 (code) — load data:
```python
raw_dir = Path('../data/raw')
parquet_files = sorted(raw_dir.glob('*.parquet'))
demo_file = [f for f in parquet_files if '10tickers' in f.name] or [parquet_files[0]]
data = pd.read_parquet(demo_file[0])
print(f'Shape: {data.shape}')
```

Cell 3 (markdown):
```
## 1. Run Backtest
```

Cell 4 (code) — build factor signals and run backtest:
```python
factors = [
    MomentumFactor(lookback=126, name='Momentum'),
    MeanReversionFactor(lookback=21, name='MeanReversion'),
    VolatilityFactor(window=63, name='Volatility'),
]
af = AdaptiveCompositeFactor(factors=factors, ic_window=63, decay_halflife=21, min_weight=0.05)
manager = AdaptiveCompositeManager(af, forward_period=21)

all_dates = sorted(data.index.get_level_values('date').unique())
factor_signals = {}

print(f'Computing factors for {len(all_dates)} dates...')
for i, date in enumerate(all_dates):
    date = pd.Timestamp(date)
    data_slice = data[data.index.get_level_values('date') <= date]
    prices_wide = data['close'].unstack('ticker')
    prices_today = prices_wide[prices_wide.index <= date].iloc[-1]
    try:
        composite, _ = manager.compute_factor_with_update(data_slice, prices_today, date)
        factor_signals[date] = composite[composite.index.get_level_values('date') == date].droplevel('date')
    except Exception:
        pass
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(all_dates)}')

print(f'Factor signals computed for {len(factor_signals)} dates')
```

Cell 5 (code) — backtest simulation:
```python
close_wide = data['close'].unstack('ticker')
tickers = close_wide.columns.tolist()

# Simple backtest: rank factor, go long top quartile
INITIAL_CAPITAL = 1_000_000
TRANSACTION_COST = 0.001
REBALANCE_FREQ = 21

portfolio_value = [INITIAL_CAPITAL]
daily_returns = []
positions = pd.Series(0.0, index=tickers)
rebalance_dates = []

signal_dates = sorted(factor_signals.keys())
for i, date in enumerate(signal_dates[1:], 1):
    prev_date = signal_dates[i - 1]

    # Daily P&L
    price_change = (close_wide.loc[date] / close_wide.loc[prev_date] - 1).fillna(0)
    daily_pnl = (positions * price_change).sum()
    portfolio_value.append(portfolio_value[-1] * (1 + daily_pnl))
    daily_returns.append(daily_pnl)

    # Rebalance
    if i % REBALANCE_FREQ == 0:
        signal = factor_signals[date].reindex(tickers).fillna(0)
        n_long = max(1, len(tickers) // 4)
        top = signal.nlargest(n_long).index
        new_positions = pd.Series(0.0, index=tickers)
        new_positions[top] = 1.0 / n_long

        turnover = (new_positions - positions).abs().sum() / 2
        cost = turnover * TRANSACTION_COST
        portfolio_value[-1] *= (1 - cost)
        positions = new_positions
        rebalance_dates.append(date)

pv = pd.Series(portfolio_value, index=[signal_dates[0]] + signal_dates[1:])
ret = pd.Series(daily_returns, index=signal_dates[1:])
print(f'\nBacktest complete: {len(pv)} days')
print(f'Final portfolio value: ${pv.iloc[-1]:,.0f}')
print(f'Total return: {(pv.iloc[-1]/pv.iloc[0]-1)*100:.1f}%')
```

Cell 6 (markdown):
```
## 2. Equity Curve
```

Cell 7 (code) — equity curve:
```python
benchmark = (close_wide.mean(axis=1).pct_change().fillna(0) + 1).cumprod() * INITIAL_CAPITAL
benchmark = benchmark.reindex(pv.index, method='ffill')

fig, ax = plt.subplots(figsize=(14, 5))
pv.plot(ax=ax, label='Strategy', linewidth=2)
benchmark.plot(ax=ax, label='Equal-Weight Benchmark', linewidth=1.5, linestyle='--', alpha=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Equity Curve')
ax.set_ylabel('Portfolio Value ($)')
ax.legend()
plt.tight_layout()
plt.show()
```

Cell 8 (markdown):
```
## 3. Monthly Returns Heatmap
```

Cell 9 (code) — heatmap:
```python
monthly_ret = ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
monthly_table = monthly_ret.to_frame('ret')
monthly_table['year'] = monthly_table.index.year
monthly_table['month'] = monthly_table.index.month
pivot = monthly_table.pivot(index='year', columns='month', values='ret')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot.columns)]

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(pivot * 100, annot=True, fmt='.1f', center=0, cmap='RdYlGn',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Return (%)'})
ax.set_title('Monthly Returns Heatmap (%)')
plt.tight_layout()
plt.show()
```

Cell 10 (markdown):
```
## 4. Drawdown
```

Cell 11 (code) — drawdown:
```python
rolling_max = pv.cummax()
drawdown = (pv - rolling_max) / rolling_max * 100

fig, ax = plt.subplots(figsize=(14, 4))
drawdown.plot(ax=ax, color='red', alpha=0.7)
ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
ax.set_title('Portfolio Drawdown (%)')
ax.set_ylabel('Drawdown (%)')
plt.tight_layout()
plt.show()
```

Cell 12 (markdown):
```
## 5. Risk Metrics
```

Cell 13 (code) — Sharpe, Sortino, max DD:
```python
rfr_daily = 0.02 / 252
ann_ret = (pv.iloc[-1] / pv.iloc[0]) ** (252 / len(pv)) - 1
ann_vol = ret.std() * np.sqrt(252)
sharpe = (ann_ret - 0.02) / ann_vol

downside = ret[ret < rfr_daily]
sortino = (ann_ret - 0.02) / (downside.std() * np.sqrt(252) + 1e-9)
max_dd = drawdown.min()

print('='*40)
print('  RISK METRICS')
print('='*40)
print(f'  Annual Return:    {ann_ret*100:.2f}%')
print(f'  Annual Volatility:{ann_vol*100:.2f}%')
print(f'  Sharpe Ratio:     {sharpe:.3f}')
print(f'  Sortino Ratio:    {sortino:.3f}')
print(f'  Max Drawdown:     {max_dd:.2f}%')
print(f'  # Rebalances:     {len(rebalance_dates)}')
print('='*40)
```

**Step 2: Verify notebook runs**

```bash
jupyter nbconvert --to notebook --execute notebooks/05_backtest_results.ipynb --output notebooks/05_backtest_results_executed.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -5
rm -f notebooks/05_backtest_results_executed.ipynb
```

---

## Task 7: Notebook 06 — Live Trading Setup

**Files:**
- Create: `notebooks/06_live_trading_setup.ipynb`

**Step 1: Create the notebook**

This notebook is documentation/tutorial only — no charts. It explains the operational pipeline end-to-end.

Cell 0 (markdown):
```
# Notebook 06: Live Trading Setup Guide

This notebook walks through every step needed to move from data to live factor-based signals.
No backtest simulation here — just the operational pipeline.

**Workflow:**
1. Load or fetch data
2. Compute adaptive factor composite
3. Monitor factor health
4. Generate trade signals
5. (Optional) Run the Streamlit dashboard
```

Cell 1 (code) — imports:
```python
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Remove default stderr sink, add notebook-friendly one
logger.remove()
logger.add(lambda msg: print(msg, end=''), colorize=False, format='{time:HH:mm:ss} | {level} | {message}')

print('✅ Imports ready')
```

Cell 2 (markdown):
```
## Step 1: Load Data

For live use you would call `YFinanceFetcher` to get fresh data. Here we load from cache.

```python
from src.data.fetcher import YFinanceFetcher
from src.data.cache_manager import CacheManager

fetcher = YFinanceFetcher()
cache = CacheManager()

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
data = cache.get_or_fetch(
    market='US', universe='live', tickers=tickers,
    start_date='2023-01-01', end_date='2024-12-31',
    fetcher=fetcher, force_refresh=True,
)
```

For the demo, we use the cached parquet instead:
```

Cell 3 (code) — load cached data:
```python
raw_dir = Path('../data/raw')
parquet_files = sorted(raw_dir.glob('*.parquet'))
data = pd.read_parquet(parquet_files[0])

print(f'Data shape: {data.shape}')
print(f'Tickers: {data.index.get_level_values("ticker").unique().tolist()}')
print(f'Date range: {data.index.get_level_values("date").min()} → {data.index.get_level_values("date").max()}')
```

Cell 4 (markdown):
```
## Step 2: Initialize Factors and Adaptive Manager

Configure with your chosen parameters. For production, load from `outputs/optimization/best_params.json`.
```

Cell 5 (code) — setup manager:
```python
import json
from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor
from src.factors.adaptive_composite import AdaptiveCompositeFactor, AdaptiveCompositeManager

# Load optimized params if available
best_params_path = Path('../outputs/optimization/best_params.json')
if best_params_path.exists():
    with open(best_params_path) as f:
        params = json.load(f)
    print(f'Loaded optimized params: {params}')
else:
    params = {
        'ic_window': 63,
        'decay_halflife': 21,
        'min_weight': 0.05,
        'normalization_method': 'softmax',
        'forward_period': 21,
    }
    print(f'Using default params: {params}')

factors = [
    MomentumFactor(lookback=126, name='Momentum'),
    MeanReversionFactor(lookback=21, name='MeanReversion'),
    VolatilityFactor(window=63, name='Volatility'),
]

af = AdaptiveCompositeFactor(
    factors=factors,
    ic_window=params['ic_window'],
    decay_halflife=params['decay_halflife'],
    min_weight=params['min_weight'],
    normalization_method=params['normalization_method'],
)
manager = AdaptiveCompositeManager(af, forward_period=params['forward_period'])
print('✅ AdaptiveCompositeManager ready')
```

Cell 6 (markdown):
```
## Step 3: Set Up FactorMonitor

The monitor tracks ICIR, weight concentration (HHI), and stale-weight alerts.
Persist to SQLite for cross-session history.
```

Cell 7 (code) — monitor setup:
```python
from src.monitoring import FactorMonitor

DB_PATH = '../outputs/factor_monitor.db'
monitor = FactorMonitor(
    manager,
    icir_threshold=0.3,
    hhi_threshold=0.5,
    db_path=DB_PATH,
)
print(f'✅ FactorMonitor ready (DB: {DB_PATH})')
```

Cell 8 (markdown):
```
## Step 4: Process New Data and Generate Signals

In a production loop you call this daily (or at your chosen rebalance frequency).
```

Cell 9 (code) — daily loop:
```python
all_dates = sorted(data.index.get_level_values('date').unique())

# Process last 90 days as a demo
recent_dates = all_dates[-90:]
print(f'Processing {len(recent_dates)} dates (demo)...\n')

for date in recent_dates:
    date = pd.Timestamp(date)
    data_slice = data[data.index.get_level_values('date') <= date]
    prices_wide = data['close'].unstack('ticker')
    prices_today = prices_wide[prices_wide.index <= date].iloc[-1]

    try:
        composite, weights = manager.compute_factor_with_update(data_slice, prices_today, date)
        metrics = monitor.check(date)
    except Exception as e:
        logger.warning(f'Skipping {date}: {e}')
        continue

# Show final signals
latest_date = pd.Timestamp(recent_dates[-1])
composite_today = composite[composite.index.get_level_values('date') == latest_date]
print(f'\n--- Factor Signals as of {latest_date.date()} ---')
print(composite_today.droplevel('date').sort_values(ascending=False).to_string())
print(f'\n--- Current Weights ---')
for k, v in weights.items():
    print(f'  {k}: {v:.3f} ({v*100:.1f}%)')
```

Cell 10 (markdown):
```
## Step 5: Generate Trade List

Rank stocks by composite factor score → top quartile = BUY, bottom = SELL (or flat for long-only).
```

Cell 11 (code) — trade list:
```python
signals = composite_today.droplevel('date').dropna()
n = len(signals)
n_long = max(1, n // 4)

ranked = signals.rank(ascending=False)
buy_list = ranked[ranked <= n_long].index.tolist()
flat_list = ranked[ranked > n_long].index.tolist()

print('TRADE LIST:')
print(f'  BUY  ({n_long} stocks): {buy_list}')
print(f'  FLAT ({len(flat_list)} stocks): {flat_list}')
print('\nTarget weights:')
for t in buy_list:
    print(f'  {t}: {1/n_long:.3f} ({100/n_long:.1f}%)')
```

Cell 12 (markdown):
```
## Step 6: Health Summary and Dashboard

Print the monitor summary. For a web dashboard, run:

```bash
streamlit run dashboard/app.py
```
```

Cell 13 (code) — summary:
```python
monitor.summary()
print('\n✅ Live trading setup complete.')
print('   Run "streamlit run dashboard/app.py" for the real-time dashboard.')
```

**Step 2: Verify notebook runs**

```bash
jupyter nbconvert --to notebook --execute notebooks/06_live_trading_setup.ipynb --output notebooks/06_live_trading_setup_executed.ipynb --ExecutePreprocessor.timeout=300 2>&1 | tail -5
rm -f notebooks/06_live_trading_setup_executed.ipynb
```

---

## Task 8: Streamlit Dashboard — `dashboard/app.py` (Issue #12)

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/app.py`

**Architecture recap:**
- Tab 1 **Factor Health** — reads from SQLite (`outputs/factor_monitor.db`)
- Tab 2 **Weight Evolution** — reads from `outputs/weight_history.parquet`
- Tab 3 **Regime State** — reads from `outputs/regime_history.parquet`
- Tab 4 **Backtest** — reads from `outputs/backtests/*.parquet`
- All reads are wrapped in `try/except` with friendly "no data yet" messages
- Auto-refresh via `st.rerun()` + `time.sleep(interval)`
- Sidebar: last-updated timestamp, refresh slider (10–120s), manual refresh button

**Step 1: Create `dashboard/__init__.py`**

```python
```
(empty file)

**Step 2: Create `dashboard/app.py`**

```python
"""
Streamlit dashboard for quant-factor-mining.

Run:  streamlit run dashboard/app.py
"""

import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_DB_PATH       = _ROOT / "outputs" / "factor_monitor.db"
_WEIGHT_PATH   = _ROOT / "outputs" / "weight_history.parquet"
_REGIME_PATH   = _ROOT / "outputs" / "regime_history.parquet"
_BACKTEST_DIR  = _ROOT / "outputs" / "backtests"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Factor Monitor",
    page_icon="📈",
    layout="wide",
)


# ── Data loaders (read-only, graceful on missing files) ────────────────────

@st.cache_data(ttl=30)
def load_sqlite_metrics():
    if not _DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(_DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT date, factor_name, icir, ic_mean, hhi, skip_count "
            "FROM factor_metrics ORDER BY recorded_at",
            conn,
        )


@st.cache_data(ttl=30)
def load_weight_history():
    if not _WEIGHT_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(_WEIGHT_PATH)


@st.cache_data(ttl=30)
def load_regime_history():
    if not _REGIME_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(_REGIME_PATH)


@st.cache_data(ttl=30)
def load_backtest():
    if not _BACKTEST_DIR.exists():
        return pd.DataFrame()
    files = sorted(_BACKTEST_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])  # most recent file


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    refresh_interval = st.slider(
        "Auto-refresh interval (s)", min_value=10, max_value=120, value=30, step=10
    )
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    last_updated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Last updated: {last_updated}")
    st.caption(f"DB: `{_DB_PATH.name}`")
    if _DB_PATH.exists():
        st.success("SQLite DB connected")
    else:
        st.warning("SQLite DB not found — run the factor monitor first")


# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Factor Health",
    "⚖️ Weight Evolution",
    "🌐 Regime State",
    "📈 Backtest",
])


# ── Tab 1: Factor Health ───────────────────────────────────────────────────
with tab1:
    st.header("Factor Health")
    metrics = load_sqlite_metrics()

    if metrics.empty:
        st.info("No data yet. Run FactorMonitor with a db_path to populate this tab.")
    else:
        # Latest snapshot per factor
        latest = metrics.groupby("factor_name").last().reset_index()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ICIR per Factor")
            fig = px.bar(
                latest, x="factor_name", y="icir",
                color="icir", color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                labels={"icir": "ICIR", "factor_name": "Factor"},
                title="Rolling ICIR (latest snapshot)",
            )
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange",
                          annotation_text="Threshold 0.3")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Weight Concentration (HHI)")
            hhi_val = latest["hhi"].iloc[0] if not latest.empty else float("nan")
            if not pd.isna(hhi_val):
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=hhi_val,
                    title={"text": "HHI"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "steelblue"},
                        "steps": [
                            {"range": [0, 0.33], "color": "#c8e6c9"},
                            {"range": [0.33, 0.5], "color": "#fff9c4"},
                            {"range": [0.5, 1.0], "color": "#ffcdd2"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3}, "value": 0.5},
                    },
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.metric("HHI", "N/A")

        st.subheader("IC Skip Count (staleness)")
        skip_table = latest[["factor_name", "skip_count"]].copy()
        skip_table["status"] = skip_table["skip_count"].apply(
            lambda x: "🔴 STALE" if x >= 21 else ("🟡 Warn" if x >= 5 else "🟢 OK")
        )
        st.dataframe(skip_table, use_container_width=True)


# ── Tab 2: Weight Evolution ────────────────────────────────────────────────
with tab2:
    st.header("Weight Evolution")
    weights = load_weight_history()

    if weights.empty:
        st.info("No weight_history.parquet found. Save weight history with:\n"
                "```python\naf.get_weight_history().to_parquet('outputs/weight_history.parquet')\n```")
    else:
        # Expect index=date, columns=factor names
        weights.index = pd.to_datetime(weights.index)
        fig = px.line(
            weights.reset_index().melt(id_vars="date", var_name="Factor", value_name="Weight"),
            x="date", y="Weight", color="Factor",
            title="Factor Weight Evolution Over Time",
            labels={"date": "Date", "Weight": "Weight"},
        )
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latest Weights")
        st.dataframe(weights.iloc[-1:].T.rename(columns={weights.index[-1]: "Weight"}),
                     use_container_width=True)


# ── Tab 3: Regime State ────────────────────────────────────────────────────
with tab3:
    st.header("Regime State")
    regimes = load_regime_history()

    if regimes.empty:
        st.info("No regime_history.parquet found. Save regime history from RegimeDetector output.")
    else:
        regimes.index = pd.to_datetime(regimes.index)
        latest_regime = regimes.iloc[-1].get("regime", "unknown") if "regime" in regimes.columns else "unknown"

        REGIME_COLORS = {
            "trending":       "#2196F3",
            "mean_reverting": "#4CAF50",
            "high_vol":       "#F44336",
            "neutral":        "#9E9E9E",
        }
        badge_color = REGIME_COLORS.get(latest_regime, "#9E9E9E")
        st.markdown(
            f"**Current regime:** "
            f"<span style='background:{badge_color};color:white;padding:4px 12px;"
            f"border-radius:12px;font-weight:bold'>{latest_regime.upper()}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        if "regime" in regimes.columns:
            # Numeric encoding for timeline
            code_map = {"trending": 1, "mean_reverting": 2, "high_vol": 3, "neutral": 0}
            regimes["regime_code"] = regimes["regime"].map(code_map).fillna(0)

            fig = px.area(
                regimes.reset_index().rename(columns={"index": "date"}),
                x="date", y="regime_code",
                color="regime",
                title="Regime Timeline",
                labels={"date": "Date", "regime_code": "Regime", "regime": "Regime"},
                color_discrete_map=REGIME_COLORS,
            )
            st.plotly_chart(fig, use_container_width=True)

            freq = regimes["regime"].value_counts().reset_index()
            freq.columns = ["regime", "count"]
            fig_pie = px.pie(freq, names="regime", values="count",
                             title="Regime Frequency",
                             color="regime", color_discrete_map=REGIME_COLORS)
            st.plotly_chart(fig_pie, use_container_width=True)


# ── Tab 4: Backtest ────────────────────────────────────────────────────────
with tab4:
    st.header("Backtest Results")
    bt = load_backtest()

    if bt.empty:
        st.info("No backtest parquet files found in `outputs/backtests/`.\n"
                "Run a backtest and save equity curve to:\n"
                "`outputs/backtests/equity_curve_YYYYMMDD.parquet`")
    else:
        bt.index = pd.to_datetime(bt.index)

        if "portfolio_value" in bt.columns:
            fig = px.line(bt.reset_index(), x=bt.index.name or "index", y="portfolio_value",
                          title="Equity Curve",
                          labels={"portfolio_value": "Portfolio Value ($)"})
            fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        if "daily_return" in bt.columns:
            monthly_ret = bt["daily_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
            pivot = pd.DataFrame({
                "year": monthly_ret.index.year,
                "month": monthly_ret.index.month_name().str[:3],
                "ret": monthly_ret.values,
            }).pivot(index="year", columns="month", values="ret")

            fig_hm = px.imshow(
                pivot * 100,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                title="Monthly Returns Heatmap (%)",
                labels={"color": "Return (%)"},
                text_auto=".1f",
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # Summary metrics
        if "portfolio_value" in bt.columns:
            pv = bt["portfolio_value"]
            total_ret = pv.iloc[-1] / pv.iloc[0] - 1
            ann_factor = 252 / len(pv)
            ann_ret = (1 + total_ret) ** ann_factor - 1
            rolling_max = pv.cummax()
            max_dd = ((pv - rolling_max) / rolling_max).min()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Return", f"{total_ret:.1%}")
            m2.metric("Annualized Return", f"{ann_ret:.1%}")
            m3.metric("Max Drawdown", f"{max_dd:.1%}")


# ── Auto-refresh ───────────────────────────────────────────────────────────
time.sleep(refresh_interval)
st.cache_data.clear()
st.rerun()
```

**Step 3: Verify dashboard starts**

```bash
cd /Users/kevinwu/Coding/quant-factor-mining
streamlit run dashboard/app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | head -5
kill %1
```

Expected: HTML output from Streamlit (any content means it started correctly). All tabs should show "No data yet" messages since the output files don't exist yet.

If `streamlit: command not found`, install first:
```bash
pip install streamlit==1.35.0 plotly==5.22.0
```

---

## Task 9: End-to-End Smoke Test

**Files:**
- No new files

Run the optimizer smoke test and confirm all pieces connect:

```bash
cd /Users/kevinwu/Coding/quant-factor-mining

# 1. Optimizer test
python3 -m pytest test_parameter_optimizer.py -v

# 2. Quick optimizer smoke test (2 trials, small data)
python3 -c "
import pandas as pd, numpy as np
from src.optimization.parameter_optimizer import ParameterOptimizer
from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor

np.random.seed(0)
dates = pd.bdate_range('2021-01-01', periods=420)
tickers = ['A','B','C']
idx = pd.MultiIndex.from_product([dates, tickers], names=['date','ticker'])
prices = np.cumprod(1 + np.random.randn(len(idx))*0.01) * 100
data = pd.DataFrame({'open':prices,'high':prices*1.01,'low':prices*0.99,'close':prices,'volume':1000.0}, index=idx)
close = data['close'].unstack('ticker')
opt = ParameterOptimizer([MomentumFactor(63), MeanReversionFactor(21), VolatilityFactor(42)],
                          data, close, n_trials=2, train_size=150, test_size=50)
p = opt.optimize()
print('Best params:', p)
"
```

Expected output includes `Best params:` dict with all 5 keys.

```bash
# 3. Confirm best_params.json was written
ls -la outputs/optimization/best_params.json
cat outputs/optimization/best_params.json
```

---

## Summary

After completing all tasks:

| Component | File | Status |
|---|---|---|
| Dependencies | `requirements.txt` | optuna + streamlit + plotly added |
| ParameterOptimizer | `src/optimization/parameter_optimizer.py` | Optuna TPE + walk-forward ICIR |
| Notebook 02 | `notebooks/02_factor_analysis.ipynb` | IC time series, decay, distribution |
| Notebook 03 | `notebooks/03_regime_analysis.ipynb` | Regime timeline, weight shifts, frequency |
| Notebook 04 | `notebooks/04_portfolio_construction.ipynb` | Efficient frontier, heatmap, turnover |
| Notebook 05 | `notebooks/05_backtest_results.ipynb` | Equity curve, monthly heatmap, metrics |
| Notebook 06 | `notebooks/06_live_trading_setup.ipynb` | Step-by-step live trading guide |
| Dashboard | `dashboard/app.py` | 4-tab Streamlit app, 30s auto-refresh |

**Run the dashboard:**
```bash
streamlit run dashboard/app.py
```

**Run the optimizer:**
```python
from src.optimization.parameter_optimizer import ParameterOptimizer
opt = ParameterOptimizer(factors, data, prices, n_trials=100)
best = opt.optimize()   # saves to outputs/optimization/best_params.json
```
