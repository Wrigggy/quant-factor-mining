"""
Volatility Factor

Academic Reference: Ang et al. (2006, 2009)
"The Cross-Section of Volatility and Expected Returns"
"High Idiosyncratic Volatility and Low Returns"

Key Finding: Low volatility stocks violate CAPM expectations, actually outperform high volatility stocks
This is called the "Low Volatility Anomaly"
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.factors.base import BaseFactor
from config.settings import VOLATILITY_WINDOW, VOLATILITY_ANNUALIZE


class VolatilityFactor(BaseFactor):
    """
    Volatility Factor (Low Volatility Anomaly)

    Formula: Volatility(t) = -1 × std(returns[t-window:t]) × √252

    Where:
    - returns = log returns log(P_t / P_{t-1})
    - window = rolling window (default 63 days = 3 months)
    - √252 = annualization factor
    - negative sign: low volatility is positive signal (buy low volatility stocks)
    """

    def __init__(
        self,
        window: int = VOLATILITY_WINDOW,
        annualize: bool = VOLATILITY_ANNUALIZE,
        use_log_returns: bool = True,
        name: str = "Volatility"
    ):
        """
        Parameters
        ----------
        window : int
            Rolling window (trading days), default 63 days (3 months)
        annualize : bool
            Whether to annualize volatility
        use_log_returns : bool
            Whether to use log returns
        name : str
            Factor name
        """
        super().__init__(name)
        self.window = window
        self.annualize = annualize
        self.use_log_returns = use_log_returns

        logger.info(f"VolatilityFactor initialized: window={window}, annualize={annualize}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility factor values

        Parameters
        ----------
        data : pd.DataFrame
            Input data, MultiIndex (date, ticker), contains 'close' column

        Returns
        -------
        pd.Series
            Volatility factor values, MultiIndex (date, ticker)
        """
        self._validate_data(data)
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)

        # Calculate returns
        if self.use_log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        # Calculate rolling standard deviation
        volatility = returns.rolling(window=self.window, min_periods=self.window // 2).std()

        if self.annualize:
            volatility = self._annualize_vol(volatility)

        # Low volatility anomaly: negative sign means low volatility is positive signal
        low_vol_factor = -1 * volatility

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(low_vol_factor)


class DownsideVolatilityFactor(BaseFactor):
    """
    Downside Volatility Factor

    Only considers volatility of negative returns, more sensitive to downside risk
    """

    def __init__(
        self,
        window: int = VOLATILITY_WINDOW,
        annualize: bool = True,
        threshold: float = 0.0,
        name: str = "DownsideVolatility"
    ):
        """
        Parameters
        ----------
        window : int
            Rolling window
        annualize : bool
            Whether to annualize
        threshold : float
            Minimum acceptable return (MAR), default 0
        name : str
            Factor name
        """
        super().__init__(name)
        self.window = window
        self.annualize = annualize
        self.threshold = threshold

        logger.info(f"DownsideVolatilityFactor initialized: window={window}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate downside volatility factor

        Formula: sqrt(mean(min(0, returns - threshold)^2))
        """
        self._validate_data(data)
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)
        returns = np.log(close / close.shift(1))

        # Only keep returns below threshold
        downside_returns = returns.apply(lambda x: np.where(x < self.threshold, x - self.threshold, 0))

        # Calculate downside standard deviation
        downside_vol = downside_returns.rolling(
            window=self.window,
            min_periods=self.window // 2
        ).apply(lambda x: np.sqrt(np.mean(x**2)))

        if self.annualize:
            downside_vol = self._annualize_vol(downside_vol)

        # Low downside volatility is positive signal
        low_downside_factor = -1 * downside_vol

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(low_downside_factor)


class IdiosyncraticVolatilityFactor(BaseFactor):
    """
    Idiosyncratic Volatility Factor

    Residual volatility after removing market and industry factors
    """

    def __init__(
        self,
        window: int = VOLATILITY_WINDOW,
        market_returns: pd.Series = None,
        annualize: bool = True,
        name: str = "IdiosyncraticVolatility"
    ):
        """
        Parameters
        ----------
        window : int
            Rolling window
        market_returns : pd.Series
            Market returns (date index)
        annualize : bool
            Whether to annualize
        name : str
            Factor name
        """
        super().__init__(name)
        self.window = window
        self.market_returns = market_returns
        self.annualize = annualize

        logger.info(f"IdiosyncraticVolatilityFactor initialized")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate idiosyncratic volatility factor

        Uses rolling regression to calculate residuals, then calculates standard deviation of residuals
        """
        self._validate_data(data)
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)
        returns = np.log(close / close.shift(1))

        if self.market_returns is None:
            # If no market returns provided, use equal-weight average as market proxy
            logger.warning("No market returns provided, using equal-weight average")
            market_proxy = returns.mean(axis=1)
        else:
            market_proxy = self.market_returns.reindex(returns.index)

        # Vectorized rolling beta: broadcast market across all tickers at once
        market_df = pd.DataFrame(
            np.tile(market_proxy.values.reshape(-1, 1), (1, len(returns.columns))),
            index=returns.index,
            columns=returns.columns,
        )

        rolling_cov = returns.rolling(self.window, min_periods=self.window // 2).cov(market_df)
        market_var = market_proxy.rolling(self.window, min_periods=self.window // 2).var()
        beta = rolling_cov.div(market_var + 1e-9, axis=0)

        # Residuals and idiosyncratic volatility — fully vectorized
        residuals = returns - beta.multiply(market_proxy, axis=0)
        idio_vol = residuals.rolling(self.window, min_periods=self.window // 2).std()

        if self.annualize:
            idio_vol = self._annualize_vol(idio_vol)

        # Low idiosyncratic volatility is positive signal
        low_idio_factor = -1 * idio_vol

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(low_idio_factor)


class RangeVolatilityFactor(BaseFactor):
    """
    Price Range Volatility Factor

    Volatility estimation based on high and low prices
    Parkinson (1980) estimator
    """

    def __init__(
        self,
        window: int = 21,
        annualize: bool = True,
        name: str = "RangeVolatility"
    ):
        """
        Parameters
        ----------
        window : int
            Rolling window
        annualize : bool
            Whether to annualize
        name : str
            Factor name
        """
        super().__init__(name)
        self.window = window
        self.annualize = annualize

        logger.info(f"RangeVolatilityFactor initialized: window={window}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate High-Low range-based volatility

        Parkinson formula:
        σ^2 = (1 / (4 * ln(2))) * E[(ln(H/L))^2]
        """
        self._validate_data(data, required=['high', 'low'])
        logger.info(f"Calculating {self.name} factor...")

        high = data['high'].unstack(level='ticker')
        low = data['low'].unstack(level='ticker')

        # Calculate log ratio
        log_hl_ratio = np.log(high / low)

        # Parkinson estimator
        parkinson_vol = log_hl_ratio.rolling(
            window=self.window,
            min_periods=self.window // 2
        ).apply(lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))))

        if self.annualize:
            parkinson_vol = self._annualize_vol(parkinson_vol)

        # Low volatility is positive signal
        low_range_factor = -1 * parkinson_vol

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(low_range_factor)


class GARCHVolatilityFactor(BaseFactor):
    """
    GARCH Model Volatility Factor

    Estimates conditional volatility using GARCH(1,1) model
    Requires arch package
    """

    def __init__(
        self,
        window: int = 252,
        name: str = "GARCHVolatility"
    ):
        """
        Parameters
        ----------
        window : int
            Estimation window
        name : str
            Factor name
        """
        super().__init__(name)
        self.window = window

        logger.info(f"GARCHVolatilityFactor initialized (requires 'arch' package)")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate GARCH volatility

        Note: This is a simplified implementation, full version requires arch package
        """
        self._validate_data(data)
        logger.warning("GARCH implementation requires 'arch' package, using simplified approach")

        close = self._unstack_close(data)
        returns = np.log(close / close.shift(1))

        # EWMA volatility (GARCH approximation)
        ewma_vol = self._annualize_vol(returns.ewm(span=self.window // 10, adjust=False).std())

        # Low volatility is positive signal
        low_garch_factor = -1 * ewma_vol

        logger.info(f"{self.name} calculation complete (simplified)")
        return self._stack_result(low_garch_factor)


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    import matplotlib.pyplot as plt

    # 配置日志
    logger.add("volatility.log", rotation="10 MB")

    # 获取测试数据
    fetcher = YFinanceFetcher()
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"]
    data = fetcher.fetch_batch(test_tickers, "2020-01-01", "2024-01-01")

    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # 测试波动率因子
    print("\n=== Testing Volatility Factor ===")
    volatility = VolatilityFactor(window=63, annualize=True)
    vol_values = volatility.compute(clean_data, normalize=True)

    print(f"\nVolatility factor shape: {vol_values.shape}")
    print(f"\nSample values:")
    print(vol_values.head(20))

    # 测试下行波动率因子
    print("\n=== Testing Downside Volatility Factor ===")
    downside_vol = DownsideVolatilityFactor(window=63)
    downside_values = downside_vol.compute(clean_data, normalize=True)

    print(f"\nDownside volatility factor shape: {downside_values.shape}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 波动率因子分布
    vol_values.hist(bins=50, ax=axes[0, 0], edgecolor='black')
    axes[0, 0].set_title('Volatility Factor Distribution (Normalized)')
    axes[0, 0].set_xlabel('Factor Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 下行波动率因子分布
    downside_values.hist(bins=50, ax=axes[0, 1], edgecolor='black')
    axes[0, 1].set_title('Downside Volatility Factor Distribution')
    axes[0, 1].set_xlabel('Factor Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 波动率因子时间序列
    vol_pivot = vol_values.unstack(level='ticker')
    for ticker in test_tickers[:3]:
        if ticker in vol_pivot.columns:
            vol_pivot[ticker].plot(ax=axes[1, 0], label=ticker, alpha=0.7)
    axes[1, 0].set_title('Volatility Factor Time Series')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Factor Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 波动率vs下行波动率相关性
    comparison = pd.DataFrame({
        'volatility': vol_values,
        'downside_vol': downside_values
    }).dropna()

    if len(comparison) > 0:
        axes[1, 1].scatter(
            comparison['volatility'],
            comparison['downside_vol'],
            alpha=0.1,
            s=1
        )
        corr = comparison.corr().iloc[0, 1]
        axes[1, 1].set_title(f'Volatility vs Downside Volatility (corr={corr:.3f})')
        axes[1, 1].set_xlabel('Volatility Factor')
        axes[1, 1].set_ylabel('Downside Volatility Factor')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('volatility_factor_analysis.png', dpi=150)
    print("\n✅ Plot saved as 'volatility_factor_analysis.png'")

    # 统计比较
    print("\n=== Factor Statistics Comparison ===")
    stats_df = pd.DataFrame({
        'Volatility': vol_values.describe(),
        'Downside_Vol': downside_values.describe()
    })
    print(stats_df)
