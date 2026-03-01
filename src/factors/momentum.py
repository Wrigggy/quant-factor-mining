"""
Momentum Factor

Academic Reference: Jegadeesh & Titman (1993)
"Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"

Key Finding: Stocks performing well in the past 12 months continue to outperform in the next 3-12 months
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.factors.base import BaseFactor
from config.settings import MOMENTUM_LOOKBACK, MOMENTUM_SKIP


class MomentumFactor(BaseFactor):
    """
    Momentum Factor

    Formula: Momentum(t) = [P(t-skip) / P(t-lookback-skip)] - 1

    Where:
    - P(t) = adjusted close price at time t
    - lookback = lookback period (default 252 days = 12 months)
    - skip = skip last N days (default 21 days = 1 month), to avoid short-term reversal effect
    """

    def __init__(
        self,
        lookback: int = MOMENTUM_LOOKBACK,
        skip: int = MOMENTUM_SKIP,
        name: str = "Momentum"
    ):
        """
        Parameters
        ----------
        lookback : int
            Lookback period (trading days), default 252 days (12 months)
        skip : int
            Skip last N days (trading days), default 21 days (1 month)
        name : str
            Factor name
        """
        super().__init__(name)
        self.lookback = lookback
        self.skip = skip

        logger.info(f"MomentumFactor initialized: lookback={lookback}, skip={skip}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum factor values

        Parameters
        ----------
        data : pd.DataFrame
            Input data, MultiIndex (date, ticker), contains 'close' column

        Returns
        -------
        pd.Series
            Momentum factor values, MultiIndex (date, ticker)
        """
        self._validate_data(data)
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)

        # Calculate momentum
        # close_skip: price at t-skip
        # close_lookback: price at t-lookback-skip
        close_skip = close.shift(self.skip)
        close_lookback = close.shift(self.lookback + self.skip)

        # Calculate returns
        momentum = (close_skip / close_lookback) - 1

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(momentum)


class MultiPeriodMomentumFactor(BaseFactor):
    """
    Multi-Period Momentum Factor

    Combines momentum signals from multiple time periods
    """

    def __init__(
        self,
        periods: list = [21, 63, 126, 252],
        weights: list = None,
        skip: int = MOMENTUM_SKIP,
        name: str = "MultiPeriodMomentum"
    ):
        """
        Parameters
        ----------
        periods : list
            Multiple lookback periods (trading days)
        weights : list, optional
            Weights for each period, default equal weight
        skip : int
            Skip last N days
        name : str
            Factor name
        """
        super().__init__(name)
        self.periods = periods
        self.weights = weights or [1.0 / len(periods)] * len(periods)
        self.skip = skip

        if len(self.periods) != len(self.weights):
            raise ValueError("Number of periods must match number of weights")

        logger.info(f"MultiPeriodMomentumFactor initialized: periods={periods}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate multi-period momentum factor"""
        self._validate_data(data)
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)
        close_skip = close.shift(self.skip)

        momentum_list = []

        for period, weight in zip(self.periods, self.weights):
            close_lookback = close.shift(period + self.skip)
            period_momentum = ((close_skip / close_lookback) - 1) * weight
            momentum_list.append(period_momentum)

        # Weighted average
        multi_momentum = sum(momentum_list)

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(multi_momentum)


class ResidualMomentumFactor(BaseFactor):
    """
    Residual Momentum Factor

    Idiosyncratic momentum after removing market and industry effects
    Requires market returns and industry classification data
    """

    def __init__(
        self,
        lookback: int = MOMENTUM_LOOKBACK,
        skip: int = MOMENTUM_SKIP,
        market_returns: pd.Series = None,
        name: str = "ResidualMomentum"
    ):
        """
        Parameters
        ----------
        lookback : int
            Lookback period
        skip : int
            Skip days
        market_returns : pd.Series
            Market returns (date index)
        name : str
            Factor name
        """
        super().__init__(name)
        self.lookback = lookback
        self.skip = skip
        self.market_returns = market_returns

        logger.info(f"ResidualMomentumFactor initialized")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate residual momentum factor"""
        self._validate_data(data)
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)

        # Calculate stock returns
        returns = close.pct_change()

        # Calculate cumulative returns
        cum_returns = (1 + returns).rolling(
            window=self.lookback,
            min_periods=self.lookback // 2
        ).apply(lambda x: x.prod()) - 1

        # If market returns provided, calculate beta-adjusted residual momentum
        if self.market_returns is not None:
            # Simplified here, should actually calculate beta and alpha per ticker via regression
            # Detailed implementation requires rolling regression
            logger.warning("Market adjustment not fully implemented, using raw momentum")

        # Skip last N days
        residual_momentum = cum_returns.shift(self.skip)

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(residual_momentum)


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    import matplotlib.pyplot as plt

    # 配置日志
    logger.add("momentum.log", rotation="10 MB")

    # 获取测试数据
    fetcher = YFinanceFetcher()
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"]
    data = fetcher.fetch_batch(test_tickers, "2020-01-01", "2024-01-01")

    preprocessor = DataPreprocessor()
    clean_data, report = preprocessor.validate_data_quality(data)

    print("\n=== Data Quality Report ===")
    print(f"Shape: {clean_data.shape}")
    print(f"Date range: {report['date_range']}")

    # 测试动量因子
    print("\n=== Testing Momentum Factor ===")
    momentum = MomentumFactor(lookback=252, skip=21)
    momentum_values = momentum.compute(clean_data, normalize=True)

    print(f"\nMomentum factor shape: {momentum_values.shape}")
    print(f"\nSample values:")
    print(momentum_values.head(20))

    # 统计分析
    print("\n=== Factor Statistics ===")
    print(momentum_values.describe())

    # 可视化因子分布
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    momentum_values.hist(bins=50, edgecolor='black')
    plt.title('Momentum Factor Distribution (Normalized)')
    plt.xlabel('Factor Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # 按ticker显示时间序列
    momentum_pivot = momentum_values.unstack(level='ticker')
    for ticker in test_tickers[:3]:
        if ticker in momentum_pivot.columns:
            momentum_pivot[ticker].plot(label=ticker, alpha=0.7)
    plt.title('Momentum Factor Time Series (Selected Stocks)')
    plt.xlabel('Date')
    plt.ylabel('Factor Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('momentum_factor_analysis.png', dpi=150)
    print("\n✅ Plot saved as 'momentum_factor_analysis.png'")

    # 测试多周期动量
    print("\n=== Testing Multi-Period Momentum Factor ===")
    multi_momentum = MultiPeriodMomentumFactor(
        periods=[21, 63, 126, 252],
        weights=[0.1, 0.2, 0.3, 0.4]
    )
    multi_momentum_values = multi_momentum.compute(clean_data, normalize=True)

    print(f"\nMulti-period momentum shape: {multi_momentum_values.shape}")
    print("\nCorrelation between single and multi-period momentum:")

    # 合并以比较
    comparison = pd.DataFrame({
        'momentum': momentum_values,
        'multi_momentum': multi_momentum_values
    }).dropna()

    if len(comparison) > 0:
        correlation = comparison.corr().iloc[0, 1]
        print(f"Correlation: {correlation:.4f}")
