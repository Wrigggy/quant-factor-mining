"""
Mean Reversion Factor

Academic Reference: Jegadeesh (1990)
"Evidence of Predictable Behavior of Security Returns"

Key Finding: Short-term (1-month) reversal effect exists, past losers outperform winners
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.factors.base import BaseFactor
from config.settings import MEAN_REVERSION_LOOKBACK


class MeanReversionFactor(BaseFactor):
    """
    Mean Reversion Factor (Short-term Reversal)

    Formula: MeanReversion(t) = -1 × [(P(t) / P(t-lookback)) - 1]

    Where:
    - P(t) = adjusted close price at time t
    - lookback = lookback period (default 21 days = 1 month)
    - negative sign: buy stocks with large declines (contrarian strategy)
    """

    def __init__(
        self,
        lookback: int = MEAN_REVERSION_LOOKBACK,
        name: str = "MeanReversion"
    ):
        """
        Parameters
        ----------
        lookback : int
            Lookback period (trading days), default 21 days (1 month)
        name : str
            Factor name
        """
        super().__init__(name)
        self.lookback = lookback

        logger.info(f"MeanReversionFactor initialized: lookback={lookback}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate mean reversion factor values

        Parameters
        ----------
        data : pd.DataFrame
            Input data, MultiIndex (date, ticker), contains 'close' column

        Returns
        -------
        pd.Series
            Mean reversion factor values, MultiIndex (date, ticker)
        """
        self._validate_data(data)
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)

        # Calculate short-term returns
        short_term_return = (close / close.shift(self.lookback)) - 1

        # Reversal: negative sign means buying stocks with large declines
        reversal = -1 * short_term_return

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(reversal)


class HighLowMeanReversionFactor(BaseFactor):
    """
    High-Low Mean Reversion Factor

    Measures current price position relative to past N-day high/low prices
    """

    def __init__(
        self,
        lookback: int = 21,
        name: str = "HighLowMeanReversion"
    ):
        """
        Parameters
        ----------
        lookback : int
            Lookback period
        name : str
            Factor name
        """
        super().__init__(name)
        self.lookback = lookback

        logger.info(f"HighLowMeanReversionFactor initialized: lookback={lookback}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate High/Low-based mean reversion factor

        Formula: (Close - Min) / (Max - Min) - 0.5
        Then apply negative sign for reversal
        """
        self._validate_data(data, required=['close', 'high', 'low'])
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)
        high = data['high'].unstack(level='ticker')
        low = data['low'].unstack(level='ticker')

        # Calculate rolling high and low
        rolling_high = high.rolling(window=self.lookback).max()
        rolling_low = low.rolling(window=self.lookback).min()

        # Current price position in [min, max] range (0 to 1)
        position = (close - rolling_low) / (rolling_high - rolling_low + 1e-9)

        # Center to [-0.5, 0.5]
        centered_position = position - 0.5

        # Reversal: higher price (near high) gives negative factor value, expecting downward reversal
        reversal = -1 * centered_position

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(reversal)


class RSIMeanReversionFactor(BaseFactor):
    """
    RSI-based Mean Reversion Factor

    RSI (Relative Strength Index) is a classic overbought/oversold indicator
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        name: str = "RSIMeanReversion"
    ):
        """
        Parameters
        ----------
        period : int
            RSI calculation period, default 14 days
        overbought : float
            Overbought threshold, default 70
        oversold : float
            Oversold threshold, default 30
        name : str
            Factor name
        """
        super().__init__(name)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        logger.info(f"RSIMeanReversionFactor initialized: period={period}")

    def _calculate_rsi(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        # Calculate price changes
        delta = close.diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate average gains/losses (using EWM, more aligned with original RSI definition)
        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI mean reversion factor

        Overbought (RSI > 70): negative factor value, expecting decline
        Oversold (RSI < 30): positive factor value, expecting rise
        """
        self._validate_data(data)
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)

        # Calculate RSI
        rsi = self._calculate_rsi(close)

        # Convert to mean reversion signal
        # Center RSI around 50
        reversal = 50 - rsi

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(reversal)


class OvernightMeanReversionFactor(BaseFactor):
    """
    Overnight Mean Reversion Factor

    Exploits reversal effect of overnight returns (previous close to today's open)
    """

    def __init__(
        self,
        lookback: int = 5,
        name: str = "OvernightMeanReversion"
    ):
        """
        Parameters
        ----------
        lookback : int
            Number of days to accumulate overnight returns
        name : str
            Factor name
        """
        super().__init__(name)
        self.lookback = lookback

        logger.info(f"OvernightMeanReversionFactor initialized: lookback={lookback}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate overnight mean reversion factor

        Formula: Negative of cumulative overnight returns
        Overnight return = Open(t) / Close(t-1) - 1
        """
        self._validate_data(data, required=['close', 'open'])
        logger.info(f"Calculating {self.name} factor...")

        close = self._unstack_close(data)
        open_price = data['open'].unstack(level='ticker')

        # Calculate overnight returns
        overnight_return = (open_price / close.shift(1)) - 1

        # Accumulate past N days of overnight returns
        cum_overnight = overnight_return.rolling(window=self.lookback).sum()

        # Reversal signal
        reversal = -1 * cum_overnight

        logger.info(f"{self.name} calculation complete")
        return self._stack_result(reversal)


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    import matplotlib.pyplot as plt

    # 配置日志
    logger.add("mean_reversion.log", rotation="10 MB")

    # 获取测试数据
    fetcher = YFinanceFetcher()
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"]
    data = fetcher.fetch_batch(test_tickers, "2020-01-01", "2024-01-01")

    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # 测试均值回归因子
    print("\n=== Testing Mean Reversion Factor ===")
    mean_reversion = MeanReversionFactor(lookback=21)
    mr_values = mean_reversion.compute(clean_data, normalize=True)

    print(f"\nMean Reversion factor shape: {mr_values.shape}")
    print(f"\nSample values:")
    print(mr_values.head(20))

    # 测试RSI因子
    print("\n=== Testing RSI Mean Reversion Factor ===")
    rsi_factor = RSIMeanReversionFactor(period=14)
    rsi_values = rsi_factor.compute(clean_data, normalize=True)

    print(f"\nRSI factor shape: {rsi_values.shape}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 均值回归因子分布
    mr_values.hist(bins=50, ax=axes[0, 0], edgecolor='black')
    axes[0, 0].set_title('Mean Reversion Factor Distribution')
    axes[0, 0].set_xlabel('Factor Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. RSI因子分布
    rsi_values.hist(bins=50, ax=axes[0, 1], edgecolor='black')
    axes[0, 1].set_title('RSI Mean Reversion Factor Distribution')
    axes[0, 1].set_xlabel('Factor Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 均值回归因子时间序列
    mr_pivot = mr_values.unstack(level='ticker')
    for ticker in test_tickers[:3]:
        if ticker in mr_pivot.columns:
            mr_pivot[ticker].plot(ax=axes[1, 0], label=ticker, alpha=0.7)
    axes[1, 0].set_title('Mean Reversion Factor Time Series')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Factor Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 两个因子的相关性
    comparison = pd.DataFrame({
        'mean_reversion': mr_values,
        'rsi': rsi_values
    }).dropna()

    if len(comparison) > 0:
        axes[1, 1].scatter(
            comparison['mean_reversion'],
            comparison['rsi'],
            alpha=0.1,
            s=1
        )
        axes[1, 1].set_title(f'Factor Correlation (corr={comparison.corr().iloc[0,1]:.3f})')
        axes[1, 1].set_xlabel('Mean Reversion Factor')
        axes[1, 1].set_ylabel('RSI Factor')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mean_reversion_factor_analysis.png', dpi=150)
    print("\n✅ Plot saved as 'mean_reversion_factor_analysis.png'")
