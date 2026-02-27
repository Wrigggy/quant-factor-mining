"""
均值回归因子 (Mean Reversion Factor)

学术依据：Jegadeesh (1990)
"Evidence of Predictable Behavior of Security Returns"

核心发现：短期（1个月）存在反转效应，过去的输家会跑赢赢家
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.factors.base import BaseFactor
from config.settings import MEAN_REVERSION_LOOKBACK


class MeanReversionFactor(BaseFactor):
    """
    均值回归因子（短期反转）

    公式：MeanReversion(t) = -1 × [(P(t) / P(t-lookback)) - 1]

    其中：
    - P(t) = t时刻的复权收盘价
    - lookback = 回看期（默认21天=1个月）
    - 负号：买入跌幅大的股票（contrarian策略）
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
            回看期（交易日），默认21天（1个月）
        name : str
            因子名称
        """
        super().__init__(name)
        self.lookback = lookback

        logger.info(f"MeanReversionFactor initialized: lookback={lookback}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算均值回归因子值

        Parameters
        ----------
        data : pd.DataFrame
            输入数据，MultiIndex (date, ticker)，包含'close'列

        Returns
        -------
        pd.Series
            均值回归因子值，MultiIndex (date, ticker)
        """
        logger.info(f"Calculating {self.name} factor...")

        # 转换为pivot格式 (date × ticker)
        close = data['close'].unstack(level='ticker')

        # 计算短期收益率
        short_term_return = (close / close.shift(self.lookback)) - 1

        # 反转：负号表示买入跌幅大的股票
        reversal = -1 * short_term_return

        # 转换回MultiIndex格式
        reversal = reversal.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return reversal


class HighLowMeanReversionFactor(BaseFactor):
    """
    基于最高最低价的均值回归因子

    衡量当前价格相对于过去N天最高/最低价的位置
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
            回看期
        name : str
            因子名称
        """
        super().__init__(name)
        self.lookback = lookback

        logger.info(f"HighLowMeanReversionFactor initialized: lookback={lookback}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算基于High/Low的均值回归因子

        公式：(Close - Min) / (Max - Min) - 0.5
        然后取负号表示反转
        """
        logger.info(f"Calculating {self.name} factor...")

        close = data['close'].unstack(level='ticker')
        high = data['high'].unstack(level='ticker')
        low = data['low'].unstack(level='ticker')

        # 计算rolling最高和最低
        rolling_high = high.rolling(window=self.lookback).max()
        rolling_low = low.rolling(window=self.lookback).min()

        # 当前价格在[min, max]区间的位置（0到1）
        position = (close - rolling_low) / (rolling_high - rolling_low + 1e-9)

        # 中心化到[-0.5, 0.5]
        centered_position = position - 0.5

        # 反转：价格越高（接近最高价），因子值越负，预期反转向下
        reversal = -1 * centered_position

        reversal = reversal.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return reversal


class RSIMeanReversionFactor(BaseFactor):
    """
    基于RSI的均值回归因子

    RSI (Relative Strength Index) 是经典的超买超卖指标
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
            RSI计算周期，默认14天
        overbought : float
            超买阈值，默认70
        oversold : float
            超卖阈值，默认30
        name : str
            因子名称
        """
        super().__init__(name)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

        logger.info(f"RSIMeanReversionFactor initialized: period={period}")

    def _calculate_rsi(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        计算RSI

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        # 计算价格变化
        delta = close.diff()

        # 分离涨跌
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # 计算平均涨跌（使用EWM，更符合RSI原始定义）
        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()

        # 计算RS和RSI
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算RSI均值回归因子

        超买（RSI > 70）：因子值为负，预期下跌
        超卖（RSI < 30）：因子值为正，预期上涨
        """
        logger.info(f"Calculating {self.name} factor...")

        close = data['close'].unstack(level='ticker')

        # 计算RSI
        rsi = self._calculate_rsi(close)

        # 转换为均值回归信号
        # RSI中心化到50
        reversal = 50 - rsi

        # 可选：只在极端情况下产生信号
        # reversal = reversal.apply(
        #     lambda x: x if (x > (50 - self.oversold) or x < (self.overbought - 50)) else 0
        # )

        reversal = reversal.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return reversal


class OvernightMeanReversionFactor(BaseFactor):
    """
    隔夜均值回归因子

    利用隔夜收益（前日收盘到今日开盘）的反转效应
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
            累积隔夜收益的天数
        name : str
            因子名称
        """
        super().__init__(name)
        self.lookback = lookback

        logger.info(f"OvernightMeanReversionFactor initialized: lookback={lookback}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算隔夜均值回归因子

        公式：累积隔夜收益的负值
        隔夜收益 = Open(t) / Close(t-1) - 1
        """
        logger.info(f"Calculating {self.name} factor...")

        close = data['close'].unstack(level='ticker')
        open_price = data['open'].unstack(level='ticker')

        # 计算隔夜收益
        overnight_return = (open_price / close.shift(1)) - 1

        # 累积过去N天的隔夜收益
        cum_overnight = overnight_return.rolling(window=self.lookback).sum()

        # 反转信号
        reversal = -1 * cum_overnight

        reversal = reversal.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return reversal


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
