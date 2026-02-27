"""
动量因子 (Momentum Factor)

学术依据：Jegadeesh & Titman (1993)
"Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"

核心发现：过去12个月表现好的股票在未来3-12个月继续跑赢
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.factors.base import BaseFactor
from config.settings import MOMENTUM_LOOKBACK, MOMENTUM_SKIP


class MomentumFactor(BaseFactor):
    """
    动量因子

    公式：Momentum(t) = [P(t-skip) / P(t-lookback-skip)] - 1

    其中：
    - P(t) = t时刻的复权收盘价
    - lookback = 回看期（默认252天=12个月）
    - skip = 跳过最后N天（默认21天=1个月），避免短期反转效应
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
            回看期（交易日），默认252天（12个月）
        skip : int
            跳过最后N天（交易日），默认21天（1个月）
        name : str
            因子名称
        """
        super().__init__(name)
        self.lookback = lookback
        self.skip = skip

        logger.info(f"MomentumFactor initialized: lookback={lookback}, skip={skip}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算动量因子值

        Parameters
        ----------
        data : pd.DataFrame
            输入数据，MultiIndex (date, ticker)，包含'close'列

        Returns
        -------
        pd.Series
            动量因子值，MultiIndex (date, ticker)
        """
        logger.info(f"Calculating {self.name} factor...")

        # 转换为pivot格式 (date × ticker)
        close = data['close'].unstack(level='ticker')

        # 计算动量
        # close_skip: t-skip时刻的价格
        # close_lookback: t-lookback-skip时刻的价格
        close_skip = close.shift(self.skip)
        close_lookback = close.shift(self.lookback + self.skip)

        # 计算收益率
        momentum = (close_skip / close_lookback) - 1

        # 转换回MultiIndex格式
        momentum = momentum.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return momentum


class MultiPeriodMomentumFactor(BaseFactor):
    """
    多周期动量因子

    结合多个时间周期的动量信号
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
            多个回看周期（交易日）
        weights : list, optional
            各周期权重，默认等权
        skip : int
            跳过最后N天
        name : str
            因子名称
        """
        super().__init__(name)
        self.periods = periods
        self.weights = weights or [1.0 / len(periods)] * len(periods)
        self.skip = skip

        if len(self.periods) != len(self.weights):
            raise ValueError("Number of periods must match number of weights")

        logger.info(f"MultiPeriodMomentumFactor initialized: periods={periods}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算多周期动量因子"""
        logger.info(f"Calculating {self.name} factor...")

        close = data['close'].unstack(level='ticker')
        close_skip = close.shift(self.skip)

        momentum_list = []

        for period, weight in zip(self.periods, self.weights):
            close_lookback = close.shift(period + self.skip)
            period_momentum = ((close_skip / close_lookback) - 1) * weight
            momentum_list.append(period_momentum)

        # 加权平均
        multi_momentum = sum(momentum_list)
        multi_momentum = multi_momentum.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return multi_momentum


class ResidualMomentumFactor(BaseFactor):
    """
    残差动量因子

    剔除市场和行业影响后的特质动量
    需要市场收益率和行业分类数据
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
            回看期
        skip : int
            跳过天数
        market_returns : pd.Series
            市场收益率（日期索引）
        name : str
            因子名称
        """
        super().__init__(name)
        self.lookback = lookback
        self.skip = skip
        self.market_returns = market_returns

        logger.info(f"ResidualMomentumFactor initialized")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算残差动量因子"""
        logger.info(f"Calculating {self.name} factor...")

        close = data['close'].unstack(level='ticker')

        # 计算股票收益率
        returns = close.pct_change()

        # 计算累积收益率
        cum_returns = (1 + returns).rolling(
            window=self.lookback,
            min_periods=self.lookback // 2
        ).apply(lambda x: x.prod()) - 1

        # 如果提供了市场收益率，计算beta调整后的残差动量
        if self.market_returns is not None:
            # 这里简化处理，实际应该按ticker回归计算beta和alpha
            # 详细实现需要rolling regression
            logger.warning("Market adjustment not fully implemented, using raw momentum")

        # 跳过最后N天
        residual_momentum = cum_returns.shift(self.skip)
        residual_momentum = residual_momentum.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return residual_momentum


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
