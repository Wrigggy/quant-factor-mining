"""
波动率因子 (Volatility Factor)

学术依据：Ang et al. (2006, 2009)
"The Cross-Section of Volatility and Expected Returns"
"High Idiosyncratic Volatility and Low Returns"

核心发现：低波动率股票违反CAPM预期，实际表现优于高波动率股票
这被称为"低波动率异象" (Low Volatility Anomaly)
"""

import pandas as pd
import numpy as np
from loguru import logger

from src.factors.base import BaseFactor
from config.settings import VOLATILITY_WINDOW, VOLATILITY_ANNUALIZE


class VolatilityFactor(BaseFactor):
    """
    波动率因子（低波异象）

    公式：Volatility(t) = -1 × std(returns[t-window:t]) × √252

    其中：
    - returns = 对数收益率 log(P_t / P_{t-1})
    - window = 滚动窗口（默认63天=3个月）
    - √252 = 年化因子
    - 负号：低波动率为正信号（买入低波动率股票）
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
            滚动窗口（交易日），默认63天（3个月）
        annualize : bool
            是否年化波动率
        use_log_returns : bool
            是否使用对数收益率
        name : str
            因子名称
        """
        super().__init__(name)
        self.window = window
        self.annualize = annualize
        self.use_log_returns = use_log_returns

        logger.info(f"VolatilityFactor initialized: window={window}, annualize={annualize}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算波动率因子值

        Parameters
        ----------
        data : pd.DataFrame
            输入数据，MultiIndex (date, ticker)，包含'close'列

        Returns
        -------
        pd.Series
            波动率因子值，MultiIndex (date, ticker)
        """
        logger.info(f"Calculating {self.name} factor...")

        # 转换为pivot格式 (date × ticker)
        close = data['close'].unstack(level='ticker')

        # 计算收益率
        if self.use_log_returns:
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()

        # 计算滚动标准差
        volatility = returns.rolling(window=self.window, min_periods=self.window // 2).std()

        # 年化
        if self.annualize:
            volatility = volatility * np.sqrt(252)

        # 低波异象：负号表示低波动率为正信号
        low_vol_factor = -1 * volatility

        # 转换回MultiIndex格式
        low_vol_factor = low_vol_factor.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return low_vol_factor


class DownsideVolatilityFactor(BaseFactor):
    """
    下行波动率因子

    只考虑负收益的波动率，对下行风险更敏感
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
            滚动窗口
        annualize : bool
            是否年化
        threshold : float
            最小可接受收益率（MAR），默认0
        name : str
            因子名称
        """
        super().__init__(name)
        self.window = window
        self.annualize = annualize
        self.threshold = threshold

        logger.info(f"DownsideVolatilityFactor initialized: window={window}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算下行波动率因子

        公式：sqrt(mean(min(0, returns - threshold)^2))
        """
        logger.info(f"Calculating {self.name} factor...")

        close = data['close'].unstack(level='ticker')
        returns = np.log(close / close.shift(1))

        # 只保留低于阈值的收益
        downside_returns = returns.apply(lambda x: np.where(x < self.threshold, x - self.threshold, 0))

        # 计算下行标准差
        downside_vol = downside_returns.rolling(
            window=self.window,
            min_periods=self.window // 2
        ).apply(lambda x: np.sqrt(np.mean(x**2)))

        # 年化
        if self.annualize:
            downside_vol = downside_vol * np.sqrt(252)

        # 低下行波动率为正信号
        low_downside_factor = -1 * downside_vol

        low_downside_factor = low_downside_factor.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return low_downside_factor


class IdiosyncraticVolatilityFactor(BaseFactor):
    """
    特质波动率因子

    剔除市场和行业因素后的残差波动率
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
            滚动窗口
        market_returns : pd.Series
            市场收益率（日期索引）
        annualize : bool
            是否年化
        name : str
            因子名称
        """
        super().__init__(name)
        self.window = window
        self.market_returns = market_returns
        self.annualize = annualize

        logger.info(f"IdiosyncraticVolatilityFactor initialized")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算特质波动率因子

        使用rolling regression计算残差，然后计算残差的标准差
        """
        logger.info(f"Calculating {self.name} factor...")

        close = data['close'].unstack(level='ticker')
        returns = np.log(close / close.shift(1))

        if self.market_returns is None:
            # 如果没有提供市场收益，使用等权平均作为市场代理
            logger.warning("No market returns provided, using equal-weight average")
            market_proxy = returns.mean(axis=1)
        else:
            market_proxy = self.market_returns

        # 简化实现：计算与市场的相关性，然后计算残差波动率
        # 完整实现应该使用rolling regression
        idio_vol = pd.DataFrame(index=returns.index, columns=returns.columns)

        for ticker in returns.columns:
            ticker_returns = returns[ticker].dropna()

            # 对齐市场收益
            aligned_market = market_proxy.reindex(ticker_returns.index)

            # 计算beta（滚动窗口）
            cov = ticker_returns.rolling(self.window).cov(aligned_market)
            market_var = aligned_market.rolling(self.window).var()
            beta = cov / (market_var + 1e-9)

            # 计算残差
            residuals = ticker_returns - beta * aligned_market

            # 计算残差波动率
            residual_vol = residuals.rolling(
                window=self.window,
                min_periods=self.window // 2
            ).std()

            idio_vol[ticker] = residual_vol

        # 年化
        if self.annualize:
            idio_vol = idio_vol * np.sqrt(252)

        # 低特质波动率为正信号
        low_idio_factor = -1 * idio_vol

        low_idio_factor = low_idio_factor.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return low_idio_factor


class RangeVolatilityFactor(BaseFactor):
    """
    价格区间波动率因子

    基于最高价和最低价的波动率估计
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
            滚动窗口
        annualize : bool
            是否年化
        name : str
            因子名称
        """
        super().__init__(name)
        self.window = window
        self.annualize = annualize

        logger.info(f"RangeVolatilityFactor initialized: window={window}")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算基于High-Low区间的波动率

        Parkinson公式：
        σ^2 = (1 / (4 * ln(2))) * E[(ln(H/L))^2]
        """
        logger.info(f"Calculating {self.name} factor...")

        high = data['high'].unstack(level='ticker')
        low = data['low'].unstack(level='ticker')

        # 计算对数比率
        log_hl_ratio = np.log(high / low)

        # Parkinson estimator
        parkinson_vol = log_hl_ratio.rolling(
            window=self.window,
            min_periods=self.window // 2
        ).apply(lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))))

        # 年化
        if self.annualize:
            parkinson_vol = parkinson_vol * np.sqrt(252)

        # 低波动率为正信号
        low_range_factor = -1 * parkinson_vol

        low_range_factor = low_range_factor.stack(dropna=False)

        logger.info(f"{self.name} calculation complete")
        return low_range_factor


class GARCHVolatilityFactor(BaseFactor):
    """
    GARCH模型波动率因子

    使用GARCH(1,1)模型估计条件波动率
    需要arch包
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
            估计窗口
        name : str
            因子名称
        """
        super().__init__(name)
        self.window = window

        logger.info(f"GARCHVolatilityFactor initialized (requires 'arch' package)")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算GARCH波动率

        注：这是一个简化实现，完整版本需要arch包
        """
        logger.warning("GARCH implementation requires 'arch' package, using simplified approach")

        # 简化：使用EWMA作为GARCH的近似
        close = data['close'].unstack(level='ticker')
        returns = np.log(close / close.shift(1))

        # EWMA波动率（类似GARCH）
        ewma_vol = returns.ewm(span=self.window // 10, adjust=False).std() * np.sqrt(252)

        # 低波动率为正信号
        low_garch_factor = -1 * ewma_vol

        low_garch_factor = low_garch_factor.stack(dropna=False)

        logger.info(f"{self.name} calculation complete (simplified)")
        return low_garch_factor


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
