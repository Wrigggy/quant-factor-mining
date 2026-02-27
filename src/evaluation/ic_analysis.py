"""
IC (Information Coefficient) 深度分析模块
包括IC时间序列、IC衰减、IC一致性等高级分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from scipy import stats
from loguru import logger


class ICAnalyzer:
    """
    IC分析器

    功能：
    1. IC时间序列分析
    2. IC衰减分析（不同前瞻期）
    3. IC一致性分析
    4. IC分布分析
    5. 滚动IC分析
    """

    def __init__(self, method: str = 'spearman'):
        """
        Parameters
        ----------
        method : str
            相关系数计算方法，'spearman'（默认）或'pearson'
        """
        self.method = method
        logger.info(f"ICAnalyzer initialized with method: {method}")

    def compute_ic_series(
        self,
        factor: pd.Series,
        returns: pd.Series,
        method: Optional[str] = None
    ) -> pd.Series:
        """
        计算IC时间序列

        Parameters
        ----------
        factor : pd.Series
            因子值，MultiIndex (date, ticker)
        returns : pd.Series
            未来收益率，MultiIndex (date, ticker)
        method : str, optional
            相关系数方法，默认使用初始化时的方法

        Returns
        -------
        pd.Series
            IC时间序列，indexed by date
        """
        method = method or self.method

        # 对齐数据
        aligned_data = pd.DataFrame({
            'factor': factor,
            'return': returns
        }).dropna()

        # 按日期分组计算IC
        ic_series = aligned_data.groupby(level='date').apply(
            lambda x: x['factor'].corr(x['return'], method=method)
        )

        logger.info(f"IC series computed: {len(ic_series)} dates")
        return ic_series

    def compute_forward_returns(
        self,
        prices: pd.DataFrame,
        periods: List[int]
    ) -> Dict[int, pd.Series]:
        """
        计算多个周期的前瞻收益率

        Parameters
        ----------
        prices : pd.DataFrame
            价格数据，MultiIndex (date, ticker)，包含'close'列
        periods : List[int]
            前瞻周期列表（交易日）

        Returns
        -------
        Dict[int, pd.Series]
            字典，key为周期，value为收益率Series
        """
        logger.info(f"Computing forward returns for periods: {periods}")

        if 'close' in prices.columns:
            close = prices['close'].unstack(level='ticker')
        else:
            close = prices

        forward_returns = {}

        for period in periods:
            # 计算前瞻收益率
            fwd_return = close.pct_change(period).shift(-period)
            fwd_return = fwd_return.stack(dropna=False)
            forward_returns[period] = fwd_return

            logger.debug(f"Period {period}: {len(fwd_return.dropna())} valid returns")

        return forward_returns

    def compute_ic_decay(
        self,
        factor: pd.Series,
        prices: pd.DataFrame,
        max_period: int = 63,
        step: int = 1
    ) -> pd.DataFrame:
        """
        计算IC衰减曲线

        Parameters
        ----------
        factor : pd.Series
            因子值
        prices : pd.DataFrame
            价格数据
        max_period : int
            最大前瞻期（默认63天=3个月）
        step : int
            步长

        Returns
        -------
        pd.DataFrame
            IC衰减数据，columns=['period', 'IC_mean', 'IC_std', 't_stat']
        """
        logger.info(f"Computing IC decay up to {max_period} periods...")

        periods = list(range(1, max_period + 1, step))
        forward_returns = self.compute_forward_returns(prices, periods)

        decay_data = []

        for period in periods:
            ic_series = self.compute_ic_series(factor, forward_returns[period])

            decay_data.append({
                'period': period,
                'IC_mean': ic_series.mean(),
                'IC_std': ic_series.std(),
                't_stat': ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series)) if ic_series.std() > 0 else 0,
                'IC_positive_pct': (ic_series > 0).mean()
            })

        decay_df = pd.DataFrame(decay_data)
        logger.info("IC decay computation complete")

        return decay_df

    def compute_rolling_ic(
        self,
        factor: pd.Series,
        returns: pd.Series,
        window: int = 63
    ) -> pd.Series:
        """
        计算滚动IC

        Parameters
        ----------
        factor : pd.Series
            因子值
        returns : pd.Series
            收益率
        window : int
            滚动窗口（默认63天=3个月）

        Returns
        -------
        pd.Series
            滚动IC时间序列
        """
        logger.info(f"Computing rolling IC with window={window}...")

        aligned_data = pd.DataFrame({
            'factor': factor,
            'return': returns
        }).dropna()

        # 获取所有日期
        dates = aligned_data.index.get_level_values('date').unique().sort_values()

        rolling_ic = []

        for i in range(window, len(dates)):
            window_dates = dates[i-window:i]
            window_data = aligned_data.loc[aligned_data.index.get_level_values('date').isin(window_dates)]

            if len(window_data) > 0:
                ic = window_data['factor'].corr(window_data['return'], method=self.method)
                rolling_ic.append({'date': dates[i], 'rolling_IC': ic})

        rolling_ic_series = pd.DataFrame(rolling_ic).set_index('date')['rolling_IC']

        logger.info(f"Rolling IC computed: {len(rolling_ic_series)} points")
        return rolling_ic_series

    def analyze_ic_consistency(
        self,
        ic_series: pd.Series,
        freq: str = 'M'
    ) -> pd.DataFrame:
        """
        分析IC一致性（按月度、季度等聚合）

        Parameters
        ----------
        ic_series : pd.Series
            IC时间序列
        freq : str
            聚合频率，'M'=月度, 'Q'=季度, 'Y'=年度

        Returns
        -------
        pd.DataFrame
            一致性统计
        """
        logger.info(f"Analyzing IC consistency with frequency: {freq}")

        # 按频率聚合
        ic_grouped = ic_series.groupby(pd.Grouper(freq=freq)).agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('positive_pct', lambda x: (x > 0).mean()),
            ('count', 'count')
        ])

        # 计算t统计量
        ic_grouped['t_stat'] = ic_grouped['mean'] / ic_grouped['std'] * np.sqrt(ic_grouped['count'])

        # 总体一致性
        overall_consistency = (ic_series > 0).mean()

        logger.info(f"Overall IC consistency (positive %): {overall_consistency:.2%}")

        return ic_grouped

    def compute_ic_statistics(self, ic_series: pd.Series) -> Dict:
        """
        计算IC的全面统计指标

        Parameters
        ----------
        ic_series : pd.Series
            IC时间序列

        Returns
        -------
        Dict
            统计指标字典
        """
        logger.info("Computing comprehensive IC statistics...")

        stats_dict = {
            'mean': ic_series.mean(),
            'std': ic_series.std(),
            'median': ic_series.median(),
            'min': ic_series.min(),
            'max': ic_series.max(),
            'skewness': ic_series.skew(),
            'kurtosis': ic_series.kurtosis(),
            'positive_pct': (ic_series > 0).mean(),
            'negative_pct': (ic_series < 0).mean(),
            't_stat': ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series)),
            'p_value': 2 * (1 - stats.t.cdf(abs(ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series))), len(ic_series) - 1)),
            'sharpe_ratio': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            'info_ratio': ic_series.mean() / ic_series.std() * np.sqrt(252) if ic_series.std() > 0 else 0,  # 年化
        }

        logger.info(f"IC Mean: {stats_dict['mean']:.4f}, t-stat: {stats_dict['t_stat']:.2f}")

        return stats_dict

    def plot_ic_analysis(
        self,
        ic_series: pd.Series,
        ic_decay: Optional[pd.DataFrame] = None,
        rolling_ic: Optional[pd.Series] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        绘制IC分析图表

        Parameters
        ----------
        ic_series : pd.Series
            IC时间序列
        ic_decay : pd.DataFrame, optional
            IC衰减数据
        rolling_ic : pd.Series, optional
            滚动IC
        figsize : Tuple[int, int]
            图表尺寸
        """
        logger.info("Generating IC analysis plots...")

        n_plots = 3 + (1 if ic_decay is not None else 0) + (1 if rolling_ic is not None else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        ax_idx = 0

        # 1. IC时间序列
        axes[ax_idx].plot(ic_series.index, ic_series.values, alpha=0.7, linewidth=1)
        axes[ax_idx].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[ax_idx].axhline(y=ic_series.mean(), color='red', linestyle='--', linewidth=0.8,
                            label=f'Mean IC: {ic_series.mean():.4f}')
        axes[ax_idx].fill_between(ic_series.index, 0, ic_series.values,
                                  where=ic_series.values > 0, alpha=0.3, color='green')
        axes[ax_idx].fill_between(ic_series.index, 0, ic_series.values,
                                  where=ic_series.values < 0, alpha=0.3, color='red')
        axes[ax_idx].set_title('IC Time Series')
        axes[ax_idx].set_ylabel('IC')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

        # 2. IC分布直方图
        axes[ax_idx].hist(ic_series.dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[ax_idx].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        axes[ax_idx].axvline(x=ic_series.mean(), color='red', linestyle='--', linewidth=0.8,
                            label=f'Mean: {ic_series.mean():.4f}')
        axes[ax_idx].set_title('IC Distribution')
        axes[ax_idx].set_xlabel('IC Value')
        axes[ax_idx].set_ylabel('Frequency')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

        # 3. IC累积分布
        ic_sorted = np.sort(ic_series.dropna())
        cumulative = np.arange(1, len(ic_sorted) + 1) / len(ic_sorted)
        axes[ax_idx].plot(ic_sorted, cumulative, linewidth=2)
        axes[ax_idx].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        axes[ax_idx].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8)
        axes[ax_idx].set_title('IC Cumulative Distribution')
        axes[ax_idx].set_xlabel('IC Value')
        axes[ax_idx].set_ylabel('Cumulative Probability')
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

        # 4. IC衰减（如果提供）
        if ic_decay is not None:
            ax = axes[ax_idx]
            ax.plot(ic_decay['period'], ic_decay['IC_mean'], marker='o', linewidth=2)
            ax.fill_between(ic_decay['period'],
                           ic_decay['IC_mean'] - ic_decay['IC_std'],
                           ic_decay['IC_mean'] + ic_decay['IC_std'],
                           alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            ax.set_title('IC Decay Analysis')
            ax.set_xlabel('Forward Period (days)')
            ax.set_ylabel('Mean IC')
            ax.grid(True, alpha=0.3)
            ax_idx += 1

        # 5. 滚动IC（如果提供）
        if rolling_ic is not None:
            ax = axes[ax_idx]
            ax.plot(rolling_ic.index, rolling_ic.values, linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
            ax.axhline(y=rolling_ic.mean(), color='red', linestyle='--', linewidth=0.8,
                      label=f'Mean: {rolling_ic.mean():.4f}')
            ax.set_title('Rolling IC')
            ax.set_xlabel('Date')
            ax.set_ylabel('Rolling IC')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        logger.info("IC analysis plots generated")


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    from src.data.cache_manager import CacheManager
    from src.factors.momentum import MomentumFactor

    logger.add("ic_analysis_test.log", rotation="10 MB")

    # 获取数据
    fetcher = YFinanceFetcher()
    cache = CacheManager()
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"]

    data = cache.get_or_fetch("US", "test", test_tickers, "2020-01-01", "2024-01-01", fetcher)

    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # 计算因子
    momentum = MomentumFactor(lookback=252, skip=21)
    factor_values = momentum.compute(clean_data, normalize=True)

    # IC分析
    analyzer = ICAnalyzer(method='spearman')

    # 1. 计算前瞻收益
    forward_returns = analyzer.compute_forward_returns(clean_data, periods=[21])

    # 2. 计算IC时间序列
    ic_series = analyzer.compute_ic_series(factor_values, forward_returns[21])

    # 3. IC统计
    ic_stats = analyzer.compute_ic_statistics(ic_series)

    print("\n=== IC Statistics ===")
    for key, value in ic_stats.items():
        print(f"{key}: {value:.4f}")

    # 4. IC衰减
    ic_decay = analyzer.compute_ic_decay(factor_values, clean_data, max_period=63, step=5)

    print("\n=== IC Decay ===")
    print(ic_decay.head(10))

    # 5. 滚动IC
    rolling_ic = analyzer.compute_rolling_ic(factor_values, forward_returns[21], window=63)

    # 6. 可视化
    analyzer.plot_ic_analysis(ic_series, ic_decay, rolling_ic)
