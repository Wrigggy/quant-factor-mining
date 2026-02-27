"""
Alphalens集成模块
用于因子有效性评估、IC分析、分位数收益分析
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from loguru import logger

try:
    import alphalens as al
    from alphalens.utils import get_clean_factor_and_forward_returns
    from alphalens.tears import create_returns_tear_sheet, create_information_tear_sheet
    from alphalens.tears import create_turnover_tear_sheet, create_event_returns_tear_sheet
    ALPHALENS_AVAILABLE = True
except ImportError:
    logger.warning("alphalens-reloaded not installed. Install with: pip install alphalens-reloaded")
    ALPHALENS_AVAILABLE = False

from config.settings import ALPHALENS_PERIODS, ALPHALENS_QUANTILES


class AlphalensEvaluator:
    """
    Alphalens因子评估器

    主要功能：
    1. 准备Alphalens格式的数据
    2. 计算前瞻收益率
    3. 生成完整的tearsheet报告
    4. 提取IC统计量
    5. 分位数收益分析
    """

    def __init__(
        self,
        periods: List[int] = None,
        quantiles: int = ALPHALENS_QUANTILES,
        bins: Optional[int] = None,
        filter_zscore: Optional[float] = None
    ):
        """
        Parameters
        ----------
        periods : List[int], optional
            前瞻收益率周期（交易日），默认[1, 5, 21]
        quantiles : int
            分位数数量，默认5
        bins : int, optional
            如果不使用分位数，可以指定固定的bins
        filter_zscore : float, optional
            过滤极端因子值（zscore阈值）
        """
        if not ALPHALENS_AVAILABLE:
            raise ImportError("alphalens-reloaded is required. Install with: pip install alphalens-reloaded")

        self.periods = periods or ALPHALENS_PERIODS
        self.quantiles = quantiles
        self.bins = bins
        self.filter_zscore = filter_zscore

        logger.info(f"AlphalensEvaluator initialized: periods={self.periods}, quantiles={quantiles}")

    def prepare_data(
        self,
        factor: pd.Series,
        prices: pd.DataFrame,
        groupby: Optional[pd.Series] = None,
        max_loss: float = 0.35
    ) -> pd.DataFrame:
        """
        准备Alphalens格式的数据

        Parameters
        ----------
        factor : pd.Series
            因子值，MultiIndex (date, ticker)
        prices : pd.DataFrame
            价格数据，MultiIndex (date, ticker)，包含'close'列
        groupby : pd.Series, optional
            分组变量（如行业），MultiIndex (date, ticker)
        max_loss : float
            允许的最大数据丢失比例

        Returns
        -------
        pd.DataFrame
            Alphalens格式的数据，包含factor和forward returns
        """
        logger.info("Preparing data for Alphalens...")

        # 确保factor是Series
        if isinstance(factor, pd.DataFrame):
            if len(factor.columns) == 1:
                factor = factor.iloc[:, 0]
            else:
                raise ValueError("factor must be a Series, not a multi-column DataFrame")

        # 提取价格
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close'].unstack(level='ticker')
            else:
                # 假设已经是pivot格式
                pass

        # 转换factor格式：Alphalens需要 MultiIndex (date, asset)
        factor_data = factor.copy()
        if not isinstance(factor_data.index, pd.MultiIndex):
            raise ValueError("factor must have MultiIndex (date, ticker)")

        # 重命名index levels以匹配Alphalens
        factor_data.index.names = ['date', 'asset']

        # 确保prices的列名与factor的asset对应
        if isinstance(prices, pd.DataFrame):
            prices.index.name = 'date'
            # 只保留factor中存在的资产
            assets = factor_data.index.get_level_values('asset').unique()
            prices = prices[prices.columns.intersection(assets)]

        try:
            # 使用Alphalens工具准备数据
            factor_data = get_clean_factor_and_forward_returns(
                factor=factor_data,
                prices=prices,
                quantiles=self.quantiles if self.bins is None else None,
                bins=self.bins,
                periods=self.periods,
                filter_zscore=self.filter_zscore,
                groupby=groupby,
                max_loss=max_loss
            )

            logger.info(f"Alphalens data prepared: {factor_data.shape}")
            logger.info(f"Periods: {self.periods}")
            logger.info(f"Assets: {len(factor_data.index.get_level_values('asset').unique())}")

            return factor_data

        except Exception as e:
            logger.error(f"Error preparing Alphalens data: {e}")
            raise

    def generate_tearsheet(
        self,
        factor_data: pd.DataFrame,
        long_short: bool = True,
        group_neutral: bool = False
    ):
        """
        生成完整的Alphalens tearsheet

        Parameters
        ----------
        factor_data : pd.DataFrame
            Alphalens格式的数据（从prepare_data获得）
        long_short : bool
            是否显示多空组合
        group_neutral : bool
            是否进行分组中性化
        """
        logger.info("Generating Alphalens tearsheets...")

        try:
            # 1. Returns Analysis
            logger.info("Creating returns tear sheet...")
            create_returns_tear_sheet(
                factor_data=factor_data,
                long_short=long_short,
                group_neutral=group_neutral
            )

            # 2. Information Analysis (IC)
            logger.info("Creating information tear sheet...")
            create_information_tear_sheet(
                factor_data=factor_data,
                group_neutral=group_neutral
            )

            # 3. Turnover Analysis
            logger.info("Creating turnover tear sheet...")
            create_turnover_tear_sheet(factor_data=factor_data)

            logger.info("✅ Tearsheets generation complete")

        except Exception as e:
            logger.error(f"Error generating tearsheets: {e}")
            raise

    def get_ic_summary(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        提取IC统计摘要

        Parameters
        ----------
        factor_data : pd.DataFrame
            Alphalens格式的数据

        Returns
        -------
        pd.DataFrame
            IC统计摘要，包含IC均值、标准差、t统计量、正值比例等
        """
        logger.info("Computing IC summary statistics...")

        try:
            from alphalens.performance import factor_information_coefficient

            # 计算IC
            ic = factor_information_coefficient(factor_data)

            # 统计摘要
            ic_summary = pd.DataFrame({
                'IC Mean': ic.mean(),
                'IC Std': ic.std(),
                'IC Mean / IC Std': ic.mean() / ic.std(),
                't-stat': ic.mean() / ic.std() * np.sqrt(len(ic)),
                'p-value': 2 * (1 - stats.t.cdf(np.abs(ic.mean() / ic.std() * np.sqrt(len(ic))), len(ic) - 1)),
                'IC Skew': ic.skew(),
                'IC Kurtosis': ic.kurtosis(),
                'IC Positive %': (ic > 0).mean() * 100
            })

            logger.info("IC Summary Statistics:")
            logger.info(f"\n{ic_summary.round(4)}")

            return ic_summary

        except Exception as e:
            logger.error(f"Error computing IC summary: {e}")
            # Fallback: 手动计算IC
            return self._compute_ic_manual(factor_data)

    def _compute_ic_manual(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """手动计算IC（当alphalens方法失败时）"""
        from scipy import stats as scipy_stats

        ic_data = {}

        for period in self.periods:
            period_col = f'{period}D'
            if period_col in factor_data.columns:
                # 按日期分组计算IC
                ic_series = factor_data.groupby(level='date').apply(
                    lambda x: x['factor'].corr(x[period_col], method='spearman')
                )

                ic_data[period_col] = {
                    'IC Mean': ic_series.mean(),
                    'IC Std': ic_series.std(),
                    't-stat': ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series)),
                    'IC Positive %': (ic_series > 0).mean() * 100
                }

        return pd.DataFrame(ic_data).T

    def get_quantile_returns(
        self,
        factor_data: pd.DataFrame,
        by_group: bool = False
    ) -> pd.DataFrame:
        """
        获取分位数收益率

        Parameters
        ----------
        factor_data : pd.DataFrame
            Alphalens格式的数据
        by_group : bool
            是否按组分别计算

        Returns
        -------
        pd.DataFrame
            分位数平均收益率
        """
        logger.info("Computing quantile returns...")

        try:
            from alphalens.performance import mean_return_by_quantile

            quantile_returns, _ = mean_return_by_quantile(
                factor_data,
                by_group=by_group,
                demeaned=True
            )

            logger.info(f"Quantile returns shape: {quantile_returns.shape}")
            return quantile_returns

        except Exception as e:
            logger.error(f"Error computing quantile returns: {e}")
            raise

    def get_turnover_analysis(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        分析因子换手率

        Parameters
        ----------
        factor_data : pd.DataFrame
            Alphalens格式的数据

        Returns
        -------
        pd.DataFrame
            换手率统计
        """
        logger.info("Analyzing turnover...")

        try:
            from alphalens.performance import factor_rank_autocorrelation

            # 计算因子排名自相关（换手率的逆指标）
            autocorr = factor_rank_autocorrelation(factor_data)

            # 换手率 = 1 - 自相关
            turnover = 1 - autocorr

            turnover_summary = pd.DataFrame({
                'Mean Turnover': turnover.mean(),
                'Std Turnover': turnover.std(),
                'Min Turnover': turnover.min(),
                'Max Turnover': turnover.max()
            })

            logger.info(f"Turnover summary:\n{turnover_summary}")
            return turnover_summary

        except Exception as e:
            logger.error(f"Error analyzing turnover: {e}")
            raise

    def evaluate_factor(
        self,
        factor: pd.Series,
        prices: pd.DataFrame,
        groupby: Optional[pd.Series] = None,
        generate_plots: bool = True
    ) -> Dict:
        """
        完整的因子评估流程

        Parameters
        ----------
        factor : pd.Series
            因子值
        prices : pd.DataFrame
            价格数据
        groupby : pd.Series, optional
            分组变量
        generate_plots : bool
            是否生成图表

        Returns
        -------
        Dict
            评估结果摘要
        """
        logger.info("Starting complete factor evaluation...")

        # 1. 准备数据
        factor_data = self.prepare_data(factor, prices, groupby)

        # 2. 计算IC
        ic_summary = self.get_ic_summary(factor_data)

        # 3. 分位数收益
        quantile_returns = self.get_quantile_returns(factor_data)

        # 4. 换手率
        turnover = self.get_turnover_analysis(factor_data)

        # 5. 生成图表
        if generate_plots:
            self.generate_tearsheet(factor_data)

        # 6. 汇总结果
        results = {
            'ic_summary': ic_summary,
            'quantile_returns': quantile_returns,
            'turnover': turnover,
            'factor_data': factor_data
        }

        # 7. 判断因子是否有效
        is_valid = self._validate_factor(ic_summary, quantile_returns)
        results['is_valid_factor'] = is_valid

        logger.info(f"Factor evaluation complete. Valid: {is_valid}")
        return results

    def _validate_factor(
        self,
        ic_summary: pd.DataFrame,
        quantile_returns: pd.DataFrame,
        ic_threshold: float = 0.02,
        tstat_threshold: float = 2.0
    ) -> bool:
        """
        验证因子是否有效

        标准：
        1. IC均值 > 0.02
        2. IC t统计量 > 2.0
        3. 分位数收益单调递增
        """
        # 检查IC
        ic_mean = ic_summary['IC Mean'].iloc[0]  # 使用第一个周期
        t_stat = ic_summary['t-stat'].iloc[0]

        ic_valid = (abs(ic_mean) > ic_threshold) and (abs(t_stat) > tstat_threshold)

        # 检查单调性（Q5 > Q1）
        try:
            q5_return = quantile_returns.loc[5].iloc[0]  # 最高分位
            q1_return = quantile_returns.loc[1].iloc[0]  # 最低分位
            monotonic = (q5_return > q1_return) if ic_mean > 0 else (q5_return < q1_return)
        except:
            monotonic = False

        logger.info(f"Validation: IC={ic_mean:.4f}, t-stat={t_stat:.2f}, Monotonic={monotonic}")

        return ic_valid and monotonic


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    from src.data.cache_manager import CacheManager
    from src.factors.momentum import MomentumFactor

    logger.add("alphalens_test.log", rotation="10 MB")

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

    # 评估因子
    evaluator = AlphalensEvaluator(periods=[1, 5, 21], quantiles=5)

    try:
        results = evaluator.evaluate_factor(
            factor=factor_values,
            prices=clean_data,
            generate_plots=False  # 在测试中不生成图表
        )

        print("\n=== Factor Evaluation Results ===")
        print(f"Valid Factor: {results['is_valid_factor']}")
        print(f"\nIC Summary:\n{results['ic_summary']}")
        print(f"\nQuantile Returns:\n{results['quantile_returns']}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
