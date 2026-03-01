"""
Alphalens Integration Module
For factor effectiveness evaluation, IC analysis, and quantile return analysis
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
    Alphalens Factor Evaluator

    Main Features:
    1. Prepare data in Alphalens format
    2. Calculate forward returns
    3. Generate complete tearsheet reports
    4. Extract IC statistics
    5. Quantile return analysis
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
            Forward return periods (trading days), default [1, 5, 21]
        quantiles : int
            Number of quantiles, default 5
        bins : int, optional
            If not using quantiles, can specify fixed bins
        filter_zscore : float, optional
            Filter extreme factor values (zscore threshold)
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
        Prepare data in Alphalens format

        Parameters
        ----------
        factor : pd.Series
            Factor values, MultiIndex (date, ticker)
        prices : pd.DataFrame
            Price data, MultiIndex (date, ticker), contains 'close' column
        groupby : pd.Series, optional
            Grouping variable (e.g., industry), MultiIndex (date, ticker)
        max_loss : float
            Maximum allowed data loss ratio

        Returns
        -------
        pd.DataFrame
            Data in Alphalens format, contains factor and forward returns
        """
        logger.info("Preparing data for Alphalens...")

        # Ensure factor is a Series
        if isinstance(factor, pd.DataFrame):
            if len(factor.columns) == 1:
                factor = factor.iloc[:, 0]
            else:
                raise ValueError("factor must be a Series, not a multi-column DataFrame")

        # Extract prices
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                prices = prices['close'].unstack(level='ticker')
            else:
                # Assume already in pivot format
                pass

        # Convert factor format: Alphalens requires MultiIndex (date, asset)
        factor_data = factor.copy()
        if not isinstance(factor_data.index, pd.MultiIndex):
            raise ValueError("factor must have MultiIndex (date, ticker)")

        # Rename index levels to match Alphalens
        factor_data.index.names = ['date', 'asset']

        # Ensure prices column names match factor assets
        if isinstance(prices, pd.DataFrame):
            prices.index.name = 'date'
            # Keep only assets that exist in factor
            assets = factor_data.index.get_level_values('asset').unique()
            prices = prices[prices.columns.intersection(assets)]

        try:
            # Use Alphalens tools to prepare data
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
        Generate complete Alphalens tearsheet

        Parameters
        ----------
        factor_data : pd.DataFrame
            Data in Alphalens format (obtained from prepare_data)
        long_short : bool
            Whether to display long-short portfolio
        group_neutral : bool
            Whether to perform group neutralization
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
        Extract IC statistics summary

        Parameters
        ----------
        factor_data : pd.DataFrame
            Data in Alphalens format

        Returns
        -------
        pd.DataFrame
            IC statistics summary, including IC mean, std, t-stat, positive ratio, etc.
        """
        logger.info("Computing IC summary statistics...")

        try:
            from alphalens.performance import factor_information_coefficient

            # Calculate IC
            ic = factor_information_coefficient(factor_data)

            # Statistical summary
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
            # Fallback: manually calculate IC
            return self._compute_ic_manual(factor_data)

    def _compute_ic_manual(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """Manually calculate IC (when alphalens method fails)"""
        from scipy import stats as scipy_stats

        ic_data = {}

        for period in self.periods:
            period_col = f'{period}D'
            if period_col in factor_data.columns:
                # Calculate IC grouped by date
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
        Get quantile returns

        Parameters
        ----------
        factor_data : pd.DataFrame
            Data in Alphalens format
        by_group : bool
            Whether to calculate by group

        Returns
        -------
        pd.DataFrame
            Quantile average returns
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
        Analyze factor turnover

        Parameters
        ----------
        factor_data : pd.DataFrame
            Data in Alphalens format

        Returns
        -------
        pd.DataFrame
            Turnover statistics
        """
        logger.info("Analyzing turnover...")

        try:
            from alphalens.performance import factor_rank_autocorrelation

            # Calculate factor rank autocorrelation (inverse indicator of turnover)
            autocorr = factor_rank_autocorrelation(factor_data)

            # Turnover = 1 - autocorrelation
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
        Complete factor evaluation workflow

        Parameters
        ----------
        factor : pd.Series
            Factor values
        prices : pd.DataFrame
            Price data
        groupby : pd.Series, optional
            Grouping variable
        generate_plots : bool
            Whether to generate plots

        Returns
        -------
        Dict
            Evaluation results summary
        """
        logger.info("Starting complete factor evaluation...")

        # 1. Prepare data
        factor_data = self.prepare_data(factor, prices, groupby)

        # 2. Calculate IC
        ic_summary = self.get_ic_summary(factor_data)

        # 3. Quantile returns
        quantile_returns = self.get_quantile_returns(factor_data)

        # 4. Turnover
        turnover = self.get_turnover_analysis(factor_data)

        # 5. Generate plots
        if generate_plots:
            self.generate_tearsheet(factor_data)

        # 6. Summarize results
        results = {
            'ic_summary': ic_summary,
            'quantile_returns': quantile_returns,
            'turnover': turnover,
            'factor_data': factor_data
        }

        # 7. Determine if factor is valid
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
        Validate if factor is effective

        Criteria:
        1. IC mean > 0.02
        2. IC t-statistic > 2.0
        3. Quantile returns are monotonically increasing
        """
        # Check IC
        ic_mean = ic_summary['IC Mean'].iloc[0]  # 使用第一个周期
        t_stat = ic_summary['t-stat'].iloc[0]

        ic_valid = (abs(ic_mean) > ic_threshold) and (abs(t_stat) > tstat_threshold)

        # Check monotonicity (Q5 > Q1)
        try:
            q5_return = quantile_returns.loc[5].iloc[0]  # 最高分位
            q1_return = quantile_returns.loc[1].iloc[0]  # 最低分位
            monotonic = (q5_return > q1_return) if ic_mean > 0 else (q5_return < q1_return)
        except Exception as e:
            logger.warning(f"Monotonicity check failed: {e}")
            monotonic = False

        logger.info(f"Validation: IC={ic_mean:.4f}, t-stat={t_stat:.2f}, Monotonic={monotonic}")

        return ic_valid and monotonic


if __name__ == "__main__":
    # Test code
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor
    from src.data.cache_manager import CacheManager
    from src.factors.momentum import MomentumFactor

    logger.add("alphalens_test.log", rotation="10 MB")

    # Fetch data
    fetcher = YFinanceFetcher()
    cache = CacheManager()
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "JPM"]

    data = cache.get_or_fetch("US", "test", test_tickers, "2020-01-01", "2024-01-01", fetcher)

    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # Calculate factor
    momentum = MomentumFactor(lookback=252, skip=21)
    factor_values = momentum.compute(clean_data, normalize=True)

    # Evaluate factor
    evaluator = AlphalensEvaluator(periods=[1, 5, 21], quantiles=5)

    try:
        results = evaluator.evaluate_factor(
            factor=factor_values,
            prices=clean_data,
            generate_plots=False  # Don't generate plots in test
        )

        print("\n=== Factor Evaluation Results ===")
        print(f"Valid Factor: {results['is_valid_factor']}")
        print(f"\nIC Summary:\n{results['ic_summary']}")
        print(f"\nQuantile Returns:\n{results['quantile_returns']}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
