"""
Factor Base Class
Defines unified factor interface and standardization methods
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import FACTOR_NORMALIZATION_METHOD


class BaseFactor(ABC):
    """
    Alpha Factor Abstract Base Class

    All concrete factors should inherit this class and implement the calculate method
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name : str, optional
            Factor name, defaults to class name
        """
        self.name = name or self.__class__.__name__
        logger.info(f"Initialized factor: {self.name}")

    def _validate_data(self, data: pd.DataFrame, required: list = None) -> None:
        """
        Validate input data structure and required columns.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to validate
        required : list, optional
            Required column names, defaults to ['close']

        Raises
        ------
        ValueError
            If data is missing required columns or does not have a MultiIndex
        """
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError(f"{self.name}: data must have MultiIndex (date, ticker)")
        cols = required or ['close']
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(f"{self.name}: missing required columns {missing}")

    def _unstack_close(self, data: pd.DataFrame) -> pd.DataFrame:
        """Unstack 'close' column from MultiIndex into date × ticker DataFrame."""
        return data['close'].unstack(level='ticker')

    def _stack_result(self, df: pd.DataFrame) -> pd.Series:
        """Stack date × ticker DataFrame back to MultiIndex Series, named after the factor."""
        return df.stack(dropna=False).rename(self.name)

    def _annualize_vol(self, daily_vol, periods: int = 252):
        """Annualize daily volatility by multiplying by sqrt(periods)."""
        return daily_vol * np.sqrt(periods)

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values

        Parameters
        ----------
        data : pd.DataFrame
            Input data, MultiIndex (date, ticker), contains OHLCV columns

        Returns
        -------
        pd.Series
            Factor values, MultiIndex (date, ticker)
        """
        pass

    def normalize(
        self,
        factor_values: pd.Series,
        method: str = FACTOR_NORMALIZATION_METHOD,
        clip_std: Optional[float] = None
    ) -> pd.Series:
        """
        Factor normalization (cross-sectional normalization)

        Parameters
        ----------
        factor_values : pd.Series
            Raw factor values, MultiIndex (date, ticker)
        method : str
            Normalization method:
            - 'zscore': Z-Score normalization (cross-sectional)
            - 'rank': Percentile ranking
            - 'minmax': Min-Max normalization
        clip_std : float, optional
            Clip standard deviation multiples (e.g., 3 means [-3σ, 3σ])

        Returns
        -------
        pd.Series
            Normalized factor values
        """
        logger.debug(f"Normalizing factor with method: {method}")

        if method == "zscore":
            # Cross-sectional Z-Score normalization
            normalized = factor_values.groupby(level='date').apply(
                lambda x: (x - x.mean()) / (x.std() + 1e-9)
            )

            # Optional: clip extreme values
            if clip_std is not None:
                normalized = normalized.clip(-clip_std, clip_std)

        elif method == "rank":
            # Cross-sectional percentile ranking
            normalized = factor_values.groupby(level='date').rank(pct=True)

        elif method == "minmax":
            # Cross-sectional Min-Max normalization to [0, 1]
            normalized = factor_values.groupby(level='date').apply(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
            )

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def neutralize(
        self,
        factor_values: pd.Series,
        industry: Optional[pd.Series] = None,
        market_cap: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Factor neutralization (remove industry and market cap exposure)

        Parameters
        ----------
        factor_values : pd.Series
            Factor values
        industry : pd.Series, optional
            Industry classification, MultiIndex (date, ticker)
        market_cap : pd.Series, optional
            Market capitalization, MultiIndex (date, ticker)

        Returns
        -------
        pd.Series
            Neutralized factor values
        """
        logger.debug("Neutralizing factor")

        neutralized = factor_values.copy()

        # Neutralize by date groups
        def neutralize_cross_section(df):
            from sklearn.linear_model import LinearRegression

            y = df['factor'].values.reshape(-1, 1)
            X = []

            # Add market cap
            if 'market_cap' in df.columns:
                X.append(np.log(df['market_cap'].values + 1e-9).reshape(-1, 1))

            # Add industry dummies
            if 'industry' in df.columns:
                industry_dummies = pd.get_dummies(df['industry'])
                X.append(industry_dummies.values)

            if not X:
                return df['factor']

            X = np.hstack(X)

            # Regression and extract residuals
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)

            return pd.Series(residuals.flatten(), index=df.index)

        # Build DataFrame
        df = pd.DataFrame({'factor': factor_values})
        if industry is not None:
            df['industry'] = industry
        if market_cap is not None:
            df['market_cap'] = market_cap

        # Neutralize by date groups
        neutralized = df.groupby(level='date').apply(neutralize_cross_section)

        return neutralized

    def validate(self, factor_values: pd.Series) -> dict:
        """
        Validate basic statistical properties of factor values

        Parameters
        ----------
        factor_values : pd.Series
            Factor values

        Returns
        -------
        dict
            Validation report
        """
        report = {
            'total_values': len(factor_values),
            'null_count': factor_values.isnull().sum(),
            'null_ratio': factor_values.isnull().mean(),
            'mean': factor_values.mean(),
            'std': factor_values.std(),
            'min': factor_values.min(),
            'max': factor_values.max(),
            'skewness': factor_values.skew(),
            'kurtosis': factor_values.kurtosis(),
            'unique_dates': len(factor_values.index.get_level_values('date').unique()),
            'unique_tickers': len(factor_values.index.get_level_values('ticker').unique()),
        }

        logger.info(f"Factor validation for {self.name}:")
        logger.info(f"  Total values: {report['total_values']}")
        logger.info(f"  Null ratio: {report['null_ratio']:.2%}")
        logger.info(f"  Mean: {report['mean']:.4f}, Std: {report['std']:.4f}")
        logger.info(f"  Range: [{report['min']:.4f}, {report['max']:.4f}]")

        return report

    def compute(
        self,
        data: pd.DataFrame,
        normalize: bool = True,
        normalization_method: Optional[str] = None,
        clip_std: Optional[float] = 3.0,
        industry: Optional[pd.Series] = None,
        market_cap: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Complete factor computation pipeline

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        normalize : bool
            Whether to normalize
        normalization_method : str, optional
            Normalization method, defaults to config
        clip_std : float, optional
            Clip standard deviation multiples
        industry : pd.Series, optional
            Industry classification (for neutralization)
        market_cap : pd.Series, optional
            Market capitalization (for neutralization)

        Returns
        -------
        pd.Series
            Processed factor values
        """
        logger.info(f"Computing factor: {self.name}")

        # 1. Calculate raw factor values
        factor_values = self.calculate(data)

        # 2. Validate
        self.validate(factor_values)

        # 3. Neutralize (if industry or market cap provided)
        if industry is not None or market_cap is not None:
            factor_values = self.neutralize(factor_values, industry, market_cap)

        # 4. Normalize
        if normalize:
            method = normalization_method or FACTOR_NORMALIZATION_METHOD
            factor_values = self.normalize(factor_values, method, clip_std)

        # 5. Remove NaN
        initial_count = len(factor_values)
        factor_values = factor_values.dropna()
        dropped_count = initial_count - len(factor_values)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} NaN values ({dropped_count/initial_count:.2%})")

        logger.info(f"Factor {self.name} computed successfully")
        return factor_values

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class CompositeFactor(BaseFactor):
    """
    Composite Factor: Weighted combination of multiple factors
    """

    def __init__(
        self,
        factors: List[BaseFactor],
        weights: Optional[List[float]] = None,
        name: str = "CompositeFactor"
    ):
        """
        Parameters
        ----------
        factors : List[BaseFactor]
            List of factors
        weights : List[float], optional
            Factor weights, default equal weight
        name : str
            Composite factor name
        """
        super().__init__(name)
        self.factors = factors
        self.weights = weights or [1.0 / len(factors)] * len(factors)

        if len(self.factors) != len(self.weights):
            raise ValueError("Number of factors must match number of weights")

        if not np.isclose(sum(self.weights), 1.0):
            logger.warning(f"Weights sum to {sum(self.weights)}, normalizing to 1.0")
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

        logger.info(f"CompositeFactor initialized with {len(factors)} factors")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate composite factor values (weighted average)

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.Series
            Composite factor values
        """
        factor_values_list = []

        for factor, weight in zip(self.factors, self.weights):
            logger.debug(f"Computing {factor.name} (weight: {weight})")
            values = factor.calculate(data)
            # Normalize individual factor
            values = factor.normalize(values)
            factor_values_list.append(values * weight)

        # Weighted sum
        composite = pd.concat(factor_values_list, axis=1).sum(axis=1)

        logger.info(f"CompositeFactor computed from {len(self.factors)} factors")
        return composite


if __name__ == "__main__":
    # 测试代码
    from src.data.fetcher import YFinanceFetcher
    from src.data.preprocessor import DataPreprocessor

    # 配置日志
    logger.add("factor_base.log", rotation="10 MB")

    # 创建简单的测试因子
    class SimpleReturnFactor(BaseFactor):
        def __init__(self, period=21):
            super().__init__(name=f"Return_{period}D")
            self.period = period

        def calculate(self, data: pd.DataFrame) -> pd.Series:
            close = data['close'].unstack(level='ticker')
            returns = close.pct_change(self.period)
            return returns.stack(dropna=False)

    # 获取测试数据
    fetcher = YFinanceFetcher()
    data = fetcher.fetch_batch(["AAPL", "MSFT", "GOOGL"], "2023-01-01", "2024-01-01")
    preprocessor = DataPreprocessor()
    clean_data, _ = preprocessor.validate_data_quality(data)

    # 测试因子
    factor = SimpleReturnFactor(period=21)
    factor_values = factor.compute(clean_data, normalize=True)

    print("\n=== Factor Values Sample ===")
    print(factor_values.head(20))
    print(f"\nFactor shape: {factor_values.shape}")
