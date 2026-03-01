"""
Data Preprocessing Module
Handles adjustments, missing values, outliers, winsorization, etc.
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from config.settings import (
    MAX_MISSING_RATIO,
    MAX_CONSECUTIVE_MISSING,
    MAX_SINGLE_DAY_RETURN,
    WINSORIZE_LOWER,
    WINSORIZE_UPPER
)


class DataPreprocessor:
    """Data Preprocessor"""

    def __init__(self):
        logger.info("DataPreprocessor initialized")

    def adjust_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate adjustment factor using Adj Close, apply to OHLC

        Parameters
        ----------
        data : pd.DataFrame
            Raw data, contains 'close', 'adj close', 'open', 'high', 'low'

        Returns
        -------
        pd.DataFrame
            Adjusted data
        """
        logger.info("Adjusting prices for stock splits and dividends")

        df = data.copy()

        # Calculate adjustment factor
        if 'adj close' in df.columns and 'close' in df.columns:
            df['adj_factor'] = df['adj close'] / df['close']

            # Apply to OHLC
            for col in ['open', 'high', 'low']:
                if col in df.columns:
                    df[col] = df[col] * df['adj_factor']

            # Use adj close as close
            df['close'] = df['adj close']

            # Remove unnecessary columns
            df = df.drop(columns=['adj close', 'adj_factor'], errors='ignore')

        logger.info("Price adjustment completed")
        return df

    def handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "ffill",
        max_consecutive: int = MAX_CONSECUTIVE_MISSING
    ) -> pd.DataFrame:
        """
        Handle missing values

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        method : str
            Fill method: 'ffill' (forward fill), 'bfill' (backward fill), 'interpolate' (interpolation)
        max_consecutive : int
            Maximum consecutive missing days

        Returns
        -------
        pd.DataFrame
            Processed data
        """
        logger.info(f"Handling missing values with method: {method}")

        df = data.copy()
        initial_nulls = df.isnull().sum().sum()

        # Process by ticker groups
        if method == "ffill":
            df = df.groupby(level='ticker').ffill(limit=max_consecutive)
        elif method == "bfill":
            df = df.groupby(level='ticker').bfill(limit=max_consecutive)
        elif method == "interpolate":
            # Linear interpolation for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df.groupby(level='ticker')[numeric_cols].apply(
                lambda x: x.interpolate(method='linear', limit=max_consecutive)
            )

        final_nulls = df.isnull().sum().sum()
        logger.info(f"Missing values: {initial_nulls} -> {final_nulls} (filled {initial_nulls - final_nulls})")

        return df

    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle anomalies

        Anomaly detection rules:
        1. Negative prices -> NaN
        2. High < Low -> Delete row
        3. Single day return > 100% -> Mark as anomaly

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Processed data
        """
        logger.info("Detecting and handling anomalies")

        df = data.copy()
        anomaly_count = 0

        # 1. Negative price detection
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    df.loc[negative_mask, col] = np.nan
                    anomaly_count += negative_count
                    logger.warning(f"Found {negative_count} negative values in '{col}'")

        # 2. High < Low detection
        if 'high' in df.columns and 'low' in df.columns:
            invalid_mask = df['high'] < df['low']
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                df.loc[invalid_mask, price_cols] = np.nan
                anomaly_count += invalid_count
                logger.warning(f"Found {invalid_count} rows where High < Low")

        # 3. Extreme return detection (grouped by ticker)
        if 'close' in df.columns:
            # Calculate daily returns
            df['_return'] = df.groupby(level='ticker')['close'].pct_change()

            # Detect extreme returns
            extreme_mask = df['_return'].abs() > MAX_SINGLE_DAY_RETURN
            extreme_count = extreme_mask.sum()
            if extreme_count > 0:
                # Mark but don't delete, let subsequent steps handle
                logger.warning(f"Found {extreme_count} extreme returns (>{MAX_SINGLE_DAY_RETURN:.0%})")
                # Optional: set to NaN
                # df.loc[extreme_mask, 'close'] = np.nan

            df = df.drop(columns=['_return'])

        logger.info(f"Total anomalies detected and handled: {anomaly_count}")
        return df

    def winsorize_data(
        self,
        data: pd.DataFrame,
        lower: float = WINSORIZE_LOWER,
        upper: float = WINSORIZE_UPPER,
        by_ticker: bool = False
    ) -> pd.DataFrame:
        """
        Winsorize processing (trimming)

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        lower : float
            Lower quantile (e.g., 0.01 means 1%)
        upper : float
            Upper quantile (e.g., 0.99 means 99%)
        by_ticker : bool
            Whether to process by ticker separately

        Returns
        -------
        pd.DataFrame
            Processed data
        """
        logger.info(f"Winsorizing data: [{lower:.2%}, {upper:.2%}]")

        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if by_ticker:
            # Winsorize by ticker groups
            for col in numeric_cols:
                df[col] = df.groupby(level='ticker')[col].transform(
                    lambda x: x.clip(lower=x.quantile(lower), upper=x.quantile(upper))
                )
        else:
            # Global winsorize
            for col in numeric_cols:
                lower_bound = df[col].quantile(lower)
                upper_bound = df[col].quantile(upper)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info("Winsorization completed")
        return df

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Comprehensive data quality validation

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        pd.DataFrame
            Cleaned data
        Dict
            Data quality report
        """
        logger.info("Starting comprehensive data quality validation")

        report = {
            'initial_shape': data.shape,
            'initial_nulls': data.isnull().sum().sum(),
            'tickers_count': len(data.index.get_level_values('ticker').unique()),
            'date_range': (
                data.index.get_level_values('date').min(),
                data.index.get_level_values('date').max()
            ),
            'anomalies': {},
            'final_shape': None,
            'final_nulls': None
        }

        # Calculate missing value ratio by ticker
        missing_by_ticker = data.groupby(level='ticker').apply(
            lambda x: x.isnull().sum().sum() / (len(x) * len(x.columns))
        )
        bad_tickers = missing_by_ticker[missing_by_ticker > MAX_MISSING_RATIO].index.tolist()

        if bad_tickers:
            logger.warning(f"Removing {len(bad_tickers)} tickers with >50% missing data")
            report['removed_tickers'] = bad_tickers
            data = data.drop(bad_tickers, level='ticker')

        # Apply all preprocessing steps
        data = self.adjust_prices(data)
        data = self.detect_anomalies(data)
        data = self.handle_missing_values(data)

        # Final statistics
        report['final_shape'] = data.shape
        report['final_nulls'] = data.isnull().sum().sum()
        report['data_coverage'] = 1 - (report['final_nulls'] / (data.shape[0] * data.shape[1]))

        logger.info("Data quality validation completed")
        logger.info(f"Shape: {report['initial_shape']} -> {report['final_shape']}")
        logger.info(f"Nulls: {report['initial_nulls']} -> {report['final_nulls']}")
        logger.info(f"Data coverage: {report['data_coverage']:.2%}")

        return data, report

    def compute_returns(self, data: pd.DataFrame, periods: list = [1, 5, 21]) -> pd.DataFrame:
        """
        Calculate multi-period returns

        Parameters
        ----------
        data : pd.DataFrame
            Price data
        periods : list
            Return periods (trading days)

        Returns
        -------
        pd.DataFrame
            Data with returns
        """
        logger.info(f"Computing returns for periods: {periods}")

        df = data.copy()

        for period in periods:
            col_name = f'return_{period}d'
            df[col_name] = df.groupby(level='ticker')['close'].pct_change(period)

        # Log returns (for volatility calculation)
        df['log_return'] = df.groupby(level='ticker')['close'].apply(
            lambda x: np.log(x / x.shift(1))
        )

        logger.info("Returns computation completed")
        return df


if __name__ == "__main__":
    # Test code
    from src.data.fetcher import YFinanceFetcher

    # Configure logging
    logger.add("data_preprocessor.log", rotation="10 MB")

    # Fetch test data
    fetcher = YFinanceFetcher()
    test_tickers = ["AAPL", "MSFT", "TSLA"]
    data = fetcher.fetch_batch(test_tickers, "2023-01-01", "2024-01-01")

    # Test preprocessing
    preprocessor = DataPreprocessor()
    clean_data, report = preprocessor.validate_data_quality(data)

    print("\n=== Data Quality Report ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\n=== Clean Data Sample ===")
    print(clean_data.head(10))
