"""
YFinance Data Fetching Module
Implements rate limiting, batch processing, parallel downloading, retry mechanism
"""

import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from config.settings import (
    YFINANCE_RATE_LIMIT,
    YFINANCE_BATCH_SIZE,
    YFINANCE_MAX_WORKERS,
    YFINANCE_RETRY_TIMES,
    YFINANCE_RETRY_DELAY,
    MAX_MISSING_RATIO
)


class RateLimiter:
    """Rate Limiter"""

    def __init__(self, max_requests_per_hour: int):
        self.max_requests = max_requests_per_hour
        self.requests = []
        self.window = 3600  # 1 hour (seconds)

    def wait_if_needed(self):
        """Wait if rate limit is reached"""
        now = time.time()

        # Remove request records from 1 hour ago
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window]

        if len(self.requests) >= self.max_requests:
            # Calculate wait time needed
            oldest_request = min(self.requests)
            wait_time = self.window - (now - oldest_request) + 1
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.requests = []

        self.requests.append(now)


class YFinanceFetcher:
    """YFinance Data Fetcher"""

    def __init__(self, rate_limit: Optional[int] = None):
        """
        Parameters
        ----------
        rate_limit : int, optional
            Maximum requests per hour, defaults to value from config file
        """
        self.rate_limiter = RateLimiter(rate_limit or YFINANCE_RATE_LIMIT)
        logger.info(f"YFinanceFetcher initialized with rate limit: {rate_limit or YFINANCE_RATE_LIMIT} req/hour")

    def fetch_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        retry_times: int = YFINANCE_RETRY_TIMES
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single stock

        Parameters
        ----------
        ticker : str
            Ticker symbol
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        retry_times : int
            Number of retries

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing OHLCV data, returns None on failure
        """
        self.rate_limiter.wait_if_needed()

        for attempt in range(retry_times):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date, auto_adjust=False)

                if df.empty:
                    logger.warning(f"{ticker}: No data returned")
                    return None

                # Check data quality
                missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
                if missing_ratio > MAX_MISSING_RATIO:
                    logger.warning(f"{ticker}: Too many missing values ({missing_ratio:.1%})")
                    return None

                # Add Ticker column
                df['Ticker'] = ticker
                df = df.reset_index()

                # Standardize column names
                df = df.rename(columns={'Date': 'date'})
                df.columns = [col.lower() for col in df.columns]

                logger.debug(f"{ticker}: Successfully fetched {len(df)} rows")
                return df

            except Exception as e:
                logger.warning(f"{ticker}: Attempt {attempt + 1}/{retry_times} failed: {e}")
                if attempt < retry_times - 1:
                    time.sleep(YFINANCE_RETRY_DELAY)
                else:
                    logger.error(f"{ticker}: All retry attempts failed")
                    return None

        return None

    def fetch_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_workers: int = YFINANCE_MAX_WORKERS
    ) -> pd.DataFrame:
        """
        Fetch data for multiple stocks in parallel

        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        max_workers : int
            Number of parallel threads

        Returns
        -------
        pd.DataFrame
            Merged data, MultiIndex (date, ticker)
        """
        logger.info(f"Fetching {len(tickers)} tickers from {start_date} to {end_date}")

        results = []
        failed_tickers = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.fetch_single_ticker, ticker, start_date, end_date): ticker
                for ticker in tickers
            }

            # Use tqdm to show progress
            with tqdm(total=len(tickers), desc="Downloading", unit="ticker") as pbar:
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        df = future.result()
                        if df is not None:
                            results.append(df)
                        else:
                            failed_tickers.append(ticker)
                    except Exception as e:
                        logger.error(f"{ticker}: Unexpected error: {e}")
                        failed_tickers.append(ticker)
                    finally:
                        pbar.update(1)

        if failed_tickers:
            logger.warning(f"Failed to fetch {len(failed_tickers)} tickers: {failed_tickers[:10]}...")

        if not results:
            logger.error("No data fetched for any ticker")
            return pd.DataFrame()

        # Merge all data
        combined_df = pd.concat(results, ignore_index=True)

        # Convert to MultiIndex
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.set_index(['date', 'ticker'])

        # Select needed columns
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume', 'adj close']
        combined_df = combined_df[[col for col in columns_to_keep if col in combined_df.columns]]

        logger.info(f"Successfully fetched data: {len(results)}/{len(tickers)} tickers, "
                   f"{len(combined_df)} total rows")

        return combined_df

    def fetch_in_batches(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        batch_size: int = YFINANCE_BATCH_SIZE
    ) -> pd.DataFrame:
        """
        Fetch large amounts of stock data in batches

        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols
        start_date : str
            Start date
        end_date : str
            End date
        batch_size : int
            Number of stocks per batch

        Returns
        -------
        pd.DataFrame
            Complete merged data
        """
        all_data = []
        num_batches = (len(tickers) + batch_size - 1) // batch_size

        logger.info(f"Fetching {len(tickers)} tickers in {num_batches} batches")

        for i in range(0, len(tickers), batch_size):
            batch_num = i // batch_size + 1
            batch_tickers = tickers[i:i + batch_size]

            logger.info(f"Processing batch {batch_num}/{num_batches}")

            batch_data = self.fetch_batch(batch_tickers, start_date, end_date)
            if not batch_data.empty:
                all_data.append(batch_data)

            # Short delay between batches
            if i + batch_size < len(tickers):
                time.sleep(2)

        if not all_data:
            logger.error("No data fetched from any batch")
            return pd.DataFrame()

        # Merge all batches
        final_df = pd.concat(all_data)
        logger.info(f"Total data fetched: {len(final_df)} rows")

        return final_df


def get_valid_date_range(start_date: str, end_date: str) -> tuple:
    """
    Validate and return valid date range

    Parameters
    ----------
    start_date : str
        Start date
    end_date : str
        End date

    Returns
    -------
    tuple
        (start_date, end_date) string tuple
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    today = pd.Timestamp.now()

    if start >= end:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")

    if end > today:
        logger.warning(f"End date ({end_date}) is in the future, using today's date")
        end = today

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


if __name__ == "__main__":
    # Test code
    from config.universes import get_universe

    # 配置日志
    logger.add("data_fetcher.log", rotation="10 MB")

    # 测试获取少量股票
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    start = "2023-01-01"
    end = "2024-01-01"

    fetcher = YFinanceFetcher()
    data = fetcher.fetch_batch(test_tickers, start, end, max_workers=3)

    print("\n=== Data Summary ===")
    print(f"Shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head(10)}")
    print(f"\nData info:\n{data.info()}")
