"""
Local Data Cache Management Module
Uses Parquet format for storage, avoids repeated API calls
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd
from loguru import logger

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_EXPIRY_DAYS,
    CACHE_FORMAT,
    CACHE_COMPRESSION
)


class CacheManager:
    """Cache Manager"""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Parameters
        ----------
        cache_dir : Path, optional
            Cache directory, defaults to RAW_DATA_DIR from config file
        """
        self.cache_dir = cache_dir or RAW_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CacheManager initialized with cache_dir: {self.cache_dir}")

    def _generate_cache_key(
        self,
        market: str,
        universe: str,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> str:
        """
        Generate cache key

        Parameters
        ----------
        market : str
            Market type
        universe : str
            Stock universe name
        tickers : List[str]
            List of ticker symbols
        start_date : str
            Start date
        end_date : str
            End date

        Returns
        -------
        str
            Cache filename
        """
        # Sort tickers to ensure consistency
        tickers_sorted = sorted(tickers)
        tickers_str = "_".join(tickers_sorted[:5])  # Only use first 5 tickers to avoid long filenames

        # Generate hash
        hash_input = f"{market}_{universe}_{len(tickers)}_{start_date}_{end_date}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        cache_key = f"{market}_{universe}_{start_date}_{end_date}_{len(tickers)}tickers_{hash_suffix}"
        return cache_key

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.parquet"

    def _is_cache_valid(self, cache_path: Path, expiry_days: int = CACHE_EXPIRY_DAYS) -> bool:
        """
        Check if cache is valid

        Parameters
        ----------
        cache_path : Path
            Cache file path
        expiry_days : int
            Expiry days

        Returns
        -------
        bool
            Whether cache is valid
        """
        if not cache_path.exists():
            return False

        # Check file modification time
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime

        if age > timedelta(days=expiry_days):
            logger.info(f"Cache expired (age: {age.days} days): {cache_path.name}")
            return False

        return True

    def save(self, data: pd.DataFrame, cache_key: str) -> Path:
        """
        Save data to cache

        Parameters
        ----------
        data : pd.DataFrame
            Data to cache
        cache_key : str
            Cache key

        Returns
        -------
        Path
            Cache file path
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            data.to_parquet(
                cache_path,
                compression=CACHE_COMPRESSION,
                index=True
            )
            file_size = cache_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Saved cache: {cache_path.name} ({file_size:.2f} MB)")
            return cache_path

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            raise

    def load(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache

        Parameters
        ----------
        cache_key : str
            Cache key

        Returns
        -------
        pd.DataFrame or None
            Cached data, returns None if not exists or expired
        """
        cache_path = self._get_cache_path(cache_key)

        if not self._is_cache_valid(cache_path):
            return None

        try:
            data = pd.read_parquet(cache_path)
            logger.info(f"Loaded cache: {cache_path.name} (shape: {data.shape})")
            return data

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def get_or_fetch(
        self,
        market: str,
        universe: str,
        tickers: List[str],
        start_date: str,
        end_date: str,
        fetcher,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get cached data, or fetch from API if not exists

        Parameters
        ----------
        market : str
            Market type
        universe : str
            Stock universe name
        tickers : List[str]
            List of ticker symbols
        start_date : str
            Start date
        end_date : str
            End date
        fetcher : YFinanceFetcher
            Data fetcher instance
        force_refresh : bool
            Whether to force refresh cache

        Returns
        -------
        pd.DataFrame
            Stock data
        """
        cache_key = self._generate_cache_key(market, universe, tickers, start_date, end_date)

        # Try to load from cache
        if not force_refresh:
            cached_data = self.load(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached data for {market} {universe}")
                return cached_data

        # Fetch from API
        logger.info(f"Fetching fresh data for {market} {universe}")
        data = fetcher.fetch_in_batches(tickers, start_date, end_date)

        if not data.empty:
            # Save to cache
            self.save(data, cache_key)

        return data

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache

        Parameters
        ----------
        older_than_days : int, optional
            Clear cache older than N days, None means clear all
        """
        logger.info(f"Clearing cache in {self.cache_dir}")

        cleared_count = 0
        cleared_size = 0

        for cache_file in self.cache_dir.glob("*.parquet"):
            should_delete = False

            if older_than_days is None:
                should_delete = True
            else:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                age = datetime.now() - mtime
                if age > timedelta(days=older_than_days):
                    should_delete = True

            if should_delete:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                cleared_count += 1
                cleared_size += file_size
                logger.debug(f"Deleted: {cache_file.name}")

        cleared_size_mb = cleared_size / (1024 * 1024)
        logger.info(f"Cleared {cleared_count} files ({cleared_size_mb:.2f} MB)")

    def get_cache_info(self) -> dict:
        """
        Get cache information

        Returns
        -------
        dict
            Cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.parquet"))

        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)

        if cache_files:
            oldest = min(cache_files, key=lambda f: f.stat().st_mtime)
            newest = max(cache_files, key=lambda f: f.stat().st_mtime)
            oldest_age = datetime.now() - datetime.fromtimestamp(oldest.stat().st_mtime)
            newest_age = datetime.now() - datetime.fromtimestamp(newest.stat().st_mtime)
        else:
            oldest_age = newest_age = None

        return {
            'cache_dir': str(self.cache_dir),
            'file_count': len(cache_files),
            'total_size_mb': total_size_mb,
            'oldest_file_age_days': oldest_age.days if oldest_age else None,
            'newest_file_age_days': newest_age.days if newest_age else None,
        }


if __name__ == "__main__":
    # Test code
    from src.data.fetcher import YFinanceFetcher

    # Configure logging
    logger.add("cache_manager.log", rotation="10 MB")

    # Initialize
    cache_manager = CacheManager()
    fetcher = YFinanceFetcher()

    # Test cache
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    market = "US"
    universe = "test"
    start = "2023-01-01"
    end = "2024-01-01"

    print("\n=== First fetch (from API) ===")
    data1 = cache_manager.get_or_fetch(market, universe, test_tickers, start, end, fetcher)
    print(f"Data shape: {data1.shape}")

    print("\n=== Second fetch (from cache) ===")
    data2 = cache_manager.get_or_fetch(market, universe, test_tickers, start, end, fetcher)
    print(f"Data shape: {data2.shape}")

    print("\n=== Cache Info ===")
    info = cache_manager.get_cache_info()
    for key, value in info.items():
        print(f"{key}: {value}")
