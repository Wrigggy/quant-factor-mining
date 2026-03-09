"""
测试所有因子实现

This script tests all implemented factors with a small dataset
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from loguru import logger

# Configure logger
logger.add("test_factors.log", rotation="10 MB")

# Import all components
from config.settings import *
from config.universes import get_universe
from src.data.fetcher import YFinanceFetcher
from src.data.preprocessor import DataPreprocessor
from src.data.cache_manager import CacheManager

from src.factors.momentum import MomentumFactor, MultiPeriodMomentumFactor
from src.factors.mean_reversion import MeanReversionFactor, RSIMeanReversionFactor
from src.factors.volatility import VolatilityFactor, DownsideVolatilityFactor


def test_data_pipeline():
    """Test data acquisition and preprocessing"""
    print("\n" + "="*60)
    print("TESTING DATA PIPELINE")
    print("="*60)

    # Initialize
    fetcher = YFinanceFetcher()
    cache = CacheManager()
    preprocessor = DataPreprocessor()

    # Get test data
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    start_date = "2020-01-01"
    end_date = "2024-01-01"

    print(f"\nFetching {len(test_tickers)} tickers from {start_date} to {end_date}...")

    data = cache.get_or_fetch(
        market="US",
        universe="test",
        tickers=test_tickers,
        start_date=start_date,
        end_date=end_date,
        fetcher=fetcher
    )

    print(f"✅ Data fetched: {data.shape}")

    # Preprocess
    clean_data, report = preprocessor.validate_data_quality(data)

    print(f"\n✅ Data preprocessed:")
    print(f"   Shape: {report['final_shape']}")
    print(f"   Coverage: {report['data_coverage']:.2%}")
    print(f"   Nulls: {report['final_nulls']}")

    return clean_data


def test_factors(data):
    """Test all factor implementations"""
    print("\n" + "="*60)
    print("TESTING FACTORS")
    print("="*60)

    factors = [
        ("Momentum", MomentumFactor(lookback=252, skip=21)),
        ("Multi-Period Momentum", MultiPeriodMomentumFactor(periods=[21, 63, 126, 252])),
        ("Mean Reversion", MeanReversionFactor(lookback=21)),
        ("RSI Mean Reversion", RSIMeanReversionFactor(period=14)),
        ("Volatility", VolatilityFactor(window=63, annualize=True)),
        ("Downside Volatility", DownsideVolatilityFactor(window=63)),
    ]

    results = {}

    for factor_name, factor in factors:
        print(f"\n--- Testing {factor_name} ---")
        try:
            factor_values = factor.compute(data, normalize=True)

            # Validate
            stats = {
                'count': len(factor_values),
                'null_ratio': factor_values.isnull().mean(),
                'mean': factor_values.mean(),
                'std': factor_values.std(),
                'min': factor_values.min(),
                'max': factor_values.max()
            }

            print(f"✅ {factor_name}:")
            print(f"   Count: {stats['count']}")
            print(f"   Null ratio: {stats['null_ratio']:.2%}")
            print(f"   Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

            results[factor_name] = factor_values

        except Exception as e:
            print(f"❌ {factor_name} failed: {e}")
            logger.exception(f"Factor {factor_name} test failed")

    return results


def test_factor_correlations(factor_results):
    """Test correlations between factors"""
    print("\n" + "="*60)
    print("FACTOR CORRELATIONS")
    print("="*60)

    # Combine all factors
    factor_df = pd.DataFrame(factor_results)

    # Compute correlation matrix
    corr_matrix = factor_df.corr()

    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))

    # Find highly correlated pairs
    print("\n--- Highly Correlated Factor Pairs (|corr| > 0.5) ---")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                print(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.3f}")

    return corr_matrix


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("QUANT FACTOR MINING - SYSTEM TEST")
    print("="*80)

    try:
        # Test 1: Data pipeline
        clean_data = test_data_pipeline()

        # Test 2: All factors
        factor_results = test_factors(clean_data)

        # Test 3: Factor correlations
        if len(factor_results) > 1:
            corr_matrix = test_factor_correlations(factor_results)

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)

        print("\n📊 Summary:")
        print(f"   Data shape: {clean_data.shape}")
        print(f"   Factors tested: {len(factor_results)}")
        print(f"   Date range: {clean_data.index.get_level_values('date').min()} to {clean_data.index.get_level_values('date').max()}")
        print(f"   Tickers: {len(clean_data.index.get_level_values('ticker').unique())}")

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TESTS FAILED")
        print("="*80)
        print(f"Error: {e}")
        logger.exception("System test failed")
        raise


if __name__ == "__main__":
    main()
