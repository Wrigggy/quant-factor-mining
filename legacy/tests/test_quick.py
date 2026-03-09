"""
简化测试脚本 - 验证核心功能
"""
import sys
sys.path.append('.')

print("\n" + "="*80)
print("快速系统验证测试")
print("="*80)

# 测试1: 配置和导入
print("\n✓ 测试1: 模块导入")
try:
    from config.settings import *
    from config.universes import get_universe
    from src.data.fetcher import YFinanceFetcher
    from src.data.cache_manager import CacheManager
    from src.data.preprocessor import DataPreprocessor
    from src.factors.momentum import MomentumFactor
    from src.factors.mean_reversion import MeanReversionFactor
    from src.factors.volatility import VolatilityFactor
    print("  ✅ 所有核心模块导入成功")
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 配置检查
print("\n✓ 测试2: 配置验证")
try:
    print(f"  - 项目根目录: {PROJECT_ROOT}")
    print(f"  - 数据目录: {DATA_DIR}")
    print(f"  - 缓存格式: {CACHE_FORMAT}")
    print(f"  - API速率限制: {YFINANCE_RATE_LIMIT}/小时")
    print("  ✅ 配置正常")
except Exception as e:
    print(f"  ❌ 配置错误: {e}")
    sys.exit(1)

# 测试3: 股票池
print("\n✓ 测试3: 股票池定义")
try:
    us_sample = get_universe("US", "sample")[:5]
    hk_sample = get_universe("HK", "sample")[:5]
    print(f"  - 美股样本 (前5): {us_sample}")
    print(f"  - 港股样本 (前5): {hk_sample}")
    print("  ✅ 股票池加载成功")
except Exception as e:
    print(f"  ❌ 股票池错误: {e}")
    sys.exit(1)

# 测试4: 因子实例化
print("\n✓ 测试4: 因子对象创建")
try:
    momentum = MomentumFactor(lookback=252, skip=21)
    mean_rev = MeanReversionFactor(lookback=21)
    volatility = VolatilityFactor(window=63)
    print(f"  - 动量因子: {momentum.name}")
    print(f"  - 均值回归因子: {mean_rev.name}")
    print(f"  - 波动率因子: {volatility.name}")
    print("  ✅ 因子对象创建成功")
except Exception as e:
    print(f"  ❌ 因子创建失败: {e}")
    sys.exit(1)

# 测试5: 数据模块初始化
print("\n✓ 测试5: 数据模块初始化")
try:
    fetcher = YFinanceFetcher()
    cache = CacheManager()
    preprocessor = DataPreprocessor()
    print("  - YFinanceFetcher: 已初始化")
    print("  - CacheManager: 已初始化")
    print("  - DataPreprocessor: 已初始化")
    print("  ✅ 数据模块就绪")
except Exception as e:
    print(f"  ❌ 数据模块错误: {e}")
    sys.exit(1)

# 测试6: 优化模块（可选，需要cvxpy）
print("\n✓ 测试6: 优化模块（可选）")
try:
    from src.optimization.risk_models import CovarianceEstimator
    from src.optimization.mean_variance import MeanVarianceOptimizer
    from src.optimization.backtester import PortfolioBacktester
    print("  - CovarianceEstimator: 已加载")
    print("  - MeanVarianceOptimizer: 已加载")
    print("  - PortfolioBacktester: 已加载")
    print("  ✅ 优化模块可用")
except ImportError as e:
    print(f"  ⚠️  优化模块不完整: {e}")
    print("  提示: 运行 'pip install cvxpy' 安装优化库")

# 测试7: 评估模块（可选，需要alphalens）
print("\n✓ 测试7: 评估模块（可选）")
try:
    from src.evaluation.ic_analysis import ICAnalyzer
    print("  - ICAnalyzer: 已加载")
    print("  ✅ 评估模块可用")
except ImportError as e:
    print(f"  ⚠️  评估模块不完整: {e}")
    print("  提示: 运行 'pip install alphalens-reloaded' 安装")

# 总结
print("\n" + "="*80)
print("✅ 核心功能验证完成！")
print("="*80)
print("\n系统状态:")
print("  - 配置: ✅ 正常")
print("  - 数据模块: ✅ 就绪")
print("  - 因子模块: ✅ 可用")
print("  - 优化模块: 检查上方输出")
print("  - 评估模块: 检查上方输出")
print("\n下一步:")
print("  1. 运行 'python3 test_factors.py' 测试因子计算")
print("  2. 运行 'python3 demo.py' 查看完整演示（需要网络）")
print("  3. 查看 'QUICKSTART.md' 获取使用示例")
print("\n")
