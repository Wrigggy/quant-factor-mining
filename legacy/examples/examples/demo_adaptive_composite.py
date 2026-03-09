"""
自适应因子组合示例

演示如何使用IC加权的自适应因子组合系统

功能展示：
1. 基础的AdaptiveCompositeFactor
2. 使用AdaptiveCompositeManager避免前视偏差
3. RegimeAwareCompositeFactor结合市场状态
4. 与静态组合的性能对比
"""

import sys
sys.path.append('.')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

print("\n" + "="*80)
print("自适应因子组合系统 - 演示")
print("="*80)

# ==================== 第1步：加载数据 ====================
print("\n📥 第1步：加载历史数据")
print("-" * 80)

from config.universes import get_universe
from src.data.fetcher import YFinanceFetcher
from src.data.cache_manager import CacheManager
from src.data.preprocessor import DataPreprocessor

fetcher = YFinanceFetcher()
cache = CacheManager()
preprocessor = DataPreprocessor()

# 使用更大的股票池以获得更稳定的IC
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]
print(f"股票池: {tickers} (共{len(tickers)}只)")

print("正在获取数据...")
data = cache.get_or_fetch(
    market="US",
    universe="adaptive_demo",
    tickers=tickers,
    start_date="2020-01-01",
    end_date="2024-01-01",
    fetcher=fetcher
)

clean_data, report = preprocessor.validate_data_quality(data)
print(f"✅ 清洗后数据: {report['final_shape']}")
print(f"   数据覆盖率: {report['data_coverage']:.2%}")

# ==================== 第2步：创建基础因子 ====================
print("\n🔢 第2步：创建基础因子")
print("-" * 80)

from src.factors.momentum import MomentumFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.volatility import VolatilityFactor

# 创建多个因子变体
factors = [
    MomentumFactor(lookback=252, skip=21, name="Momentum_12M"),
    MomentumFactor(lookback=126, skip=21, name="Momentum_6M"),
    MeanReversionFactor(lookback=21, name="Reversal_1M"),
    VolatilityFactor(window=63, name="LowVol_3M")
]

print(f"创建了{len(factors)}个基础因子：")
for factor in factors:
    print(f"  - {factor.name}")

# ==================== 第3步：静态组合（基线） ====================
print("\n📊 第3步：静态等权重组合（基线）")
print("-" * 80)

from src.factors.base import CompositeFactor

static_composite = CompositeFactor(
    factors=factors,
    weights=[0.25, 0.25, 0.25, 0.25],
    name="StaticComposite"
)

print("计算静态组合因子...")
static_values = static_composite.compute(clean_data, normalize=True)
print(f"✅ 静态组合因子: {len(static_values)} 个观测值")

# ==================== 第4步：自适应组合 ====================
print("\n🎯 第4步：IC加权自适应组合")
print("-" * 80)

from src.factors.adaptive_composite import (
    AdaptiveCompositeFactor,
    AdaptiveCompositeManager
)

# 创建自适应因子
adaptive_factor = AdaptiveCompositeFactor(
    factors=factors,
    ic_window=63,          # 使用最近3个月的IC
    decay_halflife=21,     # 1个月的衰减半衰期
    min_weight=0.05,       # 最小5%权重
    normalization_method='relu',
    name="AdaptiveComposite"
)

# 创建管理器（处理时间序列逻辑）
manager = AdaptiveCompositeManager(
    adaptive_factor=adaptive_factor,
    forward_period=21  # 使用21天前瞻收益率
)

print("逐日计算自适应因子（避免前视偏差）...")
print("注意：这个过程会比较慢，因为需要逐日更新IC历史")

# 获取所有日期
dates = clean_data.index.get_level_values('date').unique().sort_values()
print(f"  总共{len(dates)}个交易日")

# 提取价格用于IC计算
prices = clean_data['close']

# 逐日计算
adaptive_values_list = []
weight_records = []

for i, date in enumerate(dates):
    # 只使用到当前日期的数据
    data_up_to_date = clean_data[
        clean_data.index.get_level_values('date') <= date
    ]
    prices_up_to_date = prices[
        prices.index.get_level_values('date') <= date
    ]

    # 计算因子并更新IC
    composite_at_date, weights_at_date = manager.compute_factor_with_update(
        data=data_up_to_date,
        prices=prices_up_to_date,
        current_date=date
    )

    # 保存当天的因子值
    values_at_date = composite_at_date[
        composite_at_date.index.get_level_values('date') == date
    ]
    adaptive_values_list.append(values_at_date)

    # 每50天打印一次进度
    if (i + 1) % 50 == 0:
        logger.info(f"  进度: {i+1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")

# 合并所有日期的因子值
adaptive_values = pd.concat(adaptive_values_list)
print(f"✅ 自适应组合因子: {len(adaptive_values)} 个观测值")

# ==================== 第5步：因子IC分析 ====================
print("\n📈 第5步：因子IC分析")
print("-" * 80)

from src.evaluation.ic_analysis import ICAnalyzer

ic_analyzer = ICAnalyzer()

# 计算前瞻收益率
print("计算21天前瞻收益率...")
forward_returns_dict = ic_analyzer.compute_forward_returns(
    clean_data, periods=[21]
)
forward_returns = forward_returns_dict[21]

# 静态组合的IC
print("\n静态组合的IC：")
static_ic_series = ic_analyzer.compute_ic_series(
    static_values, forward_returns, method='spearman'
)
static_ic_stats = ic_analyzer.compute_ic_statistics(static_ic_series)

print(f"  IC均值: {static_ic_stats['mean']:.4f}")
print(f"  IC标准差: {static_ic_stats['std']:.4f}")
print(f"  IC IR: {static_ic_stats['sharpe_ratio']:.4f}")
print(f"  t统计量: {static_ic_stats['t_stat']:.2f}")

# 自适应组合的IC
print("\n自适应组合的IC：")
adaptive_ic_series = ic_analyzer.compute_ic_series(
    adaptive_values, forward_returns, method='spearman'
)
adaptive_ic_stats = ic_analyzer.compute_ic_statistics(adaptive_ic_series)

print(f"  IC均值: {adaptive_ic_stats['mean']:.4f}")
print(f"  IC标准差: {adaptive_ic_stats['std']:.4f}")
print(f"  IC IR: {adaptive_ic_stats['sharpe_ratio']:.4f}")
print(f"  t统计量: {adaptive_ic_stats['t_stat']:.2f}")

# IC改进
ic_improvement = adaptive_ic_stats['mean'] - static_ic_stats['mean']
print(f"\n✅ IC改进: {ic_improvement:+.4f} ({ic_improvement/abs(static_ic_stats['mean'])*100:+.1f}%)")

# 查看各基础因子的IC统计
print("\n各基础因子的IC统计：")
ic_summary = adaptive_factor.get_ic_summary()
print(ic_summary.to_string())

# ==================== 第6步：权重演化可视化 ====================
print("\n📊 第6步：权重演化可视化")
print("-" * 80)

weight_history = adaptive_factor.get_weight_history()

if len(weight_history) > 0:
    print(f"记录了{len(weight_history)}天的权重历史")

    # 可视化权重演化
    fig, ax = plt.subplots(figsize=(14, 6))

    for column in weight_history.columns:
        ax.plot(weight_history.index, weight_history[column],
               label=column, linewidth=2, alpha=0.7)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('因子权重', fontsize=12)
    ax.set_title('自适应因子权重演化', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5,
              label='等权重基线')

    plt.tight_layout()
    plt.savefig('/Users/kevinwu/Coding/quant-factor-mining/outputs/adaptive_weights_evolution.png',
               dpi=150)
    print("✅ 权重演化图已保存到 outputs/adaptive_weights_evolution.png")
    plt.close()

# ==================== 第7步：IC时间序列对比 ====================
print("\n📉 第7步：IC时间序列对比")
print("-" * 80)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# IC时间序列
axes[0].plot(static_ic_series.index, static_ic_series.values,
           label='静态组合', alpha=0.6, linewidth=1.5)
axes[0].plot(adaptive_ic_series.index, adaptive_ic_series.values,
           label='自适应组合', alpha=0.6, linewidth=1.5)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[0].set_ylabel('IC值', fontsize=12)
axes[0].set_title('IC时间序列对比', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 累积IC
cumulative_static = static_ic_series.cumsum()
cumulative_adaptive = adaptive_ic_series.cumsum()

axes[1].plot(cumulative_static.index, cumulative_static.values,
           label='静态组合', alpha=0.6, linewidth=2)
axes[1].plot(cumulative_adaptive.index, cumulative_adaptive.values,
           label='自适应组合', alpha=0.6, linewidth=2)
axes[1].set_xlabel('日期', fontsize=12)
axes[1].set_ylabel('累积IC', fontsize=12)
axes[1].set_title('累积IC对比', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/kevinwu/Coding/quant-factor-mining/outputs/ic_comparison.png',
           dpi=150)
print("✅ IC对比图已保存到 outputs/ic_comparison.png")
plt.close()

# ==================== 总结 ====================
print("\n" + "="*80)
print("✅ 演示完成！")
print("="*80)

print("\n📋 主要结果：")
print(f"  1. 静态组合IC: {static_ic_stats['mean']:.4f}")
print(f"  2. 自适应组合IC: {adaptive_ic_stats['mean']:.4f}")
print(f"  3. IC改进: {ic_improvement:+.4f} ({ic_improvement/abs(static_ic_stats['mean'])*100:+.1f}%)")
print(f"  4. 自适应组合IC IR: {adaptive_ic_stats['sharpe_ratio']:.4f}")

print("\n🎯 关键洞察：")
if adaptive_ic_stats['mean'] > static_ic_stats['mean']:
    print("  ✅ 自适应组合优于静态等权重组合")
    print("  ✅ IC加权机制成功提升了因子组合效果")
else:
    print("  ⚠️ 自适应组合未显著优于静态组合")
    print("  💡 可能的原因：")
    print("     - 样本期较短（需要更长历史数据）")
    print("     - IC窗口参数需要调整")
    print("     - 基础因子之间相关性过高")

print("\n📚 下一步：")
print("  1. 运行 python3 examples/demo_regime_aware.py 测试市场状态感知")
print("  2. 查看 outputs/ 目录的可视化结果")
print("  3. 尝试不同的ic_window和decay_halflife参数")
print("  4. 添加更多基础因子以提升多样性")

print("\n🎉 自适应因子系统已就绪！\n")
