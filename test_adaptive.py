"""
快速测试：自适应因子组合系统

验证核心功能是否正常工作
"""

import sys
sys.path.append('.')

print("\n" + "="*80)
print("自适应因子系统 - 快速验证测试")
print("="*80)

# 测试1：模块导入
print("\n✓ 测试1：模块导入")
try:
    from src.factors.adaptive_composite import (
        AdaptiveCompositeFactor,
        AdaptiveCompositeManager,
        RegimeDetector,
        RegimeAwareCompositeFactor
    )
    print("  ✅ 自适应因子模块导入成功")
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    sys.exit(1)

# 测试2：创建简单因子
print("\n✓ 测试2：创建基础因子")
try:
    from src.factors.momentum import MomentumFactor
    from src.factors.mean_reversion import MeanReversionFactor

    factors = [
        MomentumFactor(lookback=252, skip=21),
        MeanReversionFactor(lookback=21)
    ]
    print(f"  ✅ 创建了{len(factors)}个基础因子")
except Exception as e:
    print(f"  ❌ 创建失败: {e}")
    sys.exit(1)

# 测试3：创建自适应因子
print("\n✓ 测试3：创建自适应因子组合")
try:
    adaptive_factor = AdaptiveCompositeFactor(
        factors=factors,
        ic_window=63,
        decay_halflife=21,
        min_weight=0.05,
        name="TestAdaptive"
    )
    print(f"  ✅ AdaptiveCompositeFactor创建成功")
    print(f"     因子数量: {len(adaptive_factor.factors)}")
    print(f"     IC窗口: {adaptive_factor.ic_window}天")
except Exception as e:
    print(f"  ❌ 创建失败: {e}")
    sys.exit(1)

# 测试4：创建管理器
print("\n✓ 测试4：创建自适应管理器")
try:
    manager = AdaptiveCompositeManager(
        adaptive_factor=adaptive_factor,
        forward_period=21
    )
    print(f"  ✅ AdaptiveCompositeManager创建成功")
    print(f"     前瞻周期: {manager.forward_period}天")
except Exception as e:
    print(f"  ❌ 创建失败: {e}")
    sys.exit(1)

# 测试5：市场状态检测器
print("\n✓ 测试5：市场状态检测器")
try:
    import pandas as pd
    import numpy as np

    # 创建模拟市场收益率
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    market_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)

    detector = RegimeDetector(lookback=63)
    regime = detector.detect_regime(market_returns)

    print(f"  ✅ RegimeDetector工作正常")
    print(f"     检测到的市场状态: {regime}")
except Exception as e:
    print(f"  ❌ 检测失败: {e}")
    sys.exit(1)

# 测试6：权重标准化方法
print("\n✓ 测试6：权重标准化方法")
try:
    raw_weights = {
        'Momentum': 0.03,
        'Reversal': -0.01
    }

    # 测试三种标准化方法
    for method in ['softmax', 'relu', 'rank']:
        test_factor = AdaptiveCompositeFactor(
            factors=factors,
            normalization_method=method
        )
        normalized = test_factor._normalize_weights(raw_weights)
        weight_sum = sum(normalized.values())

        assert abs(weight_sum - 1.0) < 1e-6, f"{method}方法权重和不为1"
        print(f"  ✅ {method}标准化方法正常: {normalized}")

except Exception as e:
    print(f"  ❌ 标准化测试失败: {e}")
    sys.exit(1)

# 测试7：IC历史更新
print("\n✓ 测试7：IC历史更新")
try:
    # 模拟更新IC
    adaptive_factor.update_ic_history('Momentum', '2023-01-01', 0.035)
    adaptive_factor.update_ic_history('Momentum', '2023-01-02', 0.028)
    adaptive_factor.update_ic_history('Reversal', '2023-01-01', -0.015)

    # 获取IC摘要
    ic_summary = adaptive_factor.get_ic_summary()

    print(f"  ✅ IC历史更新成功")
    print(f"     Momentum观测数: {ic_summary.loc['Momentum', 'n_observations']}")
    print(f"     Momentum平均IC: {ic_summary.loc['Momentum', 'ic_mean']:.4f}")
except Exception as e:
    print(f"  ❌ IC更新失败: {e}")
    sys.exit(1)

# 总结
print("\n" + "="*80)
print("✅ 所有测试通过！自适应因子系统就绪")
print("="*80)

print("\n📋 系统状态:")
print("  - AdaptiveCompositeFactor: ✅ 正常")
print("  - AdaptiveCompositeManager: ✅ 正常")
print("  - RegimeDetector: ✅ 正常")
print("  - 权重标准化: ✅ 正常")
print("  - IC历史管理: ✅ 正常")

print("\n🎯 下一步:")
print("  1. 运行完整演示: python3 examples/demo_adaptive_composite.py")
print("  2. 查看使用指南: cat ADAPTIVE_GUIDE.md")
print("  3. 查看示例说明: cat examples/README.md")

print("\n")
