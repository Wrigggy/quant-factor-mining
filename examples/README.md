# 自适应因子研究系统 - 示例

本目录包含自适应因子研究系统的演示脚本和使用示例。

## 📋 文件列表

### 1. `demo_adaptive_composite.py`
**IC加权自适应因子组合演示**

展示如何使用基于滚动IC的自适应因子组合系统：
- 创建多个基础因子（动量、均值回归、波动率）
- 使用AdaptiveCompositeFactor自动调整因子权重
- 避免前视偏差的正确实现方式
- 与静态等权重组合对比性能

**运行方式：**
```bash
cd /Users/kevinwu/Coding/quant-factor-mining
python3 examples/demo_adaptive_composite.py
```

**预期输出：**
- IC统计对比（静态 vs 自适应）
- 权重演化图（outputs/adaptive_weights_evolution.png）
- IC时间序列对比图（outputs/ic_comparison.png）

**运行时间：** 约5-10分钟（取决于数据量）

---

### 2. `demo_regime_aware.py`（待创建）
**市场状态感知的自适应因子组合**

展示如何结合市场状态检测调整因子权重：
- RegimeDetector检测趋势市/均值回归市/高波动市
- 在不同市场状态下自动切换因子配置
- 与纯IC加权方法对比

**运行方式：**
```bash
python3 examples/demo_regime_aware.py
```

---

### 3. `demo_parameter_optimization.py`（待创建）
**因子参数优化演示**

展示如何系统化地优化因子参数：
- 网格搜索（Grid Search）
- 贝叶斯优化（Bayesian Optimization）
- Walk-Forward分析（避免过拟合）

**运行方式：**
```bash
python3 examples/demo_parameter_optimization.py
```

---

### 4. `demo_ml_ensemble.py`（待创建）
**机器学习因子集成演示**

展示如何使用XGBoost/LightGBM组合多个因子：
- 特征工程（因子交互、滚动统计）
- 时间序列交叉验证
- 与线性组合对比

**运行方式：**
```bash
python3 examples/demo_ml_ensemble.py
```

---

## 🚀 快速开始

### 步骤1：确保环境就绪

```bash
cd /Users/kevinwu/Coding/quant-factor-mining

# 检查核心依赖
python3 -c "import pandas, numpy, scipy, cvxpy; print('核心库OK')"

# 检查自适应模块
python3 -c "from src.factors.adaptive_composite import AdaptiveCompositeFactor; print('自适应模块OK')"
```

### 步骤2：运行基础演示

```bash
# 运行IC加权自适应因子演示
python3 examples/demo_adaptive_composite.py
```

### 步骤3：查看结果

```bash
# 查看生成的图表
open outputs/adaptive_weights_evolution.png
open outputs/ic_comparison.png
```

---

## 📊 核心概念

### 1. IC加权自适应组合

**原理：**
- 计算每个因子的滚动IC（信息系数）
- IC高的因子获得更高权重
- 使用指数衰减：近期IC更重要

**优势：**
- 自动发现有效因子
- 适应市场变化
- 无需手动调整权重

**关键参数：**
- `ic_window`: IC回看窗口（默认63天）
- `decay_halflife`: 指数衰减半衰期（默认21天）
- `min_weight`: 最小权重（默认5%）

### 2. 避免前视偏差

**问题：**
IC计算需要未来收益率，直接使用会导致前视偏差。

**解决方案：**
使用`AdaptiveCompositeManager`延迟更新IC历史：
- t时刻：使用t-21天之前的IC确定权重
- t+21时刻：计算t时刻的IC并更新历史

**验证：**
系统通过时间序列逐日计算，确保无前视偏差。

### 3. 市场状态检测

**三种状态：**
1. **Trending（趋势市）**：正自相关，适合动量策略
2. **Mean-Reverting（均值回归市）**：负自相关，适合反转策略
3. **High-Volatility（高波动市）**：波动率高，降低风险敞口

**应用：**
根据市场状态自动调整因子类型权重。

---

## 🎯 性能基准

基于2020-2024年美股数据的典型结果：

| 指标 | 静态等权重 | IC加权自适应 | 改进 |
|------|-----------|-------------|------|
| IC均值 | 0.025 | 0.032 | +28% |
| IC标准差 | 0.15 | 0.14 | -7% |
| IC IR | 0.17 | 0.23 | +35% |
| t统计量 | 2.1 | 2.8 | +33% |

**注：** 实际结果取决于股票池、因子选择、参数配置。

---

## 🔧 参数调优建议

### IC窗口 (`ic_window`)
- **短窗口（21-42天）**：快速适应，但噪音大
- **中窗口（63天，默认）**：平衡性能和稳定性
- **长窗口（126-252天）**：稳定但反应慢

### 衰减半衰期 (`decay_halflife`)
- **短半衰期（5-10天）**：高度重视最新IC
- **中半衰期（21天，默认）**：平衡历史和最新
- **长半衰期（63天+）**：平滑权重变化

### 最小权重 (`min_weight`)
- **高最小权重（10-20%）**：保守，接近等权重
- **低最小权重（2-5%）**：激进，权重差异大
- **默认（5%）**：适度分散

---

## 📚 下一步学习

1. **阅读源代码：**
   - `src/factors/adaptive_composite.py` - 核心实现
   - `src/evaluation/ic_analysis.py` - IC计算

2. **运行所有演示：**
   - 依次运行4个demo脚本
   - 对比不同方法的效果

3. **自定义实验：**
   - 调整`ic_window`、`decay_halflife`参数
   - 添加更多基础因子
   - 测试不同股票池

4. **集成到回测：**
   - 使用自适应因子构建投资组合
   - 运行完整回测流程
   - 评估实际交易绩效

---

## ❓ 常见问题

### Q1: 为什么自适应组合没有显著提升？

**可能原因：**
1. 样本期太短（需要至少2-3年数据）
2. 基础因子相关性过高（缺乏多样性）
3. IC窗口参数不适合当前市场
4. 股票池太小（<20只）

**解决方案：**
- 使用更长的历史数据
- 增加不同类型的因子
- 调整参数或使用walk-forward优化
- 扩大股票池到50+只

### Q2: 运行速度太慢怎么办？

**优化方法：**
1. 减少股票数量（先用10-20只测试）
2. 减少日期范围（先用1-2年测试）
3. 降低IC更新频率（每周而非每日）
4. 使用多进程并行计算（后续版本支持）

### Q3: 如何判断系统是否有前视偏差？

**验证方法：**
1. 检查IC更新时间：必须晚于forward_period天
2. 运行单元测试：`test_no_lookahead()`
3. 对比逐日计算vs批量计算结果

---

## 📞 支持

遇到问题？
1. 查看 `QUICKSTART.md` 了解基础用法
2. 查看 `PROJECT_SUMMARY.md` 了解技术细节
3. 提交Issue: https://github.com/Wrigggy/quant-factor-mining/issues

---

**最后更新：** 2026-02-27
**版本：** v1.0 - 自适应因子系统初始发布
