# 因子挖掘框架优化 - 完整方案

## 一、优化目标

1. **因子经济逻辑评估** - 评估因子是否有经济学意义
2. **独立有效因子筛选** - 筛选出独立且有效的因子
3. **因子权重优化** - 优化因子权重分配，降低冗余
4. **组合构建优化** - 构建稳健的投资组合

---

## 二、模块架构

```
analysis/multi_factor/
├── factor_engine.py           # 因子计算引擎 (6大类因子)
├── factor_analyzer.py         # 因子分析器 (IC、共线性、衰减)
├── factor_evaluator.py        # 因子评估器 (经济逻辑、分层回测、稳健性) [NEW]
├── factor_optimizer.py        # 因子优化器 (筛选、去冗余、权重优化) [NEW]
├── stock_selector.py          # 选股器 (因子打分、选股)
└── portfolio_builder.py       # 组合构建器 (组合优化、绩效评估) [NEW]
```

---

## 三、因子评估框架

### 3.1 评估维度

#### 1. 统计显著性
- **Rank IC**: Spearman秩相关系数
- **t统计量**: IC的统计显著性检验
- **p值**: 显著性水平 (5%, 1%)

#### 2. 经济显著性
- **分层回测**: 将股票按因子值分为5组，计算各组平均收益
- **多空收益**: Top组 - Bottom组的收益差
- **经济意义**: 多空收益 > 1% (月度)

#### 3. 单调性检验
- **组间单调性**: 各组收益是否单调递增/递减
- **Spearman相关**: 组号与收益的秩相关
- **Kendall tau**: 单调性强度

#### 4. 稳健性检验
- **子样本测试**: 随机抽取70%样本，重复10次
- **IC稳定性**: 子样本IC的标准差 < 0.1
- **稳健因子**: IC均值 > 0.03 且 IC标准差 < 0.1

#### 5. 综合评分 (0-100)
- 统计显著性: 30分
- 经济显著性: 30分
- 单调性: 20分
- 稳健性: 20分

### 3.2 因子评估代码

```python
from analysis.multi_factor import FactorEvaluator

evaluator = FactorEvaluator(n_groups=5)

# 评估单个因子
report = evaluator.evaluate_factor(
    factor_values=factor_series,
    forward_returns=return_series,
    factor_name="ep",
    date="2024-01-15"
)

# 生成评估报告
report_df = evaluator.generate_factor_report(evaluation_results)
print(report_df)
```

**输出示例:**
```
factor_name  n_stocks  rank_ic  t_stat  significant  long_short_return  monotonic  composite_score
ep           100       0.085    2.67    True         0.023              True       85.0
bp           100       0.072    2.25    True         0.018              True       78.0
roe          100       0.065    2.03    True         0.015              False      65.0
```

---

## 四、因子筛选框架

### 4.1 筛选标准

#### 1. 有效性筛选
- **最小IC**: |IC| >= 0.03
- **最小IC_IR**: IC_IR >= 0.5 (需要多期数据)

#### 2. 独立性筛选 (去冗余)
- **相关性阈值**: 因子间相关性 < 0.7
- **聚类去冗余**: 高相关因子对中保留IC最高的

### 4.2 因子筛选代码

```python
from analysis.multi_factor import FactorOptimizer

optimizer = FactorOptimizer(corr_threshold=0.7)

# 筛选独立有效因子
selected_factors = optimizer.screen_independent_factors(
    factors=factor_matrix,
    forward_returns=return_series,
    min_ic=0.03,
    min_ic_ir=0.5
)

print(f"筛选后因子: {selected_factors}")

# 冗余报告
redundancy_report = optimizer.compute_factor_redundancy_report(factors)
print(f"平均相关性: {redundancy_report['average_correlation']:.3f}")
print(f"高相关性因子对: {redundancy_report['n_high_corr_pairs']}")
```

### 4.3 PCA降维 (可选)

```python
# PCA降维，保留90%方差
pca_factors = optimizer.pca_factor_reduction(
    factors, variance_explained=0.90
)
print(f"PCA降维后: {pca_factors.shape[1]} 个主成分")
```

---

## 五、因子权重优化

### 5.1 优化方法

#### 1. 等权 (Equal Weight)
- 所有因子权重相等
- 优点: 简单稳健
- 缺点: 忽略因子差异

#### 2. IC加权 (IC Weight)
- 权重 ∝ |IC|
- 优点: 重视有效因子
- 缺点: 忽略因子相关性

#### 3. IC_IR加权 (IC_IR Weight)
- 权重 ∝ IC_IR (IC均值/IC标准差)
- 优点: 重视稳定因子
- 缺点: 需要多期数据

#### 4. 最大化夏普比率 (Max Sharpe)
- 优化目标: 最大化组合夏普比率
- 约束: 权重和为1，权重非负
- 优点: 理论最优
- 缺点: 可能过拟合

#### 5. 风险平价 (Risk Parity)
- 权重 ∝ 1/波动率
- 优点: 风险均衡
- 缺点: 忽略因子收益

### 5.2 权重优化代码

```python
from analysis.multi_factor import FactorOptimizer

optimizer = FactorOptimizer()

# IC加权
ic_weights = optimizer.optimize_weights(
    factors=factor_matrix,
    forward_returns=return_series,
    method="ic"
)

# 最大化夏普比率
opt_weights = optimizer.optimize_weights(
    factors=factor_matrix,
    forward_returns=return_series,
    method="max_sharpe",
    constraints={"long_only": True, "max_weight": 0.5}
)

print(f"IC权重: {ic_weights}")
print(f"优化权重: {opt_weights}")
```

---

## 六、组合构建框架

### 6.1 构建方法

#### 1. Top-N等权
- 选取得分最高的N只股票
- 等权配置
- 优点: 简单
- 缺点: 忽略得分差异

#### 2. 得分加权
- 权重 ∝ 因子得分
- 优点: 重视高分股票
- 缺点: 可能过度集中

#### 3. 优化加权
- 最大化加权得分
- 约束: 单只股票最大权重10%
- 优点: 分散风险
- 缺点: 计算复杂

#### 4. 行业中性
- 每个行业选Top-N/行业数
- 行业间等权，行业内等权
- 优点: 行业均衡
- 缺点: 需要行业数据

### 6.2 组合构建代码

```python
from analysis.multi_factor import PortfolioBuilder

builder = PortfolioBuilder(max_weight=0.1)

# Top-N等权
portfolio = builder.build_portfolio(
    stock_scores=composite_scores,
    method="top_n",
    top_n=50
)

# 得分加权
portfolio = builder.build_portfolio(
    stock_scores=composite_scores,
    method="score_weight"
)

# 行业中性
portfolio = builder.build_portfolio(
    stock_scores=composite_scores,
    method="industry_neutral",
    top_n=50,
    industry=industry_map
)

print(f"组合股票数: {len(portfolio)}")
print(f"前5大持仓: {sorted(portfolio.items(), key=lambda x: -x[1])[:5]}")
```

### 6.3 绩效评估

```python
# 计算组合绩效
stats = builder.compute_portfolio_stats(
    weights=portfolio,
    returns=historical_returns,
    risk_free_rate=0.03
)

print(f"年化收益: {stats['annualized_return']:.2%}")
print(f"年化波动: {stats['annualized_volatility']:.2%}")
print(f"夏普比率: {stats['sharpe_ratio']:.3f}")
print(f"最大回撤: {stats['max_drawdown']:.2%}")
print(f"胜率: {stats['win_rate']:.2%}")
```

### 6.4 风险贡献分析

```python
# 计算风险贡献
risk_contrib = builder.compute_risk_contribution(
    weights=portfolio,
    cov_matrix=covariance_matrix
)

print("风险贡献:")
for stock, rc in sorted(risk_contrib.items(), key=lambda x: -x[1])[:10]:
    print(f"  {stock}: {rc:.2%}")
```

### 6.5 调仓计算

```python
# 计算调仓交易
trades = builder.rebalance_portfolio(
    current_weights=current_portfolio,
    target_weights=target_portfolio,
    threshold=0.01  # 权重变化>1%才交易
)

print("调仓交易:")
for stock, trade in trades.items():
    action = "买入" if trade > 0 else "卖出"
    print(f"  {stock}: {action} {abs(trade):.2%}")
```

---

## 七、完整工作流

```python
from analysis.multi_factor import (
    FactorEngine,
    FactorAnalyzer,
    FactorEvaluator,
    FactorOptimizer,
    PortfolioBuilder,
)

# 1. 计算因子
engine = FactorEngine()
factors = engine.compute_all_factors(stock_list, date, price_data, fundamental_data)

# 2. 因子评估
evaluator = FactorEvaluator()
evaluation_results = []
for factor_name in factors.columns:
    report = evaluator.evaluate_factor(
        factors[factor_name], forward_returns, factor_name
    )
    evaluation_results.append(report)

# 3. 因子筛选
optimizer = FactorOptimizer(corr_threshold=0.7)
selected_factors = optimizer.screen_independent_factors(
    factors, forward_returns, min_ic=0.03
)

# 4. 因子权重优化
weights = optimizer.optimize_weights(
    factors[selected_factors], forward_returns, method="ic"
)

# 5. 计算综合得分
composite_score = pd.Series(0.0, index=factors.index)
for factor_name, weight in weights.items():
    factor_std = (factors[factor_name] - factors[factor_name].mean()) / factors[factor_name].std()
    composite_score += weight * factor_std

# 6. 构建组合
builder = PortfolioBuilder(max_weight=0.1)
portfolio = builder.build_portfolio(
    composite_score, method="top_n", top_n=50
)

# 7. 绩效评估
stats = builder.compute_portfolio_stats(portfolio, historical_returns)
print(f"夏普比率: {stats['sharpe_ratio']:.3f}")
```

---

## 八、自测校验清单

### 因子评估
- [ ] 因子IC计算正确 (Spearman秩相关)
- [ ] 分层回测正确 (5组，计算各组平均收益)
- [ ] 单调性检验正确 (组间收益单调)
- [ ] 稳健性检验正确 (子样本测试)
- [ ] 综合评分合理 (0-100分)

### 因子筛选
- [ ] 有效性筛选正确 (|IC| >= 0.03)
- [ ] 独立性筛选正确 (相关性 < 0.7)
- [ ] 去冗余逻辑正确 (保留IC最高的)
- [ ] PCA降维正确 (保留90%方差)

### 因子权重优化
- [ ] 等权正确 (权重和为1)
- [ ] IC加权正确 (权重 ∝ |IC|)
- [ ] 优化加权正确 (最大化夏普比率)
- [ ] 风险平价正确 (权重 ∝ 1/波动率)

### 组合构建
- [ ] Top-N选股正确 (按得分排序)
- [ ] 得分加权正确 (权重 ∝ 得分)
- [ ] 行业中性正确 (行业间等权)
- [ ] 最大权重限制正确 (<=10%)

### 绩效评估
- [ ] 年化收益正确 (月度收益 * 12)
- [ ] 夏普比率正确 ((收益-无风险)/波动)
- [ ] 最大回撤正确 (累计净值最大跌幅)
- [ ] 风险贡献正确 (边际风险 * 权重)

---

## 九、注意事项

1. **数据质量**: 因子计算前需清洗异常值 (MAD去极值)
2. **缺失值处理**: 因子缺失值填充为0或中位数
3. **因子方向**: 统一为越大越好 (负债率需取反)
4. **样本量**: 因子评估至少需要30只股票
5. **多重检验**: 多因子测试需进行Bonferroni/FDR校正
6. **过拟合风险**: 样本外测试 (walk-forward) 验证因子有效性
7. **交易成本**: 组合构建需考虑手续费、滑点、印花税

---

## 十、后续优化方向

1. **因子挖掘**: 机器学习自动发现有效因子
2. **动态权重**: 根据市场状态动态调整因子权重
3. **非线性因子**: 因子交互效应 (如价值*动量)
4. **另类数据**: 接入舆情、供应链、专利等数据
5. **组合优化**: 均值-方差优化、Black-Litterman模型
6. **风险控制**: 行业暴露限制、风格因子中性化
