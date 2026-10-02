# AGI 盈利推算及分析能力增强

## Context

AGI orchestrator 当前的自治闭环为 `perceive → diagnose → decide → act → reflect`，其中 `_diagnose` 是**纯后视**的（看当前状态+历史趋势产生 alerts），`_decide` 使用 `DynamicAllocator` 做资金分配，`_reflect` 记录 `_decision_memory`（cycle/health/pnl/equity/regime）。

**缺失能力**：无前瞻盈利推算、无盈亏归因分解、无场景推演。AGI 无法回答"未来 N 周期预期盈亏多少""哪个策略贡献最大""若 regime 切换会怎样"等关键问题，导致资金分配仅依赖后视指标而非前瞻期望。

**目标**：新增 `projection`（前瞻推算）+ `attribution`（归因分析）两个步骤，形成"诊断→归因→推算→决策→执行→反思"六段式闭环，推算结果注入分配决策与进攻性加仓 gate。

## 实现方案

### 1. 新增三个核心方法（core/quant_agi_orchestrator.py）

#### 1.1 `_attribute_pnl(perception) -> Dict`
将 `total_pnl` 按四个维度分解：
- **按策略**：`strategies[name].total_pnl` → `share = pnl_i / sum(|pnl|)`，附加 `risk_adjusted_share`
- **按 regime**：从 `_decision_memory` 取最近 N 周期，按 `regime` 分桶累加 `cycle_pnl`，输出每 regime `avg_per_cycle`（样本<3 标 `low_confidence`）
- **按方向**：聚合 `long_pnl / short_pnl / long_trades / short_trades` → `long_share / short_share / long_short_ratio`
- **按退出原因**：`take_profit_pnl / stop_loss_pnl / realized_pnl / unrealized_pnl` → 止盈贡献 vs 止损侵蚀 vs 账面依赖度

输出包含 `dominant_strategy`、`worst_strategy`、`attribution_quality`（样本量置信度）。

#### 1.2 `_project_pnl(perception, attribution) -> Dict`
基于历史表现估算未来 N 周期盈亏，按策略逐个推算后聚合：

1. **基础期望**：`base_expect = win_rate × avg_win - (1-win_rate) × avg_loss`；缺失时用 `pnl_per_trade` 兜底
2. **趋势调整**：取 `_strategy_pnl_per_trade_history[name]` 和 `_strategy_delta_pnl_history[name]` 的线性回归斜率，`trend_adj = clamp(slope × trend_weight, ±max_trend_adj_ratio × |base_expect|)`；`trend_pnl_7d_vs_30d` 做次级修正
3. **regime 调整**：复用 `self._trend_profit_take_mult`（趋势市）和 `self._range_profit_take_mult`（震荡市）；`confidence < threshold` 时按比例衰减
4. **准确度校正**：从 `_projection_accuracy` deque 取最近 N 周期 `bias_ratio`（实际/预估）的中位数作为 `correction_factor`（clamp [0.5, 2.0]，样本不足返回 1.0）
5. **累积推算**：`projected_pnl_horizon = corrected_expect × horizon_cycles`；置信带 `±1.96 × volatility × √horizon`
6. **场景推演**：`base_case`（当前 regime 持续）、`bear_case`（regime 切换到 range_bound + strength=0.3）、`bull_case`（strength+0.2 趋势加强）

输出结构：`{available, horizon_cycles, regime, correction_factor, total: {base_case/bear_case/bull_case: {per_cycle, horizon, ci_lower, ci_upper}}, per_strategy: {<name>: {base_expect, trend_adjustment, regime_mult, corrected_expect, horizon, confidence}}, projection_cycle, timestamp}`

#### 1.3 `_track_projection_accuracy(perception) -> Dict`
在 `_reflect` 中调用，闭环追踪推算准确度：
- 从 `_last_projection` 取上一周期 `total.base_case.per_cycle`
- 从 `_decision_memory[-1]` 取本周期实际 `cycle_pnl`
- `bias_ratio = actual / projected`（除零时 1.0），写入 `_projection_accuracy` deque（maxlen=20）
- `correction_factor = clamp(median(bias_ratios), 0.5, 2.0)`，样本<5 时 1.0
- 回写 `self._correction_factor` 供下周期 `_project_pnl` 使用

### 2. run_cycle 集成（L2402 附近 + L2431 附近）

**report 初始化**新增字段：
```python
"projection": {},
"attribution": {},
```

**在 `_diagnose` 之后、`_decide` 之前插入**：
```python
# 2.5 归因+推算（前瞻盈利推算）
attribution = self._attribute_pnl(perception)
projection = self._project_pnl(perception, attribution)
report["attribution"] = attribution
report["projection"] = projection
```

将 `projection` 传入 `_decide` 和 `_offensive_allocation_actions`（通过 decision 或 report 传递）。

### 3. 影响 `_decide`（L5200 附近）

在 `build_strategy_metrics` 调用之后，将推算结果注入 metrics：
```python
# 推算结果注入 strategy_metrics（供 DynamicAllocator 优先级评分参考）
if self._pnl_projection_enabled and projection.get("available"):
    for name, sp in (projection.get("per_strategy") or {}).items():
        if name in metrics:
            metrics[name]["projected_pnl_per_cycle"] = sp["corrected_expect"]
            metrics[name]["projection_confidence"] = sp["confidence"]
    # bear_case 大幅亏损 → fail-closed
    bear_horizon = (projection.get("total", {}).get("bear_case", {}) or {}).get("horizon", 0.0)
    if bear_horizon < -self._projection_fail_closed_loss:
        # 保守空计划 + warning
        ...
```

### 4. 影响 `_offensive_allocation_actions`（L9305 附近 + L9359 附近）

**新增 per-strategy gate**（在 `low_ppt` gate 之后）：
```python
# 推算预期为负的策略集合（per-strategy gate）
negative_projection_strategies: set = set()
if self._pnl_projection_enabled and projection.get("available"):
    for _n, _sp in (projection.get("per_strategy") or {}).items():
        if safe_float(_sp.get("corrected_expect"), 0.0) < -self._offensive_projection_loss_threshold:
            negative_projection_strategies.add(str(_n))
```
循环内追加：
```python
if str(name) in negative_projection_strategies:
    continue
```

**推算缩放 boost**（在 `strategy_boost = boost_step * rl_mult` 之后）：
```python
# 推算缩放：正期望高置信→加大 boost；负期望→缩小
if self._pnl_projection_enabled:
    _sp = (projection.get("per_strategy") or {}).get(str(name)) or {}
    _proj_mult = self._projection_boost_scale(
        _sp.get("corrected_expect"), _sp.get("confidence"))
    strategy_boost *= _proj_mult
```

`_projection_boost_scale(expect, confidence)` 辅助方法：基于 `sign(expect) × min(1, |expect|/unit) × confidence × span`，clamp [min_mult, max_mult]。

### 5. `_reflect` 扩充（L9657 附近）

追加推算准确度追踪：
```python
if self._pnl_projection_enabled and self._last_projection is not None:
    accuracy = self._track_projection_accuracy(perception)
    report["reflection"]["projection_accuracy"] = accuracy
self._last_projection = projection
```

### 6. config.yaml 新增配置（`agi_orchestrator` 段内，L2453 附近）

```yaml
  pnl_projection:
    enabled: true
    horizon_cycles: 5
    accuracy_window: 20
    min_accuracy_samples: 5
    min_correction: 0.5
    max_correction: 2.0
    trend_weight: 0.3
    max_trend_adj_ratio: 0.5
    confidence_level: 0.95
    projection_adaptation_span: 0.2
    projection_min_mult: 0.5
    projection_max_mult: 1.5
    fail_closed_loss_threshold: 0.05
    offensive_projection_loss_threshold: 0.0
    projection_unit_pnl: 1.0
    min_attribution_trades: 20
```

### 7. `__init__` 初始化（L1185 附近，pnl_per_trade_guard 配置块之后）

读取上述配置，初始化 `_projection_accuracy` deque、`_last_projection`、`_correction_factor` 实例字段。

### 8. 状态持久化

- `_serialize_learning_state`（L2143）：新增 `projection_accuracy`、`last_projection`（精简版）、`correction_factor`
- `_load_learning_state`（L1980 附近）：恢复上述字段

### 9. 单元测试（tests/unit/test_pnl_projection.py）

新建独立测试文件，覆盖：
1. 禁用时返回空 dict
2. 基础期望计算（win_rate×avg_win - (1-win_rate)×avg_loss）
3. 趋势调整（上升序列→正调整）
4. regime 乘子（trend vs range）
5. 准确度校正系数（中位数 clamp）
6. bear_case fail-closed
7. 归因按策略/regime/方向/退出原因分解
8. 推算准确度追踪（bias_ratio 计算+correction_factor 更新）
9. 进攻性分配跳过负期望策略
10. boost 缩放（高置信正期望→放大，负期望→缩小）
11. 状态持久化（projection_accuracy + correction_factor 恢复）
12. JSON 安全（无 NaN/Inf）

## 验证步骤

1. `py -c "import yaml; yaml.safe_load(open('config.yaml', encoding='utf-8'))"` — YAML 语法
2. 临时移除 `data/agi_orchestrator_state.json` 后 `py -m pytest tests/unit/test_pnl_projection.py -q`
3. `py -m pytest -q` — 完整回归（预期 2934+13=2947 passed）
4. `py stop.py && py start.py` — 重启系统
5. 检查 Dashboard 报告含 `projection` 和 `attribution` 字段

## 关键文件

- [quant_agi_orchestrator.py](file:///e:/新建文件夹/okx_quant_trading/core/quant_agi_orchestrator.py) — 主要修改
- [config.yaml](file:///e:/新建文件夹/okx_quant_trading/config.yaml) — 配置新增
- tests/unit/test_pnl_projection.py — 新建测试
- [contribution_analyzer.py](file:///e:/新建文件夹/okx_quant_trading/analysis/contribution_analyzer.py) — 只读参考（StrategyContribution 字段来源）
