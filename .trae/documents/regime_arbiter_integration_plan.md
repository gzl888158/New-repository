# 两套 Regime 引擎输出融合-统一仲裁实施方案

## Context

项目有两套 regime 引擎并行运行：主引擎 `MarketRegimeEngine`（services/market_regime_engine.py）服务于开仓/分配链路（orchestrator/regime_gate/signal_processor/adaptive_controller/tp_sl_engine），细粒度检测器 `MarketRegimeDetector`（app/services/adaptive_learning/market_regime_detector.py）仅服务于止损管理器（stop_loss_manager L602 `_fetch_reversal_inputs`）。两者输出未交叉融合，同一时刻可能矛盾：主引擎说 trend_bullish，检测器说 reversal，导致开仓积极但止损保守（或反之）。

用户选定"融合输出-统一仲裁"：引入 `RegimeArbiter`，融合两引擎输出，按置信度加权选 regime，统一注入所有下游（含止损），fail-closed 回退主引擎。

## 关键事实（已实证）

- 主引擎 L16-30 已含 BREAKOUT/BREAKDOWN/REVERSAL 状态 + 兼容别名（L34-38）
- 主引擎 L785-797 `get_regime()` 返回 dict: {regime, normalized_regime, state, subtype, strength, confidence, factor_scores, factor_weights, last_update}
- 检测器 L1399-1490 `detect_regime(symbol, ohlcv)` 返回 dict: {symbol, regime, probabilities, features, optimal_strategies, early_warnings, timeframe, detected_at}
- orchestrator `_perceive` L2881-2892 调 `regime_engine.get_regime()` 写入 `perception["market_regime"]`
- scheduler L1007-1009 实例化检测器注入 stop_loss_manager；L1079-1121 主引擎注入下游；L480-486 构造 orchestrator 传 `market_regime_engine`
- orchestrator `__init__` L269-275 已有 `regime_engine=None` 参数

## 改动文件清单

| 文件 | 改动 | 行号区间 |
|---|---|---|
| `services/regime_arbiter.py` | 新建 | 全文 |
| `core/quant_agi_orchestrator.py` | `__init__` 增 `regime_arbiter=None`；`_perceive` 改读 arbiter | L269-275, L2881-2897 |
| `core/scheduler.py` | L1009 后构造 `regime_arbiter`；L486 构造 orchestrator 时传入 | L1009, L480-486 |
| `core/stop_loss_manager.py` | 新增 `set_regime_arbiter`；`_fetch_reversal_inputs` 优先走 arbiter | L211-213, L602-620 |
| `tests/unit/test_regime_arbiter.py` | 新建测试 | 全文 |

## RegimeArbiter 类设计

位置：`services/regime_arbiter.py`（与 market_regime_engine.py/regime_gate.py 同目录）

```python
class RegimeArbiter:
    def __init__(self, main_engine, detector, config: Dict = None):
        # config: reversal_threshold=0.45, w_main=0.6, w_detector=0.4,
        #         conflict_log="data/regime_arbiter/conflicts.jsonl", history_len=200

    def arbitrate(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        # 输出主引擎 schema + 扩展字段
```

**输出 schema**（兼容主引擎，下游零改动）：
- 主引擎原字段：regime/normalized_regime/state/subtype/strength/confidence/factor_scores/factor_weights/last_update
- 扩展字段：detector_regime, detector_reversal_prob, early_warnings, arbiter_conflict, arbiter_strategy, source_weights, detector_raw

## 仲裁算法

1. `main_out = main_engine.get_regime()`；`det_out = detector.get_regime(symbol)`
2. det_out 缺失/异常 → 返回 main_out，strategy="main_fallback"
3. `mapped = MAP[det_out.regime]`；UNKNOWN → main_fallback
4. mapped == main_out.regime → resolved=main_out, confidence+=0.1(cap 1.0), strategy="consensus"
5. 否则 conflict=True：
   - mapped==REVERSAL 且 det_out.probabilities["reversal"]>=reversal_threshold → resolved=main_out 但 regime=REVERSAL, subtype="reversal", strategy="detector_reversal_override"
   - 否则加权：`s_main=main_conf*w_main`, `s_det=det_conf*w_detector`，胜方接管 regime/subtype，strategy="weighted"
6. blended_conf/strength = 加权均值
7. 记录 _history，conflict 时写 conflicts.jsonl

## 状态枚举映射（检测器→主引擎）

| 检测器 | 主引擎 |
|---|---|
| TRENDING_UP | TREND_BULLISH |
| TRENDING_DOWN | TREND_BEARISH |
| RANGING | RANGE_BOUND |
| HIGH_VOLATILITY | EXTREME_VOLATILITY |
| LOW_VOLATILITY | RANGE_BOUND |
| BREAKOUT | BREAKOUT |
| REVERSAL | REVERSAL |
| UNKNOWN | （不映射，main_fallback） |

注：主引擎 L34-38 已声明 TRENDING_UP/TRENDING_DOWN/RANGING/HIGH_VOL/LOW_VOL 为别名，映射可复用 `MarketRegime(det_str).value`。

## 注入点

### orchestrator `_perceive` L2881-2897
```python
source = self.regime_arbiter or self.regime_engine
get_regime = getattr(source, "arbitrate", None) or getattr(source, "get_regime", None)
regime = get_regime() if callable(get_regime) else None
```
arbiter=None 时透明回退旧路径。

### stop_loss_manager `_fetch_reversal_inputs` L602
```python
if self._regime_arbiter:
    unified = self._regime_arbiter.arbitrate(symbol)
    hmm_result = unified.get("detector_raw") or {
        "symbol": symbol,
        "regime": unified.get("detector_regime"),
        "probabilities": {"reversal": unified.get("detector_reversal_prob", 0.0)},
        "early_warnings": unified.get("early_warnings", []),
    }
else:
    hmm_result = self._market_regime_detector.get_regime(symbol)  # 旧路径兼容
```

### scheduler L1009 后
```python
self.regime_arbiter = RegimeArbiter(
    main_engine=self.market_regime_engine,
    detector=self.market_regime_detector,
    config=config.get("regime_arbiter", {}),
)
self.stop_loss_manager.set_regime_arbiter(self.regime_arbiter)
```
L486 orchestrator 构造增 `regime_arbiter=self.regime_arbiter`。

## 冲突告警与持久化

- arbiter_conflict=True 时 `logger.warning(f"[RegimeArbiter] conflict symbol={s} main={m} detector={d} strategy={st} resolved={r}")`
- 追加 `data/regime_arbiter/conflicts.jsonl`：{ts, symbol, main_regime, det_regime, resolved, strategy, main_conf, det_conf, blended_conf}
- `_history` deque(maxlen=200) 供 dashboard 调用 `get_arbiter_history(limit)`/`get_conflict_stats()`

## config.yaml 新增段（可选）

```yaml
regime_arbiter:
  enabled: true
  reversal_threshold: 0.45
  w_main: 0.6
  w_detector: 0.4
  conflict_log: data/regime_arbiter/conflicts.jsonl
  history_len: 200
```

## 测试清单（tests/unit/test_regime_arbiter.py）

1. test_consensus_when_regimes_agree — 一致时 confidence+0.1, strategy=consensus
2. test_weighted_disagree_general — 加权胜方接管
3. test_detector_reversal_override — REVERSAL 概率超阈值强制 REVERSAL
4. test_reversal_below_threshold_no_override — 走加权
5. test_fail_closed_when_detector_missing — 检测器异常回退 main_fallback
6. test_unknown_detector_regime_fallback — UNKNOWN 不映射
7. test_enum_mapping_complete — 7 个检测器 regime 全归一
8. test_stop_loss_uses_arbiter_when_set — mock arbiter 验证 _fetch_reversal_inputs
9. test_stop_loss_backward_compat_no_arbiter — arbiter=None 走旧路径
10. test_orchestrator_perceive_uses_arbiter — perception["market_regime"] 来自 arbiter
11. test_orchestrator_perceive_fallback_no_arbiter — arbiter=None 回退
12. test_conflict_logged_and_history_persisted — 验证日志+history

## 验证步骤

1. `py -m pytest tests/unit/test_regime_arbiter.py -v` — 12 新测试通过
2. `py -m pytest tests/unit/test_pnl_projection.py -q` — 50 旧测试无回归（arbiter=None 回退）
3. `py -m pytest tests/unit/test_quant_agi_orchestrator.py -q` — 无回归
4. 备份 `data/agi_orchestrator_state.json` 后跑全量回归
5. 重启系统，观察 `logs/trading_*.log` 中 `[RegimeArbiter]` 日志和 `data/regime_arbiter/conflicts.jsonl`
6. 确认 `data/kill_switch_state.json` 维持 enabled: false

## 约束

- 不改主引擎/检测器内部实现，仅消费输出
- 不改主引擎状态枚举（BREAKOUT/REVERSAL 既有）
- stop_loss_manager `set_market_regime_detector` 接口保留，新增并行 setter
- 仲裁失败 fail-closed 回退主引擎（strategy=main_fallback）
- arbiter=None 时所有下游透明回退旧路径，向后兼容
- 中文注释
