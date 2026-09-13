# 开仓全链路企业级优化 — 拒单/丢弃可溯源

## Context（背景与目标）

系统长期「不开仓」，智能体拒单率 72.7%，但每一笔「为什么不开仓」目前只能靠 `grep` 日志追溯，无法结构化聚合、无法重放审计。排查发现开仓链路存在多处**可观测性缺口**：

1. `signal_processor._process_signal` 有 15+ 处信号丢弃，全部走 `_push_dead_letter` → **仅存内存 deque**，无 traceID、不落盘、重启即丢。
2. `order_executor._execute_order_impl` 的智能体/成本/审计/风控再校验拒单只 `logger.warning`，**未发布事件**。
3. traceID 在 `RiskAdjudicator` 回写后，到 `order_executor` 多层审核之间**未复用发布**，链路在「拒绝」环节断链。
4. 各层拒单 `reason` 字符串不统一，无 `reason_code` 分类，无法按原因聚合。

**目标**：在不改变任何判定行为、任何 return 路径、任何阈值的前提下（纯增量），让开仓链路上每一个「拒绝/丢弃」都带 traceID、发布结构化事件到 EventStore，实现可重放、可聚合、可审计，从而彻底终结「靠 grep 日志排查弃仓原因」。

## 改动方案

### 1. EventType 新增（`core/unified_layer.py` L417-430）
在 `EventType` 枚举新增两个类型：
- `SIGNAL_REJECTED = "signal_rejected"` — 信号层（signal_processor）丢弃
- `ORDER_REJECTED = "order_rejected"` — 执行层（order_executor）拒绝

### 2. signal_processor 注入 event_bus + 统一拒单发布（`services/signal_processor.py`）
- `__init__`（L15）新增 `self._event_bus = None`。
- 新增 `set_event_bus(event_bus)` 方法。
- 改造 `_push_dead_letter`（L287）：新增可选参数 `layer=None, reason_code=None`，内部在保留原内存 dead letter 逻辑的同时：
  1. 确保 trace_id（复用 `signal_dict.get("trace_id")`，缺失则用 `EventIDGenerator` 生成并回写）。
  2. 发布 `SIGNAL_REJECTED` 事件（`publish_sync`，非阻塞，try/except 兜底，持久化失败不影响主流程）。
- 新增 `_classify_reject_reason(reject_reason) -> (layer, reason_code)` 归一化 helper，从前缀映射（代表项）：
  `signal_quality_rejected`→`signal_quality`、`strategy_priority_blocked`→`strategy_priority`、`signal_conflict`/`conflict_resolution_failed`→`signal_conflict`、`collaborative_trigger_blocked`→`coordinator`、`capital_focus_*`→`capital_allocation`、`decision_validation_invalid`→`decision_validation`、`meta_*`→`meta_decision`、`cba_rejected`→`cost_benefit`、`below_threshold`→`threshold`、`risk_gate_exception_*`/`trading_paused`/`strategy_risk_validation_failed`→`risk_gate`。
  - 现有 15+ 调用点**不改签名即可工作**（自动推断）；`reason_code` 默认用 `reject_reason` 全量兜底。
- `_handle_risk_block`（L828）**不改**：风险层拒单已由 `RiskAdjudicator` 发布 `RISK_ADJUDICATED`（带 trace_id + reason）覆盖，避免重复。

### 3. scheduler 注入（`core/scheduler.py`）
在 L294（`self.order_executor.set_event_bus(...)`）附近新增：
`self.signal_processor.set_event_bus(self.unified_layer.event_bus)`。

### 4. order_executor 拒单事件发布（`execution/order_executor.py`）
复用已有 `_publish_event(event_type, data)` helper（L336/L343），在 `_execute_order_impl` 四处拒绝点发布 `ORDER_REJECTED`：
- RiskGate 再校验失败（L1561-1567，`layer=risk_gate`）
- TradeCostAnalyzer BLOCKED（L1579-1583，`layer=trade_cost`）
- IntelligentAgent REJECTED（L1596-1601，`layer=intelligent_agent`）
- TradeAuditor BLOCKED（L1643-1648，`layer=trade_auditor`）

每个 payload 复用已有 `order_data.get("trace_id")`（L1211，上游同源），含 `symbol/strategy/signal_type/direction/layer/reason_code/reason/is_close`。

### 5. 统一拒单事件 data 结构
```
{ trace_id, symbol, strategy, signal_type, direction,
  layer, reason_code, reason, is_close }
```

### 6. Dashboard 聚合端点（可选，`dashboard_api.py`）
新增 `/api/rejections/stats`：从 EventStore 按 `layer/reason_code/symbol/strategy` 聚合拒单（复用 `EventStore.replay` / `stats()`），替代「grep 日志」。

## 复用点（不重复造轮子）
- `core/unified_layer.py`：`EventType`、`Event`、`LocalEventBus.publish_sync`（L524，已落盘 EventStore）。
- `execution/order_executor.py`：`_publish_event` helper（L343）、`set_event_bus`（L336）、trace_id 复用（L1210-1213）。
- `core/risk_adjudicator.py`：已有 trace_id 生成/回写 + `RISK_ADJUDICATED` 发布范式（L85-167），作为事件发布参照。
- `core/event_id.py`：`EventIDGenerator.get_instance().generate()`。

## 验证
1. 新增单测 `tests/unit/test_open_chain_rejection_trace.py`：
   - signal_processor 各门控丢弃时发布 `SIGNAL_REJECTED`（带 trace_id）。
   - order_executor agent/cost/auditor 拒单发布 `ORDER_REJECTED`（带 trace_id）。
   - trace_id 贯穿：信号 → 拒单事件同源。
   - EventStore 可重放：从写入的拒单事件 replay 还原。
2. 语法编译 + 全量 pytest（当前基线 1474 例通过）。
3. 重启后，用 `/api/events/stats` 或日志确认拒单事件成功落盘。

## 风险
- **纯增量**：不改变任何判定行为、任何 return 路径、任何阈值。
- `publish_sync` 无订阅者，不触发副作用；持久化失败有 try/except，不阻断主流程。
- 新增事件仅追加到 EventStore，磁盘占用随交易量线性增长（已有按日滚动 + 有界去重）。

## 阶段 2（可选，需逐项确认后实施）
已确认允许调整阈值/逻辑，但需逐项确认（不包含在本计划主改动内）：
- `max_rejection_rate`（当前 0.70）与 `min_signal_quality` 硬下限。
- 网格策略在 `strong_trend_down` 下的拦截策略（当前 12 次拦截的主因）。
- 信号防饥饿 `max_consecutive_blocks`（当前 5）。