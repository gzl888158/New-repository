# ghost_close 幽灵平仓专项改造方案

> 状态：草案（上线后执行）
> 关联报告：`docs/regression_test_report_2026-08-18.md` 第 6 节「遗留问题与风险」

---

## 一、根因

[scalping_strategy.py `_generate_signal`](../../strategies/scalping_strategy.py) 在 `publish_signal` **之后立即**写本地仓位：

```python
self._positions[symbol] = {"status": "open", ...}
```

而订单是限价单（`GTT` + 10s TTL），可能未成交或超时作废。**本地状态先于交易所成交回执更新**，产生「本地 open / 交易所无仓」的幽灵仓。后续靠 P22 对账事后清理，属于「补救」而非「预防」。

## 二、改造目标与原则

| 目标 | 说明 |
|------|------|
| 消除时序缺陷 | 仓位状态由「信号发布即写入」改为「成交回执驱动写入」 |
| 复用现有设施 | 走 `OrderLifecycleManager.on_fill` 钩子 + `register_position_cleanup_callback` 既有通道，不新造消息链路 |
| 保留防御层 | P22 对账 / `_verify_exchange_position` / P10 平仓前验证全部保留，作为兜底 |
| 可回滚 | 分阶段落地，每阶段可独立开关回退 |

## 三、核心设计：pending intent → fill 确认

引入「待确认开仓意图」中间态，替代乐观写入：

```
信号发布 → 记录 pending intent（不写 open）
                │
        ┌───────┴────────┐
   成交回执(on_fill)   超时/未成交
        │               │
   写入 _positions     清除 intent，允许重新发信号
   (status=open)       （配合 P22 兜底）
```

**状态机**：`pending` → `open`（成交） 或 `pending` → 丢弃（超时/撤单）。

## 四、分阶段实施（上线后执行，每阶段独立验证）

### Phase 1 — 打通成交回执桥（最小侵入）
- 在 `OrderExecutor` 新增 `register_fill_callback(callback)`（或直接复用 `OrderLifecycleManager.register_event_hook("on_fill", ...)`），在 FILLED 分支触发回调，携带 `{symbol, direction, filled_qty, avg_price, clOrdId, strategy_name}`。
- 在 `scheduler` 中把 scalping 策略注册为回调接收者（对齐 `register_position_cleanup_callback` 的注入点）。
- 本阶段只「收」不「改」，先验证回调数据完整性，不动 `_positions` 写入逻辑。

### Phase 2 — 引入 pending intent，延迟写入 open
- `_generate_signal` 改为写入 `self._pending_entries[symbol]`（含 `direction/quantity/entry_price/tp/sl/leverage/clOrdId/timestamp`），**不再直接写 `_positions` 为 open**。
- `on_fill` 回调匹配 `clOrdId`（或 `symbol+direction+quantity` 兜底匹配），命中则用**实际成交价/成交量**物化 `_positions[symbol] = {status:"open", ...}`。
- 更新同 symbol 阻塞判断：`pending` 状态同样阻塞同 symbol 新开仓，避免重复下单。

### Phase 3 — 未成交/超时清理
- 在现有 P22 `verify_exchange_positions` 中新增对 `pending` 意图的处理：
  - 若交易所已有对应仓位 → 物化为 open（补偿丢失的 fill 回调，如 WS 断连）。
  - 若超过 TTL（`ttl_seconds + 缓冲`，约 15s）仍无仓 → 丢弃 intent，允许重新发信号。
- 撤单/订单失败路径同步清理 pending intent。

### Phase 4 — 横向推广 + 观测
- 将同一机制推广到 grid / trend 策略（它们有各自的 `_positions` 写入点与 `cleanup_position`）。
- 增加指标：`pending_entry_count`、`fill_callback_hits`、`pending_timeout_cleaned`、`ghost_close` 归零趋势。
- 用 `OrderStateSynchronizer` 的 `register_order_callback` 补强（当前似乎未在 scheduler 实例化，作为 Phase 4 可选项评估是否接入）。

## 五、涉及文件

| 文件 | 改动 |
|------|------|
| `execution/order_executor.py` | 新增 `register_fill_callback`；FILLED 分支触发回调 |
| `strategies/scalping_strategy.py` | `_generate_signal` 改 pending；新增 `_on_fill` 处理；P22 增 pending 清理 |
| `core/scheduler.py` | 注入 fill 回调桥接 |
| （Phase 4）grid/trend 策略 | 复用同一 pending/fill 机制 |

## 六、测试验证

1. **单元**：pending→fill→open 状态机；fill 丢失→P22 补偿；TTL 超时→intent 丢弃；重复下单阻塞。
2. **集成**：mock `on_fill` 回调，验证从信号到仓位物化全链路；验证 `clOrdId` 匹配与 `symbol+direction` 兜底。
3. **回归**：跑 `pytest tests/unit tests/integration -q -m "not slow"`，确保 686 用例不回归。
4. **实盘观察**：上线后监控 `ghost_close` 计数应趋势归零，`fill_callback_hits` 与 `pending_timeout_cleaned` 比例健康。

## 七、回滚策略

- 每阶段用配置开关（如 `scalping.use_fill_driven_position: true/false`）控制；关闭即回退到现有乐观写入 + P22 对账的旧行为。
- Phase 1 无行为变更，可随时安全回退；Phase 2/3 回退后系统回到「可上线但有幽灵仓」的当前状态，无新增风险。

## 八、风险与注意事项

- **回调丢失**：WS 断连可能导致 `on_fill` 丢失 → 由 Phase 3 的 P22 补偿 + TTL 兜底覆盖，不依赖回调 100% 可达。
- **匹配歧义**：同 symbol 同方向快速连续下单需用 `clOrdId` 精确匹配，`symbol+direction+quantity` 仅作兜底。
- **成交量差异**：部分成交场景需以 `filled_qty` 物化，而非信号里的 `quantity`。
- **别过度耦合**：回调只传数据，不反向调用策略风控，避免循环依赖。

## 九、Phase 4 完成记录（2026-08-18）

### p4a scalping 观测指标（已完成）
`get_stats` 暴露 `pending_entry_count` / `fill_callback_hits` / `pending_timeout_cleaned` / `ghost_close_count`。

### p4b grid 推广（已完成）
- `__init__` 新增 `_use_fill_driven_position`（默认 `grid.use_fill_driven_position`）与三项观测指标。
- 新增 `_generate_clordid`（前缀 `grd`），`_trigger_grid_order` 透传 `clOrdId` 并写回对应网格层 `grid["clOrdId"]`。
- `_process_tick` 两处下单调用透传 `grid_index`。
- 新增 `on_order_filled`：按 `clOrdId` 精确匹配 pending 层，立即置 `filled=True` 并清理 `_grid_pending_at` / 重试 / 冷却。
- 保留既有 `_reconcile_pending_grids` 作为兜底（交易所持仓确认 + 180s 超时回滚，等价于 Phase 3 补偿）。
- `_reconcile_pending_grids` 超时回滚处递增 `_pending_timeout_cleaned`；`cleanup_position` 处递增 `_ghost_close_count`。

### p4c trend 推广（已完成）
- 代码在前期已就绪；本次补全：`config.yaml` trend 段新增 `use_fill_driven_position: true` + `pending_ttl_seconds: 15`。
- `_generate_trend_signal` 新增「存在 pending 时阻塞同 symbol 新开仓」检查，避免重复下单覆盖 pending intent。
- `_monitor_loop` 对账周期改为「存在 pending 时 30s，否则 300s」，并允许存在 pending 时触发 `_sync_positions_with_exchange`（原仅超限时触发）。
- `get_stats` 新增 Phase 4 观测指标。

### p4d OrderStateSynchronizer 接入评估（结论：暂不接入）
- 现状：`OrderStateSynchronizer` 未在 `scheduler` 实例化，`register_order_callback` 未被使用。
- 接口差异：其回调传递 `OrderInfo` dataclass（`client_order_id` / `strategy`=tag / `state`），与策略 `on_order_filled(fill_payload: dict)` 契约不一致；且 `_notify_order_callbacks` 对**所有**订单状态更新触发，非仅 FILLED。
- 结论：接入需新增 `OrderInfo → fill_payload` 适配层并过滤 `state == FILLED`，同时实例化 + 启动同步器并接线 WS 推送。当前 `OrderExecutor._notify_fill_callbacks` 已在 FILLED 分支提供与策略契约一致的 payload，已覆盖 ghost_close 专项需求；`OrderStateSynchronizer` 作为后续可选补强（更强 WS 实时性 + 全量对账），非本专项必要条件，故暂不接入。

