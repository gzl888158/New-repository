# fill 回执丢失根因修复 — 立项方案

> 状态：立项（待评审，未实施）
> 关联：`docs/ghost_close_remediation_plan.md`（Phase 1-4 已落地）
> 风险等级：高（涉及订单执行热路径）

---

## 一、根因

成交回执（fill）**并非来自 OKX WebSocket 推送**，而是由
`OrderExecutor._track_active_orders` 以 **REST 轮询内存态 `_active_orders`** 获得：

```
_track_active_orders
  └─ for exchange_order_id in self._active_orders           # 仅内存
       └─ okx_client.get_order(symbol, order_id)            # REST 轮询
            └─ state == "filled" 分支
                 └─ _notify_fill_callbacks(payload)          # 通知策略物化仓位
```

一旦订单不在内存 `_active_orders` 中、或轮询未命中 `filled`，回调即丢失，
策略的 `pending intent` 无法物化，最终沉沦为幽灵仓，靠 P22 对账兜底清理。

## 二、丢失点（4 类，按严重度排序）

| # | 丢失场景 | 后果 | 严重度 |
|---|---------|------|--------|
| 1 | **系统重启**：`_active_orders` 内存态清空，已成交未处理的订单回执永久丢失 | pending intent 永不能物化 | 高 |
| 2 | **轮询间隙成交**：订单在两次轮询之间成交并被移出 `_active_orders` | 回执不再被观察 | 中 |
| 3 | **`get_order` REST 失败/返回 None**：本轮跳过，后续不再跟踪则永久丢失 | 回执静默丢失 | 中 |
| 4 | **异常路径提前 return/continue**：订单移除但未触发 `_notify_fill_callbacks` | 回执链路断裂 | 中 |

## 三、现状缓解（已落地）

- Phase 1：成交回执桥 `register_fill_callback` / `_notify_fill_callbacks`（FILLED 分支触发）。
- Phase 2：`pending intent` 延迟写入，取代「信号发布即写 open」的乐观写入。
- Phase 3：P22 对账补偿 + TTL 超时清理 pending intent。
- Phase 4：grid / trend 推广 + 观测指标（`fill_callback_hits` / `pending_timeout_cleaned` / `ghost_close_count`）。

以上均为**补救**机制：依赖定期对账兜底，有延迟，且前提是订单仍被内存跟踪。

## 四、候选方案

### 方案 A — 活跃订单落盘 + REST 全量对账（推荐先做）

- `_active_orders` 持久化到 SQLite（`active_orders` 表），重启时恢复，根治丢失点 #1。
- 接入 `OrderStateSynchronizer`（p4d 已评估可选）做 REST 历史订单全量对账，
  补漏轮询间隙/异响丢失的回执（覆盖 #2/#3/#4）。
- 代价低、不碰 WebSocket 热路径，风险可控。

### 方案 B — 接入 OKX WebSocket 订单通道

- 订阅 `orders` channel，用推送替代轮询，消除轮询间隙丢失（#2）。
- 改的是订单热路径，需处理断线重连的增量对齐，风险高。

### 方案 C — A + B 结合（最彻底）

- 落盘 + WS 推送 + REST 全量对账三层兜底，覆盖全部丢失点。
- 涉及订单执行热路径 + 断线重连全量补偿，须分阶段灰度。

## 五、推荐与实施步骤（方案 A）

1. **落盘**：`_active_orders` 增删时同步写 `active_orders` 表；启动时加载恢复。
2. **全量对账**：新增 `_reconcile_fill_receipts`，拉取近期 FILLED 订单，
   比对 pending intent，命中则触发 `_notify_fill_callbacks` 补回执。
3. **观测**：新增 `fill_receipt_lost` / `fill_receipt_recovered` 指标，验证丢失归零趋势。

## 六、测试验证

1. 单元：重启恢复 `_active_orders`；REST 全量对账补漏回执；落盘增删一致性。
2. 集成：mock `get_order` 失败/轮询间隙场景，验证回执不丢。
3. 回归：`pytest tests/unit`（当前 1457 用例）零回归。
4. 实盘观察：`fill_receipt_recovered > 0` 且 `ghost_close` 趋势归零。

## 七、回滚策略

- 方案 A 用配置开关（如 `execution.persist_active_orders: true/false`）控制；
  关闭即回退到纯内存态 + P22 兜底的现状。
- 落盘为纯增量无副作用；全量对账仅补回执、不改仓，仅当策略有 pending intent 时触发物化。

## 八、风险与注意事项

- **勿在热路径加同步磁盘 IO**：落盘走异步/批量，避免拖慢下单。
- **全量对账限频**：拉取历史订单需遵守 OKX 限频，避免 `429`。
- **幂等**：`_notify_fill_callbacks` 需幂等，防止对账补漏与实时回执双触发。