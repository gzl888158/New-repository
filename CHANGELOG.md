# Changelog

本文档记录 OKX 量化交易系统的所有重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布] - 2026-09-08

### 新增

- **RealTimeRiskMonitor 接入生产主流程**：R1/R2（保证金率/回撤查询失败 fail-closed）此前仅通过单测，现完成生产装配。
  - [scheduler.py](core/scheduler.py)：实例化 `RealTimeRiskMonitor`，注入依赖（含 `equity_monitor`），`start()` 中 `await risk_monitor.start()`，`shutdown()` 中停止。
  - [strategy_engine.py](core/strategy_engine.py)：修复 `_check_risk_gate` 键名错配，逐维度 0.85 拦截真正生效。

### 修复

- **adaptive_controller NoneType 错误**：[adaptive_controller.py](risk/adaptive_controller.py) 中 `trade_records` 存在 `status='closed'` 且 `pnl IS NULL` 的记录，`dict.get("pnl", 0)` 返回 `None` 导致比较/运算抛 `TypeError`。统一改为 `(t.get("pnl") or 0)` / `(t.get("pnl") or 0.0)`，共 4 处。

- **margin 口径统一（覆盖三条链路）**：单币种 USDT 合约账户的 `/account/balance` 顶层 `mgnRatio` 字段为空/缺失，且 `availBal`/`totalMgn`/`upl` 等顶层字段同样不适用，直接读取导致误报或风控失效。
  - [risk_monitor.py](core/risk_monitor.py)：`_get_margin_ratio` 新增 `_to_float`，改为从 `details` 计算占用率，消除 `margin=100% [emergency]` 误报。
  - [position_manager.py](core/position_manager.py)：`_check_account_risk` 改为从 `details` 解析（`eq`/`availEq`/`frozenBal`/`ordFrozen`/`availBal`/`upl`），安全浮点转换，不再读顶层空字段导致 `ValueError` 静默失败或占用率恒为 0。

  统一计算口径：`used_margin = eq - availEq`（回退 `frozenBal`，下限 `ordFrozen`），`占用率 = used_margin / eq`，与 [okx_client._parse_account_info](core/okx_client.py) 完全一致。

- **pnl_reconciler 残差项入账**：[pnl_reconciler.py](risk/pnl_reconciler.py) 的 `_reconcile_account_pnl` 此前将 `unattributed` 作为单一聚合值写入快照，资金费/滑点/点差只预判不入账。现新增 `_fetch_funding_fee()` 从 OKX 资金费账单（`type=7`）拉取实际资金费率净额，将残差拆分为 `funding_fee`（真实入账）+ `slippage_spread`（滑点+点差+记录噪声残差，OKX 无独立账单无法再细分），两者与 `external_flow` 一并持久化到 [sqlite_storage.py](data/sqlite_storage.py) 的 `pnl_reconciliation` 表（新增 `funding_fee`/`slippage_spread` 列及迁移）。资金费拉取 fail-open，失败不影响 `discrepancy` 主口径。

- **TradeRecord.pnl 命名歧义补全注释**：[trade_journal.py](core/trade_journal.py) 的 `trades` 表 `pnl` 列存百分比（对应 dataclass `pnl_pct`），`pnl_usdt` 列存 USDT 净额；而 [sqlite_storage.py](data/sqlite_storage.py) ORM `TradeRecord.pnl` 存 USDT 净额。此前该映射未注释，易与已注释的 dataclass/ORM 命名混淆。已在 `_initialize_tables`、`_save_trade`、`_load_history` 三处补全口径注释，消除「同名不同义」误用风险。

### 测试

- 新增回归测试：
  - [test_p2_defensive_regressions.py](tests/unit/test_p2_defensive_regressions.py)：`mgnRatio` 空回退、高占用率触发 EMERGENCY、`availEq` 主口径（3 例）。
  - [test_execution_core.py](tests/unit/test_execution_core.py)：`test_17b_account_risk_empty_mgnratio`（1 例）。
- 全量单测：1475 passed, 18 warnings（0 失败）。
- 线上验证：重启后无 `margin=100% [emergency]` 误报，无 `Account risk check error`，账户级保证金风控恢复正常计算。

### 遗留待办

- `pnl_reconciler` 残差项（资金费率/滑点/点差）仍只预判不入账。
- `TradeRecord.pnl_pct` 与 ORM `TradeRecord.pnl`（USDT）字段名仍易混淆，建议后续统一命名或加注释。
