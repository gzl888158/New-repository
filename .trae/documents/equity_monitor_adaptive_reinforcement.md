# EquityMonitor 自适应强化：识别外部资金变动，避免误判回撤冻结

## Context（背景）

用户反馈：账户权益从历史峰值 235.21 USDT 降至 173.29 USDT，**原因是资金被人工划转/提取到他处（外部资金变动），而非交易亏损**。但 `EquityMonitor` 把这次权益下降误判为「紧急回撤」，进入 `EMERGENCY` 模式并冻结所有新开仓（`position_multiplier=0`、`max_positions=0`），且因横盘无法触发 +5% 自动恢复，被永久卡死。

目标：强化 `EquityMonitor`，使其能识别「充值/提现/划转」等外部资金变动，并据此重建回撤基准，不再把"提现"误判为"交易回撤"，达到口令中「企业级、根据实际自适应」的要求，杜绝此类问题再次发生。

## 根因

`core/equity_monitor.py`：

1. `_detect_deposit_withdrawal()`（L288-338）**已经能**检测到提现/充值（`equity_change - upl_change` 超阈值且不由 PnL 解释）。但检测到 `WITHDRAWAL` 后，`feed()`（L227-249）只调用 `_update_peaks()` 更新峰值/回撤，却**没有下调 `peak_equity` 基准**。
2. 随后 `feed()` L235-239 在检测到提现后仍调用 `_detect_emergency()`，此时 `drawdown = (235-173)/235 = 26% > 10%` 阈值 → 误触发 `EMERGENCY`（`mode_start_time=2026-09-10 17:09` 正是提现后）。
3. `EMERGENCY` 退出需从 `emergency_low_watermark`（=173.29）反弹 +5% 到 181.95，行情横盘时无法满足 → 永久冻结。
4. 兜底逻辑 `_load_state()` L847-858 的「不同账户状态检测」阈值是 `peak > last*3`（3 倍），235→173（-26%）远不足以触发，无法在重启时自救。

## 方案（推荐）

核心思路：**外部资金变动（充值/提现/划转）与交易盈亏分治——检测到外部资金变动即重建回撤基准，且不触发 EMERGENCY。**

### 改动 1：新增 `_reset_baseline(equity, reason)` 方法

在 `EquityMonitor` 内新增基准重建方法，检测到外部资金变动后调用：

- 重置 `peak_equity = equity`、`peak_equity_time = now`、`trough_equity = equity`、`max_drawdown_pct = 0.0`、`emergency_low_watermark = 0.0`
- 重置平滑指标 `sma_short/sma_long/ema = equity`（资金规模已变，历史平滑值失效）
- 若当前 `_current_mode == EMERGENCY`，退出为 `NORMAL`（提现非交易回撤，不应冻结）
- 记录 `logger.warning(f"BASELINE RESET ({reason}): equity→{equity}")`

### 改动 2：`feed()` 充值/提现分支重建基准并跳过紧急检测

修改 `feed()` L227-249 的事件处理分支：

- `event.event_type in (DEPOSIT, WITHDRAWAL)` 时，改用 `_reset_baseline(equity, event.event_type.value)` 替代 `_update_peaks(equity, now)`
- **删除** L235-239「提现可能是紧急事件，继续检查紧急模式」的 `_detect_emergency` 调用（外部资金变动不参与回撤判定）
- 其余分支不变

### 改动 3：事件持久化 + 重启自愈

- `_save_state()`（L775-802）：新增持久化 `recent_external_flows`（最近至多 20 条 `DEPOSIT/WITHDRAWAL` 事件，含 `type/timestamp/amount`）与 `baseline_equity`、`baseline_updated_at`
- `_load_state()`（L805-881）：在读取 `current_mode` 后，若 `peak_equity` 相对 `last_known_equity` 回撤 ≥ `emergency_drop_pct` 且 `recent_external_flows` 非空（或 `baseline_updated_at` 说明资金规模已变），则调用 `_reset_baseline(last_known_equity, "restart_reconcile")`，替代现有粗糙的 3 倍阈值；保留 3 倍阈值作为最终兜底

### 改动 4：配置化

- `config.yaml` 新增顶层 `equity_monitor:` 节，暴露现有硬编码默认值：

```yaml
equity_monitor:
  emergency_drop_pct: 0.10      # 触发紧急冻结的回撤阈值
  emergency_recovery_pct: 0.05  # 从低点反弹恢复阈值
  deposit_detect_pct: 0.05      # 充值/提现判定阈值
  save_interval: 60
  ema_alpha: 0.05
  trend_confirm_bars: 5
```

- `EquityMonitor.__init__` 改为从 `config.get("equity_monitor", {})` 读取，回落默认值（保持向后兼容，不传配置时行为不变）

## 关键文件

- `core/equity_monitor.py`（主要改动）
- `config.yaml`（新增 `equity_monitor` 配置节）
- 新增测试 `tests/unit/test_equity_monitor.py`

## 复用与约束

- `EquityMonitor` 已在 `core/scheduler.py` L260 以 `EquityMonitor(config)` 注入，`account_manager.py` L143 `feed()` 每约 30s 喂入 `total_equity`，无需改动调用链
- `EquityEventType.DEPOSIT / WITHDRAWAL` 已存在（L43-44），直接复用
- 不改动 `peak/trough/max_drawdown` 字段语义，仅在其"外部资金变动"场景下重建

## 验证

1. 单元测试（新增 `tests/unit/test_equity_monitor.py`）：
   - 提现检测后 `peak_equity` 被重置为提现后值，`max_drawdown_pct=0`
   - 提现不触发 `EMERGENCY`（`_current_mode` 保持/回到 `NORMAL`）
   - 提现期间恰逢 `EMERGENCY`，重置后退出 `NORMAL`
   - 充值/提现事件持久化后，`_load_state` 重启能重建基准
2. 回归：`.venv\Scripts\python.exe -m pytest tests/unit -q -p no:cacheprovider` 全绿
3. 语法编译：`py_compile core/equity_monitor.py`
4. 灰度验证：`stop.py` → 改 config/代码 → `start.py` → 观察日志出现 `BASELINE RESET` + `EquityMonitor state loaded ... mode=normal`，且不再出现 `EMERGENCY mode: blocking new positions`，正常开单