# 回归测试报告

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-08-18 |
| 测试范围 | `tests/unit` + `tests/integration`（排除 `slow` 标记） |
| 测试结果 | **686 passed / 0 failed** |
| 结论 | **无回归，全部通过** |

---

## 1. 背景与目标

本次变更围绕「企业级数据质量完善」展开，目标是对历史交易数据中暴露的缺陷进行根因修复，并确保修复不引入回归。变更包含两条主线：

1. **数据落库质量修复**（`trade_records` / `trades` 口径一致性）
2. **修复 5 个测试断言过时的失败用例**（其中 1 个为代码缺陷）

---

## 2. 变更清单

### 2.1 数据落库质量修复

| 文件 | 变更内容 |
|------|----------|
| [sqlite_storage.py](../../data/sqlite_storage.py) | 新增 `_normalize_trade_payload()`：统一 `pnl_percent = pnl/margin*100`（保证金收益率口径），`signal_type` 空值兜底 `"unknown"`；`_trade_to_dict()` 补 `signal_type`/`exit_reason` 字段；`save_trade_record`/`save_trade_records_batch`/`update_trade_record` 三处统一应用 |
| [pnl_reconciler.py](../../risk/pnl_reconciler.py) | `pnl_percent` 从「仓位价值」口径修正为「保证金」口径 |
| [trade_journal.py](../../core/trade_journal.py) | `_try_close_from_trade_records` 移除 `exit_reason == "manual"` 硬过滤（方向相反即视为平仓）；`entry_signal_type` 优先取 `signal_type` |
| [backfill_trade_quality.py](../../scripts/backfill_trade_quality.py) | 新增历史数据回填脚本（一次性修复现有脏数据） |

### 2.2 测试用例修复（5 个失败用例）

| 文件 | 变更内容 | 对应失败用例 |
|------|----------|-------------|
| [signal_processor.py](../../services/signal_processor.py) | `__init__` 补 `self._regime_gate = None`（代码缺陷：属性未初始化） | 3 个 integration 用例 |
| [test_execution_position_v2.py](../../tests/unit/test_execution_position_v2.py) | 幂等键前缀断言 `"okxqt_"` → `self.executor._idempotency_prefix` | 1 个 unit 用例 |
| [test_execution_core.py](../../tests/unit/test_execution_core.py) | 幂等键断言改为确定性验证（同时间戳+同参数相同、不同时间戳不同） | 1 个 unit 用例 |

---

## 3. 历史数据回填结果

回填前已备份数据库至 `data/trading.db.bak_20260818`。回填明细：

| 回填项 | 影响行数 | 说明 |
|--------|---------|------|
| `pnl_percent` 口径重算 | **1436 行** | 统一为保证金收益率 `pnl/margin*100` |
| `signal_type` 空值兜底 | **1955 行** | 空值 → `"unknown"`（占总数 62%，证实落库缺陷严重） |
| `exit_reason` 空值兜底 | 0 行 | 无空值，无需回填 |

> 典型案例：AVAX-USDT-SWAP scalping 单笔巨亏记录 `pnl_percent` 由 `-7.02` 修正为 `-12.45`（保证金收益率口径）。

---

## 4. 回归测试方法

- **测试框架**：pytest + pytest-asyncio（`asyncio_mode = auto`）
- **命令**：`.venv\Scripts\python.exe -m pytest tests/unit tests/integration -q -m "not slow"`
- **测试文件数**：unit 21 个 + integration 3 个
- **排除项**：`tests/perf`（性能压力测试，与本次数据质量修复无关，未纳入）

---

## 5. 测试结果

### 5.1 总体结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| unit | 636 passed / 2 failed | **638 passed / 0 failed** |
| integration | 45 passed / 3 failed | **48 passed / 0 failed** |
| **合计** | 681 passed / 5 failed | **686 passed / 0 failed** |

### 5.2 修复的失败用例明细

| 用例 | 失败原因 | 修复方式 |
|------|----------|----------|
| `test_execution_core.py::test_01_generate_idempotency_key` | 断言 `key1 != key2`，但确定性幂等键同毫秒生成相同 key | 测试改为验证确定性语义 |
| `test_execution_position_v2.py::test_01_idempotency_key_generation` | 断言 `startswith("okxqt_")`，但 clOrdId 不允许下划线 | 断言改用 `_idempotency_prefix` |
| `test_trading_flow.py::test_full_entry_signal_flow` | `SignalProcessor` 无 `_regime_gate` 属性 | 代码补初始化 `= None` |
| `test_trading_flow.py::test_signal_cooldown_blocks_duplicate` | 同上 | 同上 |
| `test_trading_flow.py::test_risk_pause_blocks_all_signals` | 同上 | 同上 |

### 5.3 本次变更相关测试全部通过

- 数据库相关：`test_trade_record_crud`、`test_database_integration_flow`、`test_capital_integration`
- 存储层规范化逻辑：另以临时脚本单独验证 5 项（closed 重算 / open 保持 / signal_type 兜底 / update 重算 / `_trade_to_dict` 字段完整），全部通过

---

## 6. 遗留问题与风险

以下问题**未在本次修复范围内**，建议作为后续独立专项处理：

| 遗留项 | 说明 | 风险等级 |
|--------|------|---------|
| `ghost_close` 幽灵平仓根因 | `_generate_signal` 在 `publish_signal` 后立即写本地 `_positions` 但未等成交确认，属「信号→成交回执」时序架构问题 | 中 |
| `fees` 历史数据为 0 | 历史记录未走预估手续费路径，回填需谨慎（避免与净额盈亏重复扣费） | 低 |
| 20 个无害 warning | deprecation（`utcnow`）、未 await 协程、测试函数 return 非 None | 低 |

---

## 7. 结论

本次「企业级数据质量完善」共修改 **6 个文件**、新增 **1 个回填脚本**，完成：

1. 统一 `pnl_percent` 计算口径（保证金收益率，单一事实来源）
2. 补齐 `signal_type`/`exit_reason` 落库与空值兜底
3. 修复 TradeJournal 平仓记录不完整（`manual` 平仓漏记）
4. 回填历史脏数据 3391 行（1436 + 1955）
5. 修复 5 个测试失败用例（含 1 个代码缺陷）

回归测试 **686 passed / 0 failed**，确认无回归。遗留的 `ghost_close` 时序问题建议独立专项处理。
