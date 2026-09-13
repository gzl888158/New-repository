# 变更总结：资金利用率 FORCE_REBALANCE 接入 Dashboard

日期：2026-09-13

## 一、目标

1. 强化资金利用率引擎，新增资本效率硬约束——资金规模足够但效率过低时触发 `FORCE_REBALANCE`（强制再平衡）。
2. 将 `force_rebalance` 状态贯通到 Dashboard 前端，以状态卡 + 告警横幅形式展示。
3. 修复过程中暴露的测试隔离问题，保证全量回归通过。

## 二、触发条件（硬约束）

在 `CapitalUtilizationEngine._determine_action` 中，硬约束优先级高于迟滞防振荡：

- `total_equity > 100.0`（`FORCE_REBALANCE_MIN_EQUITY`）
- `capital_efficiency < 0.15`（`FORCE_REBALANCE_EFFICIENCY_THRESHOLD`）

其中 `capital_efficiency = total_pnl / used_margin`。满足条件即返回 `UtilizationAction.FORCE_REBALANCE`，并采用激进再平衡参数（60% 步长、更强效率修正、更宽分配边界）。

## 三、数据流链路

```
AdaptiveController._check_capital_utilization
  └─ 写入 self._capital_utilization["status"] = "force_rebalance"
  └─ 持久化 → data/capital_utilization_state.json
        └─ DashboardEngine.get_capital_utilization()  读状态文件
        └─ dashboard_api._compute_capital_utilization() 读状态文件
              └─ /api/dashboard/full → dashboard_v2.html
                    └─ renderCapitalUtilization() → 状态卡 + 告警横幅
```

## 四、改动明细

### 1. 资金利用率引擎 — `core/capital_utilization_engine.py`

- 新增枚举 `UtilizationAction.FORCE_REBALANCE = "force_rebalance"`。
- 新增常量 `FORCE_REBALANCE_EFFICIENCY_THRESHOLD = 0.15`、`FORCE_REBALANCE_MIN_EQUITY = 100.0`。
- `_determine_action` 新增 `total_equity` 参数，硬约束优先返回 `FORCE_REBALANCE`。
- `_compute_position_boost` / `_compute_signal_relaxation` 对 `FORCE_REBALANCE` 分别返回 `1.0` / `-0.10`。
- `_compute_allocation_shift` 激进模式：步长 `0.6`、盈利放大 `1.5`、亏损压减 `0.2`、分配边界 `0.15`。

### 2. 自适应控制器 — `risk/adaptive_controller.py`

- `_check_capital_utilization` 在 `FORCE_REBALANCE` 分支写入 `status = "force_rebalance"`，并触发告警 `utilization_force_rebalance`。
- 持久化字段已含 `status` / `recommended_action` / `capital_efficiency`，无需改动 `_persist`。

### 3. Dashboard 引擎 — `core/dashboard_engine.py`

- 新增 `get_capital_utilization()`：读取 `data/capital_utilization_state.json`，返回 `status`、`recommended_action`、`capital_efficiency`、`utilization_tier`、`equity_mode`、`position_boost` 等字段。
- `get_dashboard_full()` 返回字典新增 `capital_utilization` 字段。

### 4. Dashboard 接口 — `dashboard_api.py`

- `_compute_capital_utilization()` 实时回退分支与错误分支补齐 `recommended_action`、`capital_efficiency`、`utilization_tier`、`utilization_trend`、`equity_mode` 字段。

### 5. 前端 — `static/dashboard_v2.html`

- 资金效率面板上方新增 `#capitalUtilizationBanner` 状态卡容器。
- 新增 `.util-banner` 样式及状态配色：`normal`（绿）、`low/warming_up`（蓝）、`high/force_rebalance`（橙）、`emergency_blocked/unknown`（红）。
- 新增 `renderCapitalUtilization(cu)`，按 `status` 渲染图标 + 标题 + 说明，`force_rebalance` 显示橙色 `⚠️` 告警并动态补充效率与阈值，chips 展示利用率、目标、仓位乘数、资本效率、推荐动作、层级。

### 6. 测试修复 — `tests/unit/test_dashboard_v2.py`

- `test_12_edge_cases` 原断言 `snapshot.total_equity == 0.0` 失败：未注入 OKX 客户端时，`_fetch_okx_account` 会回退到 `dashboard_api.fetch_okx_account()` 发起真实请求，返回真实账户权益污染"空数据"断言。
- 修复：用 `patch` 隔离 `dashboard_api.fetch_okx_account` / `fetch_okx_positions` / `_okx_circuit_open`，保证边界条件测试确定性。

## 五、测试与回归

- `tests/unit/test_capital_utilization_engine.py`：新增 `TestForceRebalance` 用例（触发 / 不触发 / 优先级 / 调节参数 / analyze 集成）。
- `tests/unit/test_dashboard_v2.py`：24 项全通过。
- 全量单测：`1558 passed, 0 failed`。
