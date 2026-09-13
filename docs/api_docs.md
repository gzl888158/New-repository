# Dashboard API 文档

## 概述

Dashboard API 是基于 Flask 的 HTTP REST 接口，提供交易系统的实时监控和管理功能。

- **基础地址**: `http://localhost:8080`
- **数据格式**: JSON
- **认证**: 无（仅限本地访问，如需远程访问请自行添加认证）

---

## 1. 系统状态

### GET /api/system_status
获取系统运行状态。

**响应示例**:
```json
{
  "status": "running",
  "trading_process": true,
  "dashboard_process": true,
  "ws_connected": true,
  "last_update": "2026-07-19T00:00:00",
  "redis": {
    "available": true,
    "configured": true,
    "degraded_to_memory": false,
    "memory_cache_size": 0
  },
  "api_latency_ms": 320.5,
  "api_latency_avg_ms": 285.3,
  "cpu": {
    "system": 12.5,
    "process": 5.2,
    "cores": 8
  },
  "memory": {
    "system": 65.0,
    "process": 3.5,
    "available": 12.3
  },
  "uptime": 86400
}
```

---

## 2. 账户概览

### GET /api/account_overview
获取账户总览数据。

**响应字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| account.total_equity | float | 总权益 (USDT) |
| account.available_balance | float | 可用余额 (USDT) |
| account.used_margin | float | 已用保证金 (USDT) |
| account.unrealized_pnl | float | 未实现盈亏 (USDT) |
| account.margin_rate | float | 保证金率 |
| positions | array | 当前持仓列表 |
| trading.total_trades | int | 总交易次数 |
| trading.win_rate | float | 胜率 |
| trading.total_pnl | float | 总盈亏 |
| trading.daily_pnl | float | 当日盈亏 |
| equity_curve | array | 权益曲线数据点 |

---

## 3. 订单

### GET /api/orders
获取当前挂单列表。

**响应示例**:
```json
{
  "orders": [
    {
      "id": "ordId",
      "symbol": "BTC-USDT-SWAP",
      "side": "buy",
      "type": "limit",
      "price": 50000.0,
      "quantity": 0.1,
      "status": "live",
      "strategy_name": "grid",
      "create_time": "2026-07-19T00:00:00"
    }
  ],
  "total": 1
}
```

---

## 4. 交易信号

### GET /api/signals
获取最近交易信号记录。

**查询参数**:
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| limit | int | 50 | 返回条数 |
| strategy | string | - | 按策略过滤 |

**响应示例**:
```json
{
  "signals": [
    {
      "symbol": "BTC-USDT-SWAP",
      "strategy_name": "trend",
      "direction": "long",
      "confidence": 0.75,
      "signal_type": "entry",
      "price": 50000.0,
      "create_time": "2026-07-19T00:00:00"
    }
  ],
  "total": 50
}
```

---

## 5. 策略表现

### GET /api/performance
获取各策略表现统计。

**查询参数**:
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| days | int | 30 | 统计天数 |

**响应字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| strategies | array | 各策略统计 |
| strategies[].strategy_name | string | 策略名 |
| strategies[].total_trades | int | 总交易数 |
| strategies[].winning_trades | int | 盈利交易数 |
| strategies[].win_rate | float | 胜率 |
| strategies[].total_pnl | float | 总盈亏 |
| strategies[].profit_factor | float | 盈亏比 |
| strategies[].max_drawdown | float | 最大回撤 |
| strategies[].sharpe_ratio | float | 夏普比率 |
| overall.total_pnl | float | 总盈亏 |
| overall.win_rate | float | 总胜率 |

---

## 6. 风控监控

### GET /api/risk_status
获取风控状态（优先读取实时 JSON，回退到数据库推断）。

**响应字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| source | string | 数据来源: realtime / database |
| current_equity | float | 当前权益 |
| peak_equity | float | 历史峰值权益 |
| drawdown | float | 当前回撤比例 |
| is_paused | bool | 是否暂停交易 |
| pause_reason | string | 暂停原因 |
| tier_triggered | object | 阶梯风控触发状态 |
| daily_pnl | float | 当日盈亏 |
| hourly_pnl | float | 小时盈亏 |
| consecutive_losses | int | 连续亏损次数 |

---

## 7. 因子监控

### GET /api/adaptive_factors
获取自适应资金因子和策略分配。

**响应字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| compound_factor | float | 复利因子 |
| kelly_factor | float | 凯利仓位因子 |
| drawdown_protection_factor | float | 回撤保护因子 |
| total_position_factor | float | 综合仓位因子 |
| strategy_allocations | object | 各策略资金分配比例 |
| strategy_performance | object | 各策略表现评分 |
| initial_capital | float | 初始资金基准 |
| current_equity | float | 当前权益 |

---

## 8. 权益曲线

### GET /api/equity_curve
获取权益历史曲线。

**查询参数**:
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| days | int | 30 | 天数 |

**响应示例**:
```json
{
  "curve": [
    {
      "timestamp": "2026-07-19T00:00:00",
      "equity": 100.0,
      "peak": 100.0,
      "drawdown": 0.0
    }
  ],
  "total_return": 0.15,
  "max_drawdown": 0.08,
  "sharpe_ratio": 1.5
}
```

---

## 9. 配置版本管理

### GET /api/config_versions
列出所有可回滚的配置版本。

**响应示例**:
```json
{
  "versions": [
    {
      "filename": "config_20260719_120000.yaml",
      "timestamp": "2026-07-19T12:00:00",
      "size_bytes": 1024,
      "is_pre_rollback": false
    }
  ],
  "total": 1
}
```

### POST /api/config_rollback
回滚到指定配置版本。

**请求体**:
```json
{
  "version_filename": "config_20260719_120000.yaml"
}
```

> 注意：`version_filename` 为空时回滚到上一个版本。回滚仅影响 config.yaml 文件，运行中的策略实例需要重启才生效。

**响应示例**:
```json
{
  "success": true,
  "rolled_back_to": "config_20260719_120000.yaml",
  "pre_rollback_backup": "config_pre_rollback_20260719_123000.yaml",
  "note": "Running strategy instances need restart to apply rolled-back config"
}
```

---

## 10. 成交质量（滑点）

### GET /api/fill_quality
获取订单成交质量统计（滑点分析）。

**查询参数**:
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| hours | int | 24 | 统计小时数 |

**响应字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| hours | int | 统计时长 |
| overall.total_fills | int | 总成交笔数 |
| overall.avg_slippage | float | 平均滑点（绝对值） |
| overall.max_slippage | float | 最大滑点 |
| overall.p95_slippage | float | 95分位滑点 |
| overall.avg_unfavorable_slippage | float | 平均不利滑点 |
| overall.tolerance_exceeded_count | int | 超过滑点容忍度的笔数 |
| overall.tolerance_exceeded_rate | float | 超容忍率 |
| by_strategy | array | 按策略聚合 |
| by_symbol | array | 按币种聚合（Top 10） |

---

## 11. 信号质量

### GET /api/signal_quality
策略信号质量评估。

**查询参数**:
| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| hours | int | 168 | 统计小时数（默认7天） |

**响应字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| strategies | array | 各策略信号统计 |
| strategies[].strategy_name | string | 策略名 |
| strategies[].total_signals | int | 总信号数 |
| strategies[].filled_count | int | 成交数 |
| strategies[].closed_count | int | 已平仓数 |
| strategies[].win_rate | float | 胜率 |
| strategies[].signal_fill_rate | float | 信号成交率 |
| strategies[].avg_pnl | float | 平均盈亏 |
| strategies[].total_pnl | float | 总盈亏 |
| by_symbol_strategy | array | 按策略×币种维度 |
| hourly_distribution | array | 按小时分布 |

---

## 12. A/B 测试

### GET /api/ab_tests
列出所有历史 A/B 测试报告。

**响应示例**:
```json
{
  "reports": [
    {
      "filename": "ab_test_BTC_USDT_SWAP_20260719_120000.json",
      "timestamp": "2026-07-19T12:00:00",
      "symbol": "BTC-USDT-SWAP",
      "days": 30,
      "winner": "variant_A",
      "variants_count": 3,
      "summary": "Tested 3 variants. Winner: variant_A..."
    }
  ],
  "total": 1
}
```

### GET /api/ab_tests/<filename>
获取指定 A/B 测试报告详情。

**路径参数**:
| 参数 | 类型 | 说明 |
|------|------|------|
| filename | string | 报告文件名 |

**响应字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | string | 测试币种 |
| days | int | 测试天数 |
| kline_bar | string | K线周期 |
| klines_count | int | K线数量 |
| timestamp | string | 测试时间 |
| variants | array | 各变体结果 |
| variants[].variant_name | string | 变体名 |
| variants[].total_trades | int | 总交易数 |
| variants[].win_rate | float | 胜率 |
| variants[].total_pnl | float | 总盈亏 |
| variants[].max_drawdown | float | 最大回撤 |
| variants[].sharpe_ratio | float | 夏普比率 |
| variants[].profit_factor | float | 盈亏比 |
| winner | object | 最优变体 |
| summary | string | 总结 |

---

## 13. 策略分配

### GET /api/strategy_allocation
获取策略资金分配详情。

**响应字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| allocations | object | 各策略分配比例 |
| performance_table | array | 策略表现表 |
| history | array | 分配历史 |

---

## 错误响应

所有接口在出错时返回 4xx/5xx 状态码，并包含：

```json
{
  "error": "错误描述信息"
}
```
