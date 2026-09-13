# 服务层与测试指南

本文档补充说明 P1/P2 阶段新增的服务层模块、风控增强和测试体系的使用方法。

## 一、服务层架构（services/）

### 1.1 模块总览

| 模块 | 职责 | 关键类 |
|------|------|--------|
| `signal_processor.py` | 信号路由、优先级仲裁、动态仓位调整 | `SignalProcessor` |
| `trading_scheduler.py` | 组件生命周期管理、依赖注入、健康检查 | `TradingSchedulerService` |
| `market_data_service.py` | 行情数据统一获取与缓存 | `MarketDataService` |
| `analysis_service.py` | 策略表现分析与因子计算 | `AnalysisService` |

### 1.2 SignalProcessor 使用

`SignalProcessor` 统一处理所有策略产生的交易信号，负责冷却控制、风控校验、优先级仲裁和动态仓位调整。

**初始化**：
```python
from services.signal_processor import SignalProcessor

processor = SignalProcessor(
    config=config,
    global_risk=global_risk,
    strategy_risk=strategy_risk,
    order_executor=order_executor,
    alert_manager=alert_manager,
    trade_journal=trade_journal,
    adaptive_controller=adaptive_controller,
    profit_optimizer=profit_optimizer,
    account_manager=account_manager,
)
```

**信号路由配置**：
```python
processor.setup_signal_routing(strategies=[grid, trend, scalping], redis_cache=redis)
```

**核心流程**（`_process_signal`）：
1. 信号冷却检查（每个 symbol+strategy+direction 组合独立冷却）
2. 全局风控检查（`global_risk.can_trade()`）
3. 策略风控验证（`strategy_risk.validate_signal()`）
4. 信号冲突检测（同币种反向信号）
5. 策略优先级仲裁（高优先级策略会阻止低优先级策略 120 秒）
6. 动态仓位调整（基于持仓比例、置信度、权益、自适应分配、凯利因子、复利因子）
7. 委托执行器下单

**优先级映射**（数字越小优先级越高）：
- `arbitrage`: 4（最高）
- `scalping`: 3
- `trend`: 2
- `grid` / `spot_grid` / `spot_martingale`: 1

### 1.3 TradingSchedulerService 使用

统一管理所有组件的启动、停止和健康检查。

```python
from services.trading_scheduler import TradingSchedulerService

scheduler = TradingSchedulerService(config)
scheduler.register_component("global_risk", global_risk)
scheduler.register_component("order_executor", order_executor)
scheduler.register_component("signal_processor", signal_processor)

# 启动（阻塞直到 shutdown_event 触发）
await scheduler.start()

# 优雅停止
await scheduler.stop()
```

## 二、风控系统增强（risk/）

### 2.1 三层风控架构

| 层级 | 模块 | 职责 |
|------|------|------|
| L1 全局风控 | `global_risk.py` | 账户级回撤、熔断、阶梯减仓 |
| L2 策略风控 | `strategy_risk.py` | 策略级止损、连亏、仓位限制、凯利公式 |
| L3 风险限额 | `risk_limits.py` | 单品种/策略保证金、交易频率、杠杆限制 |

### 2.2 RiskLimits

```python
from risk.risk_limits import RiskLimits

limits = RiskLimits(config, okx_client, alert_manager=alert)
signal = {"symbol": "BTC-USDT-SWAP", "leverage": 10, "quantity": 0.01}
if limits.check_signal(signal):
    # 限额内，允许下单
    pass
```

检查项：
- 单品种最大保证金占比（默认 25%）
- 单策略最大保证金占比（默认 30%）
- 最大杠杆限制（默认 20x）
- 日内/小时交易频率限制

### 2.3 熔断器（CircuitBreakers）

熔断器在 `GlobalRiskControl` 内部运行，触发后按严重级别响应：
- **minor**：仅暂停交易（加速回撤）
- **major**：暂停 + 减仓 30%（急速回撤 5%+、BTC 大幅波动、强平潮）
- **extreme**：暂停 + 强制全平（急速回撤 10%+、BTC 剧烈波动 15%+）

## 三、数据库优化（data/sqlite_storage.py）

### 3.1 复合索引

已添加以下复合索引提升查询性能：
- `idx_trade_symbol_status`：按 symbol + status 查询持仓
- `idx_trade_strategy_time`：按策略 + 时间查询交易记录
- `idx_position_symbol_time`：按品种 + 时间查询持仓历史

### 3.2 批量操作

```python
# 批量插入交易记录（自动事务，失败回滚）
storage.save_trade_records_batch(trades_list)

# 批量查询开放仓位
positions = storage.get_open_positions()
```

## 四、测试体系（tests/）

### 4.1 测试目录结构

```
tests/
├── conftest.py              # 全局 fixture（basic_config、mock 对象）
├── test_helpers.py          # 工具函数测试
├── unit/                    # 单元测试
│   ├── test_risk.py         # 风控模块测试
│   ├── test_execution.py    # 执行模块测试
│   └── test_utils.py        # 工具函数测试
├── integration/             # 集成测试
│   └── test_trading_flow.py # 端到端交易流程
└── perf/                    # 性能压测
    └── test_performance.py  # 吞吐量与延迟测试
```

### 4.2 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 仅运行单元测试
pytest tests/unit/ -v

# 仅运行集成测试
pytest tests/integration/ -v

# 仅运行性能压测（含吞吐量输出）
pytest tests/perf/ -v -s

# 运行指定测试类
pytest tests/unit/test_risk.py::TestRiskLimits -v
```

### 4.3 性能压测基准

| 测试项 | 基准 | 说明 |
|--------|------|------|
| 订单入队吞吐 | ≥ 1000 单/秒 | 单线程 add_order |
| 订单出队吞吐 | ≥ 1000 单/秒 | 单线程 get_next_order |
| 并发读写吞吐 | ≥ 500 单/秒 | 10 生产者 + 10 消费者 |
| 信号处理吞吐 | ≥ 500 信号/秒 | 含风控+动态仓位 |
| 批量插入吞吐 | ≥ 1000 条/秒 | SQLite 内存数据库 |
| 风控检查延迟 | < 1000μs | RiskLimits.check_signal |
| 策略风控延迟 | < 500μs | StrategyRiskControl.validate_signal |

### 4.4 集成测试覆盖

- **信号到执行流程**：信号 → 风控 → 队列 → 执行 完整链路
- **信号冷却**：重复信号被冷却拦截
- **风控暂停**：全局风控暂停时所有信号被拒
- **订单优先级**：止损 > 剑客 > 趋势 > 网格
- **队列容量**：满队列拒绝新订单
- **订单生命周期**：queued → executing → filled
- **数据库事务**：批量插入失败回滚

## 五、监控告警（monitoring/alert_rules.py）

### 5.1 告警分类

| 类别 | 规则数 | 关键指标 |
|------|--------|---------|
| 风控 | 7 | 回撤、日内亏损、连亏、保证金率、熔断 |
| 交易 | 5 | 拒单率、滑点、队列大小、API 错误率 |
| 系统 | 6 | 内存、CPU、磁盘、Redis/DB 连接 |
| 性能 | 4 | API 延迟、WebSocket、信号处理延迟 |

### 5.2 使用

```python
from monitoring.alert_rules import evaluate_metric, get_alert_summary, AlertSeverity

# 评估指标
triggered = evaluate_metric("drawdown_pct", 0.28)
for rule in triggered:
    print(f"[{rule.severity.value}] {rule.name}: {rule.description}")

# 查看规则摘要
summary = get_alert_summary()
```

### 5.3 严重级别

- **INFO**：提示信息，无需处理
- **WARNING**：需关注，可能影响性能
- **CRITICAL**：需立即处理，影响交易
- **EMERGENCY**：紧急，已触发熔断或强制停止
