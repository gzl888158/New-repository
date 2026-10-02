# OKX 量化交易系统架构文档

## 1. 系统概述

OKX 量化交易系统是一个多策略并行运行的自动化加密货币交易平台，通过 OKX 交易所 REST/WebSocket API 执行永续合约（USDT-SWAP）与现货交易。系统具备完整的风控体系、动态资金管理、参数自学习和监控告警能力。

### 1.1 设计理念
- **多策略协同**：6 种策略（网格/趋势/剥头皮/套利/现货网格/现货马丁格尔）并行运行，优先级仲裁 + 冷却时间避免冲突
- **阶梯式风控**：从预警到减仓到强制全平的渐进式风险控制
- **自适应资金管理**：Kelly 公式 + 复利因子 + 回撤保护 + 动态仓位调整
- **参数自学习**：6 小时周期学习，基于历史交易数据自动优化策略参数，30 分钟验证回滚
- **高可用设计**：Watchdog 进程守护、自动重启、崩溃自愈、数据库每日备份

---

## 2. 系统架构图

```
                         ┌─────────────────┐
                         │   OKX Exchange  │
                         │ (REST + WebSocket)│
                         └───────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  core/okx_client │
                        │ (REST/WS 客户端)  │
                        └───┬──────────┬───┘
                            │          │
               ┌────────────▼──┐   ┌───▼──────────────┐
               │  query_cache  │   │  market_data/    │
               │  (信号/缓存)   │   │  (行情/持久化)    │
               └───────┬───────┘   └──────────┬───────┘
                       │                      │
           ┌───────────▼──────────────────────▼─────────────┐
           │           core/scheduler (核心调度器)            │
           │  ┌──────────────────────────────────────────┐  │
           │  │  Grid │ Trend │ Scalping │ Arbitrage     │  │
           │  │       │ Spot Grid │ Spot Martingale      │  │
           │  └────┬─────────────────────────────────┬───┘  │
           │       │                                 │      │
           │  ┌────▼─────────────────────────────────▼───┐  │
           │  │          Order Queue (令牌桶限流)          │  │
           │  └──────────────┬───────────────────────────┘  │
           │                 │                              │
           │          ┌──────▼──────────┐                  │
           │          │  OrderExecutor  │                  │
           │          │  (订单执行引擎)  │                  │
           │          └──────┬──────────┘                  │
           │                 │                              │
           │  ┌──────────────▼──────────────────────────┐  │
           │  │           风控体系 (Risk Control)          │  │
           │  │  risk_gate / circuit_breaker /           │  │
           │  │  adaptive_controller / pnl_reconciler    │  │
           │  └──────────────────────────────────────────┘  │
           └────────────────────────────────────────────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │   Monitoring & Alerting   │
                   │  alert_evaluator /        │
                   │  metrics_collector /      │
                   │  health_scorer            │
                   └─────────────┬─────────────┘
                                 │
                   ┌─────────────▼─────────────┐
                   │    Dashboard (Web UI)     │
                   │    dashboard_api.py       │
                   │    http://localhost:8080  │
                   └───────────────────────────┘
```

---

## 3. 核心模块说明

### 3.1 核心层 (core/)

| 模块 | 职责 | 关键文件 |
|------|------|----------|
| OKXClient | OKX REST/WebSocket 客户端，行情订阅、订单管理、账户查询 | `core/okx_client.py` |
| TradingScheduler | 系统核心调度器，策略生命周期管理、信号路由、优先级仲裁 | `core/scheduler.py` |
| AccountManager | 账户资金管理，保证金追踪、仓位分配 | `core/account_manager.py` |
| TradeJournal | 交易日志，记录所有交易并提供统计分析 | `core/trade_journal.py` |
| IntelligentAgent | 智能代理，6 小时学习周期、策略自动暂停、参数热更新 | `core/intelligent_agent.py` |
| RiskGate | 风控门禁，信号审计、黑名单检查、仓位限制 | `core/risk_gate.py` |
| EventID | 全链路事件 ID 体系，日志脱敏 | `core/event_id.py` |
| APIKeyManager | API 密钥加密存储、轮换、失效自检 | `core/api_key_manager.py` |
| ConfigManager | 配置版本化、快照回滚、热更新 | `core/config_manager.py` |
| LocalOKXClient | 本地模拟 OKX 客户端，用于回测和测试 | `core/local_client.py` |
| MiniBacktester | 策略上线前快速回测门禁 | `core/mini_backtester.py` |

### 3.2 策略层 (6 种策略)

| 策略 | 原理 | 冷却时间 | 优先级 |
|------|------|----------|--------|
| Grid (网格) | 网格套利，非对称买/卖层，马丁格尔加仓，多时间框架确认 | 20s | 1 (最低) |
| Trend (趋势) | 9 因子评分系统，多周期确认（1H+5M），追踪止盈 | 30s | 2 |
| Scalping (剥头皮) | 短期波动交易，动量衰减检查，每日 20 笔上限 | 10s | 3 |
| Arbitrage (套利) | 跨币种相关性套利、资金费率套利 | 60s | 4 (最高) |
| Spot Grid (现货网格) | 现货市场的网格套利策略 | 20s | 1 |
| Spot Martingale (现货马丁) | 现货市场的马丁格尔策略 | 30s | 2 |

### 3.3 风控层 (risk/)

| 模块 | 职责 | 检查频率 |
|------|------|----------|
| GlobalRiskControl | 全局风控：权益回撤、仓位风险、保证金率、紧急平仓 | 5s |
| StrategyRiskControl | 策略级风控：单策略仓位限制、信号过滤 | 实时 |
| AdaptiveController | 自适应资金分配：基于策略表现动态调整资金比例 | 周期性 |
| ProfitOptimizer | 利润优化：Kelly 公式、复利因子、回撤保护曲线 | 动态 |
| CorrelationRiskControl | 相关性风控：同向高相关性仓位集中度限制 | 180s |
| PnLReconciler | PnL 对账：从 OKX 账单校正数据库 PnL | 6h |
| CircuitBreakerV2 | 熔断器 V2：急速回撤、回撤加速、BTC 波动、强平潮检测 | 实时 |

### 3.4 执行层 (execution/)

| 模块 | 职责 |
|------|------|
| OrderQueue | 异步订单队列，令牌桶限流 |
| OrderExecutor | 订单执行引擎，支持市价/限价/条件单，含滑点保护 |
| FillQualityTracker | 成交质量追踪，滑点统计分析 |
| OrderLifecycleManager | 订单生命周期管理 |
| StaleOrderManager | 过期订单清理 |
| AlgoOrders | 算法订单（TWAP/VWAP/Iceberg/DarkPool） |

### 3.5 决策层 (decision/)

| 模块 | 职责 |
|------|------|
| EnsembleDecisionMaker | 集成决策，多引擎信号融合 |
| IntelligentDecisionEngine | 智能决策引擎 |
| MLDecisionEngine | 机器学习决策引擎 |
| RuleBasedEngine | 规则引擎，价值范围重叠检查 |
| DecisionValidator | 决策验证 |
| ConfidenceCalibrator | 置信度校准（胜率→阈值自适应） |

### 3.6 分析层 (analysis/)

| 模块 | 职责 |
|------|------|
| HistoricalAnalyzer | 历史交易数据分析 |
| StrategyOptimizer | 策略参数优化器，参数热更新 + 配置版本备份 |
| ABTestingFramework | A/B 测试框架 |
| DataAnalysisEngine | 数据分析引擎 |
| ContributionAnalyzer | 策略贡献度分析 |

### 3.7 监控层 (monitoring/)

| 模块 | 职责 |
|------|------|
| AlertEngine | 告警规则引擎，阈值评估 |
| MetricsCollector | 指标采集（CPU/内存/API 延迟/胜率/Sharpe） |
| HealthScorer | 健康评分，综合多维度系统状态 |
| NotificationChannels | 通知渠道（Webhook + Telegram） |
| AutoRecovery | 自动恢复，策略降级后自动恢复 |

### 3.8 数据层 (market_data/)

| 模块 | 职责 |
|------|------|
| Manager | 行情数据管理器 |
| WebSocketFeed | WebSocket 实时行情推送 |
| Cache | 行情缓存（TTL 120s） |
| TickPersistence | Tick 数据持久化 |
| HistoricalLoader | 历史数据加载器 |

### 3.9 配置文件 (configs/)

| 模块 | 职责 |
|------|------|
| settings.py | Pydantic 配置校验，环境变量解析，多密钥池 |
| config_validator.py | 配置合法性校验 |

### 3.10 自治协调器 (core/)

| 模块 | 职责 |
|------|------|
| QuantAGIOrchestrator | 汇总市场状态、策略贡献和资金分配结果，运行资金侧自治闭环 |
| TopLevelAGICoordinator | 汇总资金、信号感知、运维自愈和自动寻优状态，生成跨闭环诊断与排序计划 |

顶层协调器输出分阶段分析、证据摘要和按优先级排序的 `decision_plan`。计划仅供人工审核，不会自动执行交易、参数修改或策略暂停；当资金健康度为 D/F 或运维建议暂停寻优时，会将冲突的寻优建议延期，并在报告中说明阻断原因。可通过 `top_level_agi.enabled` 控制是否装配。

---

## 4. 核心数据流

### 4.1 交易信号流

```
市场行情 → OKX WebSocket → 行情缓存 (TTL 120s)
                              │
                        策略实时计算
                              │
                              ▼
                    策略生成信号 (含置信度)
                              │
                              ▼
                    ┌───────────────────┐
                    │  TradingScheduler  │
                    │  信号处理流程:     │
                    │  1. 冷却时间检查   │
                    │  2. 全局风控检查   │
                    │  3. 策略风控检查   │
                    │  4. 冲突信号检查   │
                    │  5. 优先级仲裁     │
                    │  6. Kelly 仓位计算  │
                    └─────────┬─────────┘
                              │
                              ▼
                         订单队列 (令牌桶限流)
                              │
                              ▼
                         订单执行 → OKX
```

### 4.2 风控数据流

```
OKX 账户信息
     │
     ▼
GlobalRiskControl (每5秒)
 ├── 权益追踪 & 回撤计算
 ├── 阶梯式风控触发 (10%/15%/20%/25%)
 ├── 单品种仓位限制 (25%上限)
 ├── 保证金率监控 (<20%自动平仓)
 ├── 强平价距离监控 (liqPx)
 ├── 熔断器检查 (黑天鹅检测)
 │   ├── 急速回撤 (rapid_drawdown)
 │   ├── 回撤加速 (drawdown_acceleration)
 │   ├── BTC 大幅波动 (btc_movement)
 │   └── 强平潮 (liquidation_volume)
 └── 连亏检测 → 策略自动暂停

CorrelationRiskControl (每180秒)
 └── 同向高相关仓位集中度限制
```

### 4.3 资金管理数据流

```
历史交易数据 → TradeJournal → HistoricalAnalyzer
                                       │
                                       ▼
                              StrategyOptimizer (优化参数)
                                       │
                                       ▼
ProfitOptimizer (Kelly 公式 + 复利因子 + 回撤保护)
                                       │
                                       ▼
                          AdaptiveController (动态分配)
                                       │
                                       ▼
                              各策略仓位计算
```

### 4.4 自治分析与决策计划

```
资金 / 感知 / 运维 / 寻优状态
               │
               ▼
       Gather → Diagnose
               │
               ▼
风险冲突门控 → 优先级排序 → 人工审核计划
               │
               ▼
         报告与状态观测
```

报告包含 `analysis` 阶段（观测、诊断、冲突处理、计划）及 `decision_plan.steps`。每个建议附有观测证据和原因，明确标记 `auto_execute: false` 与 `requires_human_approval: true`；被风险门控延期的建议保留在 `deferred_actions` 中，不计入已生成的联动动作统计。

---

## 5. 关键设计模式

### 5.1 异步架构
- 全部使用 `async/await`，无回调风格
- 策略/风控/监控各自独立 `asyncio.Task` 运行
- 订单队列异步解耦生产与消费

### 5.2 熔断器模式
- 黑天鹅事件触发后暂停交易
- 分级响应：minor（仅暂停）→ major（减仓30%）→ extreme（全平）
- 自动恢复（minor 60s, major 300s, extreme 需手动重置）
- 网络断连熔断冷却 300s，恢复后自动复位

### 5.3 降级模式
- 429 限流检测触发全局动态退避（10s→30s→60s）
- 关键路径（行情/持仓/账户）自动绕过延迟熔断
- 网络退化时网格策略超时缩短至 60s

### 5.4 观察者模式
- 信号回调机制
- 持仓清理回调机制
- 告警通知机制

---

## 6. 配置结构

主配置文件：`config.yaml`（唯一入口，Pydantic 校验）

```
config.yaml
├── system:               # 系统配置 (version: 2.0.0)
├── okx:                  # API 密钥 (从 .env 注入)、代理设置
├── trading:              # 总资本、资金分配、风控参数
├── risk:                 # 熔断器、相关性、保证金阈值
├── strategies:           # 6 种策略参数
│   ├── grid
│   ├── trend
│   ├── scalping
│   ├── arbitrage
│   ├── spot_grid
│   └── spot_martingale
├── currencies:           # 币种分档 (tier1/2/3)
│   ├── tier1_symbols:    # BTC, ETH, SOL
│   ├── tier2_symbols:    # ADA, AVAX, BNB, DOGE, DOT, LINK, LTC, NEAR, UNI, XRP
│   └── tier3_symbols:    # 其他
├── monitoring:           # 监控端口、延迟阈值
├── anti_targeting:       # 反指纹/反识别
├── capital_attrition:    # 资金磨损分析
├── performance:          # 性能监控
├── notebook:             # 学习/记录
├── hardware:             # CPU 亲和性、资源阈值
└── telegram:             # Telegram 告警配置
```

---

## 7. 数据库结构

SQLite 数据库：`./data/trading.db`（WAL 模式，每日自动备份）

| 表名 | 说明 |
|------|------|
| trade_records | 交易记录（开仓/平仓/持仓中） |
| account_history | 账户权益历史（用于权益曲线） |
| learning_states | 策略学习状态持久化 |
| fill_quality | 订单成交质量（滑点统计） |
| config_versions | 配置版本备份 |

---

## 8. 对外接口

### 8.1 Dashboard API
- HTTP REST 接口，Flask 框架
- 端口：8080
- 健康检查：`GET /api/health`

### 8.2 告警接口
- Webhook（HTTP POST）
- Telegram Bot API

### 8.3 启动/停止
- 唯一启动入口：`python start.py`（编排 watchdog + dashboard + main）
- 停止：`python stop.py`（进程树精确识别，多轮清理）

---

## 9. 扩展点

1. **新增策略**：实现策略类，注册到 scheduler
2. **新增风控**：实现独立风控模块，在 scheduler 中启动并接入检查流程
3. **新增数据源**：扩展 OKXClient 或添加新的市场数据客户端
4. **新增告警通道**：扩展 NotificationChannels