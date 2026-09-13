# 部署文档

## 1. 环境要求

### 1.1 操作系统
- **推荐**: Windows 10 / 11
- **Python**: 3.12（.venv 虚拟环境）

### 1.2 依赖库
核心依赖：
- `okx` — OKX 官方 SDK
- `loguru` — 日志门面
- `aiohttp` — 异步 HTTP
- `flask` — Dashboard Web 服务
- `numpy`, `pandas` — 数据分析
- `pyyaml` — 配置管理
- `pydantic` — 配置校验
- `sqlalchemy` — ORM

完整依赖列表见 `requirements.txt` 或 `requirements-lock.txt`（精确版本锁定）。

---

## 2. 部署步骤

### 2.1 创建虚拟环境

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2.2 安装依赖

```bash
pip install -r requirements-lock.txt
```

### 2.3 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```env
# OKX API 密钥（需要开启合约交易权限）
OKX_API_KEY=your_api_key
OKX_SECRET_KEY=your_secret_key
OKX_PASSPHRASE=your_passphrase

# 可选：多密钥轮换（OKX_API_KEY_1, OKX_API_KEY_2, ...）
# OKX_API_KEY_1=
# OKX_SECRET_KEY_1=
# OKX_PASSPHRASE_1=

# 代理（Clash/V2Ray 等）
OKX_PROXY=http://127.0.0.1:7897

# 告警（可选）
ALERT_WEBHOOK=https://your-webhook-url.com/alert
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 加密密钥（用于 API 密钥加密存储）
TRAE_ENCRYPTION_KEY=your_encryption_key
```

### 2.4 配置 config.yaml

重点配置项：

```yaml
okx:
  api_key: "${OKX_API_KEY}"       # 从 .env 注入
  secret_key: "${OKX_SECRET_KEY}"
  passphrase: "${OKX_PASSPHRASE}"
  proxy: "${OKX_PROXY:http://127.0.0.1:7897}"

trading:
  total_capital: 100.0            # 交易资金（USDT）
  max_drawdown: 0.25              # 最大回撤 25%
```

### 2.5 启动 Redis（可选）

Windows（需先安装 Redis，或使用项目内置的 `redis/redis-server.exe`）：

```bash
redis\redis-server.exe redis\redis.windows.conf
```

> 无 Redis 时系统自动降级为内存缓存模式。

---

## 3. 启动方式

### 3.1 唯一入口：start.py

```bash
python start.py
```

`start.py` 提供完整编排：
1. 环境预检（.venv、config.yaml、main.py、API 密钥、磁盘空间）
2. 清理旧进程与锁文件
3. 配置预检（Pydantic 校验）
4. 启动 Watchdog（守护主进程，崩溃自动重启）
5. 等待交易引擎就绪
6. 启动 Dashboard（http://localhost:8080）

### 3.2 停止系统

```bash
python stop.py
```

`stop.py` 执行：
1. 优雅关闭（通知进程自行退出）
2. 强制清理残留进程（按进程树父子关系精确识别）
3. 清理锁文件与临时目录
4. 验证端口释放

### 3.3 健康检查

```bash
python health_check.py
```

输出 JSON 格式系统状态，供 Dashboard 与告警复用。

### 3.4 快速操作

| 操作 | 命令 |
|------|------|
| 启动 | `python start.py`（或双击 `启动交易系统.bat`） |
| 停止 | `python stop.py`（或双击 `stop.bat`） |
| Dashboard | http://localhost:8080 |

---

## 4. 验证体系

### 4.1 运行测试

```bash
python -m pytest tests/ -q
```

### 4.2 运行回测

```python
from backtest.backtest_engine import BacktestEngine

engine = BacktestEngine(config)
result = engine.run_backtest(
    symbol="BTC-USDT-SWAP",
    strategy_name="grid",
    days=30,
    initial_capital=100.0,
)
```

---

## 5. 目录结构

```
okx_quant_trading/
├── .venv/                  # 虚拟环境
├── config.yaml             # 主配置文件（唯一入口）
├── .env                    # 环境变量（敏感信息，不入库）
├── .env.example            # 环境变量模板
├── requirements.txt        # Python 依赖
├── requirements-lock.txt   # 精确版本锁定
├── pyproject.toml          # 项目元数据与工具配置
├── main.py                 # 交易主程序
├── start.py                # 唯一启动入口
├── stop.py                 # 停止脚本
├── watchdog.py             # 进程守护
├── dashboard_api.py        # Dashboard API 服务
├── health_check.py         # 健康检查（JSON）
├── docs/                   # 文档目录
│   ├── architecture.md     # 架构文档
│   ├── deployment.md       # 部署文档
│   └── api_docs.md         # API 文档
├── core/                   # 核心模块（调度器/客户端/风控门禁/配置等）
├── risk/                   # 风控模块（全局风控/策略风控/熔断器/对账等）
├── execution/              # 执行模块（订单队列/执行器/算法单等）
├── decision/               # 决策模块（集成决策/ML引擎/规则引擎等）
├── analysis/               # 分析模块（历史分析/策略优化/A/B测试等）
├── monitoring/             # 监控模块（告警/指标/健康评分/通知等）
├── market_data/            # 行情数据（WS推送/缓存/持久化等）
├── backtest/               # 回测模块
├── configs/                # 配置工具（Pydantic校验/环境变量解析）
├── services/               # 服务层
├── alembic/                # 数据库迁移
├── tests/                  # 测试（unit/integration/perf）
├── logs/                   # 日志（每日分割，30天保留）
├── data/                   # 运行数据
│   ├── trading.db          # SQLite 数据库（WAL模式）
│   ├── api_keys.enc        # 加密的API密钥文件
│   └── ...
├── backups/                # 数据库每日备份
└── config_versions/        # 配置版本备份
```

---

## 6. 日志系统

### 6.1 日志配置
- 单一日志门面：loguru
- 全链路事件 ID：每条日志带唯一 event_id，可回溯完整调用链
- 密钥脱敏：API Key 自动掩码为 `***`
- 按天分割文件，保留 30 天，自动 zip 压缩
- 错误日志保留 90 天

### 6.2 日志文件

| 日志文件 | 说明 |
|------|------|
| `logs/trading_YYYY-MM-DD.log` | 主交易日志（DEBUG级别） |
| `logs/error_YYYY-MM-DD.log` | 错误日志（ERROR级别） |

---

## 7. 数据备份

### 7.1 自动备份
- 启动时自动备份数据库（SQLite 在线备份 API，WAL 模式一致性快照）
- 每日凌晨 2:00 自动备份
- 备份前执行 `PRAGMA quick_check` 完整性校验

### 7.2 备份文件位置
- 数据库备份：`./backups/`
- 配置备份：`./config_versions/`

---

## 8. 常见问题

### 8.1 订单失败（错误码 51169）
**原因**：数据库中有持仓记录，但 OKX 实际无持仓（幽灵仓位）。
**解决**：系统已自动处理，会清理策略内部状态并跳过无效订单。

### 8.2 429 限流
**现象**：OKX API 返回 HTTP 429。
**解决**：系统自动触发全局动态退避（10s→30s→60s），关键路径自动绕过。

### 8.3 风控暂停
**现象**：`can_trade()` 返回 False。
**排查**：
1. 查看 Dashboard → 风控监控页面
2. 查看日志中的暂停原因

### 8.4 保证金率警告
**现象**：日志中出现 MARGIN WARNING。
**处理**：系统会自动减仓或全平，避免被交易所强平。

---

## 9. 安全建议

1. **API 密钥权限**：仅授予必要的交易权限，勿开启提币权限
2. **IP 白名单**：在 OKX 后台设置 API IP 白名单
3. **.env 不入库**：`.gitignore` 已覆盖 `.env`、`data/*.enc`、`config_versions/`
4. **定期备份**：确认数据库自动备份正常工作
5. **监控告警**：配置 Telegram 或 Webhook 告警，及时发现问题
6. **小资金试跑**：新策略先用小资金运行观察
7. **定期 review**：每周查看策略表现，必要时调整参数