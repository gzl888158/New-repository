# OKX 量化交易系统 - 生产环境部署检查清单

> 版本: 2.0.0 | 生成时间: 2026-08-05 | 目标环境: Windows Server / Windows 10+

---

## 一、基础设施检查

| # | 检查项 | 要求 | 状态 |
|---|---|---|---|
| 1.1 | Python 版本 | Python 3.11+ | [ ] |
| 1.2 | 虚拟环境 | `.venv/` 已创建且激活 | [ ] |
| 1.3 | pip 依赖 | `pip install -r requirements.txt` 无报错 | [ ] |
| 1.4 | 磁盘空间 | 数据盘剩余 > 5GB | [ ] |
| 1.5 | 内存 | 可用内存 > 2GB | [ ] |
| 1.6 | 网络 | 可访问 `api.okx.com`（443端口） | [ ] |
| 1.7 | 时区 | 系统时区 = Asia/Shanghai | [ ] |
| 1.8 | PowerShell | PowerShell 5.1+ 可用 | [ ] |
| 1.9 | CPU 核心数 | ≥ 4 核（参考 `hardware.cpu_cores`） | [ ] |

---

## 二、配置文件检查

| # | 检查项 | 文件/配置 | 状态 |
|---|---|---|---|
| 2.1 | `.env` 存在 | `OKX_API_KEY` 已填写（非占位符） | [ ] |
| 2.2 | `.env` 存在 | `OKX_SECRET_KEY` 已填写（非占位符） | [ ] |
| 2.3 | `.env` 存在 | `OKX_PASSPHRASE` 已填写（非占位符） | [ ] |
| 2.4 | `.env` 存在 | `REDIS_PASSWORD` 已配置 | [ ] |
| 2.5 | `.env` 存在 | `TELEGRAM_BOT_TOKEN` 已配置（如需通知） | [ ] |
| 2.6 | `.env` 存在 | `TELEGRAM_CHAT_ID` 已配置（如需通知） | [ ] |
| 2.7 | `.env` 存在 | `ALERT_WEBHOOK` 已配置（如需告警） | [ ] |
| 2.8 | `.env` 存在 | `DASHBOARD_TOKEN` 已设置（非默认值） | [ ] |
| 2.9 | `config.yaml` | 配置校验通过 (`validate_config()`) | [ ] |
| 2.10 | `config.yaml` | `is_testnet: false`（生产环境） | [ ] |
| 2.11 | `config.yaml` | `trading.total_capital` 与账户余额匹配 | [ ] |
| 2.12 | `config.yaml` | `trading.max_drawdown` ≤ 0.25 | [ ] |
| 2.13 | `config.yaml` | `trading.daily_max_loss` ≤ 0.04 | [ ] |
| 2.14 | `config.yaml` | `trading.max_total_leverage` ≤ 20 | [ ] |
| 2.15 | `config.yaml` | `okx.proxy` 代理地址正确（如有） | [ ] |
| 2.16 | `config.yaml` | `hardware.max_memory_usage_mb` ≤ 系统可用内存 | [ ] |
| 2.17 | `config.yaml` | `hardware.temperature_threshold` ≤ 85°C | [ ] |
| 2.18 | 配置审计 | `audit_config()` 无高危警告 | [ ] |

---

## 三、数据库与存储检查

| # | 检查项 | 详情 | 状态 |
|---|---|---|---|
| 3.1 | SQLite 数据库 | `data/trading.db` 存在且可读写 | [ ] |
| 3.2 | Redis 服务 | `redis/redis-server.exe` 已启动（端口 6379） | [ ] |
| 3.3 | Redis 连通性 | `redis_cache.health_check()` 返回 True | [ ] |
| 3.4 | Redis 认证 | `REDIS_PASSWORD` 验证通过 | [ ] |
| 3.5 | SQLite 连通性 | `sqlite_storage.health_check()` 返回 True | [ ] |
| 3.6 | 数据目录 | `data/` 目录存在且可写 | [ ] |
| 3.7 | 日志目录 | `logs/` 目录存在且可写 | [ ] |
| 3.8 | Alembic 迁移 | `alembic/` 目录存在，迁移脚本就绪 | [ ] |

---

## 四、安全加固检查

| # | 检查项 | 详情 | 状态 |
|---|---|---|---|
| 4.1 | API 密钥 | 不在代码中硬编码，仅通过 `.env` 注入 | [ ] |
| 4.2 | `.env` 权限 | `.env` 文件权限仅限当前用户读取 | [ ] |
| 4.3 | 防火墙 | 仅开放 Dashboard 端口（8080）给内网 | [ ] |
| 4.4 | Dashboard 认证 | `DASHBOARD_TOKEN` 已设置强密码 | [ ] |
| 4.5 | CSP 头 | Dashboard 响应包含 Content-Security-Policy | [ ] |
| 4.6 | 安全头 | X-Frame-Options: DENY 已启用 | [ ] |
| 4.7 | 安全头 | X-Content-Type-Options: nosniff 已启用 | [ ] |
| 4.8 | 速率限制 | API 限流 120次/分钟/IP 已启用 | [ ] |
| 4.9 | 代理安全 | 如使用代理，确认仅转发 OKX API 流量 | [ ] |
| 4.10 | `.gitignore` | `.env`、`data/*.db`、`logs/`、`backups/` 已排除 | [ ] |

---

## 五、核心模块启动检查

| # | 检查项 | 验证方式 | 状态 |
|---|---|---|---|
| 5.1 | 配置校验层 | `configs/config_validator.py` 导入无报错 | [ ] |
| 5.2 | 熔断器 | `core/circuit_breaker.py` 导入无报错 | [ ] |
| 5.3 | 健康检查器 | `core/health_checker.py` 导入无报错 | [ ] |
| 5.4 | 智能订单路由 | `execution/algo_orders/smart_order_router.py` 导入无报错 | [ ] |
| 5.5 | 风控门 | `core/risk_gate.py` 导入无报错 | [ ] |
| 5.6 | 全局风控 | `risk/global_risk.py` 导入无报错 | [ ] |
| 5.7 | 策略风控 | `risk/strategy_risk.py` 导入无报错 | [ ] |
| 5.8 | 异常处理 | `core/exception_handler.py` 导入无报错 | [ ] |
| 5.9 | 结构化日志 | `utils/structured_logger.py` 导入无报错 | [ ] |
| 5.10 | 看护进程 | `watchdog.py` 语法无报错 | [ ] |
| 5.11 | Dashboard API | `dashboard_api.py` 语法无报错 | [ ] |
| 5.12 | OKX 客户端 | `core/okx_client.py` 导入无报错 | [ ] |
| 5.13 | 策略引擎 | `core/strategy_engine.py` 导入无报错 | [ ] |
| 5.14 | 调度器 | `core/scheduler.py` 导入无报错 | [ ] |
| 5.15 | 资金管理 | `core/capital_manager.py` 导入无报错 | [ ] |

---

## 六、运行前自检

| # | 检查项 | 命令/方式 | 状态 |
|---|---|---|---|
| 6.1 | 全量单元测试 | `.venv\Scripts\python.exe -m pytest tests/unit/ -v` 全部通过 | [ ] |
| 6.2 | 集成测试 | `.venv\Scripts\python.exe -m pytest tests/integration/ -v` 全部通过 | [ ] |
| 6.3 | 性能基准测试 | `.venv\Scripts\python.exe -m pytest tests/perf/ -v` 无严重退化 | [ ] |
| 6.4 | OKX API 连通 | `GET /api/v5/public/time` 返回 200 | [ ] |
| 6.5 | OKX 账户余额 | 账户 USDT 余额 > `min_balance`（默认 5 USDT） | [ ] |
| 6.6 | 合约交易权限 | 账户已开通合约交易 | [ ] |
| 6.7 | 费率等级 | 确认 maker/taker 费率与实际一致 | [ ] |
| 6.8 | 杠杆设置 | 交易对杠杆已预设 | [ ] |
| 6.9 | 代理连通性 | 如配置代理，确认代理可达 | [ ] |
| 6.10 | 配置文件语法 | `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` | [ ] |

---

## 七、启动流程

| # | 步骤 | 操作 | 状态 |
|---|---|---|---|
| 7.1 | 启动 Redis | 运行 `redis/redis-server.exe` 或已设为 Windows 服务 | [ ] |
| 7.2 | 启动交易系统 | 双击 `启动交易系统.bat` | [ ] |
| 7.3 | 等待就绪 | 观察控制台输出，确认无 ERROR | [ ] |
| 7.4 | 验证 Dashboard | 访问 `http://localhost:8080` | [ ] |
| 7.5 | 健康检查 v2 | `GET http://localhost:8080/api/health/v2` → `score ≥ 80` | [ ] |
| 7.6 | 存活探针 | `GET /api/health/liveness` → `{"status":"healthy"}` | [ ] |
| 7.7 | 就绪探针 | `GET /api/health/readiness` → `{"status":"healthy"}` | [ ] |
| 7.8 | 启动探针 | `GET /api/health/startup` → `{"status":"healthy"}` | [ ] |
| 7.9 | 健康评分 | `GET /api/health_score` 返回正常 | [ ] |
| 7.10 | 检查心跳 | `data/heartbeat.json` 的 `last_update` 在 60 秒内 | [ ] |
| 7.11 | 检查进程 | 运行 `health_check.bat` 或 `python health_check.py` 无异常 | [ ] |
| 7.12 | 系统状态 | `GET /api/system_status` 返回正常 | [ ] |
| 7.13 | 启动状态文件 | `data/startup_status.json` 的 `phase` = `running` | [ ] |

---

## 八、监控与告警

| # | 检查项 | 要求 | 状态 |
|---|---|---|---|
| 8.1 | 看护进程 | `watchdog.py` 进程正在运行（崩溃自动重启，最多 10 次/小时） | [ ] |
| 8.2 | 健康评分 | `/api/health/v2` 的 `score` ≥ 80 | [ ] |
| 8.3 | 熔断器状态 | 所有熔断器状态为 `closed` | [ ] |
| 8.4 | 风控状态 | 无全局熔断触发 | [ ] |
| 8.5 | 错误日志 | `logs/` 目录无持续增长，无 FATAL 日志 | [ ] |
| 8.6 | 仓位监控 | 实际仓位与 Dashboard 显示一致 | [ ] |
| 8.7 | Telegram 通知 | `config.yaml` 中 `notifications.enabled: true` 时，确认能收到消息 | [ ] |
| 8.8 | Alert Webhook | `config.yaml` 中 `monitoring.alert_webhook` 已配置且可达 | [ ] |
| 8.9 | API 延迟 | `monitoring.api_latency_warning_ms` (500ms) 以下 | [ ] |
| 8.10 | 硬件监控 | CPU 使用率 < `hardware.max_cpu_usage`%，内存 < `hardware.memory_warning_threshold_mb` | [ ] |
| 8.11 | Metrics 端口 | `monitoring.metrics_port` (9090) 可访问（如启用） | [ ] |
| 8.12 | 日志轮转 | 结构化日志已配置轮转，避免磁盘占满 | [ ] |

---

## 九、回滚与应急

| # | 检查项 | 准备 | 状态 |
|---|---|---|---|
| 9.1 | 停止脚本 (bat) | `stop.bat` 可正常终止所有进程 | [ ] |
| 9.2 | 停止脚本 (ps1) | `.\stop.ps1` 可正常终止所有进程（PowerShell） | [ ] |
| 9.3 | 一键平仓 | `scripts/close_all.py` 可执行 | [ ] |
| 9.4 | 数据库备份 | `data/trading.db` 最新备份已保存至 `backups/` | [ ] |
| 9.5 | 配置备份 | `.env` 和 `config.yaml` 备份已保存 | [ ] |
| 9.6 | 紧急联系人 | 异常告警通知方式（Telegram/Webhook）已验证 | [ ] |
| 9.7 | 恢复流程 | 备份恢复步骤已文档化并测试 | [ ] |

---

## 十、最终确认

| # | 签署项 | 签字 |
|---|---|---|
| 10.1 | 以上所有检查项均已通过 | _______ |
| 10.2 | 已阅读并理解系统风险（最大回撤、杠杆、自动交易） | _______ |
| 10.3 | 部署日期：____年__月__日 | _______ |
| 10.4 | 部署人员：__________ | _______ |

---

> **快速验证命令**
> ```powershell
> # 1. 语法检查
> .\.venv\Scripts\python.exe -c "import core; import configs; print('OK')"
> 
> # 2. 全量单元测试
> .\.venv\Scripts\python.exe -m pytest tests/unit/ -v
> 
> # 3. 集成测试
> .\.venv\Scripts\python.exe -m pytest tests/integration/ -v
> 
> # 4. 性能测试
> .\.venv\Scripts\python.exe -m pytest tests/perf/ -v
> 
> # 5. 配置校验
> .\.venv\Scripts\python.exe -c "from configs.config_validator import validate_config; import yaml; cfg=yaml.safe_load(open('config.yaml','r',encoding='utf-8')); validate_config(cfg); print('Config OK')"
> 
> # 6. 配置审计
> .\.venv\Scripts\python.exe -c "from configs.config_validator import validate_config, audit_config; import yaml; cfg=yaml.safe_load(open('config.yaml','r',encoding='utf-8')); ac=validate_config(cfg); audit=audit_config(ac); print('Warnings:', audit['warnings'])"
> 
> # 7. 健康检查
> curl http://localhost:8080/api/health/v2
> 
> # 8. 进程检查
> .\.venv\Scripts\python.exe health_check.py
> 
> # 9. OKX API 连通性
> .\.venv\Scripts\python.exe -c "import requests; r=requests.get('https://www.okx.com/api/v5/public/time'); print(r.json())"
> ```