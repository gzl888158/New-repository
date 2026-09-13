"""
实盘交易前检查脚本
用于验证所有配置和环境是否就绪
"""
import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_env_variables():
    """检查环境变量配置"""
    print("=" * 60)
    print("1. 环境变量检查")
    print("=" * 60)

    required_vars = ["OKX_API_KEY", "OKX_SECRET_KEY", "OKX_PASSPHRASE"]
    missing = []

    for var in required_vars:
        value = os.getenv(var, "")
        if not value or value.startswith("your_"):
            missing.append(var)
            print(f"  [FAIL] {var}: 未配置或仍为默认值")
        else:
            masked = value[:4] + "****" + value[-4:] if len(value) > 8 else "****"
            print(f"  [PASS] {var}: {masked}")

    if missing:
        print(f"\n  错误: {len(missing)}个必需环境变量未配置")
        return False

    optional_vars = ["REDIS_PASSWORD", "ALERT_WEBHOOK", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    print(f"\n  可选变量:")
    for var in optional_vars:
        value = os.getenv(var, "")
        status = "已配置" if value else "未配置"
        print(f"    {var}: {status}")

    return True


def _load_config():
    """加载配置并进行pydantic验证"""
    try:
        from configs.settings import load_config
        config = load_config()
        return config, True
    except Exception as e:
        print(f"  [FAIL] Pydantic配置验证失败: {e}")
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f), False
        return {}, False


def check_config():
    """检查配置文件（含pydantic验证）"""
    print("\n" + "=" * 60)
    print("2. 配置文件检查")
    print("=" * 60)

    config_path = Path("config.yaml")
    if not config_path.exists():
        print("  [FAIL] config.yaml 不存在")
        return False

    config, pydantic_ok = _load_config()

    if pydantic_ok:
        print("  [PASS] Pydantic配置 schema 验证通过")
    else:
        print("  [FAIL] Pydantic配置验证未通过，请检查配置字段")
        return False

    # 检查关键配置（警告项，不阻断）
    warnings = []

    # OKX配置
    okx = config.get("okx", {})
    is_testnet = okx.get("is_testnet", True)
    if is_testnet:
        warnings.append("测试网模式: 建议实盘前先在测试网验证")
    else:
        print("  [PASS] 主网模式 (注意风险)")

    proxy_ok = bool(okx.get("proxy"))
    print(f"  [PASS] 代理配置: {'已配置' if proxy_ok else '未配置(可选)'}")

    # 交易配置
    trading = config.get("trading", {})
    total_capital = trading.get("total_capital", 0)
    if total_capital > 0:
        print(f"  [PASS] 总资金: {total_capital} USDT")
    else:
        print(f"  [FAIL] 总资金: {total_capital} USDT (无效)")
        return False

    trading_ratio = trading.get("trading_capital_ratio", 0)
    print(f"  [PASS] 交易资金比例: {trading_ratio*100:.0f}%")

    max_drawdown = trading.get("max_drawdown", 1)
    print(f"  [PASS] 最大回撤: {max_drawdown*100:.1f}%")
    if max_drawdown > 0.25:
        warnings.append(f"最大回撤阈值{max_drawdown*100:.0f}%偏高")

    daily_max_loss = trading.get("daily_max_loss", 1)
    print(f"  [PASS] 日最大亏损: {daily_max_loss*100:.1f}%")

    max_leverage = trading.get("max_total_leverage", 100)
    print(f"  [PASS] 最大总杠杆: {max_leverage}x")
    if max_leverage > 10:
        warnings.append(f"最大总杠杆{max_leverage}x偏高")

    max_positions = trading.get("max_concurrent_positions", 100)
    print(f"  [PASS] 最大持仓数: {max_positions}")

    # 策略分配总和校验
    alloc_sum = (
        trading.get("grid_allocation", 0)
        + trading.get("trend_allocation", 0)
        + trading.get("scalping_allocation", 0)
        + trading.get("arbitrage_allocation", 0)
    )
    if abs(alloc_sum - 1.0) < 1e-6:
        print(f"  [PASS] 策略分配总和: {alloc_sum:.4f}")
    else:
        print(f"  [FAIL] 策略分配总和: {alloc_sum:.4f} (不等于1.0)")
        return False

    # 资金比例总和校验
    capital_sum = (
        trading.get("trading_capital_ratio", 0)
        + trading.get("risk_reserve_ratio", 0)
        + trading.get("profit_reserve_ratio", 0)
    )
    if abs(capital_sum - 1.0) < 1e-6:
        print(f"  [PASS] 资金比例总和: {capital_sum:.4f}")
    else:
        print(f"  [FAIL] 资金比例总和: {capital_sum:.4f} (不等于1.0)")
        return False

    # 策略配置
    strategies = config.get("strategies", {})
    enabled_count = 0
    for name in ["grid", "trend", "scalping", "arbitrage"]:
        enabled = strategies.get(name, {}).get("enabled", False)
        if enabled:
            enabled_count += 1
        print(f"  [PASS] {name}策略: {'启用' if enabled else '禁用'}")

    if enabled_count >= 1:
        print(f"  [PASS] 启用策略数: {enabled_count} 个")
    else:
        print(f"  [FAIL] 启用策略数: {enabled_count} 个 (至少需要1个)")
        return False

    if warnings:
        print(f"\n  警告 ({len(warnings)}项):")
        for w in warnings:
            print(f"    - {w}")

    return True


def check_risk_params():
    """检查风险参数是否合理"""
    print("\n" + "=" * 60)
    print("3. 风险参数检查")
    print("=" * 60)

    config, _ = _load_config()

    trading = config.get("trading", {})
    risk = config.get("risk", {})

    issues = []
    warnings = []

    # 检查资金分配总和
    allocations = [
        trading.get("grid_allocation", 0),
        trading.get("trend_allocation", 0),
        trading.get("scalping_allocation", 0),
        trading.get("arbitrage_allocation", 0),
    ]
    total_alloc = sum(allocations)
    if abs(total_alloc - 1.0) > 1e-6:
        issues.append(f"策略资金分配总和({total_alloc*100:.2f}%)不等于100%")
    else:
        print(f"  [PASS] 策略资金分配: {total_alloc*100:.0f}%")

    # 检查杠杆设置
    max_leverage = trading.get("max_total_leverage", 10)
    if max_leverage > 10:
        warnings.append(f"最大总杠杆为{max_leverage}x，偏高")
    else:
        print(f"  [PASS] 最大总杠杆: {max_leverage}x")

    # 检查回撤限制
    max_dd = trading.get("max_drawdown", 0.15)
    if max_dd > 0.25:
        warnings.append(f"最大回撤阈值{max_dd*100:.0f}%偏高")
    else:
        print(f"  [PASS] 最大回撤: {max_dd*100:.0f}%")

    # 检查熔断器
    cb = risk.get("circuit_breakers", {})
    rapid_dd = cb.get("rapid_drawdown_threshold", 0.05)
    if rapid_dd > 0.10:
        warnings.append(f"快速回撤阈值{rapid_dd*100:.0f}%可能过于宽松")
    else:
        print(f"  [PASS] 快速回撤阈值: {rapid_dd*100:.0f}%")

    # 检查风险阈值顺序
    position_loss = risk.get("position_loss_threshold", 0.10)
    full_close = risk.get("full_close_threshold", 0.18)
    if position_loss >= full_close:
        issues.append(f"仓位亏损阈值({position_loss*100:.0f}%)应小于全平阈值({full_close*100:.0f}%)")
    else:
        print(f"  [PASS] 风险阈值顺序: 减仓{position_loss*100:.0f}% < 全平{full_close*100:.0f}%")

    # 检查保证金阈值顺序
    margin_call = risk.get("margin_call_threshold", 1.5)
    margin_warn = risk.get("margin_warning_threshold", 2.0)
    if margin_call >= margin_warn:
        issues.append(f"保证金告警阈值顺序错误")
    else:
        print(f"  [PASS] 保证金阈值: 告警<{margin_warn} 强平<{margin_call}")

    if issues:
        for issue in issues:
            print(f"  [FAIL] {issue}")
        return False

    if warnings:
        for warning in warnings:
            print(f"  [WARN] {warning}")

    return len(issues) == 0


def check_data_storage():
    """检查数据存储"""
    print("\n" + "=" * 60)
    print("4. 数据存储检查")
    print("=" * 60)

    # 检查数据库目录
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print(f"  [INFO] 创建数据目录: {data_dir}")

    db_file = data_dir / "trading.db"
    if db_file.exists():
        size = db_file.stat().st_size
        print(f"  [PASS] 数据库存在: {db_file} ({size/1024:.1f} KB)")
    else:
        print(f"  [WARN] 数据库不存在，将在首次运行时创建")

    return True


def check_redis():
    """检查Redis连接（使用配置文件中的参数）"""
    print("\n" + "=" * 60)
    print("5. Redis连接检查")
    print("=" * 60)

    config, _ = _load_config()
    redis_cfg = config.get("redis", {})
    host = redis_cfg.get("host", "localhost")
    port = redis_cfg.get("port", 6379)
    db = redis_cfg.get("db", 0)
    password = redis_cfg.get("password") or None

    print(f"  配置: host={host}, port={port}, db={db}")

    try:
        import redis
        r = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_connect_timeout=3,
        )
        if r.ping():
            print("  [PASS] Redis连接成功")
            # 测试基本操作
            r.setex("_pre_live_check", 10, "ok")
            print("  [PASS] Redis读写正常")
            return True
    except ImportError:
        print("  [WARN] redis模块未安装，将使用备用存储")
        return True
    except Exception as e:
        print(f"  [WARN] Redis连接失败: {e}")
        print("  [INFO] 系统将在无Redis模式下运行，部分功能受限")
        return True

    return True


def check_strategy_params():
    """检查策略参数合理性"""
    print("\n" + "=" * 60)
    print("6. 策略参数检查")
    print("=" * 60)

    config, _ = _load_config()
    strategies = config.get("strategies", {})
    all_ok = True

    # 网格策略
    grid = strategies.get("grid", {})
    if grid.get("enabled"):
        grid_min = grid.get("grid_count_min", 0)
        grid_max = grid.get("grid_count_max", 0)
        if grid_min > grid_max:
            print(f"  [FAIL] 网格策略: grid_count_min({grid_min}) > grid_count_max({grid_max})")
            all_ok = False
        else:
            print(f"  [PASS] 网格策略: {grid_min}-{grid_max}层")

    # 趋势策略
    trend = strategies.get("trend", {})
    if trend.get("enabled"):
        print(f"  [PASS] 趋势策略: 周期{trend.get('timeframe', 'N/A')}")

    #  scalping策略
    scalping = strategies.get("scalping", {})
    if scalping.get("enabled"):
        rsi_low = scalping.get("rsi_oversold", 30)
        rsi_high = scalping.get("rsi_overbought", 70)
        if rsi_low >= rsi_high:
            print(f"  [FAIL] 剥头皮策略: RSI参数顺序错误")
            all_ok = False
        else:
            print(f"  [PASS] 剥头皮策略: RSI {rsi_low}/{rsi_high}")

    # 套利策略
    arb = strategies.get("arbitrage", {})
    if arb.get("enabled"):
        print(f"  [PASS] 套利策略: 杠杆{arb.get('leverage', 'N/A')}x")

    if not any(s.get("enabled", False) for s in strategies.values()):
        print("  [WARN] 没有启用任何策略")

    return all_ok


def main():
    print("OKX量化交易系统 - 实盘前检查")
    print("=" * 60)

    results = []
    results.append(("环境变量", check_env_variables()))
    results.append(("配置文件", check_config()))
    results.append(("风险参数", check_risk_params()))
    results.append(("数据存储", check_data_storage()))
    results.append(("Redis连接", check_redis()))
    results.append(("策略参数", check_strategy_params()))

    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "通过" if passed else "未通过"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("所有关键检查通过，系统可以启动")
        print("建议: 首次运行请在测试网模式验证至少24小时")
    else:
        print("存在未通过的检查项，请修复后再启动")
        sys.exit(1)


if __name__ == "__main__":
    main()
