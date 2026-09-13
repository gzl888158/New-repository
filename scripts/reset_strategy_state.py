"""
策略状态重置脚本
清空所有策略的虚拟持仓记录，与交易所实际状态对齐。
运行前会自动备份当前状态文件 (.bak)。
"""
import json
import os
import shutil
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STRATEGY_STATE_DIR = os.path.join(BASE_DIR, "strategy_state")
NOW = datetime.now()
NOW_ISO = NOW.isoformat()
NOW_TS = NOW.timestamp()


def backup_file(filepath: str):
    """备份文件为 .bak"""
    bak_path = filepath + ".bak"
    if os.path.exists(filepath):
        shutil.copy2(filepath, bak_path)
        print(f"  [BACKUP] {os.path.basename(filepath)} -> {os.path.basename(bak_path)}")
    else:
        print(f"  [SKIP] {os.path.basename(filepath)} not found")


def write_json(filepath: str, data: dict):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [RESET] {os.path.basename(filepath)}")


def reset_grid():
    """重置 grid 策略状态"""
    filepath = os.path.join(STRATEGY_STATE_DIR, "grid.json")
    backup_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 重置所有网格的 filled 状态为 false
    for symbol, grids in data.get("_grids", {}).items():
        for g in grids:
            g["filled"] = False
            g["quantity"] = 0

    # 清空待处理订单
    data["_pending_orders"] = {}
    # 清空止损订单
    data["_stop_loss_orders"] = {}
    # 重置持仓状态为默认 sell
    for k in data.get("_position_state", {}):
        data["_position_state"][k] = "sell"
    # 重置马丁格尔状态
    for k in data.get("_martingale_state", {}):
        data["_martingale_state"][k] = 0
    # 重置趋势模式
    for k in data.get("_trend_mode", {}):
        data["_trend_mode"][k] = False
    # 清空极端波动
    data["_extreme_vol_until"] = {}
    # 清空网格重试计数
    data["_grid_retry_count"] = {}
    # 清空网格回滚冷却
    data["_grid_rollback_cooldown"] = {}

    data["_meta"] = {
        "strategy": "grid",
        "saved_at": NOW_ISO,
        "timestamp": NOW_TS,
        "version": 1,
        "reset_at": NOW_ISO,
    }

    write_json(filepath, data)


def reset_scalping():
    """重置 scalping 策略状态"""
    filepath = os.path.join(STRATEGY_STATE_DIR, "scalping.json")
    backup_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 清空所有虚拟持仓
    data["positions"] = {}
    # 重置日统计
    data["daily_pnl"] = 0.0
    data["daily_start_equity"] = data.get("daily_start_equity", 10.0)
    data["max_daily_equity"] = data.get("daily_start_equity", 10.0)
    data["current_drawdown"] = 0.0
    data["daily_reset_date"] = NOW.strftime("%Y-%m-%d")
    # 清空信号类型表现
    for k in data.get("signal_type_performance", {}):
        data["signal_type_performance"][k] = {
            "wins": 0, "losses": 0, "total_pnl": 0,
            "total_profit": 0, "total_loss": 0
        }
    # 清空持仓信号类型映射
    data["position_signal_type"] = {}

    data["_meta"] = {
        "strategy": "scalping",
        "saved_at": NOW_ISO,
        "timestamp": NOW_TS,
        "version": 1,
        "reset_at": NOW_ISO,
    }

    write_json(filepath, data)


def reset_trend():
    """重置 trend 策略状态"""
    filepath = os.path.join(STRATEGY_STATE_DIR, "trend.json")
    backup_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 清空所有虚拟持仓
    data["_position_state"] = {}
    # 清空最后信号时间
    data["_last_signal_time"] = {}

    data["_meta"] = {
        "strategy": "trend",
        "saved_at": NOW_ISO,
        "timestamp": NOW_TS,
        "version": 1,
        "reset_at": NOW_ISO,
    }

    write_json(filepath, data)


def reset_arbitrage():
    """重置 arbitrage 策略状态"""
    filepath = os.path.join(STRATEGY_STATE_DIR, "arbitrage.json")
    backup_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 清空所有虚拟持仓
    data["positions"] = {}
    # 清空对冲持仓
    data["hedge_positions"] = {}
    # 重置费率收入
    data["total_fee_earned"] = {}
    # 重置套利表现
    data["arbitrage_performance"] = {
        "funding": {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0},
        "basis": {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0},
        "correlation": {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0},
    }

    data["_meta"] = {
        "strategy": "arbitrage",
        "saved_at": NOW_ISO,
        "timestamp": NOW_TS,
        "version": 1,
        "reset_at": NOW_ISO,
    }

    write_json(filepath, data)


def reset_spot_grid():
    """重置 spot_grid 策略状态"""
    filepath = os.path.join(STRATEGY_STATE_DIR, "spot_grid.json")
    backup_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 清空活跃持仓
    data["active_positions"] = {}
    # 清空上次调整时间
    data["last_adjust_time"] = {}
    # 清空网格表现
    data["grid_performance"] = {}

    data["_meta"] = {
        "strategy": "spot_grid",
        "saved_at": NOW_ISO,
        "timestamp": NOW_TS,
        "version": 1,
        "reset_at": NOW_ISO,
    }

    write_json(filepath, data)


def reset_spot_martingale():
    """重置 spot_martingale 策略状态"""
    filepath = os.path.join(STRATEGY_STATE_DIR, "spot_martingale.json")
    backup_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 清空活跃持仓
    data["active_positions"] = {}
    # 重置日交易计数
    data["daily_trades"] = {}
    data["daily_reset_date"] = NOW.strftime("%Y-%m-%d")
    # 清空冷却期
    data["cooldown_until"] = {}

    data["_meta"] = {
        "strategy": "spot_martingale",
        "saved_at": NOW_ISO,
        "timestamp": NOW_TS,
        "version": 1,
        "reset_at": NOW_ISO,
    }

    write_json(filepath, data)


def reset_global_state():
    """重置全局状态"""
    filepath = os.path.join(BASE_DIR, "global_state.json")
    backup_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["state"]["trading.total_positions"] = 0
    data["state"]["trading.active_symbols"] = []
    data["state"]["execution.queue_size"] = 0
    data["state"]["execution.order_count_today"] = 0
    data["timestamp"] = NOW_ISO

    write_json(filepath, data)


def reset_risk_status():
    """重置风控状态（保留资金数据，只重置风控标记）"""
    filepath = os.path.join(BASE_DIR, "risk_status.json")
    backup_file(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 重置熔断状态
    data["is_paused"] = False
    data["pause_reason"] = None
    data["pause_time"] = None
    data["last_resume_time"] = NOW_ISO
    data["process_running"] = True
    data["tier_triggered"] = {"1": False, "2": False, "3": False}
    data["consecutive_losses"] = 0
    data["last_update"] = NOW_ISO

    # 重置熔断冷却
    for level in data.get("circuit_breakers", {}).get("cooldown", {}):
        data["circuit_breakers"]["cooldown"][level] = {
            "active": False,
            "remaining_seconds": 0
        }

    write_json(filepath, data)


def reset_all():
    print("=" * 60)
    print("策略状态重置工具")
    print(f"时间: {NOW_ISO}")
    print("=" * 60)

    print("\n[1/8] 备份 & 重置 grid.json ...")
    reset_grid()

    print("\n[2/8] 备份 & 重置 scalping.json ...")
    reset_scalping()

    print("\n[3/8] 备份 & 重置 trend.json ...")
    reset_trend()

    print("\n[4/8] 备份 & 重置 arbitrage.json ...")
    reset_arbitrage()

    print("\n[5/8] 备份 & 重置 spot_grid.json ...")
    reset_spot_grid()

    print("\n[6/8] 备份 & 重置 spot_martingale.json ...")
    reset_spot_martingale()

    print("\n[7/8] 备份 & 重置 global_state.json ...")
    reset_global_state()

    print("\n[8/8] 备份 & 重置 risk_status.json ...")
    reset_risk_status()

    print("\n" + "=" * 60)
    print("重置完成。所有原始文件已备份为 .bak")
    print("请重启交易系统使更改生效。")
    print("=" * 60)


if __name__ == "__main__":
    reset_all()