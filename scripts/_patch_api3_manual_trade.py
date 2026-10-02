"""一次性补记：把 2026-09-14 08:15 手动开空 API3 的真实亏损写入 trades 账本。

背景：该笔手动空头（2431 张，avgPx=0.2471）开仓后 42 秒内被 RiskGate L3 阶梯
减仓自动平掉（7 笔 reduceOnly，clOrdId 均带 okxqtrisk...redu）。真实平仓 pnl≈-2.2404、
手续费≈0.4216，净亏损≈-2.6620 USDT。但 trade_records 记为 ghost_close 且 pnl=0，
trades 表无该笔记录，导致盈亏统计失真。本脚本按 TradeJournal 口径补记一条
strategy_name=manual_override 的 trades 记录。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from sqlalchemy import text
from configs.settings import load_config
from data.sqlite_storage import SQLiteStorage
from core.okx_client import OKXClient

config = load_config()
storage = SQLiteStorage(config)
k = OKXClient(config)

print("db_path =", storage._db_path)

# 查询 API3 杠杆（仅作元数据，不影响 pnl_usdt 净额口径；查不到回退 10）
leverage = 10
try:
    for mgn_mode in ("isolated", "cross"):
        data = k._make_request(
            "GET",
            f"/api/v5/account/leverage-info?instId=API3-USDT-SWAP&mgnMode={mgn_mode}",
        )
        if data and data.get("code") == "0" and data.get("data"):
            lever = data["data"][0].get("lever")
            if lever:
                leverage = int(float(lever))
                print(f"leverage={leverage} (mgnMode={mgn_mode})")
                break
except Exception as e:
    print("leverage query err:", e)
print("使用杠杆 =", leverage)

# ---- 补记字段（口径对齐 core/trade_journal.py 的 TradeRecord）----
trade_id = "manual_override_API3-USDT-SWAP_1789344941132"
symbol = "API3-USDT-SWAP"
strategy_name = "manual_override"
direction = "short"
entry_price = 0.2471
# 平仓加权均价 = sum(avgPx*sz)/sum(sz)，7 笔 reduceOnly
exit_price = 0.2480212
quantity = 2431.0
# OKX fillTime 为毫秒 UTC 时间戳，fromtimestamp 转本地（北京）时间，与 trade_records 一致
entry_time = datetime.fromtimestamp(1789344941132 / 1000).isoformat()
exit_time = datetime.fromtimestamp(1789344983335 / 1000).isoformat()
fees = 0.42161027                      # 开仓 0.12014002 + 平仓 0.30147025
gross_pnl = -2.2404                    # OKX 7 笔平仓 pnl 之和（纯盈亏，未扣手续费）
pnl_usdt = -2.66201027                 # gross_pnl - fees，净额口径
pnl_pct = -0.003728                    # 做空收益率小数 = (entry-exit)/entry
win = 0
entry_signal_type = "manual_override"
exit_reason = "l3_auto_reduce"

print(f"entry_time={entry_time}")
print(f"exit_time={exit_time}")
print(f"pnl_usdt={pnl_usdt:.4f}  fees={fees:.4f}  pnl_pct={pnl_pct:.6f}")

conn = storage.get_connection()

# 幂等检查：已存在则跳过，避免重复补记
existing = conn.execute(
    text("SELECT trade_id FROM trades WHERE trade_id=:t"), {"t": trade_id}
).fetchone()
if existing:
    print("已存在，跳过补记:", existing[0])
else:
    conn.execute(text('''
        INSERT OR REPLACE INTO trades
        (trade_id, symbol, strategy_name, direction, entry_price, exit_price, quantity, leverage,
         entry_time, exit_time, fees, pnl, pnl_usdt, win, entry_signal_type, exit_reason,
         slippage_cost, funding_cost, spread_cost, trace_id, created_at)
        VALUES (:trade_id, :symbol, :strategy_name, :direction, :entry_price, :exit_price, :quantity, :leverage,
         :entry_time, :exit_time, :fees, :pnl, :pnl_usdt, :win, :entry_signal_type, :exit_reason,
         :slippage_cost, :funding_cost, :spread_cost, :trace_id, :created_at)
    '''), {
        "trade_id": trade_id,
        "symbol": symbol,
        "strategy_name": strategy_name,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "leverage": leverage,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "fees": fees,
        "pnl": pnl_pct,
        "pnl_usdt": pnl_usdt,
        "win": win,
        "entry_signal_type": entry_signal_type,
        "exit_reason": exit_reason,
        "slippage_cost": 0.0,
        "funding_cost": 0.0,
        "spread_cost": 0.0,
        "trace_id": "manual_override_api3_patch",
        "created_at": datetime.now().isoformat(),
    })
    conn.commit()
    print("已插入 trade_id =", trade_id)

# 验证
row = conn.execute(text(
    "SELECT trade_id, symbol, strategy_name, direction, pnl_usdt, fees, pnl, exit_reason "
    "FROM trades WHERE trade_id=:t"
), {"t": trade_id}).fetchone()
print("验证:", dict(row._mapping) if row else None)

conn.close()
print("DONE")
