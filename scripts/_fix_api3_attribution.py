"""一次性修正：API3 手动仓双账本归因与口径对齐。

发现的问题：
1. trade_records 那条 API3 记录 strategy_name='sync'（应 manual_override），
   leverage=10（真实 3）、margin=60.07（基于错杠杆）、filled_price=0.0（bills fillPx=null 未回填）、
   pnl_percent 基于错 margin 算出。
2. trades 表我上一轮补记的 fees=0.4216、pnl_usdt=-2.6620 口径错误——多扣了开仓手续费 0.12014。
   TradeJournal 口径：pnl_usdt = 毛盈亏 - 平仓手续费（开仓手续费在开仓时已单独从权益扣除，
   不进 trades）。正确值 = -2.2404 - 0.30147025 = -2.54187025，与 reconciler 回填的 trade_records.pnl 一致。
本脚本把两账本统一到 -2.54187025，并把归因/杠杆/保证金/平仓价修正到真实值。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import text
from configs.settings import load_config
from data.sqlite_storage import SQLiteStorage

config = load_config()
storage = SQLiteStorage(config)
conn = storage.get_connection()

TR_REC_ID = "7426d663-bca3-4cbe-aea4-808ca267d79a"
TRADES_ID = "manual_override_API3-USDT-SWAP_1789344941132"

# ---- 真实值 ----
entry_price = 0.2471
quantity = 2431.0
real_leverage = 3
real_margin = entry_price * quantity / real_leverage          # 200.233...
fill_px = 0.2480                                              # 7 笔 reduceOnly 加权均价
gross_pnl = -2.2404                                           # OKX 7 笔平仓 pnl 之和
close_fee = 0.30147025                                        # 平仓手续费（开仓+平仓中仅平仓部分）
net_pnl = gross_pnl - close_fee                               # -2.54187025
pnl_percent = (net_pnl / real_margin * 100) if real_margin else 0.0

print(f"real_margin={real_margin:.4f}  net_pnl={net_pnl:.8f}  pnl_percent={pnl_percent:.6f}%")

# ---- 1. 修正 trade_records（归因 + 杠杆 + 保证金 + 平仓价 + 保证金收益率）----
conn.execute(text(
    "UPDATE trade_records SET strategy_name=:s, leverage=:lev, margin=:m, "
    "filled_price=:fp, pnl_percent=:pp, signal_type=:st WHERE id=:id"
), {
    "s": "manual_override",
    "lev": real_leverage,
    "m": real_margin,
    "fp": fill_px,
    "pp": pnl_percent,
    "st": "manual_override",
    "id": TR_REC_ID,
})

# ---- 2. 修正 trades（口径：fees=平仓手续费，pnl_usdt=毛盈亏-平仓手续费）----
conn.execute(text(
    "UPDATE trades SET fees=:f, pnl_usdt=:p WHERE trade_id=:id"
), {"f": close_fee, "p": net_pnl, "id": TRADES_ID})

conn.commit()

# ---- 验证 ----
r1 = conn.execute(text(
    "SELECT strategy_name, leverage, margin, filled_price, pnl, pnl_percent, fees, exit_reason "
    "FROM trade_records WHERE id=:id"
), {"id": TR_REC_ID}).fetchone()
print("\n=== trade_records 修正后 ===")
print(dict(r1._mapping))

r2 = conn.execute(text(
    "SELECT strategy_name, direction, leverage, fees, pnl, pnl_usdt, exit_reason "
    "FROM trades WHERE trade_id=:id"
), {"id": TRADES_ID}).fetchone()
print("\n=== trades 修正后 ===")
print(dict(r2._mapping))

conn.close()
print("\nDONE")
