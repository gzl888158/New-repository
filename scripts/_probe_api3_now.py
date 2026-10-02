"""彻底排查：所有表、所有分片里是否还有 API3 的 sync/ghost_close/pnl=0 残留。"""
import sqlite3, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = os.path.join(BASE, "data", "trading.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# 所有表名
tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
print("== all tables ==")
for t in tables:
    print(" ", t)

# 在所有含 symbol 列的表中搜索 API3
print("\n== search API3 across tables ==")
for t in tables:
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
    if "symbol" not in cols:
        continue
    try:
        rows = cur.execute(f"SELECT * FROM {t} WHERE symbol LIKE '%API3%'").fetchall()
    except Exception as e:
        print(f"  [{t}] ERROR: {e}")
        continue
    for r in rows:
        print(f"  [{t}] {dict(r)}")

# strategy_performance 中 sync 相关
print("\n== strategy_performance (sync/manual_override) ==")
if "strategy_performance" in tables:
    cols = [r[1] for r in cur.execute("PRAGMA table_info(strategy_performance)").fetchall()]
    print("cols:", cols)
    try:
        rows = cur.execute(
            "SELECT * FROM strategy_performance WHERE strategy_name IN ('sync','manual_override') "
            "OR symbol LIKE '%API3%'").fetchall()
    except Exception as e:
        print("ERR", e)
        rows = []
    for r in rows:
        print(" ", dict(r))

conn.close()
print("\nDONE")
