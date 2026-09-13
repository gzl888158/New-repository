# -*- coding: utf-8 -*-
"""查看 8-12 23:00-23:30 分钟级数据，确认入金精确签名"""
import sqlite3
from datetime import datetime

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"


def parse_ts(ts):
    s = str(ts).replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, total_equity, available_balance, used_margin, unrealized_pnl, margin_rate "
        "FROM account_history WHERE timestamp >= '2026-08-12 22:55:00' "
        "AND timestamp <= '2026-08-12 23:40:00' ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()

    parsed = [(parse_ts(r["timestamp"]), r) for r in rows]
    parsed = [(t, r) for t, r in parsed if t is not None]
    parsed.sort(key=lambda x: x[0])

    print(f"count={len(parsed)}")
    for t, r in parsed:
        print(f"{t}  eq={float(r['total_equity'] or 0):.4f}  av={float(r['available_balance'] or 0):.4f}  "
              f"um={float(r['used_margin'] or 0):.4f}  up={float(r['unrealized_pnl'] or 0):.4f}  mr={r['margin_rate']}")


if __name__ == "__main__":
    main()
