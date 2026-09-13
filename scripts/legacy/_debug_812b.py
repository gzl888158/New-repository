# -*- coding: utf-8 -*-
"""扫描 8-12 全天，用 d_av≈d_eq 签名识别真实出入金"""
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
        "SELECT timestamp, total_equity, available_balance, used_margin, unrealized_pnl "
        "FROM account_history WHERE timestamp >= '2026-08-12 00:00:00' "
        "AND timestamp <= '2026-08-12 23:59:59' ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()

    parsed = []
    for r in rows:
        t = parse_ts(r["timestamp"])
        if t is None:
            continue
        parsed.append((t, float(r["total_equity"] or 0), float(r["available_balance"] or 0),
                       float(r["used_margin"] or 0), float(r["unrealized_pnl"] or 0)))
    parsed.sort(key=lambda x: x[0])

    print(f"8-12 total points: {len(parsed)}")
    print(f"first eq={parsed[0][1]:.4f}, last eq={parsed[-1][1]:.4f}")
    print()

    print("=== 8-12 出入金检测（d_eq>=3 且 |d_av-d_eq|<=0.2*|d_eq|） ===")
    for i in range(1, len(parsed)):
        pt, pe, pa, pm, pu = parsed[i-1]
        ct, ce, ca, cm, cu = parsed[i]
        d_eq = ce - pe
        d_av = ca - pa
        d_um = cm - pm
        d_up = cu - pu
        dt = (ct - pt).total_seconds()
        if abs(d_eq) >= 3.0 and abs(d_av - d_eq) <= max(1.0, 0.20 * abs(d_eq)):
            print(f"CAPITAL FLOW {pt} -> {ct} (dt={dt:.0f}s) d_eq={d_eq:+.2f} d_av={d_av:+.2f} "
                  f"d_um={d_um:+.2f} d_up={d_up:+.2f}")

    print()
    print("=== 8-12 所有 d_eq>=3 的事件（含方向/幅度） ===")
    for i in range(1, len(parsed)):
        pt, pe, pa, pm, pu = parsed[i-1]
        ct, ce, ca, cm, cu = parsed[i]
        d_eq = ce - pe
        d_av = ca - pa
        d_um = cm - pm
        d_up = cu - pu
        dt = (ct - pt).total_seconds()
        if abs(d_eq) >= 3.0:
            flag = "CAP" if abs(d_av - d_eq) <= max(1.0, 0.20 * abs(d_eq)) else "TRADE"
            print(f"[{flag}] {pt} -> {ct} (dt={dt:.0f}s) d_eq={d_eq:+.2f} d_av={d_av:+.2f} "
                  f"d_um={d_um:+.2f} d_up={d_up:+.2f}")


if __name__ == "__main__":
    main()
