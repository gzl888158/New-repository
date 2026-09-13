# -*- coding: utf-8 -*-
"""核查 7-21 / 7-24 为何被判定为出入金日"""
import sqlite3
from datetime import datetime, timedelta

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"


def parse_ts(ts):
    s = str(ts).replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def is_flow(pe, ce, pa, ca, pu, cu):
    d_eq = ce - pe
    d_av = ca - pa
    d_up = cu - pu
    if abs(d_eq) < 3.0:
        return False
    tol = max(1.0, 0.20 * abs(d_eq))
    if abs(d_av - d_eq) > tol:
        return False
    if abs(d_up) > tol:
        return False
    return True


def scan(day):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, total_equity, available_balance, used_margin, unrealized_pnl "
        "FROM account_history WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
        (day + " 00:00:00", day + " 23:59:59"),
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
    print(f"\n=== {day} ({len(parsed)} 点) ===")
    for i in range(1, len(parsed)):
        pt, pe, pa, pm, pu = parsed[i-1]
        ct, ce, ca, cm, cu = parsed[i]
        d_eq = ce - pe
        d_av = ca - pa
        d_up = cu - pu
        dt = (ct - pt).total_seconds()
        if abs(d_eq) >= 3.0:
            f = is_flow(pe, ce, pa, ca, pu, cu)
            print(f"[{'FLOW' if f else 'trade'}] {pt} -> {ct} (dt={dt:.0f}s) "
                  f"d_eq={d_eq:+.2f} d_av={d_av:+.2f} d_um={cm-pm:+.2f} d_up={d_up:+.2f}")


scan("2026-07-21")
scan("2026-07-24")
