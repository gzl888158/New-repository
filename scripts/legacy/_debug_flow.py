# -*- coding: utf-8 -*-
"""分析分钟级 account_history 以设计出入金检测签名"""
import sqlite3
from datetime import datetime, timedelta

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"


def parse_ts(ts):
    s = str(ts)
    s = s.replace("T", " ")
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute(
        "SELECT timestamp, total_equity, available_balance, used_margin, unrealized_pnl "
        "FROM account_history WHERE timestamp >= ? ORDER BY timestamp ASC",
        (cutoff,),
    ).fetchall()
    conn.close()

    parsed = []
    for r in rows:
        t = parse_ts(r["timestamp"])
        if t is None:
            continue
        eq = float(r["total_equity"] or 0)
        av = float(r["available_balance"] or 0)
        um = float(r["used_margin"] or 0)
        up = float(r["unrealized_pnl"] or 0)
        parsed.append((t, eq, av, um, up))

    parsed.sort(key=lambda x: x[0])
    print(f"total parsed points: {len(parsed)}")
    print(f"time range: {parsed[0][0]} -> {parsed[-1][0]}")
    print()

    # 扫描相邻点跳变
    print("=== 候选出入金事件（eq 相对跳变>30% 或 绝对>5） ===")
    for i in range(1, len(parsed)):
        pt, pe, pa, pm, pu = parsed[i - 1]
        ct, ce, ca, cm, cu = parsed[i]
        d_eq = ce - pe
        d_av = ca - pa
        d_up = cu - pu
        rel_eq = d_eq / pe if pe > 0 else 0
        rel_av = d_av / pa if pa > 0 else 0
        # 时间间隔（秒）过滤：只关注相邻分钟内
        dt = (ct - pt).total_seconds()
        big_eq = abs(rel_eq) > 0.30 or abs(d_eq) > 5.0
        if big_eq:
            print(f"{pt} -> {ct} (dt={dt:.0f}s) eq {pe:.4f}->{ce:.4f} ({rel_eq*100:+.1f}%, {d_eq:+.2f}) "
                  f"av {pa:.4f}->{ca:.4f} ({rel_av*100:+.1f}%, {d_av:+.2f}) um {pm:.2f}->{cm:.2f} up {pu:.4f}->{cu:.4f}")

    print()
    print("=== 候选出入金（av 绝对跳变>=5 且 eq 相对>15%） ===")
    for i in range(1, len(parsed)):
        pt, pe, pa, pm, pu = parsed[i - 1]
        ct, ce, ca, cm, cu = parsed[i]
        d_eq = ce - pe
        d_av = ca - pa
        d_up = cu - pu
        rel_eq = d_eq / pe if pe > 0 else 0
        dt = (ct - pt).total_seconds()
        if abs(d_av) >= 5.0 and abs(rel_eq) > 0.15:
            print(f"{pt} -> {ct} (dt={dt:.0f}s) eq {pe:.4f}->{ce:.4f} ({rel_eq*100:+.1f}%, {d_eq:+.2f}) "
                  f"av {pa:.4f}->{ca:.4f} ({d_av:+.2f}) um {pm:.2f}->{cm:.2f} up {pu:.4f}->{cu:.4f}")

    print()
    print("=== 所有 av 绝对跳变>=5 的记录（判断是否全是出入金） ===")
    cnt = 0
    for i in range(1, len(parsed)):
        pt, pe, pa, pm, pu = parsed[i - 1]
        ct, ce, ca, cm, cu = parsed[i]
        d_eq = ce - pe
        d_av = ca - pa
        rel_eq = d_eq / pe if pe > 0 else 0
        dt = (ct - pt).total_seconds()
        if abs(d_av) >= 5.0:
            cnt += 1
            if cnt <= 40:
                print(f"{pt} -> {ct} (dt={dt:.0f}s) eq {pe:.4f}->{ce:.4f} ({rel_eq*100:+.1f}%) "
                      f"av {pa:.4f}->{ca:.4f} ({d_av:+.2f}) um {pm:.2f}->{cm:.2f} up {pu:.4f}->{cu:.4f}")
    print(f"total av|d|>=5 events: {cnt}")


if __name__ == "__main__":
    main()
