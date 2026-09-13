"""
历史数据回填脚本：修复 trade_records 中的污染数据。

修复目标：
1. exit_reason IS NULL（趋势/剥头皮平仓未回写离场原因）→ 从 trades 表(TradeJournal)按 symbol+direction+时间匹配回填。
2. exit_reason = 'reconciled'（网格幽灵对账误标）→ relabel 为 'grid_close'（更诚实的标签）。
3. pnl = 0/null 的 closed 记录 → 优先从 OKX 平仓账单(close bill, pnl!=0)按 symbol+side+时间匹配回填。

用法：
  py -3 scripts/backfill_trade_records.py --dry-run   # 只统计不写库
  py -3 scripts/backfill_trade_records.py --apply      # 实际写入
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db")


def parse_dt(s):
    """兼容 'YYYY-MM-DD HH:MM:SS.ffffff' 和 ISO 'YYYY-MM-DDTHH:MM:SS.ffffff' 两种格式。"""
    if not s:
        return None
    s = str(s).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def norm_side(side):
    """统一方向口径：buy/long → long, sell/short → short。"""
    side = (side or "").strip().lower()
    if side in ("buy", "long"):
        return "long"
    if side in ("sell", "short"):
        return "short"
    return side


def backfill_exit_reason_from_trades(conn, dry_run, time_window_sec=120):
    """从 trades 表回填 exit_reason 与 pnl_usdt，目标为 exit_reason 为 NULL/空串 的 closed 记录。"""
    trades = conn.execute(
        "SELECT symbol, direction, exit_time, exit_reason, pnl_usdt FROM trades "
        "WHERE exit_reason IS NOT NULL AND exit_reason != ''"
    ).fetchall()
    # 解析时间
    trades_parsed = []
    for symbol, direction, exit_time, exit_reason, pnl_usdt in trades:
        dt = parse_dt(exit_time)
        if dt:
            trades_parsed.append((symbol, norm_side(direction), dt, exit_reason, pnl_usdt))

    targets = conn.execute(
        "SELECT id, symbol, side, close_time FROM trade_records "
        "WHERE status='closed' AND (exit_reason IS NULL OR exit_reason='')"
    ).fetchall()

    matched = 0
    unmatched = 0
    updates = []  # (id, exit_reason, pnl)
    for rec_id, symbol, side, close_time in targets:
        cdt = parse_dt(close_time)
        if not cdt:
            unmatched += 1
            continue
        side = norm_side(side)
        best = None
        best_diff = timedelta(seconds=time_window_sec + 1)
        for t_symbol, t_dir, t_dt, t_reason, t_pnl in trades_parsed:
            if t_symbol != symbol or t_dir != side:
                continue
            diff = abs(cdt - t_dt)
            if diff <= timedelta(seconds=time_window_sec) and diff < best_diff:
                best = (t_reason, t_pnl)
                best_diff = diff
        if best:
            updates.append((rec_id, best[0], best[1]))
            matched += 1
        else:
            unmatched += 1

    if not dry_run:
        for rec_id, reason, pnl in updates:
            if pnl is not None:
                conn.execute("UPDATE trade_records SET exit_reason=?, pnl=? WHERE id=?",
                             (reason, float(pnl), rec_id))
            else:
                conn.execute("UPDATE trade_records SET exit_reason=? WHERE id=?", (reason, rec_id))

    return {"matched": matched, "unmatched": unmatched, "updates": updates}


def backfill_empty_exit_reason(conn, dry_run):
    """兜底回填仍为 NULL/空串 的 exit_reason（trades 表匹配不到的记录）。

    口径：pnl != 0 视为真实平仓但离场原因丢失 → 'close'；pnl = 0/null 视为幽灵对账 → 'ghost_close'。
    """
    rows = conn.execute(
        "SELECT id, pnl FROM trade_records WHERE status='closed' AND (exit_reason IS NULL OR exit_reason='')"
    ).fetchall()

    n_close = 0
    n_ghost = 0
    for rec_id, pnl in rows:
        reason = "ghost_close" if (pnl is None or float(pnl) == 0) else "close"
        if not dry_run:
            conn.execute("UPDATE trade_records SET exit_reason=? WHERE id=?", (reason, rec_id))
        if reason == "ghost_close":
            n_ghost += 1
        else:
            n_close += 1

    return {"close": n_close, "ghost_close": n_ghost, "total": len(rows)}


def relabel_grid_reconciled(conn, dry_run):
    """把幽灵持仓对账误标为 'reconciled' 的记录 relabel 为 'ghost_close'。

    'reconciled' 实际含义是“本地有持仓、交易所无持仓”的幽灵持仓被清理，
    并非真实成交平仓，改为 'ghost_close' 更诚实。
    """
    rows = conn.execute(
        "SELECT COUNT(*) FROM trade_records WHERE status='closed' AND exit_reason='reconciled'"
    ).fetchone()[0]
    if not dry_run:
        conn.execute(
            "UPDATE trade_records SET exit_reason='ghost_close' WHERE status='closed' AND exit_reason='reconciled'"
        )
    return {"count": rows}


def relabel_ghost_close_with_pnl(conn, dry_run):
    """标签治理：exit_reason 为 ghost_close/ghost_cleanup 但 pnl 已非零的 closed 记录，
    说明是「成交回执丢失的真实平仓」（历史运行中已通过 OKX 账单回填了 pnl），
    但 recovered_close 重标逻辑是在回填之后才加入 pnl_reconciler 的，导致这些存量记录
    永远卡在 ghost_close 标签上。现一次性重标为 recovered_close，恢复绩效归因。

    真正的幽灵关闭（pnl=0/null，任何账单都匹配不到）保持 ghost_close 不变。
    """
    rows = conn.execute(
        "SELECT COUNT(*) FROM trade_records WHERE status='closed' "
        "AND exit_reason IN ('ghost_close','ghost_cleanup') AND pnl IS NOT NULL AND pnl != 0"
    ).fetchone()[0]
    if not dry_run:
        conn.execute(
            "UPDATE trade_records SET exit_reason='recovered_close' WHERE status='closed' "
            "AND exit_reason IN ('ghost_close','ghost_cleanup') AND pnl IS NOT NULL AND pnl != 0"
        )
    return {"count": rows}


def backfill_pnl_from_okx_bills(conn, dry_run, time_window_sec=120):
    """从 OKX 平仓账单回填 pnl，目标为 closed 且 pnl=0/null 的记录。

    平仓账单特征：pnl != 0，posSide 表示被平方向。按 symbol+side+时间窗口匹配（最近优先）。
    """
    try:
        bills = _fetch_close_bills()
    except Exception as e:
        return {"error": f"OKX API 不可用: {e}", "matched": 0, "unmatched": 0}

    if not bills:
        return {"error": "未获取到平仓账单", "matched": 0, "unmatched": 0}

    # bills: (symbol, side, ts_datetime, pnl, fee, fill_px)
    targets = conn.execute(
        "SELECT id, symbol, side, close_time, price, quantity FROM trade_records "
        "WHERE status='closed' AND (pnl IS NULL OR pnl=0)"
    ).fetchall()

    used_bills = set()
    matched = 0
    unmatched = 0
    updates = []
    for rec_id, symbol, side, close_time, price, quantity in targets:
        cdt = parse_dt(close_time)
        side = (side or "").lower()
        if not cdt:
            unmatched += 1
            continue
        best = None
        best_diff = timedelta(seconds=time_window_sec + 1)
        best_idx = -1
        for i, (b_symbol, b_side, b_dt, b_pnl, b_fee, b_px) in enumerate(bills):
            if b_symbol != symbol or b_side != side:
                continue
            if i in used_bills:
                continue
            diff = abs(cdt - b_dt)
            if diff <= timedelta(seconds=time_window_sec) and diff < best_diff:
                best = (b_pnl, b_fee, b_px)
                best_diff = diff
                best_idx = i
        if best:
            pnl, fee, px = best
            updates.append((rec_id, pnl, fee, px, price, quantity))
            used_bills.add(best_idx)
            matched += 1
        else:
            unmatched += 1

    if not dry_run:
        for rec_id, pnl, fee, px, price, quantity in updates:
            sets = ["pnl=?", "fees=?"]
            args = [float(pnl), float(fee)]
            if px and px > 0:
                sets.append("filled_price=?")
                args.append(float(px))
            args.append(rec_id)
            conn.execute(f"UPDATE trade_records SET {', '.join(sets)} WHERE id=?", args)

    return {"matched": matched, "unmatched": unmatched, "updates": updates, "bills": len(bills)}


def backfill_pnl_and_fill_px_from_trades(conn, dry_run, time_window_sec=120):
    """从权威 trades 表回填 pnl / filled_price / exit_reason，目标为 pnl 缺失(0/null) 或
    filled_price=0 的 closed 记录。

    依据：TradeJournal 平仓时在同一时刻写入 trades.exit_time 与 trade_records.close_time
    （均为 exit_dt = datetime.now()），因此 symbol+direction+时间窗口匹配极可靠，且 trades.pnl_usdt
    为权威净额。相比直接匹配 OKX 账单（需代理联网 + bill 时间对不上的 ghost_close 记录），
    这条路径是「journal 已入账但未回写 trade_records」场景的最优解。
    """
    trades = conn.execute(
        "SELECT symbol, direction, exit_time, exit_reason, pnl_usdt, exit_price FROM trades "
        "WHERE pnl_usdt IS NOT NULL"
    ).fetchall()
    trades_parsed = []
    for symbol, direction, exit_time, exit_reason, pnl_usdt, exit_price in trades:
        dt = parse_dt(exit_time)
        if dt:
            trades_parsed.append((symbol, norm_side(direction), dt, exit_reason, pnl_usdt, exit_price))

    targets = conn.execute(
        "SELECT id, symbol, side, close_time, exit_reason, pnl, filled_price FROM trade_records "
        "WHERE status='closed' AND (pnl IS NULL OR pnl=0 OR filled_price IS NULL OR filled_price=0)"
    ).fetchall()

    matched = 0
    unmatched = 0
    updates = []  # (id, pnl, filled_price, exit_reason)
    for rec_id, symbol, side, close_time, rec_reason, pnl, filled_price in targets:
        cdt = parse_dt(close_time)
        if not cdt:
            unmatched += 1
            continue
        side = norm_side(side)
        best = None
        best_diff = timedelta(seconds=time_window_sec + 1)
        for t_symbol, t_dir, t_dt, t_reason, t_pnl, t_px in trades_parsed:
            if t_symbol != symbol or t_dir != side:
                continue
            diff = abs(cdt - t_dt)
            if diff <= timedelta(seconds=time_window_sec) and diff < best_diff:
                best = (t_pnl, t_px, t_reason)
                best_diff = diff
        if best:
            t_pnl, t_px, t_reason = best
            updates.append((rec_id, t_pnl, t_px, t_reason))
            matched += 1
        else:
            unmatched += 1

    if not dry_run:
        for rec_id, t_pnl, t_px, t_reason in updates:
            sets = []
            args = []
            if t_pnl is not None:
                sets.append("pnl=?")
                args.append(float(t_pnl))
            if t_px and float(t_px) > 0:
                sets.append("filled_price=?")
                args.append(float(t_px))
            if t_reason:
                sets.append("exit_reason=?")
                args.append(t_reason)
            if sets:
                args.append(rec_id)
                conn.execute(f"UPDATE trade_records SET {', '.join(sets)} WHERE id=?", args)

    return {"matched": matched, "unmatched": unmatched, "updates": updates}


def _fetch_close_bills():
    """拉取 OKX 平仓账单（pnl != 0 且 subType 为平多/平空）。

    OKX 账单 type=2 的交易用 subType 区分方向：
      5/9/11  → 平多（close long）
      6/10/12 → 平空（close short）
    返回 [(symbol, side, dt, pnl, fee, fill_px), ...]，side 为 'long'/'short'。
    """
    import base64
    import hashlib
    import hmac

    import requests
    from dotenv import load_dotenv

    CLOSE_LONG_SUBTYPES = {5, 9, 11}
    CLOSE_SHORT_SUBTYPES = {6, 10, 12}

    load_dotenv()
    api_key = os.getenv("OKX_API_KEY")
    secret = os.getenv("OKX_SECRET_KEY")
    passphrase = os.getenv("OKX_PASSPHRASE")
    proxies = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}

    def okx_get(path):
        ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        msg = ts + "GET" + path
        sign = base64.b64encode(
            hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()
        headers = {
            "OK-ACCESS-KEY": api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json",
        }
        return requests.get("https://www.okx.com" + path, headers=headers, proxies=proxies, timeout=20).json()

    result = []
    # 分页拉取交易账单（type=2 交易，含平仓 pnl）
    before = None
    for _ in range(100):  # 最多 100 页，覆盖更久历史
        path = "/api/v5/account/bills?instType=SWAP&limit=100"
        if before:
            path += f"&before={before}"
        data = okx_get(path)
        if not data or data.get("code") != "0":
            break
        page = data.get("data", [])
        if not page:
            break
        for b in page:
            pnl = float(b.get("pnl", "0") or 0)
            if pnl == 0:
                continue
            subtype = int(b.get("subType", "0") or 0)
            if subtype in CLOSE_LONG_SUBTYPES:
                side = "long"
            elif subtype in CLOSE_SHORT_SUBTYPES:
                side = "short"
            else:
                continue  # 开仓或非平仓账单，跳过
            ts = b.get("ts", "")
            dt = datetime.fromtimestamp(int(ts) / 1000) if ts else None
            if not dt:
                continue
            fee = abs(float(b.get("fee", "0") or 0))
            px = float(b.get("px", "0") or 0)
            result.append((b.get("instId", ""), side, dt, pnl, fee, px))
        oldest_ts = min(int(b.get("ts", "0")) for b in page)
        before = str(oldest_ts)
        if len(page) < 100:
            break
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只统计不写库")
    ap.add_argument("--apply", action="store_true", help="实际写入")
    ap.add_argument("--skip-bills", action="store_true", help="跳过 OKX 账单回填（离线）")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        print("请指定 --dry-run 或 --apply")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")

    print("=" * 60)
    print("历史数据回填（trade_records）")
    print("=" * 60)

    # 1. exit_reason 回填（从 trades 表）
    r1 = backfill_exit_reason_from_trades(conn, args.dry_run)
    print(f"\n[1] exit_reason 回填（来自 trades 表）: 匹配 {r1['matched']}, 未匹配 {r1['unmatched']}")

    # 2. 网格 relabel
    r2 = relabel_grid_reconciled(conn, args.dry_run)
    print(f"[2] 'reconciled' → 'ghost_close' relabel: {r2['count']} 条")

    # 3. 兜底回填仍为 NULL/空串 的 exit_reason
    r3 = backfill_empty_exit_reason(conn, args.dry_run)
    print(f"[3] 兜底回填空 exit_reason: close {r3['close']} 条, ghost_close {r3['ghost_close']} 条 (共 {r3['total']})")

    # 4. pnl/filled_price/exit_reason 回填（从权威 trades 表，优先）
    r4 = backfill_pnl_and_fill_px_from_trades(conn, args.dry_run)
    print(f"[4] pnl/filled_price 回填（权威 trades 表）: 匹配 {r4['matched']}, 未匹配 {r4['unmatched']}")

    # 5. pnl 回填（从 OKX 账单，兜底 ghost_close 等无 trades 条目场景）
    if not args.skip_bills:
        r5 = backfill_pnl_from_okx_bills(conn, args.dry_run)
        if "error" in r5:
            print(f"[5] pnl 回填（OKX 账单）: 跳过 — {r5['error']}")
        else:
            print(f"[5] pnl 回填（OKX 账单）: 匹配 {r5['matched']}, 未匹配 {r5['unmatched']} (账单 {r5['bills']} 条)")
    else:
        print("[5] pnl 回填：跳过（--skip-bills）")

    # 6. 标签治理：pnl 非零的 ghost_close → recovered_close（必须放在第 4/5 步回填之后，
    #    确保「回填出 pnl 的 ghost_close」也被一并进行 recovered_close 重标）
    r6 = relabel_ghost_close_with_pnl(conn, args.dry_run)
    print(f"[6] ghost_close+pnl≠0 → recovered_close 重标: {r6['count']} 条")

    if args.dry_run:
        print("\n[dry-run] 未写库。确认无误后加 --apply 执行。")
        conn.rollback()
    else:
        conn.commit()
        print("\n[apply] 已提交写入。")

    conn.close()


if __name__ == "__main__":
    main()
