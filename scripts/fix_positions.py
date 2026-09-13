"""修复数据库持仓与OKX实际持仓的一致性，并回填历史盈亏"""
import sqlite3
import os
import hmac
import hashlib
import base64
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OKX_API_KEY')
secret = os.getenv('OKX_SECRET_KEY')
passphrase = os.getenv('OKX_PASSPHRASE')
proxies = {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}


def okx_request(method, path, body=''):
    ts = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())
    msg = ts + method + path + body
    sign = base64.b64encode(
        hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()
    headers = {
        'OK-ACCESS-KEY': api_key,
        'OK-ACCESS-SIGN': sign,
        'OK-ACCESS-TIMESTAMP': ts,
        'OK-ACCESS-PASSPHRASE': passphrase,
        'Content-Type': 'application/json'
    }
    url = 'https://www.okx.com' + path
    if method == 'GET':
        r = requests.get(url, headers=headers, proxies=proxies, timeout=15)
    else:
        r = requests.post(url, headers=headers, data=body, proxies=proxies, timeout=15)
    return r.json()


def get_okx_positions():
    """获取OKX实际持仓"""
    pos = okx_request('GET', '/api/v5/account/positions')
    if pos.get('code') == '0':
        return pos.get('data', [])
    return []


def get_okx_bills(limit=100):
    """获取OKX历史账单（成交记录）"""
    bills = okx_request('GET', f'/api/v5/account/bills?instType=SWAP&limit={limit}')
    if bills.get('code') == '0':
        return bills.get('data', [])
    return []


def sync_positions():
    """同步数据库持仓与OKX实际持仓"""
    print("=" * 80)
    print("=== 步骤1: 同步数据库持仓与OKX实际持仓 ===")
    print("=" * 80)

    okx_positions = get_okx_positions()
    print(f"OKX实际持仓数量: {len(okx_positions)}")

    # 构建OKX持仓键集合
    okx_pos_keys = set()
    okx_pos_map = {}
    for p in okx_positions:
        symbol = p.get('instId', '')
        pos_side = p.get('posSide', 'net')
        pos_qty = float(p.get('pos', 0))
        if abs(pos_qty) > 0:
            key = f"{symbol}:{pos_side}"
            okx_pos_keys.add(key)
            okx_pos_map[key] = p
            print(f"  OKX持仓: {symbol} {pos_side} qty={pos_qty}")

    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()

    # 获取数据库中所有open状态持仓
    cursor.execute("SELECT id, symbol, side, quantity, price, filled_price FROM trade_records WHERE status='open'")
    db_open_positions = cursor.fetchall()
    print(f"\n数据库open持仓数量: {len(db_open_positions)}")

    # 标准化方向
    def normalize_side(side):
        side = (side or '').lower()
        if side in ('long', 'buy'):
            return 'long'
        elif side in ('short', 'sell'):
            return 'short'
        return side

    ghost_count = 0
    matched_count = 0
    for row in db_open_positions:
        trade_id, symbol, side, quantity, price, filled_price = row
        norm_side = normalize_side(side)
        key = f"{symbol}:{norm_side}"

        if key in okx_pos_keys:
            matched_count += 1
            # 更新数据库中的成交价（如果为空）
            if not filled_price and price:
                okx_pos = okx_pos_map[key]
                avg_px = float(okx_pos.get('avgPx', 0))
                if avg_px > 0:
                    cursor.execute(
                        "UPDATE trade_records SET filled_price=? WHERE id=?",
                        (avg_px, trade_id)
                    )
        else:
            # 数据库中存在但OKX不存在 -> 已平仓，标记为closed
            ghost_count += 1
            cursor.execute(
                "UPDATE trade_records SET status='closed', close_time=? WHERE id=?",
                (datetime.now().isoformat(), trade_id)
            )
            print(f"  幽灵持仓已标记closed: {trade_id} {symbol} {side}")

    conn.commit()
    print(f"\n匹配成功: {matched_count}")
    print(f"幽灵持仓已清理: {ghost_count}")

    # 再次验证
    cursor.execute("SELECT COUNT(*) FROM trade_records WHERE status='open'")
    remaining_open = cursor.fetchone()[0]
    print(f"清理后数据库open持仓数量: {remaining_open}")

    conn.close()
    return matched_count, ghost_count


def backfill_pnl_from_bills():
    """从OKX账单回填历史交易盈亏"""
    print("\n" + "=" * 80)
    print("=== 步骤2: 从OKX账单回填历史交易盈亏 ===")
    print("=" * 80)

    bills = get_okx_bills(limit=100)
    print(f"获取OKX账单数量: {len(bills)}")

    # 平仓类型的账单（type 5=平多, 6=平空）
    close_bills = [b for b in bills if b.get('type') in ('5', '6')]
    print(f"平仓账单数量: {len(close_bills)}")

    if not close_bills:
        print("无平仓账单可回填")
        return 0

    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()

    backfilled = 0
    btype_map = {'5': 'long', '6': 'short'}

    for bill in close_bills:
        symbol = bill.get('instId', '')
        btype = bill.get('type', '')
        side = btype_map.get(btype, '')
        pnl = float(bill.get('pnl', 0))
        fee = float(bill.get('fee', 0))
        fill_px = float(bill.get('fillPx', 0)) if bill.get('fillPx') else 0
        ts = bill.get('ts', '')
        close_time = datetime.fromtimestamp(int(ts)/1000).isoformat() if ts else None

        # 净盈亏 = 盈亏 - 手续费
        net_pnl = pnl - abs(fee)

        # 找到对应的open持仓记录（同币种同方向，按时间最早）
        cursor.execute(
            "SELECT id, price, quantity FROM trade_records WHERE symbol=? AND side=? AND status='closed' AND pnl=0 ORDER BY create_time ASC LIMIT 1",
            (symbol, side)
        )
        row = cursor.fetchone()
        if row:
            trade_id, entry_price, qty = row
            pnl_percent = net_pnl / (entry_price * qty) if entry_price and qty and entry_price * qty > 0 else 0

            cursor.execute(
                "UPDATE trade_records SET pnl=?, pnl_percent=?, filled_price=COALESCE(filled_price, ?), close_time=COALESCE(close_time, ?) WHERE id=?",
                (net_pnl, pnl_percent, fill_px, close_time, trade_id)
            )
            backfilled += 1
            print(f"  回填: {trade_id} {symbol} {side} pnl={net_pnl:.4f} ({pnl_percent*100:.2f}%)")

    conn.commit()
    conn.close()
    print(f"\n回填完成，共回填 {backfilled} 笔交易盈亏")
    return backfilled


def show_summary():
    """显示修复后的账单摘要"""
    print("\n" + "=" * 80)
    print("=== 修复后账单摘要 ===")
    print("=" * 80)

    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_count,
            SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END) as closed_count,
            SUM(pnl) as total_pnl,
            SUM(CASE WHEN pnl != 0 THEN 1 ELSE 0 END) as pnl_count
        FROM trade_records
    """)
    r = cursor.fetchone()
    print(f"总交易: {r[0]}, 持仓中: {r[1]}, 已平仓: {r[2]}")
    print(f"总盈亏: {r[3] or 0:.4f} USDT, 已记录盈亏的交易: {r[4]}")

    # 按策略统计盈亏
    cursor.execute("""
        SELECT strategy_name, COUNT(*), SUM(pnl), AVG(pnl),
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END)
        FROM trade_records WHERE status='closed'
        GROUP BY strategy_name
    """)
    print("\n已平仓交易按策略统计:")
    print(f"{'策略':<10} {'笔数':<6} {'总盈亏':<12} {'平均':<12} {'胜':<4} {'负':<4} {'胜率'}")
    for row in cursor.fetchall():
        wins = row[4] or 0
        total = row[1] or 1
        win_rate = wins / total * 100 if total > 0 else 0
        print(f"{row[0]:<10} {row[1]:<6} {row[2] or 0:<12.4f} {row[3] or 0:<12.4f} {wins:<4} {row[5] or 0:<4} {win_rate:.1f}%")

    # 当前持仓
    cursor.execute("SELECT symbol, side, filled_price, quantity, leverage FROM trade_records WHERE status='open' ORDER BY create_time DESC")
    open_pos = cursor.fetchall()
    print(f"\n当前持仓 ({len(open_pos)}个):")
    for p in open_pos:
        print(f"  {p[0]:<16} {p[1]:<6} @ {p[2] or 0:.4f} qty={p[3] or 0:.4f} lev={p[4]}x")

    conn.close()


if __name__ == '__main__':
    matched, ghost = sync_positions()
    backfilled = backfill_pnl_from_bills()
    show_summary()
    print("\n✅ 修复完成!")
