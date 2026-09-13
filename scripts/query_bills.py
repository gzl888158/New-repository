"""查询交易账单"""
import sqlite3
from datetime import datetime

conn = sqlite3.connect('data/trading.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 查看所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print(f"=== 数据库表 ===\n{tables}\n")

# 1. 总体统计
print("=" * 80)
print("=== 交易账单总览 ===")
print("=" * 80)
cursor.execute("""
    SELECT
        COUNT(*) as total_trades,
        SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) as open_count,
        SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END) as closed_count,
        SUM(margin) as total_margin,
        SUM(pnl) as total_pnl,
        SUM(CASE WHEN pnl != 0 THEN 1 ELSE 0 END) as pnl_recorded
    FROM trade_records
""")
r = dict(cursor.fetchone())
print(f"总交易数: {r['total_trades']}")
print(f"  - 持仓中(open): {r['open_count']}")
print(f"  - 已平仓(closed): {r['closed_count']}")
print(f"总保证金: {r['total_margin']:.4f} USDT" if r['total_margin'] else "总保证金: 0")
print(f"总盈亏: {r['total_pnl']:.4f} USDT" if r['total_pnl'] else "总盈亏: 0")
print(f"已记录盈亏的交易数: {r['pnl_recorded']}")

# 2. 按策略统计
print("\n" + "=" * 80)
print("=== 按策略统计 ===")
print("=" * 80)
cursor.execute("""
    SELECT
        strategy_name,
        COUNT(*) as count,
        SUM(margin) as total_margin,
        SUM(pnl) as total_pnl,
        AVG(pnl) as avg_pnl,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses
    FROM trade_records
    GROUP BY strategy_name
    ORDER BY count DESC
""")
print(f"{'策略':<10} {'笔数':<6} {'总保证金':<12} {'总盈亏':<12} {'平均盈亏':<12} {'胜':<4} {'负':<4}")
print("-" * 80)
for row in cursor.fetchall():
    r = dict(row)
    win_rate = (r['wins'] / r['count'] * 100) if r['count'] > 0 and r['wins'] else 0
    print(f"{r['strategy_name']:<10} {r['count']:<6} {r['total_margin'] or 0:<12.4f} {r['total_pnl'] or 0:<12.4f} {r['avg_pnl'] or 0:<12.4f} {r['wins'] or 0:<4} {r['losses'] or 0:<4}")

# 3. 按币种统计
print("\n" + "=" * 80)
print("=== 按币种统计 ===")
print("=" * 80)
cursor.execute("""
    SELECT
        symbol,
        side,
        COUNT(*) as count,
        SUM(margin) as total_margin,
        SUM(pnl) as total_pnl,
        AVG(price) as avg_price
    FROM trade_records
    GROUP BY symbol, side
    ORDER BY symbol, side
""")
print(f"{'币种':<18} {'方向':<6} {'笔数':<6} {'总保证金':<12} {'总盈亏':<12} {'均价':<12}")
print("-" * 80)
for row in cursor.fetchall():
    r = dict(row)
    print(f"{r['symbol']:<18} {r['side']:<6} {r['count']:<6} {r['total_margin'] or 0:<12.4f} {r['total_pnl'] or 0:<12.4f} {r['avg_price'] or 0:<12.4f}")

# 4. 最近20笔交易明细
print("\n" + "=" * 80)
print("=== 最近20笔交易明细 ===")
print("=" * 80)
cursor.execute("""
    SELECT
        id,
        symbol,
        strategy_name,
        side,
        price,
        filled_price,
        quantity,
        margin,
        leverage,
        pnl,
        pnl_percent,
        status,
        create_time,
        close_time
    FROM trade_records
    ORDER BY create_time DESC
    LIMIT 20
""")
rows = cursor.fetchall()
print(f"{'时间':<21} {'币种':<16} {'策略':<8} {'方向':<6} {'价格':<10} {'成交价':<10} {'数量':<10} {'保证金':<10} {'杠杆':<4} {'盈亏':<10} {'状态':<8}")
print("-" * 140)
for row in rows:
    r = dict(row)
    create_time = (r['create_time'] or '')[:19]
    pnl_str = f"{r['pnl']:.4f}" if r['pnl'] else "0.0000"
    price_str = f"{r['price']:.4f}" if r['price'] else "0.0000"
    fill_str = f"{r['filled_price']:.4f}" if r['filled_price'] else "-"
    qty_str = f"{r['quantity']:.4f}" if r['quantity'] else "0.0000"
    margin_str = f"{r['margin']:.4f}" if r['margin'] else "0.0000"
    print(f"{create_time:<21} {r['symbol']:<16} {r['strategy_name']:<8} {r['side']:<6} {price_str:<10} {fill_str:<10} {qty_str:<10} {margin_str:<10} {r['leverage'] or 0:<4} {pnl_str:<10} {r['status']:<8}")

# 5. 持仓中的交易
print("\n" + "=" * 80)
print("=== 当前持仓中的交易 ===")
print("=" * 80)
cursor.execute("""
    SELECT
        id,
        symbol,
        strategy_name,
        side,
        price,
        filled_price,
        quantity,
        margin,
        leverage,
        create_time
    FROM trade_records
    WHERE status='open'
    ORDER BY create_time DESC
""")
open_positions = cursor.fetchall()
if open_positions:
    print(f"{'时间':<21} {'币种':<16} {'策略':<8} {'方向':<6} {'成交价':<10} {'数量':<10} {'保证金':<10} {'杠杆':<4}")
    print("-" * 120)
    for row in open_positions:
        r = dict(row)
        create_time = (r['create_time'] or '')[:19]
        fill_str = f"{r['filled_price']:.4f}" if r['filled_price'] else f"{r['price']:.4f}" if r['price'] else "-"
        qty_str = f"{r['quantity']:.4f}" if r['quantity'] else "0.0000"
        margin_str = f"{r['margin']:.4f}" if r['margin'] else "0.0000"
        print(f"{create_time:<21} {r['symbol']:<16} {r['strategy_name']:<8} {r['side']:<6} {fill_str:<10} {qty_str:<10} {margin_str:<10} {r['leverage'] or 0:<4}")
else:
    print("当前无持仓")

# 6. 时间分布
print("\n" + "=" * 80)
print("=== 交易时间分布 ===")
print("=" * 80)
cursor.execute("""
    SELECT
        DATE(create_time) as date,
        COUNT(*) as count,
        SUM(margin) as total_margin,
        SUM(pnl) as total_pnl
    FROM trade_records
    GROUP BY DATE(create_time)
    ORDER BY date DESC
""")
print(f"{'日期':<12} {'笔数':<6} {'总保证金':<12} {'总盈亏':<12}")
print("-" * 50)
for row in cursor.fetchall():
    r = dict(row)
    print(f"{r['date']:<12} {r['count']:<6} {r['total_margin'] or 0:<12.4f} {r['total_pnl'] or 0:<12.4f}")

conn.close()

# 7. 查询OKX实际账户余额和持仓
print("\n" + "=" * 80)
print("=== OKX实际账户状态 ===")
print("=" * 80)
try:
    import sys
    sys.path.insert(0, '.')
    import os
    import hmac
    import hashlib
    import base64
    import time
    import requests
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

    # 账户余额
    bal = okx_request('GET', '/api/v5/account/balance')
    if bal.get('code') == '0' and bal.get('data'):
        b = bal['data'][0]
        print(f"账户权益: {b.get('totalEq')} USDT")
        print(f"可用余额: {b.get('availBal')} USDT")
        print(f"已用保证金: {b.get('imr')} USDT")
        print(f"保证金余额: {b.get('margin')} USDT")
        print(f"未实现盈亏: {b.get('upl')} USDT")
        print(f"账户杠杆: {b.get('lever')}x")
        for detail in b.get('details', []):
            if float(detail.get('cashBal', 0)) > 0:
                print(f"  {detail.get('ccy')}: 余额={detail.get('cashBal')}, 可用={detail.get('availBal')}")

    # 当前持仓
    print("\n--- OKX实际持仓 ---")
    pos = okx_request('GET', '/api/v5/account/positions')
    if pos.get('code') == '0':
        positions = pos.get('data', [])
        if positions:
            print(f"{'币种':<18} {'方向':<6} {'数量':<12} {'均价':<12} {'标记价':<12} {'保证金':<12} {'未实现盈亏':<12} {'杠杆':<6} {'收益率'}")
            print("-" * 120)
            total_pnl = 0
            total_margin = 0
            for p in positions:
                symbol = p.get('instId', '')
                pos_side = p.get('posSide', '')
                pos_qty = float(p.get('pos', 0))
                avg_px = float(p.get('avgPx', 0))
                mark_px = float(p.get('markPx', 0))
                margin = float(p.get('margin', 0))
                upl = float(p.get('upl', 0))
                lever = p.get('lever', '')
                upl_ratio = float(p.get('uplRatio', 0)) * 100
                direction = '多' if pos_side == 'long' else '空' if pos_side == 'short' else pos_side
                print(f"{symbol:<18} {direction:<6} {pos_qty:<12.4f} {avg_px:<12.4f} {mark_px:<12.4f} {margin:<12.4f} {upl:<12.4f} {lever:<6} {upl_ratio:.2f}%")
                total_pnl += upl
                total_margin += margin
            print("-" * 120)
            print(f"总保证金: {total_margin:.4f} USDT, 总未实现盈亏: {total_pnl:.4f} USDT")
        else:
            print("当前无持仓")
    else:
        print(f"查询持仓失败: {pos.get('msg')}")

    # 最近成交记录
    print("\n--- OKX最近成交记录（最近5笔）---")
    bills = okx_request('GET', '/api/v5/account/bills?instType=SWAP&limit=5')
    if bills.get('code') == '0':
        bill_data = bills.get('data', [])
        if bill_data:
            print(f"{'时间':<21} {'币种':<16} {'类型':<8} {'方向':<6} {'数量':<10} {'价格':<10} {'盈亏':<10} {'费用':<10}")
            print("-" * 100)
            for b in bill_data:
                ts = b.get('ts', '')
                if ts:
                    t = datetime.fromtimestamp(int(ts)/1000).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    t = '-'
                symbol = b.get('instId', '')
                btype = b.get('type', '')
                btype_map = {'1': '买入', '2': '卖出', '3': '开多', '4': '开空', '5': '平多', '6': '平空'}
                btype_str = btype_map.get(btype, btype)
                pnl = float(b.get('pnl', 0))
                fee = float(b.get('fee', 0))
                px = float(b.get('fillPx', 0)) if b.get('fillPx') else float(b.get('px', 0))
                qty = float(b.get('fillSz', 0)) if b.get('fillSz') else float(b.get('sz', 0))
                side = b.get('posSide', '')
                print(f"{t:<21} {symbol:<16} {btype_str:<8} {side:<6} {qty:<10.4f} {px:<10.4f} {pnl:<10.4f} {fee:<10.4f}")
        else:
            print("无成交记录")
    else:
        print(f"查询账单失败: {bills.get('msg')}")

except Exception as e:
    print(f"查询OKX失败: {e}")
