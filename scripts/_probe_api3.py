import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import load_config
from core.okx_client import OKXClient

k = OKXClient(load_config())

print('=== 1. API3 订单历史（filled，最近100条，过滤API3）===')
try:
    for o in (k.get_order_history(limit=100) or []):
        if 'API3' not in (o.get('instId') or ''):
            continue
        print(json.dumps({
            'instId': o.get('instId'), 'ordId': o.get('ordId'),
            'clOrdId': o.get('clOrdId'),
            'side': o.get('side'), 'posSide': o.get('posSide'),
            'sz': o.get('sz'), 'accFillSz': o.get('accFillSz'),
            'avgPx': o.get('avgPx'), 'pnl': o.get('pnl'),
            'fee': o.get('fee'), 'state': o.get('state'),
            'fillTime': o.get('fillTime'), 'tag': o.get('tag'),
            'reduceOnly': o.get('reduceOnly'), 'closeOrderAlgo': o.get('closeOrderAlgo'),
        }, ensure_ascii=False))
except Exception as e:
    print('ERR', e)

print('\n=== 2. API3 当前持仓 ===')
try:
    for p in (k.get_positions() or []):
        if 'API3' in (p.get('instId') or ''):
            print(json.dumps(p, ensure_ascii=False))
except Exception as e:
    print('ERR', e)

print('\n=== 3. API3 账单（type=2 交易，最近30条）===')
try:
    bills = k.get_all_bills_paginated(bill_type='2', max_pages=3) or []
    for b in bills:
        if 'API3' in (b.get('instId') or ''):
            print(json.dumps({
                'instId': b.get('instId'), 'subType': b.get('subType'),
                'side': b.get('side'), 'sz': b.get('sz'),
                'pnl': b.get('pnl'), 'fee': b.get('fee'),
                'fillPx': b.get('fillPx'), 'ts': b.get('ts'),
                'execType': b.get('execType'),
            }, ensure_ascii=False))
except Exception as e:
    print('ERR', e)
