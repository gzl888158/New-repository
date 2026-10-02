import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import load_config
from core.okx_client import OKXClient

k = OKXClient(load_config())

print("=== 1. 当前持仓 ===")
for p in (k.get_positions() or []):
    if float(p.get('pos', 0) or 0) != 0:
        print(json.dumps({
            'instId': p.get('instId'), 'posSide': p.get('posSide'),
            'pos': p.get('pos'), 'avgPx': p.get('avgPx'),
            'upl': p.get('upl'), 'uplRatio': p.get('uplRatio'),
            'lever': p.get('lever'), 'mgnMode': p.get('mgnMode'),
            'margin': p.get('margin'), 'liqPx': p.get('liqPx'),
            'cTime': p.get('cTime'), 'uTime': p.get('uTime'),
        }, ensure_ascii=False))

print("=== 2. 最近订单历史 ===")
for o in (k.get_order_history(limit=30) or []):
    print(json.dumps({
        'instId': o.get('instId'), 'side': o.get('side'),
        'posSide': o.get('posSide'), 'state': o.get('state'),
        'sz': o.get('sz'), 'avgPx': o.get('avgPx'),
        'pnl': o.get('pnl'), 'fee': o.get('fee'),
        'ordType': o.get('ordType'), 'clOrdId': o.get('clOrdId'),
        'tag': o.get('tag'), 'fillTime': o.get('fillTime'),
    }, ensure_ascii=False))

print("=== 3. 普通挂单 ===")
for o in (k.get_orders('SWAP') or []):
    print(json.dumps({
        'instId': o.get('instId'), 'side': o.get('side'),
        'posSide': o.get('posSide'), 'ordType': o.get('ordType'),
        'sz': o.get('sz'), 'px': o.get('px'), 'state': o.get('state'),
        'clOrdId': o.get('clOrdId'), 'tag': o.get('tag'), 'ordId': o.get('ordId'),
    }, ensure_ascii=False))

print("=== 4. 条件单(algo/止盈止损) ===")
for o in (k.get_algo_orders(ord_type='conditional') or []):
    print(json.dumps({
        'instId': o.get('instId'), 'algoId': o.get('algoId'),
        'ordType': o.get('ordType'), 'side': o.get('side'),
        'posSide': o.get('posSide'), 'sz': o.get('sz'),
        'tpTriggerPx': o.get('tpTriggerPx'), 'tpOrdPx': o.get('tpOrdPx'),
        'slTriggerPx': o.get('slTriggerPx'), 'slOrdPx': o.get('slOrdPx'),
        'state': o.get('state'),
    }, ensure_ascii=False))

print("=== 5. 账户 ===")
a = k.get_account_info() or {}
print(json.dumps({
    'totalEq': a.get('totalEq'), 'upl': a.get('upl'),
    'availBal': a.get('availBal'), 'mgnRatio': a.get('mgnRatio'),
}, ensure_ascii=False))
