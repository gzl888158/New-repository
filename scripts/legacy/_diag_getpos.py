# -*- coding: utf-8 -*-
import os, sys, json
sys.path.insert(0, r'e:\新建文件夹\okx_quant_trading')
os.chdir(r'e:\新建文件夹\okx_quant_trading')

from configs.settings import load_config
from core.okx_client import OKXClient

c = load_config()
k = OKXClient(c)

print("=== get_positions() 同步方法返回 ===")
ps = k.get_positions()
print(f"共 {len(ps)} 条")
for p in ps:
    qty = float(p.get("pos", "0") or 0)
    if qty != 0:
        print(f"  {p.get('instId'):20s} posSide={p.get('posSide'):6s} pos={p.get('pos')} avgPx={p.get('avgPx')} liqPx={p.get('liqPx')}")

print("\n=== _parse_position 结果 (margin>0) ===")
for p in ps:
    pos = k._parse_position(p)
    if pos and pos.margin > 0:
        print(f"  {pos.symbol:20s} side={pos.side:6s} qty={pos.quantity} avg_cost={pos.avg_cost} mark={pos.mark_price} margin={pos.margin} lever={pos.leverage} upl={pos.unrealized_pnl}")

k.close()
